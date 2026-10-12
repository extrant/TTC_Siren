using System.Collections.Concurrent;
using TTC.Siren.Search;

namespace TTC.Siren.Engine;

/// <summary>
/// Everything one search needs that is expensive to create and must not be shared: the Fast transposition table and the
/// exact-endgame memo. A context is owned by exactly one request at a time (it is leased from <see cref="WorkerPool"/>);
/// each request starts a new table generation bound to its own <see cref="SearchContext"/> (review H4-1), so nothing a
/// previous request stored can be read. The memo is passed explicitly to <c>EndgameRobust.Select</c> (review L6-4): the
/// engine never relies on the <c>[ThreadStatic]</c> fallback.
/// </summary>
public sealed class WorkerContext
{
    public WorkerContext(int id, int ttEntries, int memoEntries)
    {
        Id = id;
        Tt = new FastTranspositionTable(ttEntries);
        Memo = ExactEndgameSolver.CreateMemo(memoEntries);
    }

    public int Id { get; }
    public FastTranspositionTable Tt { get; }
    public FastTranspositionTable Memo { get; }
    public long RequestsServed { get; set; }

    /// <summary>The access class of the request that holds the context right now (set by the pool when it is handed out; the pool counts running searches by class).</summary>
    internal AccessClass Leased;
}

public enum BusyReason
{
    None,
    QueueFull,
    QueueTimeout,

    /// <summary>The caller's HTTP connection was aborted before a worker was handed over: nothing is searched for it.</summary>
    Abandoned,

    /// <summary>Anonymous requests are limited separately (<see cref="CapacityOptions.AnonymousQueueLimit"/>): too many of them are waiting.</summary>
    AnonymousQueueFull,

    /// <summary>An anonymous request waited longer than <see cref="CapacityOptions.AnonymousQueueWaitMs"/>.</summary>
    AnonymousQueueTimeout,
}

/// <summary>Who is asking (decision A-27). The order is the priority: a plugin request is served before a web request, and a web request before an anonymous one.</summary>
public enum AccessClass
{
    /// <summary>The game plugin, authenticated with an API key: real games, the highest priority, a share of the workers reserved for it.</summary>
    Plugin = 0,

    /// <summary>A browser logged in with a key (session cookie).</summary>
    Web = 1,

    /// <summary>Nobody logged in: read-only pages, and (8D) game coaching within tight limits. Lowest priority and a capped share of the workers.</summary>
    Anonymous = 2,
}

/// <summary>
/// How the workers are shared between the access classes (decision A-27). A <b>reserved</b> number of workers is never taken by web or anonymous work: a plugin
/// request finds a free worker whenever at most (workers - reserved) searches of the other classes are running, so it waits for nothing but its own class.
/// Anonymous work may use at most <see cref="AnonymousMaxWorkers"/> workers at a time and has its own, shorter queue.
/// </summary>
public sealed class CapacityOptions
{
    /// <summary>Workers that only the plugin may use (0: none). Must be less than the number of workers.</summary>
    public int PluginReservedWorkers { get; init; }

    /// <summary>The most workers anonymous requests may occupy at once (0: no limit other than the pool size).</summary>
    public int AnonymousMaxWorkers { get; init; }

    /// <summary>The most anonymous requests that may wait for a worker; more are answered "busy" (0: no limit other than the queue limit).</summary>
    public int AnonymousQueueLimit { get; init; }

    /// <summary>How long an anonymous request waits for a worker before it is answered "busy".</summary>
    public int AnonymousQueueWaitMs { get; init; } = 10_000;

    /// <summary>No sharing: one queue, every request is a plugin request (the engine host and the tests that do not care).</summary>
    public static readonly CapacityOptions None = new();

    /// <summary>The numbers for <paramref name="workers"/> workers from shares: the reserved share goes to the plugin, anonymous work gets at most its share (at least one worker).</summary>
    public static CapacityOptions FromShares(int workers, double pluginReservedShare, double anonymousMaxShare, int anonymousQueueLimit, int anonymousQueueWaitMs)
    {
        int reserved = workers <= 1 ? 0 : Math.Clamp((int)Math.Floor(workers * pluginReservedShare), 0, workers - 1);
        int anonymous = Math.Clamp((int)Math.Floor(workers * anonymousMaxShare), 1, Math.Max(1, workers - reserved));
        return new CapacityOptions { PluginReservedWorkers = reserved, AnonymousMaxWorkers = anonymous, AnonymousQueueLimit = anonymousQueueLimit, AnonymousQueueWaitMs = anonymousQueueWaitMs };
    }
}

/// <summary>
/// The estimated cost of a request, used to order the wait queue (review O8-1): a cheap request (a few empty cells, or the robust endgame) must not
/// wait behind a dozen openings that take seconds each.
/// </summary>
public static class RequestClass
{
    public const int Cheap = 0;
    public const int Medium = 1;
    public const int Expensive = 2;

    /// <summary>Openings and the first moves are expensive, mid-games medium, endgames cheap.</summary>
    public static int Estimate(int emptyCells, bool robustEndgame) =>
        robustEndgame || emptyCells <= 5 ? Cheap : emptyCells <= 7 ? Medium : Expensive;
}

/// <summary>
/// The outcome of waiting for a worker. <see cref="Grant"/> numbers the grants of the pool in the order in which workers were handed out
/// (1, 2, 3 ...; 0 when nothing was granted): it is the service order, which tests assert instead of measuring wall-clock times.
/// </summary>
public readonly record struct AcquireResult(WorkerContext? Worker, BusyReason Reason, double WaitedMs, long Grant = 0);

/// <summary>
/// A fixed number of worker contexts and a bounded wait queue: at most <c>workers</c> searches run at once, at most
/// <c>queueLimit</c> requests wait for a worker, and a request waits at most <c>queueWaitMs</c>. Contexts are created on
/// first use (a table is 96 MiB by default, so an idle engine stays small).
/// A queued request whose caller has gone away (<c>ct</c> cancelled: the HTTP connection was aborted) is dropped when its turn
/// comes, without a search (review M7-1). A search that is already running is never interrupted, and nothing is checked inside it.
///
/// <para><b>Scheduling inside a class</b> (review O8-1). A freed worker goes to the waiting request with the smallest EFFECTIVE cost class, ties broken by arrival
/// order (FIFO within a class). The effective class is the request's <see cref="RequestClass"/> lowered by one for every <c>agingMs</c> it has
/// waited, down to <see cref="RequestClass.Cheap"/>: so cheap requests overtake expensive ones, but an expensive request that has waited
/// <c>2 * agingMs</c> is as good as a cheap one and then goes strictly by arrival: it is overtaken only by requests that arrived before it.
/// The wait of a request is therefore bounded by <c>2 * agingMs</c> plus the time the requests that arrived earlier need (and by
/// <c>queueWaitMs</c>, after which it is answered "busy"). With every request of the same class the order is plain FIFO.</para>
///
/// <para><b>Scheduling between access classes</b> (decision A-27, Phase 8B). The classes are strictly ordered: plugin, then web, then anonymous; the cost order with aging
/// applies inside each. <see cref="CapacityOptions"/> reserves workers for the plugin and caps the workers anonymous work may hold, so that a flood of anonymous
/// requests can neither take the plugin's workers nor make a plugin request wait behind queued work: the only thing a plugin request can wait for is a search of its own
/// class that is already running (a running search is never interrupted). A request of a lower class whose class is at its limit stays queued while requests of a
/// class that is not at its limit are served past it.</para>
/// </summary>
public sealed class WorkerPool
{
    private sealed class Waiter
    {
        public required int Rank;
        public required AccessClass Access;
        public required long Seq;
        public required long ArrivedTimestamp;
        public long Grant;
        public readonly TaskCompletionSource Granted = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool IsGranted;
    }

    private readonly int _workers, _ttEntries, _memoEntries, _queueLimit, _queueWaitMs, _agingMs;
    private readonly CapacityOptions _capacity;
    private readonly object _lock = new();
    private readonly List<Waiter> _queue = new();
    private readonly ConcurrentQueue<WorkerContext> _idle = new();
    private readonly int[] _running = new int[3];
    private readonly int[] _queued = new int[3];
    private readonly long[] _served = new long[3];
    private readonly long[] _busy = new long[3];
    private int _freePermits;
    private int _created;
    private long _seq;
    private long _abandoned;
    private long _grants;
    private readonly TimeProvider _time;

    /// <param name="time">The clock that measures how long a request has waited (aging, <see cref="AcquireResult.WaitedMs"/>). Tests pass a logical clock, so the
    /// scheduling rules are asserted on the order of service and not on how fast the machine happens to be (review M8-2). The queue-wait timeout is real time.</param>
    /// <param name="capacity">How the workers are shared between the access classes; null: not at all.</param>
    public WorkerPool(int workers, int ttEntries, int memoEntries, int queueLimit, int queueWaitMs, int agingMs = 2000, TimeProvider? time = null, CapacityOptions? capacity = null)
    {
        if (agingMs < 1) throw new ArgumentOutOfRangeException(nameof(agingMs));
        _capacity = capacity ?? CapacityOptions.None;
        if (_capacity.PluginReservedWorkers < 0 || (workers > 1 && _capacity.PluginReservedWorkers >= workers) || (workers == 1 && _capacity.PluginReservedWorkers > 0))
            throw new ArgumentOutOfRangeException(nameof(capacity), "the reserved workers must be fewer than the workers");
        if (_capacity.AnonymousMaxWorkers < 0 || _capacity.AnonymousQueueLimit < 0 || _capacity.AnonymousQueueWaitMs < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _time = time ?? TimeProvider.System;
        _workers = workers;
        _ttEntries = ttEntries;
        _memoEntries = memoEntries;
        _queueLimit = queueLimit;
        _queueWaitMs = queueWaitMs;
        _agingMs = agingMs;
        _freePermits = workers;
    }

    public int Workers => _workers;
    public int QueueLimit => _queueLimit;
    public int QueueWaitMs => _queueWaitMs;
    public int AgingMs => _agingMs;
    public CapacityOptions Capacity => _capacity;
    public int Created => Volatile.Read(ref _created);

    public int Waiting
    {
        get
        {
            lock (_lock) return _queue.Count;
        }
    }

    /// <summary>Requests of the class that are waiting for a worker.</summary>
    public int WaitingOf(AccessClass access)
    {
        lock (_lock) return _queued[(int)access];
    }

    /// <summary>Searches of the class that are running (or whose worker has just been granted).</summary>
    public int RunningOf(AccessClass access)
    {
        lock (_lock) return _running[(int)access];
    }

    /// <summary>Contexts not currently leased (including those not created yet).</summary>
    public int Idle
    {
        get
        {
            lock (_lock) return _freePermits;
        }
    }

    public long BusyAnswers
    {
        get
        {
            lock (_lock) return _busy[0] + _busy[1] + _busy[2];
        }
    }

    /// <summary>"Busy" answers given to the class.</summary>
    public long BusyAnswersOf(AccessClass access)
    {
        lock (_lock) return _busy[(int)access];
    }

    /// <summary>Requests dropped without a search because their caller had already gone away.</summary>
    public long AbandonedSkipped => Interlocked.Read(ref _abandoned);

    /// <summary>Workers handed back after a request (searches that ran, whatever their outcome).</summary>
    public long Served
    {
        get
        {
            lock (_lock) return _served[0] + _served[1] + _served[2];
        }
    }

    public long ServedOf(AccessClass access)
    {
        lock (_lock) return _served[(int)access];
    }

    public long MemoryBytes => (long)Created * (_ttEntries + (long)_memoEntries) * FastTranspositionTable.EntryBytes;

    /// <summary>Leases a worker with the default class (<see cref="RequestClass.Medium"/>): plain FIFO among equals.</summary>
    public async Task<(WorkerContext? Worker, BusyReason Reason)> AcquireAsync(CancellationToken ct)
    {
        var r = await AcquireAsync(ct, RequestClass.Medium).ConfigureAwait(false);
        return (r.Worker, r.Reason);
    }

    /// <summary>
    /// Whether a request of the class may start now: a worker is free and the class limits allow it. The plugin may use every worker; the other classes
    /// leave the reserved ones alone, and anonymous work also stays within its own maximum.
    /// </summary>
    private bool Admissible(AccessClass access)
    {
        if (_freePermits <= 0) return false;
        if (access == AccessClass.Plugin) return true;
        int others = _running[(int)AccessClass.Web] + _running[(int)AccessClass.Anonymous];
        if (others >= _workers - _capacity.PluginReservedWorkers) return false;
        if (access == AccessClass.Anonymous && _capacity.AnonymousMaxWorkers > 0 && _running[(int)AccessClass.Anonymous] >= _capacity.AnonymousMaxWorkers) return false;
        return true;
    }

    private long Start(AccessClass access)
    {
        _freePermits--;
        _running[(int)access]++;
        return ++_grants;
    }

    /// <summary>
    /// Leases a worker, waiting in the bounded queue if all are busy. The worker is <c>null</c> + reason when it answers "busy", or when the
    /// caller has gone away (<see cref="BusyReason.Abandoned"/>: <paramref name="ct"/> was cancelled while queued or just as the
    /// worker became free; the worker is handed on untouched and nothing is searched). <see cref="AcquireResult.WaitedMs"/> is the time spent queued.
    /// </summary>
    public Task<AcquireResult> AcquireAsync(CancellationToken ct, int requestClass) => AcquireAsync(ct, requestClass, AccessClass.Plugin);

    public async Task<AcquireResult> AcquireAsync(CancellationToken ct, int requestClass, AccessClass access)
    {
        long arrived = _time.GetTimestamp();
        Waiter? waiter = null;
        long immediateGrant = 0;
        int waitMs = access == AccessClass.Anonymous ? Math.Min(_queueWaitMs, _capacity.AnonymousQueueWaitMs) : _queueWaitMs;
        lock (_lock)
        {
            if (Admissible(access))
            {
                immediateGrant = Start(access);
            }
            else if (_queue.Count >= _queueLimit)
            {
                _busy[(int)access]++;
                return new AcquireResult(null, BusyReason.QueueFull, 0);
            }
            else if (access == AccessClass.Anonymous && _capacity.AnonymousQueueLimit > 0 && _queued[(int)AccessClass.Anonymous] >= _capacity.AnonymousQueueLimit)
            {
                _busy[(int)access]++;
                return new AcquireResult(null, BusyReason.AnonymousQueueFull, 0);
            }
            else
            {
                waiter = new Waiter { Rank = Math.Clamp(requestClass, RequestClass.Cheap, RequestClass.Expensive), Access = access, Seq = ++_seq, ArrivedTimestamp = arrived };
                _queue.Add(waiter);
                _queued[(int)access]++;
            }
        }

        if (waiter is not null)
        {
            try
            {
                await waiter.Granted.Task.WaitAsync(TimeSpan.FromMilliseconds(waitMs), ct).ConfigureAwait(false);
            }
            catch (Exception e) when (e is OperationCanceledException or TimeoutException)
            {
                lock (_lock)
                {
                    if (!waiter.IsGranted)
                    {
                        _queue.Remove(waiter);
                        _queued[(int)access]--;
                        if (e is TimeoutException)
                        {
                            _busy[(int)access]++;
                            return new AcquireResult(null, access == AccessClass.Anonymous ? BusyReason.AnonymousQueueTimeout : BusyReason.QueueTimeout, _time.GetElapsedTime(arrived).TotalMilliseconds);
                        }
                        Interlocked.Increment(ref _abandoned);
                        return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
                    }
                }
                // granted at the very moment the wait ended: a timed-out caller simply takes the worker; a cancelled one gives it back
                if (e is OperationCanceledException)
                {
                    Interlocked.Increment(ref _abandoned);
                    GiveBack(access);
                    return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
                }
            }
        }

        if (ct.IsCancellationRequested)
        {
            // the caller left while it waited and a worker has just become free (or was free all along): give the permit back
            Interlocked.Increment(ref _abandoned);
            GiveBack(access);
            return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
        }

        double waited = _time.GetElapsedTime(arrived).TotalMilliseconds;
        long grant = waiter?.Grant ?? immediateGrant;
        if (!_idle.TryDequeue(out var worker))
        {
            int id = Interlocked.Increment(ref _created) - 1; // the permit guarantees id < workers
            try
            {
                worker = new WorkerContext(id, _ttEntries, _memoEntries);
            }
            catch
            {
                Interlocked.Decrement(ref _created);
                GiveBack(access); // do not leak the permit if the tables cannot be allocated
                throw;
            }
        }
        worker.Leased = access;
        return new AcquireResult(worker, BusyReason.None, waited, grant);
    }

    public void Release(WorkerContext worker)
    {
        worker.RequestsServed++;
        var access = worker.Leased;
        lock (_lock) _served[(int)access]++;
        _idle.Enqueue(worker);
        GiveBack(access);
    }

    /// <summary>Returns a permit of a class: the freed capacity goes to the best waiting requests that the class limits now allow, or becomes free.</summary>
    private void GiveBack(AccessClass access)
    {
        lock (_lock)
        {
            _running[(int)access]--;
            _freePermits++;
            Dispatch();
        }
    }

    /// <summary>Grants workers to waiting requests while one can start: the highest class first, inside it the best effective cost class, then the earliest arrival.</summary>
    private void Dispatch()
    {
        while (_queue.Count > 0)
        {
            long now = _time.GetTimestamp();
            Waiter? best = null;
            int bestRank = 0;
            foreach (var w in _queue)
            {
                if (!Admissible(w.Access)) continue;
                int rank = Effective(w, now);
                if (best is null || w.Access < best.Access || (w.Access == best.Access && (rank < bestRank || (rank == bestRank && w.Seq < best.Seq))))
                {
                    best = w;
                    bestRank = rank;
                }
            }
            if (best is null) return;
            _queue.Remove(best);
            _queued[(int)best.Access]--;
            best.IsGranted = true;
            best.Grant = Start(best.Access);
            best.Granted.SetResult();
        }
    }

    private int Effective(Waiter w, long now)
    {
        long waitedMs = (long)_time.GetElapsedTime(w.ArrivedTimestamp, now).TotalMilliseconds;
        return Math.Max(RequestClass.Cheap, w.Rank - (int)Math.Min(waitedMs / _agingMs, RequestClass.Expensive));
    }
}
