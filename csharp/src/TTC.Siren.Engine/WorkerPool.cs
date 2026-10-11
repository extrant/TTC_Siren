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
}

public enum BusyReason
{
    None,
    QueueFull,
    QueueTimeout,

    /// <summary>The caller's HTTP connection was aborted before a worker was handed over: nothing is searched for it.</summary>
    Abandoned,
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
/// Scheduling (review O8-1). A freed worker goes to the waiting request with the smallest EFFECTIVE class, ties broken by arrival order
/// (FIFO within a class). The effective class is the request's <see cref="RequestClass"/> lowered by one for every <c>agingMs</c> it has
/// waited, down to <see cref="RequestClass.Cheap"/>: so cheap requests overtake expensive ones, but an expensive request that has waited
/// <c>2 * agingMs</c> is as good as a cheap one and then goes strictly by arrival: it is overtaken only by requests that arrived before it.
/// The wait of a request is therefore bounded by <c>2 * agingMs</c> plus the time the requests that arrived earlier need (and by
/// <c>queueWaitMs</c>, after which it is answered "busy"). With every request of the same class the order is plain FIFO.
/// </summary>
public sealed class WorkerPool
{
    private sealed class Waiter
    {
        public required int Rank;
        public required long Seq;
        public required long ArrivedTimestamp;
        public long Grant;
        public readonly TaskCompletionSource Granted = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool IsGranted;
    }

    private readonly int _workers, _ttEntries, _memoEntries, _queueLimit, _queueWaitMs, _agingMs;
    private readonly object _lock = new();
    private readonly List<Waiter> _queue = new();
    private readonly ConcurrentQueue<WorkerContext> _idle = new();
    private int _freePermits;
    private int _created;
    private long _seq;
    private long _busyAnswers;
    private long _abandoned;
    private long _served;
    private long _grants;
    private readonly TimeProvider _time;

    /// <param name="time">The clock that measures how long a request has waited (aging, <see cref="AcquireResult.WaitedMs"/>). Tests pass a logical clock, so the
    /// scheduling rules are asserted on the order of service and not on how fast the machine happens to be (review M8-2). The queue-wait timeout is real time.</param>
    public WorkerPool(int workers, int ttEntries, int memoEntries, int queueLimit, int queueWaitMs, int agingMs = 2000, TimeProvider? time = null)
    {
        if (agingMs < 1) throw new ArgumentOutOfRangeException(nameof(agingMs));
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
    public int Created => Volatile.Read(ref _created);

    public int Waiting
    {
        get
        {
            lock (_lock) return _queue.Count;
        }
    }

    /// <summary>Contexts not currently leased (including those not created yet).</summary>
    public int Idle
    {
        get
        {
            lock (_lock) return _freePermits;
        }
    }

    public long BusyAnswers => Interlocked.Read(ref _busyAnswers);

    /// <summary>Requests dropped without a search because their caller had already gone away.</summary>
    public long AbandonedSkipped => Interlocked.Read(ref _abandoned);

    /// <summary>Workers handed back after a request (searches that ran, whatever their outcome).</summary>
    public long Served => Interlocked.Read(ref _served);

    public long MemoryBytes => (long)Created * (_ttEntries + (long)_memoEntries) * FastTranspositionTable.EntryBytes;

    /// <summary>Leases a worker with the default class (<see cref="RequestClass.Medium"/>): plain FIFO among equals.</summary>
    public async Task<(WorkerContext? Worker, BusyReason Reason)> AcquireAsync(CancellationToken ct)
    {
        var r = await AcquireAsync(ct, RequestClass.Medium).ConfigureAwait(false);
        return (r.Worker, r.Reason);
    }

    /// <summary>
    /// Leases a worker, waiting in the bounded queue if all are busy. The worker is <c>null</c> + reason when it answers "busy", or when the
    /// caller has gone away (<see cref="BusyReason.Abandoned"/>: <paramref name="ct"/> was cancelled while queued or just as the
    /// worker became free; the worker is handed on untouched and nothing is searched). <see cref="AcquireResult.WaitedMs"/> is the time spent queued.
    /// </summary>
    public async Task<AcquireResult> AcquireAsync(CancellationToken ct, int requestClass)
    {
        long arrived = _time.GetTimestamp();
        Waiter? waiter = null;
        long immediateGrant = 0;
        lock (_lock)
        {
            if (_freePermits > 0)
            {
                _freePermits--;
                immediateGrant = ++_grants;
            }
            else if (_queue.Count >= _queueLimit)
            {
                Interlocked.Increment(ref _busyAnswers);
                return new AcquireResult(null, BusyReason.QueueFull, 0);
            }
            else
            {
                waiter = new Waiter { Rank = Math.Clamp(requestClass, RequestClass.Cheap, RequestClass.Expensive), Seq = ++_seq, ArrivedTimestamp = arrived };
                _queue.Add(waiter);
            }
        }

        if (waiter is not null)
        {
            try
            {
                await waiter.Granted.Task.WaitAsync(TimeSpan.FromMilliseconds(_queueWaitMs), ct).ConfigureAwait(false);
            }
            catch (Exception e) when (e is OperationCanceledException or TimeoutException)
            {
                lock (_lock)
                {
                    if (!waiter.IsGranted)
                    {
                        _queue.Remove(waiter);
                        if (e is TimeoutException)
                        {
                            Interlocked.Increment(ref _busyAnswers);
                            return new AcquireResult(null, BusyReason.QueueTimeout, _time.GetElapsedTime(arrived).TotalMilliseconds);
                        }
                        Interlocked.Increment(ref _abandoned);
                        return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
                    }
                }
                // granted at the very moment the wait ended: a timed-out caller simply takes the worker; a cancelled one gives it back
                if (e is OperationCanceledException)
                {
                    Interlocked.Increment(ref _abandoned);
                    GiveBack();
                    return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
                }
            }
        }

        if (ct.IsCancellationRequested)
        {
            // the caller left while it waited and a worker has just become free (or was free all along): give the permit back
            Interlocked.Increment(ref _abandoned);
            GiveBack();
            return new AcquireResult(null, BusyReason.Abandoned, _time.GetElapsedTime(arrived).TotalMilliseconds);
        }

        double waited = _time.GetElapsedTime(arrived).TotalMilliseconds;
        long grant = waiter?.Grant ?? immediateGrant;
        if (_idle.TryDequeue(out var worker)) return new AcquireResult(worker, BusyReason.None, waited, grant);
        int id = Interlocked.Increment(ref _created) - 1; // the permit guarantees id < workers
        try
        {
            return new AcquireResult(new WorkerContext(id, _ttEntries, _memoEntries), BusyReason.None, waited, grant);
        }
        catch
        {
            Interlocked.Decrement(ref _created);
            GiveBack(); // do not leak the permit if the tables cannot be allocated
            throw;
        }
    }

    public void Release(WorkerContext worker)
    {
        worker.RequestsServed++;
        Interlocked.Increment(ref _served);
        _idle.Enqueue(worker);
        GiveBack();
    }

    /// <summary>Returns a permit: it goes to the best waiting request, or becomes free.</summary>
    private void GiveBack()
    {
        lock (_lock)
        {
            if (_queue.Count == 0)
            {
                _freePermits++;
                return;
            }
            long now = _time.GetTimestamp();
            Waiter best = _queue[0];
            int bestRank = Effective(best, now);
            for (int i = 1; i < _queue.Count; i++)
            {
                var w = _queue[i];
                int rank = Effective(w, now);
                if (rank < bestRank || (rank == bestRank && w.Seq < best.Seq))
                {
                    best = w;
                    bestRank = rank;
                }
            }
            _queue.Remove(best);
            best.IsGranted = true;
            best.Grant = ++_grants;
            best.Granted.SetResult();
        }
    }

    private int Effective(Waiter w, long now)
    {
        long waitedMs = (long)_time.GetElapsedTime(w.ArrivedTimestamp, now).TotalMilliseconds;
        return Math.Max(RequestClass.Cheap, w.Rank - (int)Math.Min(waitedMs / _agingMs, RequestClass.Expensive));
    }
}
