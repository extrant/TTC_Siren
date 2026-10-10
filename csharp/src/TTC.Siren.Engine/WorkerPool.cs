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
/// A fixed number of worker contexts and a bounded wait queue: at most <c>workers</c> searches run at once, at most
/// <c>queueLimit</c> requests wait for a worker, and a request waits at most <c>queueWaitMs</c>. Contexts are created on
/// first use (a table is 96 MiB by default, so an idle engine stays small).
/// A queued request whose caller has gone away (<c>ct</c> cancelled: the HTTP connection was aborted) is dropped when its turn
/// comes, without a search (review M7-1). A search that is already running is never interrupted, and nothing is checked inside it.
/// </summary>
public sealed class WorkerPool
{
    private readonly int _workers, _ttEntries, _memoEntries, _queueLimit, _queueWaitMs;
    private readonly SemaphoreSlim _free;
    private readonly ConcurrentQueue<WorkerContext> _idle = new();
    private int _created;
    private int _waiting;
    private long _busyAnswers;
    private long _abandoned;
    private long _served;

    public WorkerPool(int workers, int ttEntries, int memoEntries, int queueLimit, int queueWaitMs)
    {
        _workers = workers;
        _ttEntries = ttEntries;
        _memoEntries = memoEntries;
        _queueLimit = queueLimit;
        _queueWaitMs = queueWaitMs;
        _free = new SemaphoreSlim(workers, workers);
    }

    public int Workers => _workers;
    public int QueueLimit => _queueLimit;
    public int QueueWaitMs => _queueWaitMs;
    public int Created => Volatile.Read(ref _created);
    public int Waiting => Volatile.Read(ref _waiting);

    /// <summary>Contexts not currently leased (including those not created yet).</summary>
    public int Idle => _free.CurrentCount;

    public long BusyAnswers => Interlocked.Read(ref _busyAnswers);

    /// <summary>Requests dropped without a search because their caller had already gone away.</summary>
    public long AbandonedSkipped => Interlocked.Read(ref _abandoned);

    /// <summary>Workers handed back after a request (searches that ran, whatever their outcome).</summary>
    public long Served => Interlocked.Read(ref _served);

    public long MemoryBytes => (long)Created * (_ttEntries + (long)_memoEntries) * FastTranspositionTable.EntryBytes;

    /// <summary>
    /// Leases a worker, waiting in the bounded queue if all are busy. <c>null</c> + reason when it answers "busy", or when the
    /// caller has gone away (<see cref="BusyReason.Abandoned"/>: <paramref name="ct"/> was cancelled while queued or just as the
    /// worker became free; the worker is handed on untouched and nothing is searched).
    /// </summary>
    public async Task<(WorkerContext? Worker, BusyReason Reason)> AcquireAsync(CancellationToken ct)
    {
        if (!_free.Wait(0))
        {
            if (Interlocked.Increment(ref _waiting) > _queueLimit)
            {
                Interlocked.Decrement(ref _waiting);
                Interlocked.Increment(ref _busyAnswers);
                return (null, BusyReason.QueueFull);
            }
            bool got;
            try
            {
                got = await _free.WaitAsync(_queueWaitMs, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                Interlocked.Increment(ref _abandoned);
                return (null, BusyReason.Abandoned);
            }
            finally
            {
                Interlocked.Decrement(ref _waiting);
            }
            if (!got)
            {
                Interlocked.Increment(ref _busyAnswers);
                return (null, BusyReason.QueueTimeout);
            }
        }

        if (ct.IsCancellationRequested)
        {
            // the caller left while it waited and a worker has just become free (or was free all along): give the permit back
            Interlocked.Increment(ref _abandoned);
            _free.Release();
            return (null, BusyReason.Abandoned);
        }

        if (_idle.TryDequeue(out var worker)) return (worker, BusyReason.None);
        int id = Interlocked.Increment(ref _created) - 1; // the semaphore guarantees id < workers
        try
        {
            return (new WorkerContext(id, _ttEntries, _memoEntries), BusyReason.None);
        }
        catch
        {
            Interlocked.Decrement(ref _created);
            _free.Release(); // do not leak the permit if the tables cannot be allocated
            throw;
        }
    }

    public void Release(WorkerContext worker)
    {
        worker.RequestsServed++;
        Interlocked.Increment(ref _served);
        _idle.Enqueue(worker);
        _free.Release();
    }
}
