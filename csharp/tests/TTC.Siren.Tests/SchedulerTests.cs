using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Engine;
using TTC.Siren.Server;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// A clock that moves only when a test moves it (one tick = one millisecond). With it the aging of the wait queue is exercised on logical time:
/// what a test asserts is the ORDER in which requests are served, never how long anything took on the machine that runs the test (review M8-2).
/// </summary>
internal sealed class LogicalClock : TimeProvider
{
    private long _ms;

    public override long TimestampFrequency => 1000;

    public override long GetTimestamp() => Interlocked.Read(ref _ms);

    public void Advance(long milliseconds) => Interlocked.Add(ref _ms, milliseconds);
}

/// <summary>
/// Review O8-1: the wait queue is ordered by the estimated cost of a request, with aging. Cheap requests (endgames, the robust endgame) overtake
/// openings; an opening that has waited long enough is as good as a cheap request and then goes by arrival, so it cannot be overtaken for ever.
/// The first tests pin the rules on the pool alone, on a logical clock; the last reproduces the review's probe (one worker, twelve openings, then
/// five cheap requests) through the real service, with the order of service read from the pool's grant numbers.
/// Nothing here depends on how fast the machine is: a request is queued by a call that has registered it before it returns, the aging clock is
/// logical, and each assertion is about who was served before whom.
/// </summary>
public sealed class SchedulerTests
{
    private readonly ITestOutputHelper _output;

    public SchedulerTests(ITestOutputHelper output) => _output = output;

    private static WorkerPool Pool(LogicalClock clock, int agingMs = 60_000, int queue = 64) =>
        new(workers: 1, ttEntries: 1 << 8, memoEntries: 1 << 8, queueLimit: queue, queueWaitMs: 60_000, agingMs, clock);

    /// <summary>
    /// Queues a request. <c>AcquireAsync</c> runs up to its first wait synchronously, so when this returns the request is in the queue
    /// (<see cref="WorkerPool.Waiting"/> has grown): arrival order is exactly the order of these calls.
    /// </summary>
    private static Task<AcquireResult> Queue(WorkerPool pool, int rank)
    {
        int before = pool.Waiting;
        var task = pool.AcquireAsync(CancellationToken.None, rank);
        Assert.Equal(before + 1, pool.Waiting);
        return task;
    }

    /// <summary>Serves every pending request, one after the other, and returns the names in the order of service.</summary>
    private static async Task<List<string>> Drain(WorkerPool pool, Dictionary<string, Task<AcquireResult>> pending)
    {
        var served = new List<string>();
        while (pending.Count > 0)
        {
            var all = pending.ToDictionary(kv => kv.Value, kv => kv.Key);
            var done = await Task.WhenAny(pending.Values);
            string name = all[done];
            pending.Remove(name);
            served.Add(name);
            pool.Release((await done).Worker!);
        }
        return served;
    }

    [Fact]
    public async Task Requests_of_one_class_are_served_first_come_first_served()
    {
        var pool = Pool(new LogicalClock());
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Medium)).Worker!;
        var pending = new Dictionary<string, Task<AcquireResult>>();
        for (int i = 0; i < 6; i++) pending["e" + i] = Queue(pool, RequestClass.Expensive);
        pool.Release(running);
        Assert.Equal(new[] { "e0", "e1", "e2", "e3", "e4", "e5" }, await Drain(pool, pending));
    }

    [Fact]
    public async Task Grant_numbers_are_the_order_of_service()
    {
        var pool = Pool(new LogicalClock());
        var first = await pool.AcquireAsync(CancellationToken.None, RequestClass.Medium);
        Assert.Equal(1, first.Grant);
        var a = Queue(pool, RequestClass.Expensive);
        var b = Queue(pool, RequestClass.Cheap);
        pool.Release(first.Worker!);
        var order = await Drain(pool, new Dictionary<string, Task<AcquireResult>> { ["a"] = a, ["b"] = b });
        Assert.Equal(new[] { "b", "a" }, order);
        Assert.Equal(2, (await b).Grant);
        Assert.Equal(3, (await a).Grant);
    }

    [Fact]
    public async Task A_cheap_request_overtakes_the_openings_that_were_queued_before_it()
    {
        var pool = Pool(new LogicalClock());
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive)).Worker!;
        var pending = new Dictionary<string, Task<AcquireResult>>();
        for (int i = 0; i < 4; i++) pending["opening" + i] = Queue(pool, RequestClass.Expensive);
        pending["cheap-a"] = Queue(pool, RequestClass.Cheap);
        pending["medium"] = Queue(pool, RequestClass.Medium);
        pending["cheap-b"] = Queue(pool, RequestClass.Cheap);
        pool.Release(running);
        // cheap before medium before the openings, each class first come first served
        Assert.Equal(new[] { "cheap-a", "cheap-b", "medium", "opening0", "opening1", "opening2", "opening3" }, await Drain(pool, pending));
    }

    [Theory]
    [InlineData(0, "late-cheap")]
    [InlineData(99, "late-cheap")]
    [InlineData(100, "late-cheap")] // one aging period: the opening counts as medium, which a cheap request still overtakes
    [InlineData(199, "late-cheap")]
    [InlineData(200, "old-opening")] // two aging periods: as good as a cheap request, and it arrived first
    [InlineData(250, "old-opening")]
    [InlineData(5000, "old-opening")]
    public async Task An_opening_is_overtaken_by_a_later_cheap_request_only_until_it_has_waited_two_aging_periods(int waitedMs, string firstServed)
    {
        var clock = new LogicalClock();
        var pool = Pool(clock, agingMs: 100);
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive)).Worker!;
        var pending = new Dictionary<string, Task<AcquireResult>> { ["old-opening"] = Queue(pool, RequestClass.Expensive) };
        clock.Advance(waitedMs);
        pending["late-cheap"] = Queue(pool, RequestClass.Cheap);
        pool.Release(running);
        var order = await Drain(pool, pending);
        Assert.Equal(firstServed, order[0]);
        Assert.Equal(2, order.Count);
    }

    [Theory]
    [InlineData(99, "late-cheap")]
    [InlineData(100, "medium")] // one aging period: a medium request has become cheap, and it arrived first
    [InlineData(101, "medium")]
    public async Task A_medium_request_ages_into_the_cheap_class_after_one_aging_period(int waitedMs, string firstServed)
    {
        var clock = new LogicalClock();
        var pool = Pool(clock, agingMs: 100);
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive)).Worker!;
        var pending = new Dictionary<string, Task<AcquireResult>> { ["medium"] = Queue(pool, RequestClass.Medium) };
        clock.Advance(waitedMs);
        pending["late-cheap"] = Queue(pool, RequestClass.Cheap);
        pool.Release(running);
        Assert.Equal(firstServed, (await Drain(pool, pending))[0]);
    }

    [Fact]
    public async Task Under_a_constant_stream_of_cheap_requests_an_opening_is_overtaken_for_exactly_two_aging_periods()
    {
        const int aging = 100, step = 10;
        var clock = new LogicalClock();
        var pool = Pool(clock, agingMs: aging, queue: 1000);
        var holder = await pool.AcquireAsync(CancellationToken.None, RequestClass.Medium);
        var opening = Queue(pool, RequestClass.Expensive);

        // every 10 logical ms one cheap request arrives and the worker that is running finishes: the stream alone would keep the single worker busy for ever
        int cheapServed = 0;
        var worker = holder.Worker!;
        Task<AcquireResult> cheap;
        while (true)
        {
            cheap = Queue(pool, RequestClass.Cheap);
            clock.Advance(step);
            pool.Release(worker);
            var winner = await Task.WhenAny(opening, cheap);
            if (ReferenceEquals(winner, opening)) break;
            cheapServed++;
            worker = (await cheap).Worker!;
        }
        // releases happen at 10, 20, ... ms: the ones at 10 ... 190 serve a cheap request, the one at 200 = 2 * aging serves the opening
        var served = await opening;
        _output.WriteLine($"the opening was served after {cheapServed} cheap requests, having waited {served.WaitedMs} logical ms (aging {aging} ms)");
        Assert.Equal(2 * aging / step - 1, cheapServed);
        Assert.Equal(2 * aging, served.WaitedMs);
        pool.Release(served.Worker!);
        pool.Release((await cheap).Worker!); // the cheap request that arrived with it comes next
    }

    // =============================================================================================================== the probe of the review

    private static JsonElement Opening(string tag, int maxTimeMs)
    {
        var node = JsonNode.Parse("""
            {"board":[],"myHand":[{"numU":4,"numL":2,"numD":3,"numR":4},{"numU":5,"numL":4,"numD":6,"numR":5},{"numU":2,"numL":2,"numD":7,"numR":2},
            {"numU":4,"numL":3,"numD":3,"numR":3},{"numU":1,"numL":3,"numD":7,"numR":2}],
            "oppHand":[{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0}],
            "myOwner":2,"currentPlayer":2,"rules":"加算,同数","max_depth":10}
            """)!.AsObject();
        node["max_time"] = maxTimeMs / 1000.0;
        node["npcName"] = tag;
        return JsonDocument.Parse(node.ToJsonString()).RootElement.Clone();
    }

    private static async Task WaitFor(Func<bool> condition, string what)
    {
        // a hang detector, not a performance assertion: the condition becomes true as soon as the thread pool has run the request up to the queue
        var deadline = DateTime.UtcNow.AddSeconds(120);
        while (!condition())
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("never happened: " + what);
            await Task.Delay(2);
        }
    }

    /// <summary>
    /// One worker is held by a blocker (it stands for the opening that was already running), 12 openings and then the 5 cheap requests are queued
    /// in that arrival order, and the blocker is released. The order of service is returned (the labels sorted by the grant number of their request).
    /// </summary>
    private static async Task<List<string>> Probe(bool priority, List<JsonElement> cheap)
    {
        var service = AiMoveParityTests.CreateService(out var pool, workers: 1, time: new LogicalClock()); // a frozen clock: no aging, only the classes
        service.PriorityScheduling = priority;
        var blocker = await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive);
        Assert.Equal(0, pool.Idle);

        var outcomes = new ConcurrentDictionary<string, AiMoveOutcome>();
        var tasks = new List<Task>();
        int queued = 0;
        async Task Submit(string label, JsonElement request, ulong seed)
        {
            tasks.Add(Task.Run(async () => outcomes[label] = await service.HandleAsync(request, seed, CancellationToken.None)));
            queued++;
            await WaitFor(() => pool.Waiting == queued, label + " reaching the queue");
        }
        for (int i = 0; i < 12; i++) await Submit("opening-" + i, Opening("opening-" + i, 200), (ulong)(100 + i));
        for (int i = 0; i < cheap.Count; i++) await Submit("cheap-" + i, cheap[i], (ulong)(500 + i));

        pool.Release(blocker.Worker!);
        await Task.WhenAll(tasks);
        Assert.All(outcomes.Values, o => Assert.Equal(200, o.Status));
        Assert.Equal(12 + cheap.Count, outcomes.Count);
        return outcomes.OrderBy(kv => kv.Value.PoolGrant).Select(kv => kv.Key).ToList();
    }

    [Fact]
    public async Task The_probe_of_the_review_one_worker_twelve_openings_five_cheap_requests()
    {
        // five endgame requests from the corpus (robust endgame / few empty cells)
        var cheap = new List<JsonElement>();
        foreach (var doc in Golden8.ReadRecords(AiMoveParityTests.EnsureGolden()))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                if (rec.GetProperty("status").GetInt32() != 200 || !rec.GetProperty("response").TryGetProperty("pos", out _)) continue;
                if (!rec.GetProperty("tags").EnumerateArray().Any(t => t.GetString() == "robust_candidate")) continue;
                cheap.Add(rec.GetProperty("request").Clone());
                if (cheap.Count == 5) break;
            }
        }
        Assert.Equal(5, cheap.Count);

        var openings = Enumerable.Range(0, 12).Select(i => "opening-" + i).ToArray();
        var cheapLabels = Enumerable.Range(0, 5).Select(i => "cheap-" + i).ToArray();

        var fifo = await Probe(priority: false, cheap);
        var scheduled = await Probe(priority: true, cheap);
        _output.WriteLine("first come first served: " + string.Join(" ", fifo));
        _output.WriteLine("priority with aging    : " + string.Join(" ", scheduled));

        // old behaviour: every request waits for the ones that came first, so the cheap requests are stuck behind all twelve openings
        Assert.Equal(openings.Concat(cheapLabels), fifo);
        // new: the five cheap requests are served first, then the openings, which keep their arrival order (none is starved or reordered)
        Assert.Equal(cheapLabels.Concat(openings), scheduled);
    }
}
