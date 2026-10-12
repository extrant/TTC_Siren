using TTC.Siren.Engine;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Access classes and capacity (decision A-27): the plugin comes first and has workers that nothing else can take, a web user comes next, a visitor who is not logged in
/// comes last and may hold only a small share of the workers. Every assertion is about the ORDER of service or the NUMBER of running searches, never about how long
/// anything took (review M8-2); the aging clock is logical.
/// </summary>
public sealed class AccessClassTests
{
    private readonly ITestOutputHelper _output;

    public AccessClassTests(ITestOutputHelper output) => _output = output;

    private static WorkerPool Pool(int workers = 8, int reserved = 2, int anonymousMax = 2, int anonymousQueue = 0, int anonymousWaitMs = 10_000, int queue = 1000, LogicalClock? clock = null,
        int agingMs = 100, int queueWaitMs = 60_000) =>
        new(workers, 1 << 8, 1 << 8, queue, queueWaitMs, agingMs, clock,
            new CapacityOptions { PluginReservedWorkers = reserved, AnonymousMaxWorkers = anonymousMax, AnonymousQueueLimit = anonymousQueue, AnonymousQueueWaitMs = anonymousWaitMs });

    private static Task<AcquireResult> Ask(WorkerPool pool, AccessClass access, int cost = RequestClass.Medium) => pool.AcquireAsync(CancellationToken.None, cost, access);

    /// <summary>A request that the pool served at once: its task completed before the call returned (no queueing at all).</summary>
    private static async Task<WorkerContext> Immediately(Task<AcquireResult> task)
    {
        Assert.True(task.IsCompletedSuccessfully, "the request had to wait");
        var result = await task;
        Assert.Equal(BusyReason.None, result.Reason);
        return result.Worker!;
    }

    // ============================================================================================================================ the shares

    [Theory]
    [InlineData(32, 0.25, 0.25, 8, 8)]
    [InlineData(16, 0.25, 0.25, 4, 4)]
    [InlineData(12, 0.25, 0.25, 3, 3)]
    [InlineData(8, 0.25, 0.25, 2, 2)]
    [InlineData(4, 0.25, 0.25, 1, 1)]
    [InlineData(3, 0.25, 0.25, 0, 1)]
    [InlineData(2, 0.25, 0.25, 0, 1)]
    [InlineData(1, 0.25, 0.25, 0, 1)]   // a single worker cannot be shared: nothing is reserved, and anonymous work may use it when nothing else asks
    [InlineData(8, 0.5, 1.0, 4, 4)]     // anonymous work can never have more than what the reserve leaves
    [InlineData(8, 0.9, 0.25, 7, 1)]    // the reserve never takes the last worker
    public void The_shares_become_numbers_of_workers_that_always_leave_something_for_every_class(int workers, double reservedShare, double anonymousShare, int expectedReserved, int expectedAnonymous)
    {
        var c = CapacityOptions.FromShares(workers, reservedShare, anonymousShare, 16, 10_000);
        Assert.Equal(expectedReserved, c.PluginReservedWorkers);
        Assert.Equal(expectedAnonymous, c.AnonymousMaxWorkers);
        Assert.True(c.PluginReservedWorkers <= Math.Max(0, workers - 1));
        Assert.True(c.AnonymousMaxWorkers >= 1);
        _ = new WorkerPool(workers, 1 << 4, 1 << 4, 10, 1000, capacity: c); // and the pool accepts it
    }

    [Fact]
    public void A_capacity_that_leaves_no_worker_to_web_and_anonymous_work_is_refused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorkerPool(4, 1 << 4, 1 << 4, 10, 1000, capacity: new CapacityOptions { PluginReservedWorkers = 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorkerPool(1, 1 << 4, 1 << 4, 10, 1000, capacity: new CapacityOptions { PluginReservedWorkers = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorkerPool(4, 1 << 4, 1 << 4, 10, 1000, capacity: new CapacityOptions { PluginReservedWorkers = -1 }));
    }

    // ============================================================================================================================ the plugin is never in the way of anybody else's queue

    [Fact]
    public async Task Under_a_flood_of_anonymous_requests_a_plugin_request_is_served_at_once()
    {
        var pool = Pool(workers: 8, reserved: 2, anonymousMax: 2, anonymousQueue: 0);
        var anonymous = new List<Task<AcquireResult>>();
        for (int i = 0; i < 200; i++) anonymous.Add(Ask(pool, AccessClass.Anonymous, RequestClass.Expensive));
        Assert.Equal(2, pool.RunningOf(AccessClass.Anonymous));           // only its share runs
        Assert.Equal(198, pool.WaitingOf(AccessClass.Anonymous));          // the rest queues, and does not touch the other workers
        Assert.Equal(6, pool.Idle);

        // six plugin requests find workers at once: the two reserved ones and the four that anonymous work may not use
        var plugin = new List<WorkerContext>();
        for (int i = 0; i < 6; i++) plugin.Add(await Immediately(Ask(pool, AccessClass.Plugin)));
        Assert.Equal(0, pool.Idle);
        // the seventh has to wait for a plugin search -- not for the 198 anonymous requests in the queue
        var seventh = Ask(pool, AccessClass.Plugin);
        Assert.False(seventh.IsCompleted);
        Assert.Equal(1, pool.WaitingOf(AccessClass.Plugin));
        pool.Release(plugin[0]);
        var served = await seventh;                                          // the freed worker went to the plugin request, not to an anonymous one
        Assert.Equal(BusyReason.None, served.Reason);
        Assert.Equal(2, pool.RunningOf(AccessClass.Anonymous));
        Assert.Equal(198, pool.WaitingOf(AccessClass.Anonymous));
        foreach (var w in plugin.Skip(1)) pool.Release(w);
        pool.Release(served.Worker!);
    }

    [Fact]
    public async Task Anonymous_work_never_holds_more_than_its_share_however_many_workers_are_idle()
    {
        var pool = Pool(workers: 8, reserved: 2, anonymousMax: 2, anonymousQueue: 0);
        var pending = new List<Task<AcquireResult>>();
        for (int i = 0; i < 20; i++) pending.Add(Ask(pool, AccessClass.Anonymous));
        Assert.Equal(2, pool.RunningOf(AccessClass.Anonymous));
        Assert.Equal(6, pool.Idle); // six idle workers, and the eighteen waiting anonymous requests are NOT given one
        int served = 0;
        while (pending.Count > 0)
        {
            var done = await Task.WhenAny(pending);   // a request that has been given its worker
            pending.Remove(done);
            Assert.True(pool.RunningOf(AccessClass.Anonymous) <= 2, $"{pool.RunningOf(AccessClass.Anonymous)} anonymous searches running");
            Assert.Equal(8 - pool.RunningOf(AccessClass.Anonymous), pool.Idle); // nothing else is running: the idle workers stay idle, they are not lent to anonymous work
            pool.Release((await done).Worker!);       // its place is taken by the next waiting one, so two keep running until the queue is empty
            served++;
        }
        Assert.Equal(20, served);
        Assert.Equal(20, (int)pool.ServedOf(AccessClass.Anonymous));
        Assert.Equal(8, pool.Idle);
    }

    [Fact]
    public async Task Web_and_anonymous_work_together_leave_the_reserved_workers_alone()
    {
        var pool = Pool(workers: 8, reserved: 2, anonymousMax: 2);
        var web = new List<Task<AcquireResult>>();
        for (int i = 0; i < 10; i++) web.Add(Ask(pool, AccessClass.Web));
        Assert.Equal(6, pool.RunningOf(AccessClass.Web));                  // 8 workers - 2 reserved
        Assert.Equal(4, pool.WaitingOf(AccessClass.Web));
        Assert.Equal(2, pool.Idle);                                          // the reserve stays free
        var anonymous = Ask(pool, AccessClass.Anonymous);
        Assert.False(anonymous.IsCompleted);                                 // web + anonymous may not pass six
        await Immediately(Ask(pool, AccessClass.Plugin));
        await Immediately(Ask(pool, AccessClass.Plugin));
        Assert.Equal(0, pool.Idle);
    }

    [Fact]
    public async Task A_freed_worker_goes_to_the_highest_class_that_waits_then_to_the_cheapest_request_of_that_class()
    {
        var clock = new LogicalClock();
        var pool = Pool(workers: 3, reserved: 0, anonymousMax: 3, clock: clock);
        var hold = new List<WorkerContext>();
        for (int i = 0; i < 3; i++) hold.Add(await Immediately(Ask(pool, AccessClass.Plugin)));

        var pending = new Dictionary<string, Task<AcquireResult>>
        {
            ["anonymous-cheap"] = Ask(pool, AccessClass.Anonymous, RequestClass.Cheap),
            ["web-expensive"] = Ask(pool, AccessClass.Web, RequestClass.Expensive),
            ["web-cheap"] = Ask(pool, AccessClass.Web, RequestClass.Cheap),
            ["plugin-expensive"] = Ask(pool, AccessClass.Plugin, RequestClass.Expensive),
            ["anonymous-expensive"] = Ask(pool, AccessClass.Anonymous, RequestClass.Expensive),
        };
        // the workers come back one at a time; each goes to the best waiting request, and we see which one before the next worker is released
        var order = new List<string>();
        var granted = new List<WorkerContext>();
        async Task NextGrant()
        {
            var map = pending.ToDictionary(kv => kv.Value, kv => kv.Key);
            var done = await Task.WhenAny(pending.Values);
            order.Add(map[done]);
            pending.Remove(map[done]);
            granted.Add((await done).Worker!);
        }
        foreach (var w in hold)
        {
            pool.Release(w);
            await NextGrant();
        }
        foreach (var w in granted.ToList())
        {
            pool.Release(w);
            if (pending.Count > 0) await NextGrant();
        }
        Assert.Equal(new[] { "plugin-expensive", "web-cheap", "web-expensive", "anonymous-cheap", "anonymous-expensive" }, order);
    }

    [Fact]
    public async Task Inside_a_class_nothing_starves_an_old_expensive_request_is_not_overtaken_for_ever_by_cheap_ones()
    {
        const int aging = 100, step = 10;
        foreach (var access in new[] { AccessClass.Web, AccessClass.Anonymous })
        {
            var clock = new LogicalClock();
            var pool = Pool(workers: 1, reserved: 0, anonymousMax: 1, clock: clock, agingMs: aging);
            var holder = await Immediately(Ask(pool, access));
            var opening = Ask(pool, access, RequestClass.Expensive);
            var worker = holder;
            int cheapServed = 0;
            Task<AcquireResult> cheap;
            while (true)
            {
                cheap = Ask(pool, access, RequestClass.Cheap);
                clock.Advance(step);
                pool.Release(worker);
                var winner = await Task.WhenAny(opening, cheap);
                if (ReferenceEquals(winner, opening)) break;
                cheapServed++;
                worker = (await cheap).Worker!;
            }
            Assert.Equal(2 * aging / step - 1, cheapServed);                // the same bound as without classes: two aging periods
            pool.Release((await opening).Worker!);
            pool.Release((await cheap).Worker!);
        }
    }

    // ============================================================================================================================ the queues

    [Fact]
    public async Task The_anonymous_queue_has_its_own_cap_and_answers_busy_at_once_when_it_is_full()
    {
        var pool = Pool(workers: 2, reserved: 0, anonymousMax: 1, anonymousQueue: 3, queue: 100);
        var first = await Immediately(Ask(pool, AccessClass.Anonymous));     // the one allowed anonymous search
        var queued = Enumerable.Range(0, 3).Select(_ => Ask(pool, AccessClass.Anonymous)).ToList();
        Assert.Equal(3, pool.WaitingOf(AccessClass.Anonymous));
        var refused = await Ask(pool, AccessClass.Anonymous);                // the fifth anonymous request: the queue of three is full
        Assert.Equal(BusyReason.AnonymousQueueFull, refused.Reason);
        Assert.Null(refused.Worker);
        Assert.Equal(1, pool.BusyAnswersOf(AccessClass.Anonymous));
        Assert.Equal(3, pool.WaitingOf(AccessClass.Anonymous));              // nothing else changed

        // the other classes are not affected by it: the plugin is served, a web request is served and the plugin queue is not the anonymous one
        var plugin = await Immediately(Ask(pool, AccessClass.Plugin));
        Assert.Equal(0, pool.BusyAnswersOf(AccessClass.Plugin));
        pool.Release(plugin);
        pool.Release(first);
        foreach (var t in queued)
        {
            pool.Release((await t).Worker!);
        }
    }

    [Fact]
    public async Task A_queued_anonymous_request_gives_up_after_its_own_shorter_wait_while_a_plugin_request_waits_longer()
    {
        var pool = Pool(workers: 1, reserved: 0, anonymousMax: 1, anonymousQueue: 4, anonymousWaitMs: 50, queueWaitMs: 60_000);
        var hold = await Immediately(Ask(pool, AccessClass.Plugin));
        var anonymous = await Ask(pool, AccessClass.Anonymous);               // waits at most 50 ms (real time: the outcome does not depend on how fast the machine is)
        Assert.Equal(BusyReason.AnonymousQueueTimeout, anonymous.Reason);
        Assert.Equal(0, pool.WaitingOf(AccessClass.Anonymous));                // it left the queue
        var plugin = Ask(pool, AccessClass.Plugin);
        await Task.Delay(120);                                                  // longer than the anonymous wait: the plugin request is still queued, not timed out
        Assert.False(plugin.IsCompleted);
        pool.Release(hold);
        pool.Release((await plugin).Worker!);
    }

    [Fact]
    public async Task The_global_queue_limit_applies_to_every_class_and_a_cancelled_request_gives_nothing_away()
    {
        var pool = Pool(workers: 1, reserved: 0, anonymousMax: 1, queue: 2);
        var hold = await Immediately(Ask(pool, AccessClass.Plugin));
        var a = Ask(pool, AccessClass.Web);
        using var cts = new CancellationTokenSource();
        var b = pool.AcquireAsync(cts.Token, RequestClass.Medium, AccessClass.Anonymous);
        Assert.Equal(BusyReason.QueueFull, (await Ask(pool, AccessClass.Plugin)).Reason);
        cts.Cancel();
        Assert.Equal(BusyReason.Abandoned, (await b).Reason);                  // the caller went away: no worker for it
        Assert.Equal(1, pool.Waiting);
        Assert.Equal(0, pool.WaitingOf(AccessClass.Anonymous));
        pool.Release(hold);
        pool.Release((await a).Worker!);
        Assert.Equal(1, pool.Idle);
        Assert.Equal(0, pool.RunningOf(AccessClass.Plugin) + pool.RunningOf(AccessClass.Web) + pool.RunningOf(AccessClass.Anonymous));
    }

    // ============================================================================================================================ random operations, fixed seeds

    [Theory]
    [InlineData(11)]
    [InlineData(2026101)]
    [InlineData(987654321)]
    public async Task Whatever_the_order_of_arrivals_and_departures_the_limits_hold_and_every_request_is_served(int seed)
    {
        const int workers = 6, reserved = 2, anonymousMax = 2;
        var rng = new Random(seed);
        var clock = new LogicalClock();
        var pool = Pool(workers, reserved, anonymousMax, clock: clock, agingMs: 50);
        var running = new List<(Task<AcquireResult> Task, AccessClass Access)>();
        var waiting = new List<(Task<AcquireResult> Task, AccessClass Access)>();
        int served = 0, asked = 0;

        void Check(string where)
        {
            int p = pool.RunningOf(AccessClass.Plugin), w = pool.RunningOf(AccessClass.Web), a = pool.RunningOf(AccessClass.Anonymous);
            Assert.True(a <= anonymousMax, $"seed {seed} {where}: {a} anonymous searches running");
            Assert.True(w + a <= workers - reserved, $"seed {seed} {where}: {w} web + {a} anonymous running, the plugin's {reserved} workers are reserved");
            Assert.True(p + w + a <= workers, $"seed {seed} {where}: {p + w + a} running on {workers} workers");
            // a plugin request waits only for plugin searches: if one is queued, every worker that is not running a web or anonymous search is running a plugin search
            if (pool.WaitingOf(AccessClass.Plugin) > 0) Assert.True(p >= workers - (workers - reserved), $"seed {seed} {where}: a plugin request waits with only {p} plugin searches running");
            if (pool.WaitingOf(AccessClass.Plugin) > 0) Assert.Equal(workers, p + w + a);
        }

        for (int op = 0; op < 600; op++)
        {
            clock.Advance(rng.Next(0, 30));
            if (rng.Next(100) < 55)
            {
                var access = (AccessClass)rng.Next(3);
                var t = Ask(pool, access, rng.Next(3));
                asked++;
                waiting.Add((t, access));
            }
            else if (running.Count > 0)
            {
                int i = rng.Next(running.Count);
                pool.Release((await running[i].Task).Worker!);
                running.RemoveAt(i);
            }
            // requests whose grant has happened move from "waiting" to "running" (a completed task is a granted request)
            foreach (var w in waiting.Where(x => x.Task.IsCompleted).ToList())
            {
                waiting.Remove(w);
                running.Add(w);
                served++;
            }
            await Task.Yield();
            foreach (var w in waiting.Where(x => x.Task.IsCompleted).ToList())
            {
                waiting.Remove(w);
                running.Add(w);
                served++;
            }
            Check($"after operation {op}");
        }
        // drain: everybody that asked is served, nobody is lost
        while (running.Count > 0 || waiting.Count > 0)
        {
            if (running.Count > 0)
            {
                pool.Release((await running[0].Task).Worker!);
                running.RemoveAt(0);
            }
            else await Task.WhenAny(waiting.Select(x => x.Task));
            foreach (var w in waiting.Where(x => x.Task.IsCompleted).ToList())
            {
                waiting.Remove(w);
                running.Add(w);
                served++;
            }
        }
        Assert.Equal(asked, served);
        Assert.Equal(workers, pool.Idle);
        Assert.Equal(asked, (int)pool.Served);
        _output.WriteLine($"seed {seed}: {asked} requests, served by class plugin {pool.ServedOf(AccessClass.Plugin)} web {pool.ServedOf(AccessClass.Web)} anonymous {pool.ServedOf(AccessClass.Anonymous)}");
    }
}
