using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Server;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Review O8-1: the wait queue is ordered by the estimated cost of a request, with aging. Cheap requests (endgames, the robust endgame) overtake
/// openings; an opening that has waited long enough is as good as a cheap request and then goes by arrival, so it cannot be overtaken for ever.
/// The first tests pin the rules on the pool alone; the last reproduces the review's probe (one worker, twelve openings, then five cheap requests).
/// </summary>
public sealed class SchedulerTests
{
    private readonly ITestOutputHelper _output;

    public SchedulerTests(ITestOutputHelper output) => _output = output;

    private static WorkerPool Pool(int agingMs = 60_000, int queue = 64) => new(workers: 1, ttEntries: 1 << 8, memoEntries: 1 << 8, queueLimit: queue, queueWaitMs: 60_000, agingMs);

    /// <summary>Queues a request of the given class and returns the order in which it is served (appended to <paramref name="served"/>).</summary>
    private static Task<AcquireResult> Queue(WorkerPool pool, int rank, string name, List<string> served)
    {
        var task = pool.AcquireAsync(CancellationToken.None, rank);
        return task.ContinueWith(t =>
        {
            lock (served) served.Add(name);
            pool.Release(t.Result.Worker!);
            return t.Result;
        });
    }

    private static async Task WaitUntilQueued(WorkerPool pool, int count)
    {
        for (int i = 0; i < 500 && pool.Waiting < count; i++) await Task.Delay(5);
        Assert.Equal(count, pool.Waiting);
    }

    [Fact]
    public async Task Requests_of_one_class_are_served_first_come_first_served()
    {
        var pool = Pool();
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Medium)).Worker!;
        var served = new List<string>();
        var tasks = new List<Task>();
        for (int i = 0; i < 6; i++)
        {
            tasks.Add(Queue(pool, RequestClass.Expensive, "e" + i, served));
            await WaitUntilQueued(pool, i + 1);
        }
        pool.Release(running);
        await Task.WhenAll(tasks);
        Assert.Equal(new[] { "e0", "e1", "e2", "e3", "e4", "e5" }, served);
    }

    [Fact]
    public async Task A_cheap_request_overtakes_the_openings_that_were_queued_before_it()
    {
        var pool = Pool();
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive)).Worker!;
        var served = new List<string>();
        var tasks = new List<Task>();
        for (int i = 0; i < 4; i++)
        {
            tasks.Add(Queue(pool, RequestClass.Expensive, "opening" + i, served));
            await WaitUntilQueued(pool, i + 1);
        }
        tasks.Add(Queue(pool, RequestClass.Cheap, "cheap-a", served));
        await WaitUntilQueued(pool, 5);
        tasks.Add(Queue(pool, RequestClass.Medium, "medium", served));
        await WaitUntilQueued(pool, 6);
        tasks.Add(Queue(pool, RequestClass.Cheap, "cheap-b", served));
        await WaitUntilQueued(pool, 7);
        pool.Release(running);
        await Task.WhenAll(tasks);
        // cheap before medium before the openings, each class first come first served
        Assert.Equal(new[] { "cheap-a", "cheap-b", "medium", "opening0", "opening1", "opening2", "opening3" }, served);
    }

    [Fact]
    public async Task An_opening_that_has_waited_long_enough_is_no_longer_overtaken_by_later_cheap_requests()
    {
        var pool = Pool(agingMs: 80);
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Expensive)).Worker!;
        var served = new List<string>();
        var tasks = new List<Task> { Queue(pool, RequestClass.Expensive, "old-opening", served) };
        await WaitUntilQueued(pool, 1);
        await Task.Delay(250); // more than two aging periods: the opening now counts as cheap, and it arrived first
        tasks.Add(Queue(pool, RequestClass.Cheap, "late-cheap", served));
        await WaitUntilQueued(pool, 2);
        pool.Release(running);
        await Task.WhenAll(tasks);
        Assert.Equal(new[] { "old-opening", "late-cheap" }, served);
    }

    [Fact]
    public async Task An_opening_waits_at_most_twice_the_aging_period_for_requests_that_arrive_after_it_even_under_a_constant_stream_of_cheap_ones()
    {
        const int aging = 100;
        var pool = Pool(agingMs: aging, queue: 1000);
        var running = (await pool.AcquireAsync(CancellationToken.None, RequestClass.Medium)).Worker!;
        var order = new List<string>();
        var opening = Queue(pool, RequestClass.Expensive, "opening", order);
        await WaitUntilQueued(pool, 1);
        var sw = Stopwatch.StartNew();
        // a cheap request every 10 ms for 600 ms; each takes 20 ms of "work" (one worker): the stream alone would keep the worker busy for ever
        var stream = Task.Run(async () =>
        {
            var tasks = new List<Task>();
            for (int i = 0; i < 60; i++)
            {
                tasks.Add(Task.Run(async () =>
                {
                    var r = await pool.AcquireAsync(CancellationToken.None, RequestClass.Cheap);
                    lock (order) order.Add("cheap");
                    await Task.Delay(20);
                    pool.Release(r.Worker!);
                }));
                await Task.Delay(10);
            }
            await Task.WhenAll(tasks);
        });
        await Task.Delay(30);
        pool.Release(running);
        var result = await opening;
        double waited = result.WaitedMs;
        await stream;
        int cheapBefore;
        lock (order) cheapBefore = order.IndexOf("opening");
        _output.WriteLine($"the opening waited {waited:F0} ms (aging {aging} ms) and was served after {cheapBefore} cheap requests");
        // after 2 * aging it counts as cheap and, having arrived first, goes first: the wait is bounded (generous margin for a loaded machine)
        Assert.True(waited < 2 * aging + 400, $"the opening waited {waited} ms");
        Assert.True(cheapBefore >= 1, "the cheap requests did overtake it at first");
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

    private static async Task<(List<double> OpeningDone, List<double> CheapDone, List<double> OpeningWait, List<double> CheapWait, int Ok)> Probe(bool priority, List<JsonElement> cheap)
    {
        var service = AiMoveParityTests.CreateService(workers: 1);
        service.PriorityScheduling = priority;
        var sw = Stopwatch.StartNew();
        var openingDone = new double[12];
        var openingWait = new double[12];
        var cheapDone = new double[cheap.Count];
        var cheapWait = new double[cheap.Count];
        int ok = 0;
        var tasks = new List<Task>();
        for (int i = 0; i < 12; i++)
        {
            int n = i;
            tasks.Add(Task.Run(async () =>
            {
                var o = await service.HandleAsync(Opening("opening-" + n, 300), (ulong)(100 + n), CancellationToken.None);
                openingDone[n] = sw.Elapsed.TotalMilliseconds;
                openingWait[n] = o.QueueWaitMs;
                if (o.Status == 200) Interlocked.Increment(ref ok);
            }));
            await Task.Delay(5);
        }
        await Task.Delay(100); // the openings are queued behind the first one
        for (int i = 0; i < cheap.Count; i++)
        {
            int n = i;
            tasks.Add(Task.Run(async () =>
            {
                var o = await service.HandleAsync(cheap[n], (ulong)(500 + n), CancellationToken.None);
                cheapDone[n] = sw.Elapsed.TotalMilliseconds;
                cheapWait[n] = o.QueueWaitMs;
                if (o.Status == 200) Interlocked.Increment(ref ok);
            }));
            await Task.Delay(5);
        }
        await Task.WhenAll(tasks);
        return (openingDone.ToList(), cheapDone.ToList(), openingWait.ToList(), cheapWait.ToList(), ok);
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

        var fifo = await Probe(priority: false, cheap);
        var scheduled = await Probe(priority: true, cheap);

        string Line(string name, (List<double> OpeningDone, List<double> CheapDone, List<double> OpeningWait, List<double> CheapWait, int Ok) r) =>
            $"{name}: cheap requests answered after {r.CheapDone.Min():F0}..{r.CheapDone.Max():F0} ms (queue wait {r.CheapWait.Average():F0} ms on average), " +
            $"openings after {r.OpeningDone.Min():F0}..{r.OpeningDone.Max():F0} ms (longest queue wait {r.OpeningWait.Max():F0} ms), {r.Ok}/17 answered";
        _output.WriteLine(Line("first come first served ", fifo));
        _output.WriteLine(Line("priority with aging      ", scheduled));

        // everything is answered, in both modes
        Assert.Equal(17, fifo.Ok);
        Assert.Equal(17, scheduled.Ok);
        // old behaviour: the cheap requests are stuck behind the openings; new: they are answered while the openings still queue
        Assert.True(fifo.CheapDone.Min() > fifo.OpeningDone.OrderBy(x => x).ElementAt(8), "FIFO: the cheap requests wait for most openings");
        Assert.True(scheduled.CheapDone.Max() < scheduled.OpeningDone.OrderBy(x => x).ElementAt(3), "scheduled: the cheap requests are answered while the openings queue");
        Assert.True(scheduled.CheapWait.Max() < fifo.CheapWait.Min() / 2);
        // no opening starves: the longest wait is about what the work in front of it takes (12 x 300 ms + the cheap requests), plus slack
        Assert.True(scheduled.OpeningWait.Max() < 12 * 300 + 5 * 150 + 2500, $"the longest opening wait was {scheduled.OpeningWait.Max()} ms");
    }
}
