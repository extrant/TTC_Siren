using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Phase 7: determinism under concurrency, no leakage between requests, the bounded queue.</summary>
public sealed class EngineConcurrencyTests : IClassFixture<EngineFixture>
{
    private readonly EngineFixture _fx;
    private readonly ITestOutputHelper _output;

    public EngineConcurrencyTests(EngineFixture fx, ITestOutputHelper output)
    {
        _fx = fx;
        _output = output;
    }

    /// <summary>At least 200 different requests: standard searches on golden positions and robust-endgame requests.</summary>
    private static List<string> Requests(int standard, int robust)
    {
        var list = new List<string>();
        string path = GoldenFiles.EnsureSearchGolden("search");
        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                if (list.Count >= standard) break;
                var rec = doc.RootElement;
                if (rec.TryGetProperty("name", out _)) continue; // the benchmark items: heavier
                var rules = RuleParser.FromPythonList(rec.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
                var snapshot = StateSnapshot.FromJson(rec.GetProperty("state"), rules);
                if (snapshot.Hands[snapshot.Current].Count == 0) continue;
                list.Add(EngineJson.Request(snapshot, rec.GetProperty("aggr").GetBoolean(), maxDepth: "4", requestId: "s" + list.Count));
            }
        }
        int standardCount = list.Count;
        foreach (var doc in EndgameGolden.Records("robust"))
        {
            using (doc)
            {
                if (list.Count - standardCount >= robust) break;
                var rec = doc.RootElement;
                var rules = EndgameGolden.Rules(rec);
                var snapshot = EndgameGolden.Snapshot(rec.GetProperty("state"), rules);
                if (snapshot.Hands[snapshot.Current].Count == 0) continue;
                string r = EngineJson.Robust(snapshot.Current, rec.GetProperty("opp").GetInt32(), EndgameGolden.Generated(rec),
                    rec.GetProperty("sample").GetInt32(), EndgameGolden.NpcIds(rec));
                list.Add(EngineJson.Request(snapshot, aggressive: (list.Count % 2) == 1, maxDepth: "3", robust: r, requestId: "r" + list.Count));
            }
        }
        return list;
    }

    private static async Task<string[]> RunAll(HttpClient client, IReadOnlyList<string> requests, int parallelism, int seed)
    {
        var results = new string[requests.Count];
        var shuffled = Enumerable.Range(0, requests.Count).ToArray();
        new Random(seed).Shuffle(shuffled);
        using var gate = new SemaphoreSlim(parallelism);
        var tasks = shuffled.Select(async i =>
        {
            await gate.WaitAsync();
            try
            {
                var (status, body) = await EngineJson.Post(client, requests[i]);
                Assert.True(status == 200, $"request {i}: HTTP {status}: {body}");
                Assert.Equal("ok", body["status"]!.GetValue<string>());
                results[i] = EngineJson.Normalized(body);
            }
            finally
            {
                gate.Release();
            }
        });
        await Task.WhenAll(tasks);
        return results;
    }

    private static async Task<(EngineHost Host, HttpClient Client)> StartSingleWorkerEngine()
    {
        var host = EngineHost.Create(new EngineOptions { Port = 0, Workers = 1, QueueLimit = 1000, QueueWaitMs = 600_000, TtEntries = 1 << 18, MemoEntries = 1 << 15 });
        await host.StartAsync();
        return (host, new HttpClient { BaseAddress = new Uri(host.BaseUrl), Timeout = TimeSpan.FromMinutes(10) });
    }

    [Fact]
    public async Task Concurrent_responses_equal_a_sequential_run_on_a_fresh_engine_field_by_field()
    {
        var requests = Requests(standard: 170, robust: 60);
        Assert.True(requests.Count >= 200, $"only {requests.Count} requests");
        Assert.Equal(requests.Count, requests.Distinct().Count());

        // reference: one worker, one request at a time, a fresh engine
        var (refHost, refClient) = await StartSingleWorkerEngine();
        string[] reference;
        try
        {
            reference = await RunAll(refClient, requests, parallelism: 1, seed: 1);
        }
        finally
        {
            refClient.Dispose();
            await refHost.StopAsync();
            await refHost.DisposeAsync();
        }

        // 16 requests in flight against 4 workers: more requests than workers, twice with different orders
        foreach (int seed in new[] { 7, 8 })
        {
            var concurrent = await RunAll(_fx.Client, requests, parallelism: 16, seed: seed);
            for (int i = 0; i < requests.Count; i++)
                Assert.True(reference[i] == concurrent[i], $"request {i} differs under concurrency (seed {seed})\n  sequential: {reference[i]}\n  concurrent: {concurrent[i]}");
        }
        var health = JsonNode.Parse(await _fx.Client.GetStringAsync("/v1/health"))!;
        _output.WriteLine($"{requests.Count} requests x2 concurrent runs identical to the sequential run; workers created={health["workersCreated"]}, busy answers={health["busyAnswers"]}");
        Assert.True(health["workersCreated"]!.GetValue<int>() <= 4);
    }

    [Fact]
    public async Task A_request_never_sees_what_an_earlier_request_left_on_the_same_worker()
    {
        var requests = Requests(standard: 40, robust: 20);
        var (host, client) = await StartSingleWorkerEngine();
        try
        {
            var first = await RunAll(client, requests, 1, seed: 3);          // every request after other requests on the one worker
            var again = await RunAll(client, requests, 1, seed: 4);          // a different order, the same single worker
            for (int i = 0; i < requests.Count; i++) Assert.Equal(first[i], again[i]);

            // and each one equals a run on a worker that had never served anything
            for (int i = 0; i < requests.Count; i += 7)
            {
                var (fresh, freshClient) = await StartSingleWorkerEngine();
                try
                {
                    var (status, body) = await EngineJson.Post(freshClient, requests[i]);
                    Assert.Equal(200, status);
                    Assert.Equal(first[i], EngineJson.Normalized(body));
                }
                finally
                {
                    freshClient.Dispose();
                    await fresh.StopAsync();
                    await fresh.DisposeAsync();
                }
            }
        }
        finally
        {
            client.Dispose();
            await host.StopAsync();
            await host.DisposeAsync();
        }
    }

    [Fact]
    public async Task A_full_queue_answers_busy_and_the_engine_recovers()
    {
        // one worker, room for one waiting request, short wait: a burst of slow searches must produce "busy"
        var host = EngineHost.Create(new EngineOptions { Port = 0, Workers = 1, QueueLimit = 1, QueueWaitMs = 150, TtEntries = 1 << 20, MemoEntries = 1 << 15 });
        await host.StartAsync();
        using var client = new HttpClient { BaseAddress = new Uri(host.BaseUrl), Timeout = TimeSpan.FromMinutes(2) };
        try
        {
            // an opening searched to depth 9 takes about a second: long enough to keep the single worker busy
            var slow = EngineJson.Request(ReferenceScenarios.Create("open_none", SearchGolden.Db).ToSnapshot(), maxDepth: "9");
            var tasks = Enumerable.Range(0, 6).Select(_ => EngineJson.Post(client, slow)).ToArray();
            var results = await Task.WhenAll(tasks);
            int ok = results.Count(r => r.Status == 200), busy = results.Count(r => r.Status == 503);
            Assert.True(ok >= 1, "at least one request is served");
            Assert.True(busy >= 1, "a burst beyond workers + queue is answered busy");
            Assert.Equal(6, ok + busy);
            foreach (var r in results.Where(r => r.Status == 503))
            {
                Assert.Equal("busy", r.Body["status"]!.GetValue<string>());
                Assert.Contains(r.Body["reason"]!.GetValue<string>(), new[] { "queue_full", "queue_timeout" });
            }
            // the served ones are all equal (same request)
            Assert.Single(results.Where(r => r.Status == 200).Select(r => EngineJson.Normalized(r.Body)).Distinct());

            var health = JsonNode.Parse(await client.GetStringAsync("/v1/health"))!;
            Assert.Equal(busy, (int)health["busyAnswers"]!.GetValue<long>());
            Assert.Equal(1, health["idleWorkers"]!.GetValue<int>());

            // after the burst the engine serves again
            var (status, _) = await EngineJson.Post(client, EngineJson.Request(ReferenceScenarios.Create("end5_same_plus", SearchGolden.Db).ToSnapshot()));
            Assert.Equal(200, status);
            _output.WriteLine($"burst of 6: {ok} served, {busy} busy");
        }
        finally
        {
            await host.StopAsync();
            await host.DisposeAsync();
        }
    }

    [Fact]
    public async Task The_worker_pool_is_bounded_and_creates_contexts_lazily()
    {
        var pool = new WorkerPool(workers: 2, ttEntries: 1 << 10, memoEntries: 1 << 10, queueLimit: 0, queueWaitMs: 10);
        Assert.Equal(0, pool.Created);
        var (a, _) = await pool.AcquireAsync(CancellationToken.None);
        var (b, _) = await pool.AcquireAsync(CancellationToken.None);
        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.NotEqual(a!.Id, b!.Id);
        Assert.Equal(2, pool.Created);
        var (none, reason) = await pool.AcquireAsync(CancellationToken.None); // queue limit 0: no waiting allowed
        Assert.Null(none);
        Assert.Equal(BusyReason.QueueFull, reason);
        pool.Release(a);
        var (c, _) = await pool.AcquireAsync(CancellationToken.None);
        Assert.Same(a, c); // reused, not created again
        Assert.Equal(2, pool.Created);
        Assert.Equal(1, pool.BusyAnswers);
    }
}
