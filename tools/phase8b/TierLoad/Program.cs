using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;

// Phase 8B measurement: what does a flood of anonymous work do to the latency of the plugin?
//   dotnet run -c Release --project tools/phase8b/TierLoad -- [--workers 12] [--plugin-rounds 3] [--flood 48] [--out result.json]
// One server process with real HTTP for the plugin (an API key) and an in-process flood of anonymous requests (8B has no anonymous compute endpoint over HTTP; the
// flood goes through the same AiMoveService and WorkerPool a future endpoint will use). Three phases with the same plugin requests, one at a time:
//   A  quiet                        -- nothing else is running
//   B  anonymous flood              -- the access classes of 8B (reserved workers for the plugin, anonymous capped, its own queue)
//   C  the same flood, no classes   -- control: the flood is made of ordinary (plugin-class) requests, the way every request was handled before 8B
// The numbers describe THIS machine with THIS CPU set (the caller pins the process) and are not an estimate for another one.
int workers = 12, rounds = 3, flood = 48;
string? outPath = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--workers": workers = int.Parse(args[++i]); break;
        case "--plugin-rounds": rounds = int.Parse(args[++i]); break;
        case "--flood": flood = int.Parse(args[++i]); break;
        case "--out": outPath = args[++i]; break;
        default: Console.Error.WriteLine("unknown argument " + args[i]); return 2;
    }
}

string tmp = Path.Combine(Path.GetTempPath(), "ttc-tier-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(tmp);
try
{
    var keys = new ApiKeyStore(Path.Combine(tmp, "keys.json"));
    var (_, secret) = keys.Add("load", 1_000_000, 100_000, 1000);
    var options = new ServerOptions
    {
        Host = "127.0.0.1", Port = 0, Workers = workers, GameLog = new GameLogOptions { Directory = Path.Combine(tmp, "logs") },
        Security = new SecurityOptions { KeyStoreFile = Path.Combine(tmp, "keys.json"), Sessions = { AuditLogFile = "" } },
    };
    await using var host = ServerHost.Create(options);
    var warm = await host.WarmUpAsync();
    await host.StartAsync();
    Console.WriteLine($"# server up on {host.BaseUrl}: {workers} workers, reserved for the plugin {host.Pool.Capacity.PluginReservedWorkers}, anonymous at most {host.Pool.Capacity.AnonymousMaxWorkers}, " +
                      $"anonymous queue {host.Pool.Capacity.AnonymousQueueLimit}, warm-up {warm.Milliseconds:N0} ms; logical CPUs seen {Environment.ProcessorCount}");

    var db = host.Ai.Catalog.Db;
    // the plugin's workload: mid-games and robust endgames at production settings (depth 10, 10 s), and openings with a 1 s limit (so their latency is the limit, not luck)
    var pluginRequests = new List<(string Kind, string Body)>();
    for (int r = 0; r < 12; r++)
    {
        pluginRequests.Add(("mid", WarmUp.BuildRequest(db, 3, 10, 10)));
        pluginRequests.Add(("robust", WarmUp.BuildRequest(db, 5, 10, 10)));
    }
    for (int r = 0; r < 3; r++) pluginRequests.Add(("opening", WarmUp.BuildRequest(db, 0, 1.0, 10)));

    using var http = new HttpClient { BaseAddress = new Uri(host.BaseUrl), Timeout = TimeSpan.FromMinutes(5) };
    http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", secret);

    async Task<Dictionary<string, List<double>>> PluginPassAsync(int every = 1)
    {
        var latencies = new Dictionary<string, List<double>> { ["mid"] = new(), ["robust"] = new(), ["opening"] = new() };
        for (int round = 0; round < rounds; round++)
            foreach (var (kind, body) in pluginRequests.Where((_, i) => i % every == 0))
            {
                long t0 = Stopwatch.GetTimestamp();
                using var response = await http.PostAsync("/ai_move", new StringContent(body, Encoding.UTF8, "application/json"));
                await response.Content.ReadAsStringAsync();
                double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                if (!response.IsSuccessStatusCode) throw new InvalidOperationException($"plugin request answered {(int)response.StatusCode}");
                latencies[kind].Add(ms);
            }
        return latencies;
    }

    // ---- A: quiet
    var quiet = await PluginPassAsync();

    // ---- B / C: a flood of work that keeps asking, as fast as it is answered
    async Task<(Dictionary<string, List<double>> Plugin, FloodStats Flood)> UnderFloodAsync(AccessClass floodClass, int every)
    {
        var stats = new FloodStats();
        using var stop = new CancellationTokenSource();
        string floodBody = WarmUp.BuildRequest(db, 0, 3, 8); // the search an anonymous request gets at most (depth 8, 3 s): the same work in both flood phases
        var tasks = new List<Task>();
        for (int i = 0; i < flood; i++)
        {
            int id = i;
            tasks.Add(Task.Run(async () =>
            {
                ulong seed = (ulong)(1000 + id);
                while (!stop.IsCancellationRequested)
                {
                    using var doc = JsonDocument.Parse(floodBody);
                    var outcome = await host.Ai.HandleAsync(doc.RootElement, seed++, CancellationToken.None, null, floodClass);
                    if (outcome.Status == 200) Interlocked.Increment(ref stats.Answered);
                    else
                    {
                        Interlocked.Increment(ref stats.Busy);
                        await Task.Delay(5);
                    }
                }
            }));
        }
        var sampler = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                stats.MaxRunningAnonymous = Math.Max(stats.MaxRunningAnonymous, host.Pool.RunningOf(AccessClass.Anonymous));
                stats.MaxQueuedAnonymous = Math.Max(stats.MaxQueuedAnonymous, host.Pool.WaitingOf(AccessClass.Anonymous));
                stats.MaxQueuedPlugin = Math.Max(stats.MaxQueuedPlugin, host.Pool.WaitingOf(AccessClass.Plugin));
                stats.MaxQueuedTotal = Math.Max(stats.MaxQueuedTotal, host.Pool.Waiting);
                await Task.Delay(2);
            }
        });
        await Task.Delay(1500); // let the flood fill the workers and the queue before the first plugin request
        var plugin = await PluginPassAsync(every);
        stop.Cancel();
        await Task.WhenAll(tasks);
        await sampler;
        return (plugin, stats);
    }

    var (saturated, floodStats) = await UnderFloodAsync(AccessClass.Anonymous, 1);
    var (control, controlStats) = await UnderFloodAsync(AccessClass.Plugin, 5); // without classes every plugin request waits behind the flood: a smaller set keeps the control short

    // ---- report
    static string Row(string name, List<double> v)
    {
        var s = v.OrderBy(x => x).ToList();
        if (s.Count == 0) return $"{name,-9} n=  0";
        double Pct(double p) => s[Math.Min(s.Count - 1, (int)Math.Round(p * (s.Count - 1)))];
        return $"{name,-9} n={s.Count,3}  p50 {Pct(0.5),9:F1}  p95 {Pct(0.95),9:F1}  max {s[^1],9:F1} ms";
    }
    var result = new Dictionary<string, object?>
    {
        ["workers"] = workers, ["reserved"] = host.Pool.Capacity.PluginReservedWorkers, ["anonymousMax"] = host.Pool.Capacity.AnonymousMaxWorkers, ["flood"] = flood,
        ["logicalCpus"] = Environment.ProcessorCount,
    };
    foreach (var (title, set, fs) in new (string, Dictionary<string, List<double>>, FloodStats?)[] { ("A  quiet", quiet, null), ("B  anonymous flood, access classes", saturated, floodStats), ("C  control: same flood without classes", control, controlStats) })
    {
        Console.WriteLine();
        Console.WriteLine(title);
        foreach (var kind in new[] { "mid", "robust", "opening" }) Console.WriteLine("  " + Row(kind, set[kind]));
        if (fs is not null)
            Console.WriteLine($"  flood: {fs.Answered} answered, {fs.Busy} busy answers; at most {fs.MaxRunningAnonymous} anonymous searches running, {fs.MaxQueuedAnonymous} anonymous queued, " +
                              $"{fs.MaxQueuedPlugin} plugin queued, {fs.MaxQueuedTotal} queued in all");
        result[title.Substring(0, 1)] = new Dictionary<string, object?>
        {
            ["title"] = title,
            ["latenciesMs"] = set.ToDictionary(kv => kv.Key, kv => (object?)kv.Value.Select(x => Math.Round(x, 2)).ToList()),
            ["flood"] = fs is null ? null : new Dictionary<string, object?>
            {
                ["answered"] = fs.Answered, ["busy"] = fs.Busy, ["maxRunningAnonymous"] = fs.MaxRunningAnonymous, ["maxQueuedAnonymous"] = fs.MaxQueuedAnonymous,
                ["maxQueuedPlugin"] = fs.MaxQueuedPlugin, ["maxQueuedTotal"] = fs.MaxQueuedTotal,
            },
        };
    }
    if (outPath is not null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
    }
    return 0;
}
finally
{
    try
    {
        Directory.Delete(tmp, recursive: true);
    }
    catch (IOException)
    {
    }
}

sealed class FloodStats
{
    public long Answered, Busy;
    public int MaxRunningAnonymous, MaxQueuedAnonymous, MaxQueuedPlugin, MaxQueuedTotal;
}
