using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Explain;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Live;
using TTC.Siren.Server.Security;

namespace TTC.Siren.Server;

/// <summary>A JSON text answer in the form Flask's <c>jsonify</c> uses (compact, sorted keys, trailing newline).</summary>
internal sealed class JsonTextResult : IResult
{
    private readonly int _status;
    private readonly byte[] _body;

    public JsonTextResult(int status, string json)
    {
        _status = status;
        _body = Encoding.UTF8.GetBytes(json + "\n");
    }

    public Task ExecuteAsync(HttpContext httpContext)
    {
        httpContext.Response.StatusCode = _status;
        httpContext.Response.ContentType = "application/json";
        httpContext.Response.ContentLength = _body.Length;
        return httpContext.Response.Body.WriteAsync(_body).AsTask();
    }
}

/// <summary>
/// The product server (Phase 8A): <c>POST /ai_move</c> served natively (the plugin contract of API.md), with the engine in process.
/// Used by <c>Program</c> and, with port 0, by the tests (a real socket, the same code path as production).
/// </summary>
public sealed class ServerHost
{
    public const int MaxBodyBytes = 1 << 20;

    private readonly WebApplication _app;

    private ServerHost(WebApplication app, ServerOptions options, AiMoveService ai, WorkerPool pool, string cardDbSha256, GameLogService gameLog)
    {
        GameLog = gameLog;
        _app = app;
        Options = options;
        Ai = ai;
        Pool = pool;
        CardDbSha256 = cardDbSha256;
    }

    /// <summary>Raised after every <c>/ai_move</c> request has been answered (the game log and the live events subscribe).</summary>
    public event Action<AiMoveOutcome>? RequestCompleted;

    public ServerOptions Options { get; }
    public AiMoveService Ai { get; }
    public WorkerPool Pool { get; }
    public string CardDbSha256 { get; }
    public GameLogService GameLog { get; }
    public LiveHub Hub => GameLog.Hub;
    public string BaseUrl { get; private set; } = "";

    public static ServerHost Create(ServerOptions options, string[]? args = null, IApiKeyAuthenticator? authenticator = null, TimeProvider? time = null)
    {
        authenticator ??= new OpenAuthenticator();
        ServerHost? self = null;
        options.Validate();
        string dbPath = options.CardDbPath ?? CardDb.LocateDataFile(AppContext.BaseDirectory);
        if (!File.Exists(dbPath)) throw new FileNotFoundException($"Card database not found: {dbPath}");
        string sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dbPath))).ToLowerInvariant();
        var db = CardDb.Load(dbPath);
        var catalog = new CardCatalog(db);

        ThreadPool.GetMinThreads(out int minWorker, out int minIo);
        ThreadPool.SetMinThreads(Math.Max(minWorker, options.Workers + 8), minIo);

        var builder = WebApplication.CreateSlimBuilder(args ?? Array.Empty<string>());
        builder.Host.UseWindowsService();
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.WebHost.ConfigureKestrel(k =>
        {
            k.Listen(options.Address, options.Port, l => l.Protocols = HttpProtocols.Http1);
            k.Limits.MaxRequestBodySize = MaxBodyBytes;
            k.AddServerHeader = false;
        });

        var pool = new WorkerPool(options.Workers, options.TtEntries, options.MemoEntries, options.QueueLimit, options.QueueWaitMs);
        var templates = options.TemplatesDirectory is { } dir ? TemplateSet.Load(Path.Combine(dir, "explain.zh-CN.json")) : TemplateSet.Default();
        var ai = new AiMoveService(catalog, new SolveService(db), pool, templates, options.TtEntries);
        string version = typeof(ServerHost).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
        var store = new GameLogStore(options.GameLog, time);
        var hub = new LiveHub(options.LiveBuffer);
        var gameLog = new GameLogService(store, hub, db, "TTC.Siren.Server " + version, time);
        builder.Services.AddSingleton(new GameLogCleanup(store));
        builder.Services.AddHostedService(sp => sp.GetRequiredService<GameLogCleanup>());
        var app = builder.Build();
        var started = Stopwatch.GetTimestamp();

        app.MapGet("/health", () =>
        {
            var body = new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["server"] = "TTC.Siren.Server",
                ["version"] = version,
                ["cardDbSha256"] = sha,
                ["cardCount"] = db.Count,
                ["workers"] = pool.Workers,
                ["workersCreated"] = pool.Created,
                ["idleWorkers"] = pool.Idle,
                ["queued"] = pool.Waiting,
                ["queueLimit"] = pool.QueueLimit,
                ["queueWaitMs"] = pool.QueueWaitMs,
                ["busyAnswers"] = pool.BusyAnswers,
                ["served"] = pool.Served,
                ["uptimeSeconds"] = Stopwatch.GetElapsedTime(started).TotalSeconds,
            };
            return new JsonTextResult(200, PyJson.Serialize(body));
        });

        app.MapPost("/ai_move", async (HttpRequest http, CancellationToken ct) =>
        {
            var auth = authenticator.Authenticate(http);
            if (!auth.Ok) return new JsonTextResult(auth.Status, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = auth.Error }));

            string text;
            JsonDocument doc;
            try
            {
                using var ms = new MemoryStream();
                await http.Body.CopyToAsync(ms, ct).ConfigureAwait(false);
                var bytes = ms.ToArray();
                text = Encoding.UTF8.GetString(bytes);
                doc = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 64 });
            }
            catch (Exception e) when (e is JsonException or Microsoft.AspNetCore.Http.BadHttpRequestException)
            {
                return new JsonTextResult(400, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = "Invalid input data" }));
            }

            using (doc)
            {
                var session = gameLog.StartSession(auth.KeyId, text);
                var outcome = await ai.HandleAsync(doc.RootElement, ChooseSeed(options, http), ct, session).ConfigureAwait(false);
                session.Complete(outcome);
                var result = new JsonTextResult(outcome.Status, PyJson.Serialize(outcome.Body));
                self?.RequestCompleted?.Invoke(outcome);
                return result;
            }
        });

        // ---- the game logs of the key (a key sees only its own games)
        app.MapGet("/v1/games", (HttpRequest http) =>
        {
            var auth = authenticator.Authenticate(http);
            if (!auth.Ok) return Error(auth.Status, auth.Error!);
            int offset = int.TryParse(http.Query["offset"], out int o) ? o : 0, limit = int.TryParse(http.Query["limit"], out int l) ? l : 50;
            var (items, total) = store.List(auth.KeyId, offset, limit);
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?>
            {
                ["total"] = total, ["offset"] = offset, ["limit"] = Math.Clamp(limit, 1, 200),
                ["games"] = items.Select(g => (object?)new Dictionary<string, object?>
                {
                    ["game_id"] = g.GameId, ["started"] = JsonConv.Utc(g.Started), ["last_write"] = JsonConv.Utc(g.LastWrite), ["records"] = g.Records,
                    ["requests"] = g.Requests, ["ended"] = g.Ended, ["winner"] = g.Winner, ["my_owner"] = g.MyOwner, ["rules"] = g.Rules, ["id_source"] = g.IdSource,
                }).ToList(),
            }));
        });

        app.MapGet("/v1/games/{id}", (string id, HttpRequest http) =>
        {
            var auth = authenticator.Authenticate(http);
            if (!auth.Ok) return Error(auth.Status, auth.Error!);
            if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
            var lines = store.ReadLines(auth.KeyId, id);
            if (lines.Count == 0) return Error(404, "game not found");
            var records = lines.Select(line =>
            {
                using var d = JsonDocument.Parse(line);
                return JsonConv.ToObject(d.RootElement);
            }).ToList();
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["game_id"] = id, ["schema"] = 1, ["records"] = records }));
        });

        app.MapGet("/v1/games/{id}/download", (string id, HttpRequest http, HttpResponse response) =>
        {
            var auth = authenticator.Authenticate(http);
            if (!auth.Ok) return Error(auth.Status, auth.Error!);
            if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
            var lines = store.ReadLines(auth.KeyId, id);
            if (lines.Count == 0) return Error(404, "game not found");
            response.Headers.ContentDisposition = $"attachment; filename=\"{id}.jsonl\"";
            return Results.Text(string.Join("\n", lines) + "\n", "application/x-ndjson", Encoding.UTF8);
        });

        app.MapDelete("/v1/games/{id}", (string id, HttpRequest http) =>
        {
            var auth = authenticator.Authenticate(http);
            if (!auth.Ok) return Error(auth.Status, auth.Error!);
            if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
            return store.Delete(auth.KeyId, id)
                ? new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["deleted"] = id }))
                : Error(404, "game not found");
        });

        // ---- the live event stream (Server-Sent Events) of the key
        app.MapGet("/v1/live", async (HttpContext ctx) =>
        {
            var auth = authenticator.Authenticate(ctx.Request);
            if (!auth.Ok)
            {
                await Error(auth.Status, auth.Error!).ExecuteAsync(ctx).ConfigureAwait(false);
                return;
            }
            string? gameFilter = ctx.Request.Query["game_id"].ToString() is { Length: > 0 } g && GameLogStore.GameIdPattern.IsMatch(g) ? g : null;
            await LiveStream.RunAsync(ctx, hub, auth.KeyId, gameFilter).ConfigureAwait(false);
        });

        self = new ServerHost(app, options, ai, pool, sha, gameLog);
        return self;
    }

    private static JsonTextResult Error(int status, string message) => new(status, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = message }));

    /// <summary>Production: a fresh 64-bit cryptographic seed per request. The header seed exists for the parity tests only, behind a configuration switch.</summary>
    private static ulong ChooseSeed(ServerOptions options, HttpRequest http)
    {
        if (options.TestSeedHeader && http.Headers.TryGetValue("X-Ttc-Test-Seed", out var header) && ulong.TryParse(header.ToString(), out ulong seed)) return seed;
        Span<byte> bytes = stackalloc byte[8];
        RandomNumberGenerator.Fill(bytes);
        return BitConverter.ToUInt64(bytes);
    }

    public async Task StartAsync(CancellationToken ct = default)
    {
        await _app.StartAsync(ct).ConfigureAwait(false);
        var addresses = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses;
        BaseUrl = addresses.First().Replace("[::]", "127.0.0.1", StringComparison.Ordinal);
    }

    public Task StopAsync(CancellationToken ct = default) => _app.StopAsync(ct);

    public Task WaitForShutdownAsync(CancellationToken ct = default) => _app.WaitForShutdownAsync(ct);

    public async ValueTask DisposeAsync() => await _app.DisposeAsync().ConfigureAwait(false);
}
