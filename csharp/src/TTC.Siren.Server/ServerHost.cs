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

    private ServerHost(WebApplication app, ServerOptions options, AiMoveService ai, WorkerPool pool, string cardDbSha256)
    {
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
    public string BaseUrl { get; private set; } = "";

    public static ServerHost Create(ServerOptions options, string[]? args = null)
    {
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
        var ai = new AiMoveService(catalog, new SolveService(db), pool);
        var app = builder.Build();
        var started = Stopwatch.GetTimestamp();
        string version = typeof(ServerHost).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

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
            JsonDocument doc;
            try
            {
                using var ms = new MemoryStream();
                await http.Body.CopyToAsync(ms, ct).ConfigureAwait(false);
                doc = JsonDocument.Parse(ms.ToArray(), new JsonDocumentOptions { MaxDepth = 64 });
            }
            catch (Exception e) when (e is JsonException or Microsoft.AspNetCore.Http.BadHttpRequestException)
            {
                return new JsonTextResult(400, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = "Invalid input data" }));
            }

            using (doc)
            {
                var outcome = await ai.HandleAsync(doc.RootElement, ChooseSeed(options, http), ct).ConfigureAwait(false);
                var result = new JsonTextResult(outcome.Status, PyJson.Serialize(outcome.Body));
                self?.RequestCompleted?.Invoke(outcome);
                return result;
            }
        });

        self = new ServerHost(app, options, ai, pool, sha);
        return self;
    }

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
