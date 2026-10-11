using System.Diagnostics;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Explain;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Live;
using TTC.Siren.Server.Proxy;
using TTC.Siren.Server.Security;

namespace TTC.Siren.Server;

/// <summary>A JSON text answer in the form Flask's <c>jsonify</c> uses (compact, sorted keys, trailing newline).</summary>
internal sealed class JsonTextResult : IResult
{
    private readonly int _status;
    private readonly byte[] _body;
    private readonly (string Name, string Value)[] _headers;

    public JsonTextResult(int status, string json, params (string Name, string Value)[] headers)
    {
        _status = status;
        _body = Encoding.UTF8.GetBytes(json + "\n");
        _headers = headers;
    }

    public Task ExecuteAsync(HttpContext httpContext)
    {
        httpContext.Response.StatusCode = _status;
        httpContext.Response.ContentType = "application/json";
        httpContext.Response.ContentLength = _body.Length;
        foreach (var (name, value) in _headers) httpContext.Response.Headers[name] = value;
        return httpContext.Response.Body.WriteAsync(_body).AsTask();
    }
}

/// <summary>
/// The product server (Phase 8A): <c>POST /ai_move</c> served natively (the plugin contract of API.md), with the engine in process; the game logs,
/// the live event stream, API keys with per-key limits, HTTPS, the proxy to the Python app for the three <c>/api/*</c> endpoints, the placeholder page.
/// Used by <c>Program</c> and, with port 0, by the tests (a real socket, the same code path as production).
/// </summary>
public sealed class ServerHost
{
    public const int MaxBodyBytes = 1 << 20;

    private readonly WebApplication _app;

    private ServerHost(WebApplication app, ServerOptions options, AiMoveService ai, WorkerPool pool, string cardDbSha256, GameLogService gameLog, ApiKeyStore? keys)
    {
        GameLog = gameLog;
        _app = app;
        Options = options;
        Ai = ai;
        Pool = pool;
        CardDbSha256 = cardDbSha256;
        Keys = keys;
    }

    /// <summary>Raised after every <c>/ai_move</c> request has been answered (tests subscribe).</summary>
    public event Action<AiMoveOutcome>? RequestCompleted;

    public ServerOptions Options { get; }
    public AiMoveService Ai { get; }
    public WorkerPool Pool { get; }
    public string CardDbSha256 { get; }
    public GameLogService GameLog { get; }
    public ApiKeyStore? Keys { get; }
    public LiveHub Hub => GameLog.Hub;

    /// <summary>The plain HTTP address (valid after <see cref="StartAsync"/>), or empty when only HTTPS is served.</summary>
    public string BaseUrl { get; private set; } = "";

    /// <summary>The HTTPS address, or empty when HTTPS is off.</summary>
    public string HttpsUrl { get; private set; } = "";

    public static ServerHost Create(ServerOptions options, string[]? args = null, IApiKeyAuthenticator? authenticator = null, TimeProvider? time = null,
        ILoggerProvider? loggerProvider = null)
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

        X509Certificate2? certificate = null;
        if (options.Security.Https.Enabled)
        {
            string? password = options.Security.Https.PfxPassword ?? Environment.GetEnvironmentVariable(options.Security.Https.PfxPasswordEnvironmentVariable);
            certificate = X509CertificateLoader.LoadPkcs12FromFile(options.Security.Https.PfxPath!, password);
        }

        var builder = WebApplication.CreateSlimBuilder(args ?? Array.Empty<string>());
        builder.Host.UseWindowsService();
        builder.Logging.ClearProviders();
        if (loggerProvider is null)
        {
            builder.Logging.AddConsole();
            builder.Logging.SetMinimumLevel(LogLevel.Warning);
        }
        else
        {
            builder.Logging.AddProvider(loggerProvider);
            builder.Logging.SetMinimumLevel(LogLevel.Trace);
        }
        builder.WebHost.ConfigureKestrel(k =>
        {
            if (options.HttpEnabled) k.Listen(options.Address, options.Port, l => l.Protocols = HttpProtocols.Http1);
            if (certificate is not null)
                k.Listen(options.Address, options.Security.Https.Port, l =>
                {
                    l.UseHttps(certificate);
                    l.Protocols = HttpProtocols.Http1;
                });
            k.Limits.MaxRequestBodySize = BodyReader.HardLimit; // each endpoint reads with its own, smaller limit (BodyReader)
            k.Limits.MaxRequestHeadersTotalSize = 32 * 1024;
            k.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(15);
            k.Limits.KeepAliveTimeout = TimeSpan.FromSeconds(120);
            k.AddServerHeader = false;
        });

        var pool = new WorkerPool(options.Workers, options.TtEntries, options.MemoEntries, options.QueueLimit, options.QueueWaitMs, options.AgingMs);
        var templates = options.TemplatesDirectory is { } dir ? TemplateSet.Load(Path.Combine(dir, "explain.zh-CN.json")) : TemplateSet.Default();
        var ai = new AiMoveService(catalog, new SolveService(db), pool, templates, options.TtEntries) { PriorityScheduling = options.PriorityScheduling };
        string version = typeof(ServerHost).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
        var store = new GameLogStore(options.GameLog, time);
        var hub = new LiveHub(options.LiveBuffer);
        var gameLog = new GameLogService(store, hub, db, "TTC.Siren.Server " + version, time);
        builder.Services.AddSingleton(new GameLogCleanup(store));
        builder.Services.AddHostedService(sp => sp.GetRequiredService<GameLogCleanup>());

        // ---- authentication and limits
        var security = options.Security;
        ApiKeyStore? keys = null;
        KeyLimiter? limiter = null;
        Action<string>? authLog = null; // bound to the host's logger once it exists
        if (authenticator is null)
        {
            if (security.RequireKey)
            {
                keys = new ApiKeyStore(security.KeyStoreFile, time);
                authenticator = new KeyAuthenticator(keys, security, message => authLog?.Invoke(message));
                limiter = new KeyLimiter(keys, security, time);
            }
            else authenticator = new OpenAuthenticator();
        }
        else limiter = new KeyLimiter(null, security, time);
        var proxy = options.Proxy.Enabled ? new PythonProxy(options.Proxy) : null;

        var app = builder.Build();
        var logger = app.Logger;
        authLog = message => logger.LogWarning("{Message}", message);
        var started = Stopwatch.GetTimestamp();
        var corsOrigins = new HashSet<string>(security.CorsOrigins.Select(o => o.TrimEnd('/')), StringComparer.Ordinal);

        // ---- headers: security headers on every answer, CORS for the configured origins, HSTS over HTTPS
        app.Use(async (ctx, next) =>
        {
            var headers = ctx.Response.Headers;
            headers["X-Content-Type-Options"] = "nosniff";
            headers["Referrer-Policy"] = "no-referrer";
            headers["X-Frame-Options"] = "DENY";
            if (ctx.Request.IsHttps && security.Https.Hsts) headers["Strict-Transport-Security"] = $"max-age={security.Https.HstsMaxAgeSeconds}; includeSubDomains";
            if (ctx.Request.Headers.TryGetValue("Origin", out var origin) && corsOrigins.Contains(origin.ToString().TrimEnd('/')))
            {
                headers["Access-Control-Allow-Origin"] = origin.ToString();
                headers["Vary"] = "Origin";
                headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Api-Key";
                headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS";
                headers["Access-Control-Max-Age"] = "600";
                if (HttpMethods.IsOptions(ctx.Request.Method))
                {
                    ctx.Response.StatusCode = 204;
                    return;
                }
            }
            else if (HttpMethods.IsOptions(ctx.Request.Method))
            {
                ctx.Response.StatusCode = 403; // a preflight from an origin that is not allowed
                return;
            }
            await next().ConfigureAwait(false);
        });

        if (options.StaticDirectory is { } staticDir && Directory.Exists(staticDir))
        {
            var files = new PhysicalFileProvider(Path.GetFullPath(staticDir));
            app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = files });
            app.UseStaticFiles(new StaticFileOptions
            {
                FileProvider = files,
                OnPrepareResponse = c => c.Context.Response.Headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'",
            });
        }
        else
        {
            app.MapGet("/", () => Results.Content(
                Placeholder.Html(version, pool.Workers, pool.Idle, Stopwatch.GetElapsedTime(started).TotalSeconds), "text/html; charset=utf-8"));
        }

        // ---- admission: key, then limits (the lease ends with the request)
        (AuthResult Auth, IDisposable? Lease, IResult? Failure) Admit(HttpRequest http, bool stream = false)
        {
            var auth = authenticator!.Authenticate(http);
            if (!auth.Ok)
                return (auth, null, new JsonTextResult(auth.Status, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = auth.Error }),
                    auth.Status == 401 ? new[] { ("WWW-Authenticate", "Bearer") } : Array.Empty<(string, string)>()));
            if (limiter is null) return (auth, null, null);
            var decision = stream ? limiter.TryOpenStream(auth.KeyId) : limiter.TryEnter(auth.KeyId);
            if (!decision.Allowed)
                return (auth, null, new JsonTextResult(429, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = decision.Reason, ["retry_after_seconds"] = decision.RetryAfterSeconds }),
                    ("Retry-After", decision.RetryAfterSeconds.ToString(System.Globalization.CultureInfo.InvariantCulture))));
            return (auth, decision.Lease, null);
        }

        app.MapGet("/health", () =>
        {
            var body = new Dictionary<string, object?>
            {
                ["status"] = "ok", ["server"] = "TTC.Siren.Server", ["version"] = version, ["cardDbSha256"] = sha, ["cardCount"] = db.Count,
                ["workers"] = pool.Workers, ["workersCreated"] = pool.Created, ["idleWorkers"] = pool.Idle, ["queued"] = pool.Waiting, ["queueLimit"] = pool.QueueLimit,
                ["queueWaitMs"] = pool.QueueWaitMs, ["agingMs"] = pool.AgingMs, ["busyAnswers"] = pool.BusyAnswers, ["served"] = pool.Served,
                ["uptimeSeconds"] = Stopwatch.GetElapsedTime(started).TotalSeconds,
            };
            return new JsonTextResult(200, PyJson.Serialize(body));
        });

        // ---- the plugin's endpoint
        app.MapPost("/ai_move", async (HttpRequest http, CancellationToken ct) =>
        {
            var (auth, lease, failure) = Admit(http);
            if (failure is not null) return failure;
            using (lease)
            {
                var (bytes, tooLarge) = await BodyReader.ReadAsync(http, MaxBodyBytes, ct).ConfigureAwait(false);
                if (tooLarge is not null) return tooLarge; // review L8-3: 413, in the usual error form
                string text = Encoding.UTF8.GetString(bytes!).TrimStart('\uFEFF');
                JsonDocument doc;
                try
                {
                    doc = JsonDocument.Parse(bytes!, new JsonDocumentOptions { MaxDepth = 64 });
                }
                catch (JsonException)
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
            }
        });

        // ---- the Python app's three endpoints, behind the same key (until Phase 8C ports them)
        if (proxy is not null)
            foreach (var path in PythonProxy.Paths)
            {
                string captured = path;
                app.MapPost(captured, async (HttpRequest http, CancellationToken ct) =>
                {
                    var (_, lease, failure) = Admit(http);
                    if (failure is not null) return failure;
                    using (lease) return await proxy.ForwardAsync(http, captured, ct).ConfigureAwait(false);
                });
            }

        // ---- the game logs of the key (a key sees only its own games)
        app.MapGet("/v1/games", (HttpRequest http) =>
        {
            var (auth, lease, failure) = Admit(http);
            if (failure is not null) return failure;
            using (lease)
            {
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
            }
        });

        app.MapGet("/v1/games/{id}", (string id, HttpRequest http) =>
        {
            var (auth, lease, failure) = Admit(http);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                var lines = store.ReadLines(auth.KeyId, id);
                if (lines.Count == 0) return Error(404, "game not found");
                // the log's own lines, as they were written (review L8-2): numbers keep their form, nothing is parsed and written again
                var records = lines.Select(line => (object?)new RawJson(line)).ToList();
                return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["game_id"] = id, ["schema"] = SchemaOf(lines), ["records"] = records }));
            }
        });

        app.MapGet("/v1/games/{id}/download", (string id, HttpRequest http, HttpResponse response) =>
        {
            var (auth, lease, failure) = Admit(http);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                var lines = store.ReadLines(auth.KeyId, id);
                if (lines.Count == 0) return Error(404, "game not found");
                response.Headers.ContentDisposition = $"attachment; filename=\"{id}.jsonl\"";
                return Results.Text(string.Join("\n", lines) + "\n", "application/x-ndjson", Encoding.UTF8);
            }
        });

        app.MapDelete("/v1/games/{id}", (string id, HttpRequest http) =>
        {
            var (auth, lease, failure) = Admit(http);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                return store.Delete(auth.KeyId, id)
                    ? new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["deleted"] = id }))
                    : Error(404, "game not found");
            }
        });

        // ---- the live event stream (Server-Sent Events) of the key
        app.MapGet("/v1/live", async (HttpContext ctx) =>
        {
            var (auth, lease, failure) = Admit(ctx.Request, stream: true);
            if (failure is not null)
            {
                await failure.ExecuteAsync(ctx).ConfigureAwait(false);
                return;
            }
            using (lease)
            {
                string? gameFilter = ctx.Request.Query["game_id"].ToString() is { Length: > 0 } g && GameLogStore.GameIdPattern.IsMatch(g) ? g : null;
                await LiveStream.RunAsync(ctx, hub, auth.KeyId, gameFilter).ConfigureAwait(false);
            }
        });

        // ---- operator endpoints: loopback only, never through a proxy
        bool IsOperator(HttpRequest r) =>
            r.HttpContext.Connection.RemoteIpAddress is { } ip && IPAddress.IsLoopback(ip) &&
            !r.Headers.ContainsKey("X-Forwarded-For") && !r.Headers.ContainsKey("Forwarded") && !r.Headers.ContainsKey("X-Real-IP");
        app.MapGet("/admin/status", (HttpRequest http) =>
        {
            if (!IsOperator(http)) return Error(403, "operator endpoints are reachable from this machine only");
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?>
            {
                ["version"] = version, ["workers"] = pool.Workers, ["workers_created"] = pool.Created, ["idle"] = pool.Idle, ["queued"] = pool.Waiting,
                ["queue_limit"] = pool.QueueLimit, ["busy_answers"] = pool.BusyAnswers, ["served"] = pool.Served, ["table_memory_bytes"] = pool.MemoryBytes,
                ["keys"] = keys?.List().Count(k => k.Enabled), ["https"] = security.Https.Enabled, ["proxy"] = proxy is not null,
                ["game_log_dir"] = options.GameLog.Directory, ["uptime_seconds"] = Stopwatch.GetElapsedTime(started).TotalSeconds,
            }));
        });
        app.MapPost("/admin/reload-keys", (HttpRequest http) =>
        {
            if (!IsOperator(http)) return Error(403, "operator endpoints are reachable from this machine only");
            keys?.ReloadNow();
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["keys"] = keys?.List().Count ?? 0 }));
        });

        self = new ServerHost(app, options, ai, pool, sha, gameLog, keys);
        return self;
    }

    /// <summary>The highest schema version in the lines of a game (files written before Phase 8B hold version 1; a game may continue in version 2).</summary>
    private static int SchemaOf(List<string> lines)
    {
        int version = 1;
        foreach (string line in lines)
        {
            using var d = JsonDocument.Parse(line);
            if (d.RootElement.TryGetProperty("v", out var v) && v.TryGetInt32(out int n)) version = Math.Max(version, n);
        }
        return version;
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
        var addresses = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Select(a => a.Replace("[::]", "127.0.0.1", StringComparison.Ordinal)).ToList();
        BaseUrl = addresses.FirstOrDefault(a => a.StartsWith("http://", StringComparison.Ordinal)) ?? "";
        HttpsUrl = addresses.FirstOrDefault(a => a.StartsWith("https://", StringComparison.Ordinal)) ?? "";
    }

    public Task StopAsync(CancellationToken ct = default) => _app.StopAsync(ct);

    public Task WaitForShutdownAsync(CancellationToken ct = default) => _app.WaitForShutdownAsync(ct);

    public async ValueTask DisposeAsync() => await _app.DisposeAsync().ConfigureAwait(false);
}

/// <summary>The page at <c>/</c> until the web UI (Phase 8B) exists: the product name and the status of the server. No script, no external resource.</summary>
internal static class Placeholder
{
    public static string Html(string version, int workers, int idle, double uptimeSeconds) => $$"""
        <!doctype html>
        <html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'">
        <title>TTC Siren</title>
        <style>body{font-family:system-ui,sans-serif;margin:0;display:grid;place-items:center;min-height:100vh;background:#0b1a24;color:#d8e6ee}
        main{max-width:32rem;padding:2rem}h1{margin:0 0 .25rem;font-weight:600}p{color:#8fb0c2}code{color:#bfe3f2}</style></head>
        <body><main><h1>TTC Siren</h1><p>幻卡求解服务 · Triple Triad solver</p>
        <p>状态：<strong>运行中</strong> · 版本 <code>{{System.Net.WebUtility.HtmlEncode(version)}}</code> · 工作线程 {{idle}}/{{workers}} 空闲 · 已运行 {{(int)uptimeSeconds}} 秒</p>
        <p>网页界面尚未提供；插件请使用 <code>POST /ai_move</code>（需要 API 密钥）。</p></main></body></html>
        """;
}
