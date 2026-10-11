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

    private ServerHost(WebApplication app, ServerOptions options, AiMoveService ai, WorkerPool pool, string cardDbSha256, GameLogService gameLog, ApiKeyStore? keys,
        SessionStore sessions, AuditLog audit, AnonymousGate anonymousGate)
    {
        Sessions = sessions;
        Audit = audit;
        AnonymousGate = anonymousGate;
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

    /// <summary>The web sessions (the browser holds the id in an HttpOnly cookie, the server only its hash).</summary>
    public SessionStore Sessions { get; }

    public AuditLog Audit { get; }

    /// <summary>The per-address limits for visitors who are not logged in. 8B has no anonymous compute endpoint; the coaching of 8D will use this gate and the anonymous class of the pool.</summary>
    public AnonymousGate AnonymousGate { get; }
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

        var pool = new WorkerPool(options.Workers, options.TtEntries, options.MemoEntries, options.QueueLimit, options.QueueWaitMs, options.AgingMs,
            capacity: options.Access.CapacityFor(options.Workers));
        var templates = options.TemplatesDirectory is { } dir ? TemplateSets.FromDirectory(dir) : TemplateSets.Default();
        var ai = new AiMoveService(catalog, new SolveService(db), pool, templates, options.TtEntries)
        {
            PriorityScheduling = options.PriorityScheduling,
            Caps = new SearchCaps(MaxSeconds: options.MaxSearchSeconds > 0 ? options.MaxSearchSeconds : null),
            AnonymousCaps = options.Access.AnonymousCaps,
        };
        string version = typeof(ServerHost).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
        var store = new GameLogStore(options.GameLog, time);
        var hub = new LiveHub(options.LiveBuffer, options.Live.ReplayEvents, TimeSpan.FromMinutes(options.Live.ReplayMinutes), time, options.Live.ReplayBytes);
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
        var clientIps = new ClientIpResolver(security.TrustedProxies, AppContext.BaseDirectory); // fails now, not at the first request, when the range file is missing
        var sessions = new SessionStore(keys, security.Sessions, time);
        var cookies = new SessionCookie(security);
        Action<string>? infoLog = null; // bound to the host's logger once it exists
        var audit = new AuditLog(security.Sessions.AuditLogFile, message => infoLog?.Invoke(message), time);
        var loginLimiter = new IpLimiter(time);
        var anonymousGate = new AnonymousGate(options.Access, time);

        var app = builder.Build();
        var logger = app.Logger;
        authLog = message => logger.LogWarning("{Message}", message);
        infoLog = message => logger.LogInformation("{Message}", message);
        var started = Stopwatch.GetTimestamp();
        var corsOrigins = new HashSet<string>(security.CorsOrigins.Select(o => o.TrimEnd('/')), StringComparer.Ordinal);

        // ---- who is the client (real address behind a trusted proxy), then headers: security headers on every answer, no caching of dynamic answers,
        //      CORS for the configured origins, HSTS over HTTPS
        app.Use(async (ctx, next) =>
        {
            ctx.Items[HttpItems.ClientIp] = clientIps.Resolve(ctx);
            var headers = ctx.Response.Headers;
            // Cloudflare caches by file extension unless told not to; a dynamic answer must never be stored (the static files set their own policy below)
            ctx.Response.OnStarting(() =>
            {
                if (!ctx.Response.Headers.ContainsKey("Cache-Control")) ctx.Response.Headers["Cache-Control"] = "no-store";
                return Task.CompletedTask;
            });
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
                OnPrepareResponse = c =>
                {
                    var h = c.Context.Response.Headers;
                    h["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'";
                    // files under /assets/ have a content hash in their name (the build writes them so): they never change, cache them for a year;
                    // everything else (index.html) is revalidated each time, so a new release is picked up
                    h["Cache-Control"] = c.Context.Request.Path.StartsWithSegments("/assets") ? "public, max-age=31536000, immutable" : "no-cache";
                },
            });
        }
        else
        {
            app.MapGet("/", () => Results.Content(
                Placeholder.Html(version, pool.Workers, pool.Idle, Stopwatch.GetElapsedTime(started).TotalSeconds), "text/html; charset=utf-8"));
        }

        // ---- admission: who is it -- an API key in a header (the plugin, or any program), or, on the endpoints that accept it, the session cookie of the web login --
        //      then the limits of that key for that class, and the class the work is done as. The lease ends with the request.
        //      A request that carries a key is judged by the key even if it also carries a cookie. A request authenticated by the cookie must also pass the CSRF check
        //      when it changes something.
        (AuthResult Auth, AccessClass Class, IDisposable? Lease, IResult? Failure) Admit(HttpRequest http, bool acceptSession, bool stream = false)
        {
            AuthResult auth;
            var cls = AccessClass.Plugin;
            if (acceptSession && KeyAuthenticator.ExtractKey(http) is null && cookies.Has(http))
            {
                var session = sessions.Find(cookies.Read(http));
                if (session is null)
                {
                    cookies.Clear(http.HttpContext.Response, http); // the browser forgets the dead session
                    return (default, cls, null, Error(401, "session expired or invalid; log in again"));
                }
                if (OriginPolicy.IsStateChanging(http.Method) && !OriginPolicy.IsSameOrigin(http)) return (default, cls, null, Error(403, "cross-site request refused"));
                auth = AuthResult.FromSession(session.KeyId);
                cls = AccessClass.Web;
            }
            else auth = authenticator!.Authenticate(http);
            if (!auth.Ok)
                return (auth, cls, null, new JsonTextResult(auth.Status, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = auth.Error }),
                    auth.Status == 401 ? new[] { ("WWW-Authenticate", "Bearer") } : Array.Empty<(string, string)>()));
            if (limiter is null) return (auth, cls, null, null);
            var decision = stream ? limiter.TryOpenStream(auth.KeyId, cls) : limiter.TryEnter(auth.KeyId, cls);
            if (!decision.Allowed)
                return (auth, cls, null, new JsonTextResult(429, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = decision.Reason, ["retry_after_seconds"] = decision.RetryAfterSeconds }),
                    ("Retry-After", decision.RetryAfterSeconds.ToString(System.Globalization.CultureInfo.InvariantCulture))));
            return (auth, cls, decision.Lease, null);
        }

        Dictionary<string, object?> ClassStats()
        {
            var classes = new Dictionary<string, object?>();
            foreach (var c in new[] { AccessClass.Plugin, AccessClass.Web, AccessClass.Anonymous })
                classes[c.ToString().ToLowerInvariant()] = new Dictionary<string, object?>
                {
                    ["running"] = pool.RunningOf(c), ["queued"] = pool.WaitingOf(c), ["served"] = pool.ServedOf(c), ["busyAnswers"] = pool.BusyAnswersOf(c),
                };
            return classes;
        }

        app.MapGet("/health", () =>
        {
            var body = new Dictionary<string, object?>
            {
                ["status"] = "ok", ["server"] = "TTC.Siren.Server", ["version"] = version, ["cardDbSha256"] = sha, ["cardCount"] = db.Count,
                ["workers"] = pool.Workers, ["workersCreated"] = pool.Created, ["idleWorkers"] = pool.Idle, ["queued"] = pool.Waiting, ["queueLimit"] = pool.QueueLimit,
                ["queueWaitMs"] = pool.QueueWaitMs, ["agingMs"] = pool.AgingMs, ["busyAnswers"] = pool.BusyAnswers, ["served"] = pool.Served,
                ["reservedForPlugin"] = pool.Capacity.PluginReservedWorkers, ["anonymousMaxWorkers"] = pool.Capacity.AnonymousMaxWorkers, ["classes"] = ClassStats(),
                ["uptimeSeconds"] = Stopwatch.GetElapsedTime(started).TotalSeconds,
            };
            return new JsonTextResult(200, PyJson.Serialize(body));
        });

        // ---- the plugin's endpoint
        app.MapPost("/ai_move", async (HttpRequest http, CancellationToken ct) =>
        {
            var (auth, _, lease, failure) = Admit(http, acceptSession: false); // the plugin's endpoint: an API key, never a cookie
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
                    var outcome = await ai.HandleAsync(doc.RootElement, ChooseSeed(options, http), ct, session, language: ExplanationLanguage.Requested(http)).ConfigureAwait(false);
                    session.Complete(outcome);
                    var result = outcome.RetryAfterSeconds > 0
                        ? new JsonTextResult(outcome.Status, PyJson.Serialize(outcome.Body), ("Retry-After", outcome.RetryAfterSeconds.ToString(System.Globalization.CultureInfo.InvariantCulture)))
                        : new JsonTextResult(outcome.Status, PyJson.Serialize(outcome.Body));
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
                    var (_, _, lease, failure) = Admit(http, acceptSession: false);
                    if (failure is not null) return failure;
                    using (lease) return await proxy.ForwardAsync(http, captured, ct).ConfigureAwait(false);
                });
            }

        // ---- the game logs of the key (a key sees only its own games)
        app.MapGet("/v1/games", (HttpRequest http) =>
        {
            var (auth, _, lease, failure) = Admit(http, acceptSession: true);
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
            var (auth, _, lease, failure) = Admit(http, acceptSession: true);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                var lines = store.ReadLines(auth.KeyId, id);
                if (lines.Count == 0) return Error(404, "game not found");
                // the log's own lines, as they were written (review L8-2): numbers keep their form, nothing is parsed and written again
                string? lang = ExplanationLanguage.Requested(http);
                var records = lines.Select(line => (object?)new RawJson(lang is null ? line : ExplanationLanguage.RewriteRecord(line, lang))).ToList();
                return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["game_id"] = id, ["schema"] = SchemaOf(lines), ["records"] = records }));
            }
        });

        app.MapGet("/v1/games/{id}/download", (string id, HttpRequest http, HttpResponse response) =>
        {
            var (auth, _, lease, failure) = Admit(http, acceptSession: true);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                var lines = store.ReadLines(auth.KeyId, id);
                if (lines.Count == 0) return Error(404, "game not found");
                response.Headers.ContentDisposition = $"attachment; filename=\"{id}.jsonl\"";
                string? downloadLang = ExplanationLanguage.Requested(http); // a download is the file as it was written, unless a language is asked for
                return Results.Text(string.Join("\n", downloadLang is null ? lines : lines.Select(l => ExplanationLanguage.RewriteRecord(l, downloadLang))) + "\n", "application/x-ndjson", Encoding.UTF8);
            }
        });

        app.MapDelete("/v1/games/{id}", (string id, HttpRequest http) =>
        {
            var (auth, _, lease, failure) = Admit(http, acceptSession: true);
            if (failure is not null) return failure;
            using (lease)
            {
                if (!GameLogStore.GameIdPattern.IsMatch(id)) return Error(400, "invalid game id");
                return store.Delete(auth.KeyId, id)
                    ? new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["deleted"] = id }))
                    : Error(404, "game not found");
            }
        });

        // ---- the web login: a key is exchanged for a session cookie (decision A-27). No accounts, no passwords. The key is read from the body of this one request and
        //      is not stored, logged or put into the cookie; the cookie holds a random session id, the server holds only its hash.
        string ClientIpText(HttpRequest r) => (r.HttpContext.Items[HttpItems.ClientIp] as IPAddress ?? IPAddress.Loopback).ToString();

        app.MapPost("/v1/session", async (HttpRequest http, HttpResponse response, CancellationToken ct) =>
        {
            string ip = ClientIpText(http);
            if (keys is null) return Error(501, "login is not available on this server");
            var attempt = loginLimiter.TryEnter(ip, security.Sessions.LoginAttemptsPerMinute, security.Sessions.LoginBurst, 8);
            if (!attempt.Allowed)
            {
                audit.Login(ip, "throttled", null);
                return new JsonTextResult(429, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = "too many login attempts, try again later", ["retry_after_seconds"] = attempt.RetryAfterSeconds }),
                    ("Retry-After", attempt.RetryAfterSeconds.ToString(System.Globalization.CultureInfo.InvariantCulture)));
            }
            using (attempt.Lease)
            {
                // a login form posted by another site is not ours (a browser always sends Origin with a POST; a program that sends none is fine)
                if (http.Headers.ContainsKey("Origin") && !OriginPolicy.IsSameOrigin(http))
                {
                    audit.Login(ip, "failed", null);
                    return Error(403, "cross-site request refused");
                }
                var (bytes, tooLarge) = await BodyReader.ReadAsync(http, 4096, ct).ConfigureAwait(false);
                if (tooLarge is not null) return tooLarge;
                string? secret = null;
                try
                {
                    using var doc = JsonDocument.Parse(bytes!, new JsonDocumentOptions { MaxDepth = 4 });
                    if (doc.RootElement.ValueKind == JsonValueKind.Object && doc.RootElement.TryGetProperty("key", out var k) && k.ValueKind == JsonValueKind.String) secret = k.GetString();
                }
                catch (JsonException)
                {
                }
                var record = string.IsNullOrWhiteSpace(secret) || secret.Length > 200 ? null : keys.Find(secret.Trim());
                if (record is null || !record.Enabled)
                {
                    audit.Login(ip, "failed", null);
                    return Error(401, "login failed"); // the same answer for an unknown, a mistyped and a revoked key
                }
                var (id, session) = sessions.Create(record.Id);
                cookies.Write(response, http, id, session.Expires, security.Sessions.Lifetime);
                audit.Login(ip, "success", record.Id);
                return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?>
                {
                    ["ok"] = true, ["key"] = new Dictionary<string, object?> { ["id"] = record.Id, ["name"] = record.Name }, ["expires"] = JsonConv.Utc(session.Expires),
                }));
            }
        });

        app.MapDelete("/v1/session", (HttpRequest http, HttpResponse response) =>
        {
            string? id = cookies.Read(http);
            if (id is null) return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["ok"] = true, ["logged_out"] = false }));
            if (!OriginPolicy.IsSameOrigin(http)) return Error(403, "cross-site request refused"); // the cookie authenticates this request: CSRF check
            var session = sessions.Find(id);
            sessions.Remove(id);
            cookies.Clear(response, http);
            audit.Logout(ClientIpText(http), session?.KeyId);
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["ok"] = true, ["logged_out"] = true }));
        });

        app.MapGet("/v1/me", (HttpRequest http) =>
        {
            var (auth, cls, lease, failure) = Admit(http, acceptSession: true);
            if (failure is not null) return failure;
            using (lease)
            {
                var record = keys?.GetById(auth.KeyId);
                var me = new Dictionary<string, object?>
                {
                    ["authenticated"] = true, ["via"] = auth.Source == AuthSource.Session ? "session" : "api_key", ["class"] = cls.ToString().ToLowerInvariant(),
                    // never the key and never its hash: the public id, the name and the limits
                    ["key"] = new Dictionary<string, object?>
                    {
                        ["id"] = auth.KeyId, ["name"] = record?.Name ?? auth.KeyId, ["created"] = record is null ? null : JsonConv.Utc(record.Created),
                        ["rate_per_minute"] = record?.RatePerMinute ?? security.DefaultRatePerMinute, ["burst"] = record?.Burst ?? security.DefaultBurst,
                        ["max_in_flight"] = record?.MaxInFlight ?? security.DefaultMaxInFlight,
                    },
                    ["session"] = auth.Source == AuthSource.Session && sessions.Find(cookies.Read(http)) is { } s
                        ? new Dictionary<string, object?> { ["created"] = JsonConv.Utc(s.Created), ["expires"] = JsonConv.Utc(s.Expires) } : null,
                    ["server"] = new Dictionary<string, object?> { ["version"] = version },
                };
                return new JsonTextResult(200, PyJson.Serialize(me));
            }
        });

        // ---- the live event stream (Server-Sent Events) of the key
        app.MapGet("/v1/live", async (HttpContext ctx) =>
        {
            var (auth, _, lease, failure) = Admit(ctx.Request, acceptSession: true, stream: true);
            if (failure is not null)
            {
                await failure.ExecuteAsync(ctx).ConfigureAwait(false);
                return;
            }
            using (lease)
            {
                string? gameFilter = ctx.Request.Query["game_id"].ToString() is { Length: > 0 } g && GameLogStore.GameIdPattern.IsMatch(g) ? g : null;
                await LiveStream.RunAsync(ctx, hub, auth.KeyId, gameFilter, options.Live.Heartbeat, LiveStream.LastEventId(ctx.Request), ExplanationLanguage.Requested(ctx.Request)).ConfigureAwait(false);
            }
        });

        // ---- operator endpoints: loopback only, never through a proxy
        //      A request that a proxy handled (Cloudflare, a local tunnel, a reverse proxy: any of the proxy headers) is refused even when its TCP peer is the local machine,
        //      because that is exactly what a tunnel looks like. The operator sits at the machine and sends none of them.
        bool IsOperator(HttpRequest r) =>
            r.HttpContext.Connection.RemoteIpAddress is { } ip && IPAddress.IsLoopback(ip) && !ClientIpResolver.CarriesProxyHeaders(r);
        app.MapGet("/admin/status", (HttpRequest http) =>
        {
            if (!IsOperator(http)) return Error(403, "operator endpoints are reachable from this machine only");
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?>
            {
                ["version"] = version, ["workers"] = pool.Workers, ["workers_created"] = pool.Created, ["idle"] = pool.Idle, ["queued"] = pool.Waiting,
                ["queue_limit"] = pool.QueueLimit, ["busy_answers"] = pool.BusyAnswers, ["served"] = pool.Served, ["table_memory_bytes"] = pool.MemoryBytes,
                ["keys"] = keys?.List().Count(k => k.Enabled), ["sessions"] = sessions.Count, ["classes"] = ClassStats(), ["https"] = security.Https.Enabled, ["proxy"] = proxy is not null,
                ["game_log_dir"] = options.GameLog.Directory, ["uptime_seconds"] = Stopwatch.GetElapsedTime(started).TotalSeconds,
            }));
        });
        app.MapPost("/admin/reload-keys", (HttpRequest http) =>
        {
            if (!IsOperator(http)) return Error(403, "operator endpoints are reachable from this machine only");
            keys?.ReloadNow();
            return new JsonTextResult(200, PyJson.Serialize(new Dictionary<string, object?> { ["keys"] = keys?.List().Count ?? 0 }));
        });

        if (options.TestSeedHeader) logger.LogWarning("Server:TestSeedHeader is ON: the X-Ttc-Test-Seed header of a local request fixes the sampling seed. This is a test switch and must never be on in production.");
        if (security.TrustedProxies.AnyTrusted)
            logger.LogInformation("Client addresses are read from {Header} for requests from trusted proxies ({Ranges} configured ranges, loopback {Loopback})",
                security.TrustedProxies.ClientIpHeader, clientIps.RangeCount, security.TrustedProxies.TrustLoopback);

        self = new ServerHost(app, options, ai, pool, sha, gameLog, keys, sessions, audit, anonymousGate);
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
        // the header is believed only from a local caller that no proxy handled (review L8-1: and only when the switch was allowed at start)
        if (options.TestSeedHeader && http.HttpContext.Connection.RemoteIpAddress is { } peer && IPAddress.IsLoopback(peer) && !ClientIpResolver.CarriesProxyHeaders(http) &&
            http.Headers.TryGetValue("X-Ttc-Test-Seed", out var header) && ulong.TryParse(header.ToString(), out ulong seed)) return seed;
        Span<byte> bytes = stackalloc byte[8];
        RandomNumberGenerator.Fill(bytes);
        return BitConverter.ToUInt64(bytes);
    }

    /// <summary>Runs the warm-up (see <see cref="WarmUp"/>) on every worker. <c>Program</c> calls it before the server starts listening when <c>Server:WarmUp</c> is on.</summary>
    public Task<WarmUpReport> WarmUpAsync(CancellationToken ct = default) => WarmUp.RunAsync(Ai, Pool, Ai.Catalog.Db, ct: ct);

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
