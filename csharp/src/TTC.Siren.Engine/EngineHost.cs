using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TTC.Siren.Core;

namespace TTC.Siren.Engine;

/// <summary>A JSON body that is already serialised (no re-encoding) with an explicit HTTP status.</summary>
internal sealed class RawJsonResult : IResult
{
    private readonly int _status;
    private readonly byte[] _body;

    public RawJsonResult(int status, byte[] body)
    {
        _status = status;
        _body = body;
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
/// The engine web host: Kestrel on one loopback address, <c>POST /v1/solve</c> and <c>GET /v1/health</c>. Used by
/// <c>Program</c> and, with port 0, by the tests (a real socket, the same code path as production).
/// </summary>
public sealed class EngineHost
{
    private readonly WebApplication _app;

    private EngineHost(WebApplication app, EngineOptions options, WorkerPool pool, string cardDbSha256)
    {
        _app = app;
        Options = options;
        Pool = pool;
        CardDbSha256 = cardDbSha256;
    }

    public EngineOptions Options { get; }
    public WorkerPool Pool { get; }
    public string CardDbSha256 { get; }

    /// <summary>The address the host listens on, for example <c>http://127.0.0.1:5300</c> (valid after <see cref="StartAsync"/>).</summary>
    public string BaseUrl { get; private set; } = "";

    public static EngineHost Create(EngineOptions options, string[]? args = null)
    {
        options.Validate();
        string dbPath = options.CardDbPath ?? CardDb.LocateDataFile(AppContext.BaseDirectory);
        if (!File.Exists(dbPath)) throw new FileNotFoundException($"Card database not found: {dbPath}");
        string sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dbPath))).ToLowerInvariant();
        var db = CardDb.Load(dbPath);

        ThreadPool.GetMinThreads(out int minWorker, out int minIo);
        ThreadPool.SetMinThreads(Math.Max(minWorker, options.Workers + 8), minIo);

        var builder = WebApplication.CreateSlimBuilder(args ?? Array.Empty<string>());
        builder.Host.UseWindowsService(); // no effect unless started by the Windows service manager
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.WebHost.ConfigureKestrel(k =>
        {
            k.Listen(options.Address, options.Port, l => l.Protocols = HttpProtocols.Http1);
            k.Limits.MaxRequestBodySize = Protocol.MaxBodyBytes;
            k.AddServerHeader = false;
        });

        var pool = new WorkerPool(options.Workers, options.TtEntries, options.MemoEntries, options.QueueLimit, options.QueueWaitMs);
        var service = new SolveService(db);
        var app = builder.Build();
        var started = Stopwatch.GetTimestamp();
        string version = typeof(EngineHost).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

        app.MapGet("/v1/health", () =>
        {
            var buffer = new System.Buffers.ArrayBufferWriter<byte>(512);
            using (var w = new System.Text.Json.Utf8JsonWriter(buffer))
            {
                w.WriteStartObject();
                w.WriteNumber("protocol", Protocol.Version);
                w.WriteString("status", "ok");
                w.WriteString("engine", "TTC.Siren.Engine");
                w.WriteString("version", version);
                w.WriteString("cardDbSha256", sha);
                w.WriteNumber("cardCount", db.Count);
                w.WriteNumber("workers", pool.Workers);
                w.WriteNumber("workersCreated", pool.Created);
                w.WriteNumber("idleWorkers", pool.Idle);
                w.WriteNumber("queued", pool.Waiting);
                w.WriteNumber("queueLimit", pool.QueueLimit);
                w.WriteNumber("busyAnswers", pool.BusyAnswers);
                w.WriteNumber("tableMemoryBytes", pool.MemoryBytes);
                w.WriteNumber("uptimeSeconds", Stopwatch.GetElapsedTime(started).TotalSeconds);
                w.WriteEndObject();
            }
            return Json(200, buffer.WrittenSpan.ToArray());
        });

        app.MapPost("/v1/solve", async (HttpRequest http, CancellationToken ct) =>
        {
            long arrived = Stopwatch.GetTimestamp();
            byte[] body;
            try
            {
                using var ms = new MemoryStream();
                await http.Body.CopyToAsync(ms, ct).ConfigureAwait(false);
                body = ms.ToArray();
            }
            catch (Microsoft.AspNetCore.Http.BadHttpRequestException e)
            {
                return Json(400, ResponseWriter.Error("invalid_request", e.Message));
            }

            SolveRequest request;
            try
            {
                request = RequestReader.Parse(body);
            }
            catch (RequestException e)
            {
                return Json(400, ResponseWriter.Error("invalid_request", e.Message));
            }

            var (worker, reason) = await pool.AcquireAsync(ct).ConfigureAwait(false);
            if (worker is null)
            {
                return Json(503, ResponseWriter.Error("busy", "all workers are busy",
                    reason == BusyReason.QueueFull ? "queue_full" : "queue_timeout"));
            }

            try
            {
                var result = service.Solve(request, worker);
                return Json(200, ResponseWriter.Solve(result));
            }
            catch (RequestException e)
            {
                return Json(400, ResponseWriter.Error("invalid_request", e.Message));
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                return Json(500, ResponseWriter.Error("error", e.GetType().Name + ": " + e.Message));
            }
            finally
            {
                pool.Release(worker);
            }
        });

        return new EngineHost(app, options, pool, sha);
    }

    private static IResult Json(int status, byte[] body) => new RawJsonResult(status, body);

    public async Task StartAsync(CancellationToken ct = default)
    {
        await _app.StartAsync(ct).ConfigureAwait(false);
        var addresses = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses;
        BaseUrl = addresses.First().Replace("[::]", "127.0.0.1", StringComparison.Ordinal);
    }

    public Task StopAsync(CancellationToken ct = default) => _app.StopAsync(ct);

    public Task WaitForShutdownAsync(CancellationToken ct = default) => _app.WaitForShutdownAsync(ct);

    public async ValueTask DisposeAsync()
    {
        await _app.DisposeAsync().ConfigureAwait(false);
    }
}
