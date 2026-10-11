using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Proxy;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

internal sealed class CapturingLoggerProvider : ILoggerProvider
{
    public List<string> Messages { get; } = new();

    public ILogger CreateLogger(string categoryName) => new Capture(this, categoryName);

    public void Dispose()
    {
    }

    private sealed class Capture : ILogger
    {
        private readonly CapturingLoggerProvider _owner;
        private readonly string _category;

        public Capture(CapturingLoggerProvider owner, string category)
        {
            _owner = owner;
            _category = category;
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            lock (_owner.Messages) _owner.Messages.Add($"{_category} [{logLevel}] {formatter(state, exception)} {exception}");
        }
    }
}

/// <summary>Stops and disposes a started server at the end of a test.</summary>
internal sealed class HostDisposer : IAsyncDisposable
{
    private readonly ServerHost _host;

    public HostDisposer(ServerHost host) => _host = host;

    public async ValueTask DisposeAsync()
    {
        await _host.StopAsync();
        await _host.DisposeAsync();
    }
}

/// <summary>
/// Step D, cloud foundation: API keys (stored as hashes, handed out once by the admin CLI, honoured from either header, revoked without a restart),
/// per-key rate limits and in-flight limits with 429 and Retry-After, HTTPS with HSTS, CORS for exact origins, security headers, loopback-only operator
/// endpoints, the proxy to the Python app (401/502/504), the placeholder page and static files — and that no key ever appears in a log.
/// </summary>
public sealed class SecurityTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-sec-" + Guid.NewGuid().ToString("N"));
    private readonly List<IAsyncDisposable> _toDispose = new();

    public SecurityTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        foreach (var d in _toDispose) d.DisposeAsync().AsTask().GetAwaiter().GetResult();
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private string KeyFile => Path.Combine(_dir, "keys.json");

    private async Task<ServerHost> StartAsync(Action<ServerOptions>? tweak = null, TimeProvider? time = null, ILoggerProvider? logger = null)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries,
            GameLog = new GameLogOptions { Directory = Path.Combine(_dir, "logs") },
            Security = new SecurityOptions { KeyStoreFile = KeyFile },
        };
        tweak?.Invoke(options);
        var host = ServerHost.Create(options, time: time, loggerProvider: logger);
        await host.StartAsync();
        _toDispose.Add(new HostDisposer(host));
        return host;
    }

    private static HttpRequestMessage Req(HttpMethod method, string url, string? bearer = null, string? apiKey = null, string? body = null)
    {
        var m = new HttpRequestMessage(method, url);
        if (bearer is not null) m.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
        if (apiKey is not null) m.Headers.Add("X-Api-Key", apiKey);
        if (body is not null) m.Content = new StringContent(body, Encoding.UTF8, "application/json");
        return m;
    }

    private static string OpeningBody(int maxTimeMs = 200, int maxDepth = 4) =>
        "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4},{\"numU\":5,\"numL\":4,\"numD\":6,\"numR\":5},{\"numU\":2,\"numL\":2,\"numD\":7,\"numR\":2}," +
        "{\"numU\":4,\"numL\":3,\"numD\":3,\"numR\":3},{\"numU\":1,\"numL\":3,\"numD\":7,\"numR\":2}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}," +
        "{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}]," +
        $"\"myOwner\":2,\"currentPlayer\":2,\"rules\":\"加算\",\"max_depth\":{maxDepth},\"max_time\":{maxTimeMs / 1000.0:0.###}}}";

    // =============================================================================================================== keys

    [Fact]
    public void Keys_are_random_shown_once_and_stored_only_as_hashes()
    {
        var store = new ApiKeyStore(KeyFile);
        var (a, secretA) = store.Add("alice", 60, 5, 2);
        var (b, secretB) = store.Add("bob", 120, 20, 4);
        Assert.Matches("^ttc_[A-Za-z0-9_-]{43}$", secretA);
        Assert.NotEqual(secretA, secretB);
        Assert.NotEqual(a.Id, b.Id);
        string file = File.ReadAllText(KeyFile);
        Assert.DoesNotContain(secretA, file);
        Assert.DoesNotContain(secretB, file);
        Assert.Contains(ApiKeyStore.HashOf(secretA), file);
        Assert.Equal(a.Id, store.Find(secretA)!.Id);
        Assert.Null(store.Find(secretA + "x"));
        Assert.Throws<InvalidOperationException>(() => store.Add("Alice", 1, 1, 1)); // names are unique, case-insensitively

        // the second store object sees what the first wrote, and the other way round (the CLI edits the file while the server runs)
        var other = new ApiKeyStore(KeyFile);
        Assert.Equal(2, other.List().Count);
        Assert.True(other.Revoke("alice"));
        var time = new ManualTimeProvider(DateTimeOffset.UtcNow);
        var watching = new ApiKeyStore(KeyFile, time);
        Assert.False(watching.Find(secretA)!.Enabled);
        Assert.True(other.SetLimit(b.Id, 30, null, 1));
        time.Advance(TimeSpan.FromSeconds(2));
        Assert.Equal(30, watching.Find(secretB)!.RatePerMinute);
        Assert.Equal(1, watching.Find(secretB)!.MaxInFlight);
        Assert.Equal(20, watching.Find(secretB)!.Burst);
    }

    [Fact]
    public void The_admin_cli_creates_lists_revokes_and_limits_keys_and_prints_the_secret_once()
    {
        var security = new SecurityOptions { KeyStoreFile = KeyFile };
        string Run(params string[] args)
        {
            var output = new StringWriter();
            var error = new StringWriter();
            int code = KeyCli.Run(args, security, output, error);
            Assert.True(code == 0, $"keys {string.Join(' ', args)} exit {code}: {error}");
            return output.ToString();
        }
        string added = Run("keys", "add", "--name", "carol", "--rate", "90", "--burst", "7", "--inflight", "3");
        string secret = added.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).First(l => l.StartsWith("ttc_", StringComparison.Ordinal));
        Assert.Equal(1, added.Split(secret).Length - 1); // printed exactly once
        string list = Run("keys", "list");
        Assert.Contains("carol", list);
        Assert.Contains("enabled", list);
        Assert.DoesNotContain(secret, list);
        Assert.DoesNotContain(secret[4..20], list);
        Run("keys", "set-limit", "carol", "--rate", "10");
        Assert.Equal(10, new ApiKeyStore(KeyFile).List()[0].RatePerMinute);
        Run("keys", "revoke", "carol");
        Assert.Contains("REVOKED", Run("keys", "list"));
        var error = new StringWriter();
        Assert.Equal(1, KeyCli.Run(new[] { "keys", "revoke", "nobody" }, security, new StringWriter(), error));
        Assert.Equal(2, KeyCli.Run(new[] { "keys", "add" }, security, new StringWriter(), error));
        Assert.Equal(2, KeyCli.Run(new[] { "keys", "frobnicate" }, security, new StringWriter(), error));
    }

    [Fact]
    public async Task A_key_is_required_either_header_works_a_revoked_key_is_refused_without_a_restart()
    {
        var time = new ManualTimeProvider(DateTimeOffset.UtcNow);
        var store = new ApiKeyStore(KeyFile);
        var (_, secret) = store.Add("alice", 6000, 100, 8);
        await using var _ = new Nothing();
        var host = await StartAsync(time: time);
        using var http = new HttpClient();

        var none = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", body: OpeningBody()));
        Assert.Equal(401, (int)none.StatusCode);
        Assert.Contains("Bearer", none.Headers.WwwAuthenticate.ToString());
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret + "0", body: OpeningBody()))).StatusCode);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: OpeningBody()))).StatusCode);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", apiKey: secret, body: OpeningBody()))).StatusCode);
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games"))).StatusCode);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games", apiKey: secret))).StatusCode);
        // the health check and the placeholder page need no key
        Assert.Equal(200, (int)(await http.GetAsync(host.BaseUrl + "/health")).StatusCode);

        // revoke through the file (as the CLI does): refused within a second, with 403, no restart
        Assert.True(new ApiKeyStore(KeyFile).Revoke("alice"));
        time.Advance(TimeSpan.FromSeconds(2));
        var revoked = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: OpeningBody()));
        Assert.Equal(403, (int)revoked.StatusCode);
        Assert.Contains("revoked", await revoked.Content.ReadAsStringAsync());
    }

    private sealed class Nothing : IAsyncDisposable
    {
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    [Fact]
    public async Task The_loopback_exemption_is_off_by_default_and_gives_the_development_key_when_switched_on()
    {
        new ApiKeyStore(KeyFile).Add("alice", 6000, 100, 8);
        using var http = new HttpClient();
        var strict = await StartAsync();
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Post, strict.BaseUrl + "/ai_move", body: OpeningBody()))).StatusCode);
        var dev = await StartAsync(o => o.Security.AllowLoopbackWithoutKey = true);
        var response = await http.SendAsync(Req(HttpMethod.Post, dev.BaseUrl + "/ai_move", body: OpeningBody()));
        Assert.Equal(200, (int)response.StatusCode);
        Assert.Single(dev.GameLog.Store.List("local-dev", 0, 10).Items); // the games of the exempt caller belong to the key "local-dev"
        Assert.Throws<ArgumentException>(() => new SecurityOptions { AllowLoopbackWithoutKey = true }.Validate("0.0.0.0", false));
    }

    // =============================================================================================================== limits

    [Fact]
    public async Task The_rate_limit_answers_429_with_retry_after_and_each_key_has_its_own_bucket_which_refills()
    {
        var time = new ManualTimeProvider(DateTimeOffset.UtcNow);
        var store = new ApiKeyStore(KeyFile, time);
        var (_, slow) = store.Add("slow", ratePerMinute: 60, burst: 3, maxInFlight: 8);
        var (_, other) = store.Add("other", 60, 3, 8);
        var host = await StartAsync(time: time);
        using var http = new HttpClient();
        HttpRequestMessage Call(string key) => Req(HttpMethod.Get, host.BaseUrl + "/v1/games", bearer: key);

        for (int i = 0; i < 3; i++) Assert.Equal(200, (int)(await http.SendAsync(Call(slow))).StatusCode);
        var limited = await http.SendAsync(Call(slow));
        Assert.Equal(429, (int)limited.StatusCode);
        Assert.True(limited.Headers.RetryAfter!.Delta!.Value.TotalSeconds >= 1);
        using (var doc = JsonDocument.Parse(await limited.Content.ReadAsStringAsync()))
        {
            Assert.Contains("rate limit", doc.RootElement.GetProperty("error").GetString());
            Assert.True(doc.RootElement.GetProperty("retry_after_seconds").GetInt32() >= 1);
        }
        Assert.Equal(200, (int)(await http.SendAsync(Call(other))).StatusCode); // another key is unaffected

        time.Advance(TimeSpan.FromSeconds(1.1)); // one request per second refills
        Assert.Equal(200, (int)(await http.SendAsync(Call(slow))).StatusCode);
        Assert.Equal(429, (int)(await http.SendAsync(Call(slow))).StatusCode);

        // set-limit takes effect without a restart
        Assert.True(new ApiKeyStore(KeyFile).SetLimit("slow", 6000, 50, null));
        time.Advance(TimeSpan.FromSeconds(2));
        for (int i = 0; i < 10; i++) Assert.Equal(200, (int)(await http.SendAsync(Call(slow))).StatusCode);
    }

    [Fact]
    public async Task The_in_flight_limit_refuses_a_second_concurrent_request_of_the_key_only()
    {
        var store = new ApiKeyStore(KeyFile);
        var (_, one) = store.Add("one", 6000, 100, maxInFlight: 1);
        var (_, two) = store.Add("two", 6000, 100, maxInFlight: 1);
        var host = await StartAsync();
        using var http = new HttpClient();
        var first = http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: one, body: OpeningBody(1500, maxDepth: 10)));
        for (int i = 0; i < 200 && host.Pool.Idle == host.Pool.Workers; i++) await Task.Delay(10); // the search has started
        var second = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: one, body: OpeningBody(100)));
        Assert.Equal(429, (int)second.StatusCode);
        Assert.Equal(1, second.Headers.RetryAfter!.Delta!.Value.TotalSeconds);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: two, body: OpeningBody(100)))).StatusCode);
        Assert.Equal(200, (int)(await first).StatusCode);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: one, body: OpeningBody(100)))).StatusCode); // free again
    }

    [Fact]
    public async Task The_number_of_open_event_streams_per_key_is_limited()
    {
        var (_, key) = new ApiKeyStore(KeyFile).Add("alice", 6000, 100, 8);
        var host = await StartAsync(o => o.Security.MaxLiveStreamsPerKey = 2);
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var open = new List<HttpResponseMessage>();
        for (int i = 0; i < 2; i++)
        {
            var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", bearer: key), HttpCompletionOption.ResponseHeadersRead);
            Assert.Equal(200, (int)r.StatusCode);
            open.Add(r);
        }
        var third = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", bearer: key), HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(429, (int)third.StatusCode);
        foreach (var r in open) r.Dispose();
        for (int i = 0; i < 100 && host.Hub.SubscriberCount(new ApiKeyStore(KeyFile).List()[0].Id) > 0; i++) await Task.Delay(20);
        var again = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", bearer: key), HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(200, (int)again.StatusCode);
        again.Dispose();
    }

    // =============================================================================================================== secrets in logs

    [Fact]
    public async Task No_key_appears_in_any_log_message_or_game_log_file()
    {
        var store = new ApiKeyStore(KeyFile);
        var (_, secret) = store.Add("alice", 6000, 100, 8);
        var capture = new CapturingLoggerProvider();
        var host = await StartAsync(logger: capture);
        using var http = new HttpClient();
        const string fakeKey = "ttc_THIS-IS-NOT-A-REAL-KEY-BUT-LOOKS-LIKE-ONE-0123456789";

        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: OpeningBody()));
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", apiKey: secret, body: "{broken"));
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: fakeKey, body: OpeningBody()));
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games", apiKey: fakeKey));
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", bearer: fakeKey), HttpCompletionOption.ResponseHeadersRead);
        new ApiKeyStore(KeyFile).Revoke("alice");
        await Task.Delay(1100);
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: OpeningBody()));

        string everything;
        lock (capture.Messages) everything = string.Join("\n", capture.Messages);
        Assert.NotEmpty(capture.Messages);
        Assert.DoesNotContain(secret, everything);
        Assert.DoesNotContain(secret[4..], everything);
        Assert.DoesNotContain(fakeKey, everything);
        Assert.DoesNotContain("Bearer", everything);
        Assert.DoesNotContain("ttc_THIS", everything);                     // not even a prefix of a presented key is logged
        Assert.DoesNotContain(secret[..8], everything);
        Assert.Contains("rejected unknown API key", everything);
        Assert.Contains("rejected revoked API key k-", everything);       // a revoked key by its public id
        foreach (var file in Directory.EnumerateFiles(_dir, "*", SearchOption.AllDirectories).Where(f => !f.EndsWith("keys.json", StringComparison.Ordinal)))
            Assert.DoesNotContain(secret, File.ReadAllText(file));
        _output.WriteLine($"{capture.Messages.Count} log messages checked");
    }

    // =============================================================================================================== HTTPS, headers, CORS, operator endpoints

    private string MakePfx(string password)
    {
        using var rsa = RSA.Create(2048);
        var request = new CertificateRequest("CN=localhost", rsa, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var san = new SubjectAlternativeNameBuilder();
        san.AddDnsName("localhost");
        san.AddIpAddress(IPAddress.Loopback);
        request.CertificateExtensions.Add(san.Build());
        using var cert = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddDays(30));
        string path = Path.Combine(_dir, "test.pfx");
        File.WriteAllBytes(path, cert.Export(X509ContentType.Pfx, password));
        return path;
    }

    [Fact]
    public async Task Https_serves_with_hsts_and_the_security_headers_and_misconfigurations_are_refused()
    {
        string pfx = MakePfx("test-password");
        var host = await StartAsync(o =>
        {
            o.Security.RequireKey = false;
            o.Security.Https = new HttpsOptions { Enabled = true, Port = 0, PfxPath = pfx, PfxPassword = "test-password" };
        });
        Assert.StartsWith("https://", host.HttpsUrl);
        var handler = new HttpClientHandler { ServerCertificateCustomValidationCallback = (_, cert, _, _) => cert!.Subject == "CN=localhost" };
        using var https = new HttpClient(handler);
        var secure = await https.GetAsync(host.HttpsUrl + "/health");
        Assert.Equal(200, (int)secure.StatusCode);
        Assert.Contains("max-age=31536000", secure.Headers.GetValues("Strict-Transport-Security").Single());
        Assert.Equal("nosniff", secure.Headers.GetValues("X-Content-Type-Options").Single());
        Assert.Equal("DENY", secure.Headers.GetValues("X-Frame-Options").Single());
        Assert.Equal("no-referrer", secure.Headers.GetValues("Referrer-Policy").Single());
        using var plain = new HttpClient();
        var insecure = await plain.GetAsync(host.BaseUrl + "/health");
        Assert.False(insecure.Headers.Contains("Strict-Transport-Security")); // HSTS only over HTTPS
        Assert.Equal("nosniff", insecure.Headers.GetValues("X-Content-Type-Options").Single());

        // what must not start
        var options = new ServerOptions { Host = "0.0.0.0", Port = 0, Security = new SecurityOptions { KeyStoreFile = KeyFile } };
        Assert.Contains("without HTTPS", Assert.Throws<ArgumentException>(options.Validate).Message);
        options.Security.BehindTlsProxy = true;
        options.Validate(); // a reverse proxy terminates TLS in front: allowed
        Assert.Throws<ArgumentException>(() => new SecurityOptions { Https = new HttpsOptions { Enabled = true } }.Validate("127.0.0.1", true)); // no certificate
        Assert.Throws<ArgumentException>(() => new SecurityOptions { CorsOrigins = { "*" } }.Validate("127.0.0.1", true));
        Assert.Throws<ArgumentException>(() => new ServerOptions { HttpEnabled = false }.Validate());
    }

    [Fact]
    public async Task Cors_allows_only_the_configured_origins_and_never_a_wildcard()
    {
        var (_, key) = new ApiKeyStore(KeyFile).Add("alice", 6000, 100, 8);
        var host = await StartAsync(o => o.Security.CorsOrigins.Add("https://app.example.org"));
        using var http = new HttpClient();

        var preflight = new HttpRequestMessage(HttpMethod.Options, host.BaseUrl + "/ai_move");
        preflight.Headers.Add("Origin", "https://app.example.org");
        preflight.Headers.Add("Access-Control-Request-Method", "POST");
        var ok = await http.SendAsync(preflight);
        Assert.Equal(204, (int)ok.StatusCode);
        Assert.Equal("https://app.example.org", ok.Headers.GetValues("Access-Control-Allow-Origin").Single());
        Assert.Contains("Authorization", ok.Headers.GetValues("Access-Control-Allow-Headers").Single());
        Assert.Equal("Origin", ok.Headers.GetValues("Vary").Single());

        var bad = new HttpRequestMessage(HttpMethod.Options, host.BaseUrl + "/ai_move");
        bad.Headers.Add("Origin", "https://evil.example.org");
        var refused = await http.SendAsync(bad);
        Assert.Equal(403, (int)refused.StatusCode);
        Assert.False(refused.Headers.Contains("Access-Control-Allow-Origin"));

        var real = Req(HttpMethod.Get, host.BaseUrl + "/v1/games", bearer: key);
        real.Headers.Add("Origin", "https://app.example.org");
        Assert.Equal("https://app.example.org", (await http.SendAsync(real)).Headers.GetValues("Access-Control-Allow-Origin").Single());
        var foreign = Req(HttpMethod.Get, host.BaseUrl + "/v1/games", bearer: key);
        foreign.Headers.Add("Origin", "https://evil.example.org");
        var foreignAnswer = await http.SendAsync(foreign);
        Assert.Equal(200, (int)foreignAnswer.StatusCode);
        Assert.False(foreignAnswer.Headers.Contains("Access-Control-Allow-Origin")); // the browser will not hand the answer to that page
    }

    [Fact]
    public async Task Operator_endpoints_work_from_this_machine_but_not_through_a_proxy_and_oversized_bodies_are_refused()
    {
        var (_, key) = new ApiKeyStore(KeyFile).Add("alice", 6000, 100, 8);
        var host = await StartAsync();
        using var http = new HttpClient();
        var status = await http.GetAsync(host.BaseUrl + "/admin/status");
        Assert.Equal(200, (int)status.StatusCode);
        using (var doc = JsonDocument.Parse(await status.Content.ReadAsStringAsync()))
        {
            Assert.Equal(1, doc.RootElement.GetProperty("keys").GetInt32());
            Assert.Equal(2, doc.RootElement.GetProperty("workers").GetInt32());
        }
        var proxied = new HttpRequestMessage(HttpMethod.Get, host.BaseUrl + "/admin/status");
        proxied.Headers.Add("X-Forwarded-For", "203.0.113.9");
        Assert.Equal(403, (int)(await http.SendAsync(proxied)).StatusCode);
        Assert.Equal(200, (int)(await http.PostAsync(host.BaseUrl + "/admin/reload-keys", null)).StatusCode);

        var big = Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: key, body: "{\"x\":\"" + new string('a', 1_200_000) + "\"}");
        var answer = await http.SendAsync(big);
        Assert.True((int)answer.StatusCode is 413 or 400, $"status {(int)answer.StatusCode}");
    }

    // =============================================================================================================== proxy

    private async Task<(WebApplication App, string Url)> StartUpstreamAsync(RequestDelegate handler)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var app = builder.Build();
        app.Run(handler);
        await app.StartAsync();
        _toDispose.Add(new AppDisposer(app));
        return (app, app.Urls.First());
    }

    private sealed class AppDisposer : IAsyncDisposable
    {
        private readonly WebApplication _app;

        public AppDisposer(WebApplication app) => _app = app;

        public async ValueTask DisposeAsync()
        {
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }

    [Fact]
    public async Task The_three_api_endpoints_are_forwarded_to_the_python_app_behind_the_key_and_failures_are_502_and_504()
    {
        var (_, key) = new ApiKeyStore(KeyFile).Add("alice", 6000, 100, 8);
        var seen = new List<(string Path, string Body, string? Type, bool HadAuth, bool HadKey)>();
        var (_, upstreamUrl) = await StartUpstreamAsync(async ctx =>
        {
            using var reader = new StreamReader(ctx.Request.Body);
            string body = await reader.ReadToEndAsync();
            lock (seen) seen.Add((ctx.Request.Path, body, ctx.Request.ContentType, ctx.Request.Headers.ContainsKey("Authorization"), ctx.Request.Headers.ContainsKey("X-Api-Key")));
            if (ctx.Request.Path == "/api/winrate" && body.Contains("slow")) await Task.Delay(3000);
            ctx.Response.StatusCode = body.Contains("teapot") ? 418 : 200;
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync("{\"echo\":" + (body.Length == 0 ? "null" : body) + "}");
        });
        var host = await StartAsync(o =>
        {
            o.Proxy = new ProxyOptions { PythonUrl = upstreamUrl, TimeoutSeconds = 1 };
        });
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

        foreach (var path in PythonProxy.Paths)
        {
            Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + path, body: "{\"a\":1}"))).StatusCode); // no key, nothing reaches Python
            var answer = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + path, bearer: key, body: "{\"ownedIds\":[1,2,3]}"));
            Assert.Equal(200, (int)answer.StatusCode);
            Assert.Equal("{\"echo\":{\"ownedIds\":[1,2,3]}}", await answer.Content.ReadAsStringAsync());
            Assert.StartsWith("application/json", answer.Content.Headers.ContentType!.ToString());
        }
        lock (seen)
        {
            Assert.Equal(PythonProxy.Paths, seen.Select(s => s.Path));      // only the authenticated calls arrived, at their own paths
            Assert.All(seen, s => Assert.False(s.HadAuth || s.HadKey));      // the caller's key is not passed on
            Assert.All(seen, s => Assert.StartsWith("application/json", s.Type));
        }
        // the upstream's status is passed on
        var teapot = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/api/recommend", bearer: key, body: "{\"teapot\":true}"));
        Assert.Equal(418, (int)teapot.StatusCode);
        // other paths and other methods are not proxied
        Assert.Equal(404, (int)(await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/api/other", bearer: key, body: "{}"))).StatusCode);
        Assert.Equal(405, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/api/winrate", bearer: key))).StatusCode);
        // too slow: 504
        var slow = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/api/winrate", bearer: key, body: "{\"slow\":true}"));
        Assert.Equal(504, (int)slow.StatusCode);

        // nothing listens: 502
        var dead = await StartAsync(o => o.Proxy = new ProxyOptions { PythonUrl = "http://127.0.0.1:1", TimeoutSeconds = 5 });
        var down = await http.SendAsync(Req(HttpMethod.Post, dead.BaseUrl + "/api/recommend", bearer: key, body: "{}"));
        Assert.Equal(502, (int)down.StatusCode);
        Assert.Contains("not reachable", await down.Content.ReadAsStringAsync());
    }

    // =============================================================================================================== page

    [Fact]
    public async Task The_placeholder_page_shows_the_product_and_the_status_and_a_static_directory_replaces_it()
    {
        using var http = new HttpClient();
        var host = await StartAsync();
        var page = await http.GetAsync(host.BaseUrl + "/");
        Assert.Equal(200, (int)page.StatusCode);
        string html = await page.Content.ReadAsStringAsync();
        Assert.Contains("TTC Siren", html);
        Assert.Contains("运行中", html);
        Assert.Contains("default-src 'none'", html);
        Assert.DoesNotContain("<script", html);

        string wwwroot = Path.Combine(_dir, "www");
        Directory.CreateDirectory(wwwroot);
        File.WriteAllText(wwwroot + "/index.html", "<!doctype html><title>real ui</title>");
        File.WriteAllText(wwwroot + "/app.js", "console.log(1)");
        var withUi = await StartAsync(o => o.StaticDirectory = wwwroot);
        Assert.Contains("real ui", await http.GetStringAsync(withUi.BaseUrl + "/"));
        var js = await http.GetAsync(withUi.BaseUrl + "/app.js");
        Assert.Equal(200, (int)js.StatusCode);
        Assert.Contains("default-src 'self'", js.Headers.GetValues("Content-Security-Policy").Single());
    }
}
