using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// The web login (decision A-27): a key is exchanged ONCE for a session cookie (HttpOnly, SameSite=Strict, Secure behind TLS); the key never goes back to the browser,
/// is not stored and is not logged; the session ends on logout, expiry or when its key is revoked; state-changing requests that the cookie authenticates must come from the
/// site's own pages (CSRF); logins are throttled per REAL client address (a header cannot choose the address) and audited; the plugin's endpoint accepts a key only.
/// </summary>
public sealed class WebSessionTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-web-" + Guid.NewGuid().ToString("N"));
    private readonly List<IAsyncDisposable> _hosts = new();
    private readonly ManualTimeProvider _time = new(DateTimeOffset.UtcNow);

    public WebSessionTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        foreach (var h in _hosts) h.DisposeAsync().AsTask().GetAwaiter().GetResult();
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private string KeyFile => Path.Combine(_dir, "keys.json");
    private string AuditFile => Path.Combine(_dir, "audit.jsonl");

    private async Task<ServerHost> StartAsync(Action<ServerOptions>? tweak = null, Microsoft.Extensions.Logging.ILoggerProvider? logger = null)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries,
            GameLog = new GameLogOptions { Directory = Path.Combine(_dir, "logs") },
            Security = new SecurityOptions { KeyStoreFile = KeyFile, Sessions = { AuditLogFile = AuditFile, LoginBurst = 100, LoginAttemptsPerMinute = 6000 } },
        };
        tweak?.Invoke(options);
        var host = ServerHost.Create(options, time: _time, loggerProvider: logger);
        await host.StartAsync();
        _hosts.Add(new HostDisposer(host));
        return host;
    }

    private (string Id, string Secret) NewKey(string name = "alice", int rate = 6000, int burst = 100, int inFlight = 8)
    {
        var (record, secret) = new ApiKeyStore(KeyFile, _time).Add(name, rate, burst, inFlight);
        return (record.Id, secret);
    }

    private static readonly HttpClientHandler NoCookies = new() { UseCookies = false };

    private static HttpClient Client() => new(new HttpClientHandler { UseCookies = false });

    private static HttpRequestMessage Req(HttpMethod method, string url, string? cookie = null, string? bearer = null, string? body = null, params (string Name, string Value)[] headers)
    {
        var m = new HttpRequestMessage(method, url);
        if (cookie is not null) m.Headers.Add("Cookie", cookie);
        if (bearer is not null) m.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
        if (body is not null) m.Content = new StringContent(body, Encoding.UTF8, "application/json");
        foreach (var (n, v) in headers) m.Headers.TryAddWithoutValidation(n, v);
        return m;
    }

    private static string LoginBody(string key) => JsonSerializer.Serialize(new { key });

    /// <summary>Logs in and returns the cookie as the browser would send it back (<c>name=value</c>) and the raw Set-Cookie header.</summary>
    private static async Task<(string Cookie, string SetCookie, string Id, JsonDocument Body)> LoginAsync(HttpClient http, ServerHost host, string key, params (string, string)[] headers)
    {
        using var response = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(key), headers: headers));
        string text = await response.Content.ReadAsStringAsync();
        Assert.True(response.StatusCode == HttpStatusCode.OK, $"login: {(int)response.StatusCode} {text}");
        string setCookie = response.Headers.GetValues("Set-Cookie").Single();
        string pair = setCookie.Split(';')[0];
        return (pair, setCookie, pair[(pair.IndexOf('=') + 1)..], JsonDocument.Parse(text));
    }

    private string Origin(ServerHost host) => host.BaseUrl; // http://127.0.0.1:port: the page's own origin

    // ============================================================================================================================ the cookie

    [Fact]
    public async Task Login_answers_with_a_random_256_bit_session_in_a_cookie_that_script_cannot_read_and_other_sites_cannot_send()
    {
        var (keyId, secret) = NewKey();
        var host = await StartAsync();
        using var http = Client();

        var (cookie1, setCookie, id1, body) = await LoginAsync(http, host, secret);
        _output.WriteLine("Set-Cookie: " + Regex.Replace(setCookie, "=[A-Za-z0-9_-]{43}", "=<id>"));
        Assert.Matches("^ttc_session=[A-Za-z0-9_-]{43}(;|$)", setCookie);            // 32 random bytes in base64url
        Assert.Contains("httponly", setCookie.ToLowerInvariant());
        Assert.Contains("samesite=strict", setCookie.ToLowerInvariant());
        Assert.Contains("path=/", setCookie.ToLowerInvariant());
        Assert.Contains("max-age=43200", setCookie.ToLowerInvariant());              // 12 hours
        Assert.DoesNotContain("secure", setCookie.ToLowerInvariant());               // a plain http loopback server: there is no TLS to be secure about
        Assert.DoesNotContain("domain=", setCookie.ToLowerInvariant());              // host-only
        Assert.NotEqual(secret, id1);                                                // the cookie is not the key
        Assert.DoesNotContain(secret, setCookie);

        // the body names the key by its public id and name -- never the secret
        Assert.True(body.RootElement.GetProperty("ok").GetBoolean());
        Assert.Equal(keyId, body.RootElement.GetProperty("key").GetProperty("id").GetString());
        Assert.Equal("alice", body.RootElement.GetProperty("key").GetProperty("name").GetString());
        Assert.DoesNotContain(secret, body.RootElement.GetRawText());
        Assert.DoesNotContain("hash", body.RootElement.GetRawText());

        var (cookie2, _, id2, _) = await LoginAsync(http, host, secret);              // every login is a new session
        Assert.NotEqual(id1, id2);
        Assert.Equal(2, host.Sessions.Count);
        Assert.NotEmpty(cookie2);
        Assert.Equal("no-store", (await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie1))).Headers.CacheControl?.ToString());
    }

    [Fact]
    public async Task The_cookie_is_secure_and_host_only_behind_a_tls_proxy_and_over_https()
    {
        var (_, secret) = NewKey();
        var behind = await StartAsync(o => o.Security.BehindTlsProxy = true);
        using var http = Client();
        var (cookie, setCookie, _, _) = await LoginAsync(http, behind, secret);
        Assert.Matches("^__Host-ttc_session=[A-Za-z0-9_-]{43}(;|$)", setCookie);     // __Host-: the browser accepts it only as Secure, host-only, Path=/
        Assert.Contains("secure", setCookie.ToLowerInvariant());
        Assert.Contains("httponly", setCookie.ToLowerInvariant());
        Assert.Contains("samesite=strict", setCookie.ToLowerInvariant());
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, behind.BaseUrl + "/v1/me", cookie: cookie))).StatusCode);
        // the plain-named cookie is not accepted where the secure one is expected
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Get, behind.BaseUrl + "/v1/me", cookie: cookie.Replace("__Host-", "")))).StatusCode);

        // directly over HTTPS (Kestrel with a certificate)
        string pfx = Path.Combine(_dir, "test.pfx");
        using (var rsa = RSA.Create(2048))
        {
            var request = new CertificateRequest("CN=localhost", rsa, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            var san = new SubjectAlternativeNameBuilder();
            san.AddIpAddress(IPAddress.Loopback);
            request.CertificateExtensions.Add(san.Build());
            using var cert = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddDays(30));
            File.WriteAllBytes(pfx, cert.Export(X509ContentType.Pfx, "pw"));
        }
        var tls = await StartAsync(o => o.Security.Https = new HttpsOptions { Enabled = true, Port = 0, PfxPath = pfx, PfxPassword = "pw" });
        using var https = new HttpClient(new HttpClientHandler { UseCookies = false, ServerCertificateCustomValidationCallback = (_, _, _, _) => true });
        using var response = await https.SendAsync(Req(HttpMethod.Post, tls.HttpsUrl + "/v1/session", body: LoginBody(secret)));
        Assert.Equal(200, (int)response.StatusCode);
        string tlsCookie = response.Headers.GetValues("Set-Cookie").Single();
        Assert.StartsWith("__Host-ttc_session=", tlsCookie);
        Assert.Contains("secure", tlsCookie.ToLowerInvariant());
    }

    // ============================================================================================================================ what a failed login says

    [Fact]
    public async Task A_login_that_fails_says_only_login_failed_for_every_reason_and_sets_no_cookie()
    {
        var (_, good) = NewKey("alice");
        var (_, revokedSecret) = NewKey("bob");
        new ApiKeyStore(KeyFile, _time).Revoke("bob");
        _time.Advance(TimeSpan.FromSeconds(2));
        var host = await StartAsync();
        using var http = Client();

        var bodies = new[]
        {
            LoginBody("ttc_" + new string('x', 43)), LoginBody(good + "0"), LoginBody(good[..^1]), LoginBody(""), LoginBody(revokedSecret),
            "{}", "{\"key\":5}", "{\"key\":null}", "[\"x\"]", "null", "not json", "", "{\"key\":\"" + new string('k', 500) + "\"}",
        };
        string? first = null;
        foreach (var b in bodies)
        {
            using var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: b));
            string text = await r.Content.ReadAsStringAsync();
            Assert.True(r.StatusCode == HttpStatusCode.Unauthorized, $"{b}: {(int)r.StatusCode} {text}");
            Assert.False(r.Headers.Contains("Set-Cookie"));
            Assert.Equal("{\"error\":\"login failed\"}\n", text);   // the same words for an unknown, a mistyped, a revoked and a malformed key
            first ??= text;
        }
        Assert.Equal(0, host.Sessions.Count);
        // a body over the small limit of the login (4 KiB) is refused as such
        using (var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: "{\"key\":\"" + new string('a', 20_000) + "\"}")))
            Assert.Equal(413, (int)r.StatusCode);
        // the right key still works
        await LoginAsync(http, host, good);
    }

    // ============================================================================================================================ which endpoint accepts what

    [Fact]
    public async Task The_cookie_is_accepted_on_the_game_logs_the_stream_and_me_and_never_on_the_plugins_endpoints()
    {
        var (keyId, secret) = NewKey();
        var host = await StartAsync();
        using var http = Client();
        var (cookie, _, _, _) = await LoginAsync(http, host, secret);

        // accepted
        foreach (string path in new[] { "/v1/games", "/v1/me" })
        {
            using var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + path, cookie: cookie));
            Assert.Equal(200, (int)r.StatusCode);
        }
        using (var live = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", cookie: cookie), HttpCompletionOption.ResponseHeadersRead))
        {
            Assert.Equal(200, (int)live.StatusCode);
            Assert.Equal("text/event-stream", live.Content.Headers.ContentType?.MediaType);
        }
        using (var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games/nothing", cookie: cookie))) Assert.Equal(404, (int)r.StatusCode); // authenticated, just not there

        // refused: the plugin's endpoint and the proxied ones want a key
        string opening = "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}],\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"\",\"max_depth\":2,\"max_time\":1}";
        foreach (string path in new[] { "/ai_move", "/api/recommend", "/api/draft/analyze", "/api/winrate" })
        {
            using var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + path, cookie: cookie, body: opening, headers: new[] { ("Origin", Origin(host)) }));
            Assert.True(r.StatusCode == HttpStatusCode.Unauthorized, $"{path} with a session cookie answered {(int)r.StatusCode}");
            Assert.Contains("API key required", await r.Content.ReadAsStringAsync());
        }
        Assert.Empty(host.GameLog.Store.List(keyId, 0, 10).Items);                       // and nothing was searched

        // nothing at all: refused where a login is needed, open where it is not
        foreach (string path in new[] { "/v1/games", "/v1/me", "/v1/live" })
        {
            using var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + path), HttpCompletionOption.ResponseHeadersRead);
            Assert.Equal(401, (int)r.StatusCode);
        }
        using (var r = await http.GetAsync(host.BaseUrl + "/health")) Assert.Equal(200, (int)r.StatusCode);

        // a key and a cookie together: the key decides (a stale cookie does not break an API client)
        using (var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: "ttc_session=" + new string('z', 43), bearer: secret)))
        {
            Assert.Equal(200, (int)r.StatusCode);
            using var me = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
            Assert.Equal("api_key", me.RootElement.GetProperty("via").GetString());
            Assert.Equal("plugin", me.RootElement.GetProperty("class").GetString());
        }
    }

    [Fact]
    public async Task Me_describes_the_key_and_the_session_and_never_reveals_a_secret()
    {
        var (keyId, secret) = NewKey("alice", rate: 90, burst: 7, inFlight: 3);
        var host = await StartAsync();
        using var http = Client();
        var (cookie, _, sessionId, _) = await LoginAsync(http, host, secret);

        using var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie));
        string text = await r.Content.ReadAsStringAsync();
        using var me = JsonDocument.Parse(text);
        Assert.Equal("session", me.RootElement.GetProperty("via").GetString());
        Assert.Equal("web", me.RootElement.GetProperty("class").GetString());
        var key = me.RootElement.GetProperty("key");
        Assert.Equal(keyId, key.GetProperty("id").GetString());
        Assert.Equal("alice", key.GetProperty("name").GetString());
        Assert.Equal(90, key.GetProperty("rate_per_minute").GetInt32());
        Assert.Equal(7, key.GetProperty("burst").GetInt32());
        Assert.Equal(3, key.GetProperty("max_in_flight").GetInt32());
        Assert.True(me.RootElement.GetProperty("session").GetProperty("expires").GetString()!.EndsWith('Z'));
        Assert.DoesNotContain(secret, text);
        Assert.DoesNotContain(ApiKeyStore.HashOf(secret), text);
        Assert.DoesNotContain(sessionId, text);

        using var viaKey = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", bearer: secret));
        using var meKey = JsonDocument.Parse(await viaKey.Content.ReadAsStringAsync());
        Assert.Equal("api_key", meKey.RootElement.GetProperty("via").GetString());
        Assert.Equal(JsonValueKind.Null, meKey.RootElement.GetProperty("session").ValueKind);
        Assert.DoesNotContain(secret, meKey.RootElement.GetRawText());
    }

    // ============================================================================================================================ ending a session

    [Fact]
    public async Task A_session_expires_after_its_lifetime_and_the_browser_is_told_to_forget_the_cookie()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync(o => o.Security.Sessions.LifetimeHours = 1);
        using var http = Client();
        var (cookie, setCookie, _, _) = await LoginAsync(http, host, secret);
        Assert.Contains("max-age=3600", setCookie.ToLowerInvariant());

        _time.Advance(TimeSpan.FromMinutes(59));
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie))).StatusCode);
        _time.Advance(TimeSpan.FromMinutes(2));
        using var expired = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie));
        Assert.Equal(401, (int)expired.StatusCode);
        Assert.Contains("expired", await expired.Content.ReadAsStringAsync());
        string cleared = expired.Headers.GetValues("Set-Cookie").Single().ToLowerInvariant();
        Assert.Contains("max-age=0", cleared);
        Assert.StartsWith("ttc_session=;", cleared);
        Assert.Equal(0, host.Sessions.Count);
    }

    [Fact]
    public async Task A_session_dies_when_its_key_is_revoked_without_a_restart_and_a_revoked_key_cannot_log_in()
    {
        var (_, secret) = NewKey();
        var (_, other) = NewKey("bob");
        var host = await StartAsync();
        using var http = Client();
        var (alice, _, _, _) = await LoginAsync(http, host, secret);
        var (bob, _, _, _) = await LoginAsync(http, host, other);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: alice))).StatusCode);

        Assert.True(new ApiKeyStore(KeyFile, _time).Revoke("alice"));                 // what `keys revoke alice` does
        _time.Advance(TimeSpan.FromSeconds(2));                                         // the server notices the file within a second
        using var dead = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games", cookie: alice));
        Assert.Equal(401, (int)dead.StatusCode);
        Assert.Contains("max-age=0", dead.Headers.GetValues("Set-Cookie").Single().ToLowerInvariant());
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: bob))).StatusCode); // another key's session is not touched
        Assert.Equal(1, host.Sessions.Count);

        using var login = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(secret)));
        Assert.Equal(401, (int)login.StatusCode);
        Assert.Equal("{\"error\":\"login failed\"}\n", await login.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Logout_ends_the_session_and_one_key_may_hold_only_so_many_sessions()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync(o => o.Security.Sessions.MaxPerKey = 2);
        using var http = Client();
        var (first, _, _, _) = await LoginAsync(http, host, secret);
        var (second, _, _, _) = await LoginAsync(http, host, secret);
        var (third, _, _, _) = await LoginAsync(http, host, secret);
        Assert.Equal(2, host.Sessions.Count);
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: first))).StatusCode);   // the oldest was ended to make room
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: second))).StatusCode);

        using var logout = await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session", cookie: third, headers: new[] { ("Origin", Origin(host)) }));
        Assert.Equal(200, (int)logout.StatusCode);
        Assert.Contains("\"logged_out\":true", await logout.Content.ReadAsStringAsync());
        Assert.Contains("max-age=0", logout.Headers.GetValues("Set-Cookie").Single().ToLowerInvariant());
        Assert.Equal(401, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: third))).StatusCode);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: second))).StatusCode); // only that session ended

        using var nothing = await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session"));   // no cookie: nothing to do, and nothing to forge
        Assert.Equal(200, (int)nothing.StatusCode);
        Assert.Contains("\"logged_out\":false", await nothing.Content.ReadAsStringAsync());
    }

    // ============================================================================================================================ CSRF

    [Fact]
    public async Task A_request_that_the_cookie_authenticates_and_that_changes_something_must_come_from_the_sites_own_pages()
    {
        var (keyId, secret) = NewKey();
        var host = await StartAsync();
        using var http = Client();
        var (cookie, _, _, _) = await LoginAsync(http, host, secret);

        // a game to protect (made with the key, as the plugin does)
        string gameBody = "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4},{\"numU\":5,\"numL\":4,\"numD\":6,\"numR\":5}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}],\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"\",\"max_depth\":2,\"max_time\":2,\"gameId\":\"csrf-1\"}";
        using (var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: gameBody))) Assert.Equal(200, (int)r.StatusCode);
        string url = host.BaseUrl + "/v1/games/csrf-1";
        string own = Origin(host);

        async Task<int> Delete(params (string, string)[] headers)
        {
            using var r = await http.SendAsync(Req(HttpMethod.Delete, url, cookie: cookie, headers: headers));
            return (int)r.StatusCode;
        }
        // every way a forged request could look: refused, and the game is still there
        Assert.Equal(403, await Delete());                                                                       // no Origin, no Referer
        Assert.Equal(403, await Delete(("Origin", "https://evil.example")));
        Assert.Equal(403, await Delete(("Origin", "null")));                                                     // a sandboxed frame or a redirect
        Assert.Equal(403, await Delete(("Origin", "http://127.0.0.1:1")));                                       // same host, other port
        Assert.Equal(403, await Delete(("Origin", "http://localhost:" + new Uri(host.BaseUrl).Port)));          // the same server under another name is another origin
        Assert.Equal(403, await Delete(("Referer", "https://evil.example/page")));
        Assert.Equal(403, await Delete(("Origin", own), ("Sec-Fetch-Site", "cross-site")));                      // the browser's own statement beats a header
        Assert.Equal(403, await Delete(("Origin", own), ("Sec-Fetch-Site", "same-site")));                       // a sibling subdomain is not the page either
        Assert.Equal(403, await Delete(("Origin", own + ".evil.example")));
        using (var still = await http.SendAsync(Req(HttpMethod.Get, url, cookie: cookie))) Assert.Equal(200, (int)still.StatusCode);
        Assert.Single(host.GameLog.Store.List(keyId, 0, 10).Items);

        // logout is protected the same way, and the session survives a forged logout
        using (var forged = await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session", cookie: cookie, headers: new[] { ("Origin", "https://evil.example") })))
            Assert.Equal(403, (int)forged.StatusCode);
        using (var forged = await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session", cookie: cookie)))
            Assert.Equal(403, (int)forged.StatusCode);
        Assert.Equal(1, host.Sessions.Count);
        Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie))).StatusCode);

        // the site's own page: Origin, or the Referer when there is no Origin, or the browser saying same-origin
        Assert.Equal(200, await Delete(("Origin", own), ("Sec-Fetch-Site", "same-origin")));
        using (var gone = await http.SendAsync(Req(HttpMethod.Get, url, cookie: cookie))) Assert.Equal(404, (int)gone.StatusCode);
        Assert.Equal(404, await Delete(("Referer", own + "/games/csrf-1")));        // allowed (and the game is already gone)
        Assert.Equal(404, await Delete(("Origin", own)));

        // reads are not state-changing: a foreign Origin on a GET does not stop it (the browser would not send the cookie from a foreign site anyway)
        using (var read = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games", cookie: cookie, headers: new[] { ("Origin", "https://evil.example") }))) Assert.Equal(200, (int)read.StatusCode);
    }

    [Fact]
    public async Task A_request_with_an_API_key_needs_no_origin_because_nothing_is_sent_by_itself()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync();
        using var http = Client();
        string gameBody = "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}],\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"\",\"max_depth\":2,\"max_time\":2,\"gameId\":\"key-1\"}";
        using (var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", bearer: secret, body: gameBody, headers: new[] { ("Origin", "https://anything.example") }))) Assert.Equal(200, (int)r.StatusCode);
        using (var r = await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/games/key-1", bearer: secret))) Assert.Equal(200, (int)r.StatusCode);
    }

    [Fact]
    public async Task A_login_form_posted_from_another_site_is_refused_and_a_program_without_an_origin_is_not()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync();
        using var http = Client();
        using (var r = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(secret), headers: new[] { ("Origin", "https://evil.example") })))
        {
            Assert.Equal(403, (int)r.StatusCode);
            Assert.False(r.Headers.Contains("Set-Cookie"));
        }
        Assert.Equal(0, host.Sessions.Count);
        await LoginAsync(http, host, secret, ("Origin", Origin(host)));
        await LoginAsync(http, host, secret);
    }

    [Fact]
    public async Task Cors_never_allows_credentials_so_another_origin_cannot_use_the_cookie()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync(o => o.Security.CorsOrigins.Add("https://app.example.org"));
        using var http = Client();
        using var r = await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/health", headers: new[] { ("Origin", "https://app.example.org") }));
        Assert.Equal("https://app.example.org", r.Headers.GetValues("Access-Control-Allow-Origin").Single());
        Assert.False(r.Headers.Contains("Access-Control-Allow-Credentials"));
        await LoginAsync(http, host, secret);
    }

    // ============================================================================================================================ throttling and the real address

    [Fact]
    public async Task Logins_are_throttled_per_address_and_the_throttle_refills_with_time()
    {
        var (_, secret) = NewKey();
        var host = await StartAsync(o => { o.Security.Sessions.LoginBurst = 3; o.Security.Sessions.LoginAttemptsPerMinute = 6; });   // 3 at once, then one every 10 s
        using var http = Client();
        for (int i = 0; i < 3; i++)
        {
            using var wrong = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody("ttc_wrong" + i)));
            Assert.Equal(401, (int)wrong.StatusCode);
        }
        using var throttled = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(secret)));   // even the right key waits
        Assert.Equal(429, (int)throttled.StatusCode);
        Assert.False(throttled.Headers.Contains("Set-Cookie"));
        Assert.InRange(throttled.Headers.RetryAfter!.Delta!.Value.TotalSeconds, 1, 10);
        using (var doc = JsonDocument.Parse(await throttled.Content.ReadAsStringAsync()))
        {
            Assert.Contains("try again later", doc.RootElement.GetProperty("error").GetString());
            Assert.InRange(doc.RootElement.GetProperty("retry_after_seconds").GetInt32(), 1, 10);
        }
        _time.Advance(TimeSpan.FromSeconds(11));
        await LoginAsync(http, host, secret);                                    // one attempt is available again
        using var again = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(secret)));
        Assert.Equal(429, (int)again.StatusCode);                                // and a successful one used it up
    }

    [Fact]
    public async Task A_header_cannot_choose_the_address_that_is_throttled_unless_the_peer_is_a_trusted_proxy()
    {
        var (_, secret) = NewKey();
        // not trusted (the default): the peer is the client, whatever the header says
        var plain = await StartAsync(o => { o.Security.Sessions.LoginBurst = 3; o.Security.Sessions.LoginAttemptsPerMinute = 6; });
        using var http = Client();
        var statuses = new List<int>();
        for (int i = 0; i < 6; i++)
        {
            using var r = await http.SendAsync(Req(HttpMethod.Post, plain.BaseUrl + "/v1/session", body: LoginBody("ttc_wrong"), headers: new[] { ("CF-Connecting-IP", $"198.51.100.{i + 1}"), ("X-Forwarded-For", $"203.0.113.{i + 1}") }));
            statuses.Add((int)r.StatusCode);
        }
        Assert.Equal(new[] { 401, 401, 401, 429, 429, 429 }, statuses);          // a new "address" in every request did not help

        // a trusted proxy (here: loopback, as with a local cloudflared) is believed: every client address has its own allowance
        var tunnel = await StartAsync(o =>
        {
            o.Security.Sessions.LoginBurst = 3;
            o.Security.Sessions.LoginAttemptsPerMinute = 6;
            o.Security.TrustedProxies.TrustLoopback = true;
        });
        statuses.Clear();
        foreach (int client in new[] { 1, 1, 1, 1, 2, 2, 3 })
        {
            using var r = await http.SendAsync(Req(HttpMethod.Post, tunnel.BaseUrl + "/v1/session", body: LoginBody("ttc_wrong"), headers: new[] { ("CF-Connecting-IP", $"198.51.100.{client}") }));
            statuses.Add((int)r.StatusCode);
        }
        Assert.Equal(new[] { 401, 401, 401, 429, 401, 401, 401 }, statuses);     // the fourth attempt of client 1 only
        // a trusted proxy that sends rubbish: the peer's own bucket
        for (int i = 0; i < 4; i++)
        {
            using var r = await http.SendAsync(Req(HttpMethod.Post, tunnel.BaseUrl + "/v1/session", body: LoginBody("ttc_wrong"), headers: new[] { ("CF-Connecting-IP", "not-an-address") }));
            Assert.Equal(i < 3 ? 401 : 429, (int)r.StatusCode);
        }
        GC.KeepAlive(secret);
    }

    [Fact]
    public async Task Every_login_attempt_is_audited_with_time_address_and_outcome_and_the_key_id_only_on_success()
    {
        var (keyId, secret) = NewKey();
        var capture = new CapturingLoggerProvider();
        var host = await StartAsync(o => { o.Security.Sessions.LoginBurst = 3; o.Security.Sessions.LoginAttemptsPerMinute = 6; o.Security.TrustedProxies.TrustLoopback = true; }, capture);
        using var http = Client();
        using (await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody("ttc_nope"), headers: new[] { ("CF-Connecting-IP", "198.51.100.7") }))) { }
        var (cookie, _, sessionId, _) = await LoginAsync(http, host, secret, ("CF-Connecting-IP", "198.51.100.7"));
        using (await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody("ttc_nope"), headers: new[] { ("CF-Connecting-IP", "198.51.100.7") }))) { }
        using (var throttled = await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(secret), headers: new[] { ("CF-Connecting-IP", "198.51.100.7") }))) Assert.Equal(429, (int)throttled.StatusCode);
        using (await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session", cookie: cookie, headers: new[] { ("Origin", Origin(host)), ("CF-Connecting-IP", "198.51.100.7") }))) { }

        var lines = File.ReadAllLines(AuditFile).Select(l => JsonDocument.Parse(l).RootElement.Clone()).ToList();
        Assert.Equal(new[] { "failed", "success", "failed", "throttled", "success" }, lines.Select(l => l.GetProperty("outcome").GetString()));
        Assert.Equal(new[] { "login", "login", "login", "login", "logout" }, lines.Select(l => l.GetProperty("event").GetString()));
        Assert.All(lines, l => Assert.Equal("198.51.100.7", l.GetProperty("ip").GetString()));    // the real client, as the trusted proxy reported it
        Assert.All(lines, l => Assert.True(DateTimeOffset.TryParse(l.GetProperty("t").GetString(), out _)));
        Assert.Equal(new bool[] { false, true, false, false, true }, lines.Select(l => l.GetProperty("success").GetBoolean()));
        Assert.Equal(new string?[] { null, keyId, null, null, keyId }, lines.Select(l => l.GetProperty("key_id").ValueKind == JsonValueKind.Null ? null : l.GetProperty("key_id").GetString()));

        string everything = File.ReadAllText(AuditFile);
        lock (capture.Messages) everything += "\n" + string.Join("\n", capture.Messages);
        Assert.Contains("audit", everything);
        AssertNoSecrets(everything, secret, sessionId, "ttc_nope");
    }

    private static void AssertNoSecrets(string text, string secret, string sessionId, params string[] presented)
    {
        Assert.DoesNotContain(secret, text);
        Assert.DoesNotContain(secret[4..], text);
        Assert.DoesNotContain(secret[..12], text);                    // not even a prefix of a key
        Assert.DoesNotContain(ApiKeyStore.HashOf(secret), text);
        Assert.DoesNotContain(sessionId, text);
        Assert.DoesNotContain(sessionId[..16], text);                 // nor a prefix of a session id
        Assert.DoesNotContain("ttc_session=", text);
        foreach (string p in presented) Assert.DoesNotContain(p, text);
    }

    [Fact]
    public async Task No_key_and_no_session_id_appears_in_any_log_message_or_file_through_login_use_expiry_logout_and_failures()
    {
        var (_, secret) = NewKey();
        var (_, revoked) = NewKey("bob");
        var capture = new CapturingLoggerProvider();
        var host = await StartAsync(o => o.Security.Sessions.LifetimeHours = 1, capture);
        using var http = Client();

        var ids = new List<string>();
        var (c1, _, id1, _) = await LoginAsync(http, host, secret);
        ids.Add(id1);
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: c1));
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/live", cookie: c1), HttpCompletionOption.ResponseHeadersRead);
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/ai_move", cookie: c1, body: "{}"));                         // refused: a cookie is not a key
        await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/games/x", cookie: c1));                                // CSRF refusal
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: "ttc_session=" + new string('q', 43)));      // an unknown id
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody("ttc_" + new string('y', 43))));
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/games", bearer: secret + "x"));
        new ApiKeyStore(KeyFile, _time).Revoke("bob");
        _time.Advance(TimeSpan.FromSeconds(2));
        await http.SendAsync(Req(HttpMethod.Post, host.BaseUrl + "/v1/session", body: LoginBody(revoked)));
        _time.Advance(TimeSpan.FromHours(2));                                                                                   // the session expires
        await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: c1));
        var (c2, _, id2, _) = await LoginAsync(http, host, secret);
        ids.Add(id2);
        await http.SendAsync(Req(HttpMethod.Delete, host.BaseUrl + "/v1/session", cookie: c2, headers: new[] { ("Origin", Origin(host)) }));

        string everything;
        lock (capture.Messages) everything = string.Join("\n", capture.Messages);
        everything += "\n" + File.ReadAllText(AuditFile);
        foreach (string file in Directory.EnumerateFiles(_dir, "*", SearchOption.AllDirectories).Where(f => !f.EndsWith("keys.json", StringComparison.Ordinal)))
            everything += "\n" + File.ReadAllText(file);
        Assert.NotEmpty(capture.Messages);
        foreach (string id in ids) AssertNoSecrets(everything, secret, id, revoked);
        Assert.DoesNotContain(revoked[..12], everything);
        Assert.DoesNotContain(new string('q', 43), everything);
        Assert.DoesNotContain("y" + new string('y', 20), everything);
    }

    // ============================================================================================================================ capacity classes and limits

    [Fact]
    public async Task The_web_login_has_its_own_rate_limit_so_a_browser_cannot_use_up_the_plugins()
    {
        var (_, secret) = NewKey("alice", rate: 1, burst: 2, inFlight: 4);
        var host = await StartAsync();
        using var http = Client();
        var (cookie, _, _, _) = await LoginAsync(http, host, secret);

        for (int i = 0; i < 2; i++) Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", bearer: secret))).StatusCode);
        Assert.Equal(429, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", bearer: secret))).StatusCode);   // the plugin's bucket is empty
        for (int i = 0; i < 2; i++) Assert.Equal(200, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie))).StatusCode);   // the browser's is not
        Assert.Equal(429, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", cookie: cookie))).StatusCode);
        Assert.Equal(429, (int)(await http.SendAsync(Req(HttpMethod.Get, host.BaseUrl + "/v1/me", bearer: secret))).StatusCode);
    }

    [Fact]
    public async Task Health_shows_the_capacity_of_the_classes_without_any_secret()
    {
        var host = await StartAsync(o =>
        {
            o.Workers = 8;
            o.Access.PluginReservedShare = 0.25;
            o.Access.AnonymousMaxShare = 0.25;
        });
        using var http = Client();
        using var health = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/health"));
        Assert.Equal(8, health.RootElement.GetProperty("workers").GetInt32());
        Assert.Equal(2, health.RootElement.GetProperty("reservedForPlugin").GetInt32());
        Assert.Equal(2, health.RootElement.GetProperty("anonymousMaxWorkers").GetInt32());
        var classes = health.RootElement.GetProperty("classes");
        foreach (string c in new[] { "plugin", "web", "anonymous" })
        {
            Assert.Equal(0, classes.GetProperty(c).GetProperty("running").GetInt32());
            Assert.Equal(0, classes.GetProperty(c).GetProperty("queued").GetInt32());
        }
        Assert.False(health.RootElement.TryGetProperty("sessions", out _)); // the number of sessions is for the operator
        using var status = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/admin/status"));
        Assert.Equal(0, status.RootElement.GetProperty("sessions").GetInt32());
    }
}
