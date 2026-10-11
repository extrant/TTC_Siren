using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Live;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// The server behind Cloudflare (decision A-27): the real client address from a trusted proxy only, operator endpoints that refuse proxied requests, a stream
/// that survives the proxy (heartbeat, resuming), answers that are never cached, and an answer time that fits the proxy's timeout.
/// </summary>
public sealed class CloudflareReadinessTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-cf-" + Guid.NewGuid().ToString("N"));
    private readonly List<IAsyncDisposable> _hosts = new();

    public CloudflareReadinessTests(ITestOutputHelper output)
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

    private async Task<ServerHost> StartAsync(Action<ServerOptions>? tweak = null)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries,
            GameLog = new GameLogOptions { Directory = Path.Combine(_dir, "logs-" + Guid.NewGuid().ToString("N")) },
            Security = new SecurityOptions { RequireKey = false, DefaultBurst = 1000, DefaultRatePerMinute = 100000, DefaultMaxInFlight = 64 },
        };
        tweak?.Invoke(options);
        var host = ServerHost.Create(options, authenticator: new HeaderAuthenticator());
        await host.StartAsync();
        _hosts.Add(new HostDisposer(host));
        return host;
    }

    // ============================================================================================================================ CIDR and the resolver

    [Theory]
    [InlineData("173.245.48.0/20", "173.245.48.1", true)]
    [InlineData("173.245.48.0/20", "173.245.63.255", true)]
    [InlineData("173.245.48.0/20", "173.245.64.0", false)]
    [InlineData("173.245.48.0/20", "173.245.47.255", false)]
    [InlineData("104.16.0.0/13", "104.23.255.255", true)]
    [InlineData("104.16.0.0/13", "104.24.0.0", false)]
    [InlineData("2606:4700::/32", "2606:4700:1::1", true)]
    [InlineData("2606:4700::/32", "2606:4701::1", false)]
    [InlineData("2a06:98c0::/29", "2a06:98c7:ffff::1", true)]
    [InlineData("2a06:98c0::/29", "2a06:98c8::1", false)]
    [InlineData("10.0.0.1", "10.0.0.1", true)]       // a plain address is a /32
    [InlineData("10.0.0.1", "10.0.0.2", false)]
    [InlineData("0.0.0.0/0", "8.8.8.8", true)]
    [InlineData("173.245.48.0/20", "2606:4700::1", false)] // never across families
    [InlineData("173.245.48.0/20", "::ffff:173.245.48.9", true)] // an IPv4 peer reported as IPv4-mapped IPv6
    public void Cidr_matches_the_addresses_of_its_network_and_no_others(string cidr, string address, bool expected)
    {
        Assert.True(Cidr.TryParse(cidr, out var c));
        Assert.Equal(expected, c.Contains(IPAddress.Parse(address)));
    }

    [Theory]
    [InlineData("")]
    [InlineData("not-an-address")]
    [InlineData("10.0.0.0/33")]
    [InlineData("::/129")]
    [InlineData("10.0.0.0/-1")]
    [InlineData("10.0.0.0/8/8")]
    public void Cidr_refuses_what_is_not_an_address_or_a_network(string text) => Assert.False(Cidr.TryParse(text, out _));

    [Fact]
    public void The_shipped_cloudflare_range_file_parses_and_covers_the_published_ranges()
    {
        string file = Path.Combine(GoldenFiles.RepoRoot, "csharp", "src", "TTC.Siren.Server", "data", "cloudflare-ips.txt");
        var options = new TrustedProxyOptions { UseCloudflareRanges = true, CidrFile = file };
        var resolver = new ClientIpResolver(options, AppContext.BaseDirectory);
        Assert.Equal(22, resolver.RangeCount); // 15 IPv4 and 7 IPv6 ranges (fetched 2026-10-11)
        Assert.True(resolver.IsTrustedPeer(IPAddress.Parse("173.245.48.5")));
        Assert.True(resolver.IsTrustedPeer(IPAddress.Parse("2606:4700:4700::1111")));
        Assert.True(resolver.IsTrustedPeer(IPAddress.Parse("131.0.72.1")));
        Assert.False(resolver.IsTrustedPeer(IPAddress.Parse("8.8.8.8")));
        Assert.False(resolver.IsTrustedPeer(IPAddress.Parse("127.0.0.1")));      // loopback is trusted only when asked for
        Assert.False(resolver.IsTrustedPeer(IPAddress.Parse("192.168.1.1")));
        Assert.False(resolver.IsTrustedPeer(null));
    }

    [Fact]
    public void The_cloudflare_ranges_are_trusted_only_when_asked_for_and_a_missing_or_empty_file_stops_the_start()
    {
        Assert.False(new ClientIpResolver(new TrustedProxyOptions(), AppContext.BaseDirectory).IsTrustedPeer(IPAddress.Parse("173.245.48.5")));
        Assert.Throws<FileNotFoundException>(() => new ClientIpResolver(new TrustedProxyOptions { UseCloudflareRanges = true, CidrFile = Path.Combine(_dir, "nothing.txt") }, _dir));
        string empty = Path.Combine(_dir, "empty.txt");
        File.WriteAllText(empty, "# only a comment\n\n");
        Assert.Throws<InvalidDataException>(() => new ClientIpResolver(new TrustedProxyOptions { UseCloudflareRanges = true, CidrFile = empty }, _dir));
        string bad = Path.Combine(_dir, "bad.txt");
        File.WriteAllText(bad, "173.245.48.0/20\nnot a range\n");
        Assert.Throws<InvalidDataException>(() => new ClientIpResolver(new TrustedProxyOptions { UseCloudflareRanges = true, CidrFile = bad }, _dir));
        Assert.Throws<ArgumentException>(() => new ServerOptions { Security = new SecurityOptions { TrustedProxies = { Cidrs = { "300.1.1.1/8" } } } }.Validate());
    }

    private static DefaultHttpContext Context(string peer, params (string Name, string Value)[] headers)
    {
        var ctx = new DefaultHttpContext();
        ctx.Connection.RemoteIpAddress = IPAddress.Parse(peer);
        foreach (var (n, v) in headers) ctx.Request.Headers.Append(n, v);
        return ctx;
    }

    [Fact]
    public void The_client_address_comes_from_the_header_only_when_the_peer_is_a_trusted_proxy()
    {
        var resolver = new ClientIpResolver(new TrustedProxyOptions
        {
            UseCloudflareRanges = true, CidrFile = Path.Combine(GoldenFiles.RepoRoot, "csharp", "src", "TTC.Siren.Server", "data", "cloudflare-ips.txt"), Cidrs = { "10.1.0.0/16" },
        }, AppContext.BaseDirectory);

        // from Cloudflare: the header is believed
        Assert.Equal(IPAddress.Parse("198.51.100.7"), resolver.Resolve(Context("173.245.48.5", ("CF-Connecting-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("2001:db8::1"), resolver.Resolve(Context("2606:4700::5", ("CF-Connecting-IP", "2001:db8::1"))));
        Assert.Equal(IPAddress.Parse("198.51.100.7"), resolver.Resolve(Context("10.1.2.3", ("CF-Connecting-IP", "198.51.100.7")))); // a configured proxy
        // from Cloudflare without the header (a health check of the proxy itself): the peer
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5")));

        // spoofing attempts: the same header from anybody else is ignored
        Assert.Equal(IPAddress.Parse("203.0.113.50"), resolver.Resolve(Context("203.0.113.50", ("CF-Connecting-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("127.0.0.1"), resolver.Resolve(Context("127.0.0.1", ("CF-Connecting-IP", "198.51.100.7")))); // loopback is not trusted here
        Assert.Equal(IPAddress.Parse("203.0.113.50"), resolver.Resolve(Context("203.0.113.50", ("X-Forwarded-For", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("203.0.113.50"), resolver.Resolve(Context("203.0.113.50", ("Forwarded", "for=198.51.100.7"))));
        // X-Forwarded-For is never read, not even from a trusted proxy (only the configured header is)
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5", ("X-Forwarded-For", "198.51.100.7"))));

        // a trusted proxy that sends rubbish, a list or two headers: the peer, never a guess
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5", ("CF-Connecting-IP", "not-an-ip"))));
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5", ("CF-Connecting-IP", "1.1.1.1, 2.2.2.2"))));
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5", ("CF-Connecting-IP", "1.1.1.1"), ("CF-Connecting-IP", "2.2.2.2"))));
        Assert.Equal(IPAddress.Parse("173.245.48.5"), resolver.Resolve(Context("173.245.48.5", ("CF-Connecting-IP", ""))));
    }

    [Fact]
    public void A_local_tunnel_is_trusted_through_the_loopback_switch_only()
    {
        var off = new ClientIpResolver(new TrustedProxyOptions(), AppContext.BaseDirectory);
        var on = new ClientIpResolver(new TrustedProxyOptions { TrustLoopback = true }, AppContext.BaseDirectory);
        Assert.Equal(IPAddress.Loopback, off.Resolve(Context("127.0.0.1", ("CF-Connecting-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("198.51.100.7"), on.Resolve(Context("127.0.0.1", ("CF-Connecting-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("198.51.100.7"), on.Resolve(Context("::1", ("CF-Connecting-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Parse("203.0.113.50"), on.Resolve(Context("203.0.113.50", ("CF-Connecting-IP", "198.51.100.7")))); // loopback only
        Assert.Equal(IPAddress.Loopback, on.Resolve(Context("127.0.0.1"))); // the operator at the machine
    }

    [Fact]
    public void Another_header_name_can_be_configured()
    {
        var resolver = new ClientIpResolver(new TrustedProxyOptions { TrustLoopback = true, ClientIpHeader = "X-Real-IP" }, AppContext.BaseDirectory);
        Assert.Equal(IPAddress.Parse("198.51.100.7"), resolver.Resolve(Context("127.0.0.1", ("X-Real-IP", "198.51.100.7"))));
        Assert.Equal(IPAddress.Loopback, resolver.Resolve(Context("127.0.0.1", ("CF-Connecting-IP", "198.51.100.7"))));
    }

    // ============================================================================================================================ operator endpoints

    [Theory]
    [InlineData("CF-Connecting-IP", "198.51.100.7")]
    [InlineData("CF-Connecting-IPv6", "2001:db8::1")]
    [InlineData("CF-Ray", "8a1b2c3d4e5f-FRA")]
    [InlineData("CF-IPCountry", "DE")]
    [InlineData("CF-Visitor", "{\"scheme\":\"https\"}")]
    [InlineData("True-Client-IP", "198.51.100.7")]
    [InlineData("X-Forwarded-For", "198.51.100.7")]
    [InlineData("X-Forwarded-Host", "example.org")]
    [InlineData("X-Forwarded-Proto", "https")]
    [InlineData("Forwarded", "for=198.51.100.7")]
    [InlineData("X-Real-IP", "198.51.100.7")]
    [InlineData("Via", "1.1 cloudflare")]
    public async Task Operator_endpoints_refuse_a_request_that_a_proxy_handled_even_from_the_local_machine(string header, string value)
    {
        var host = await StartAsync();
        using var http = new HttpClient();
        // the operator at the machine: allowed
        using (var ok = await http.GetAsync(host.BaseUrl + "/admin/status")) Assert.Equal(200, (int)ok.StatusCode);
        using (var ok = await http.PostAsync(host.BaseUrl + "/admin/reload-keys", null)) Assert.Equal(200, (int)ok.StatusCode);

        foreach (var (method, path) in new[] { (HttpMethod.Get, "/admin/status"), (HttpMethod.Post, "/admin/reload-keys") })
        {
            var request = new HttpRequestMessage(method, host.BaseUrl + path);
            request.Headers.TryAddWithoutValidation(header, value);
            request.Headers.TryAddWithoutValidation("X-Test-Key", "operator");
            using var response = await http.SendAsync(request);
            Assert.Equal(403, (int)response.StatusCode);
            Assert.Contains("this machine only", await response.Content.ReadAsStringAsync());
        }
    }

    [Fact]
    public async Task Operator_endpoints_refuse_the_proxy_headers_also_when_the_loopback_is_a_trusted_proxy()
    {
        var host = await StartAsync(o => o.Security.TrustedProxies.TrustLoopback = true);
        using var http = new HttpClient();
        using (var ok = await http.GetAsync(host.BaseUrl + "/admin/status")) Assert.Equal(200, (int)ok.StatusCode);
        var tunnelled = new HttpRequestMessage(HttpMethod.Get, host.BaseUrl + "/admin/status");
        tunnelled.Headers.TryAddWithoutValidation("CF-Connecting-IP", "198.51.100.7");
        tunnelled.Headers.TryAddWithoutValidation("CF-Ray", "abc");
        using var refused = await http.SendAsync(tunnelled);
        Assert.Equal(403, (int)refused.StatusCode);
    }

    // ============================================================================================================================ no caching

    [Fact]
    public async Task Dynamic_answers_are_never_cacheable_and_static_files_have_their_own_policy()
    {
        string site = Path.Combine(_dir, "site");
        Directory.CreateDirectory(Path.Combine(site, "assets"));
        File.WriteAllText(Path.Combine(site, "index.html"), "<!doctype html><title>x</title>");
        File.WriteAllText(Path.Combine(site, "assets", "app.1a2b3c4d.js"), "console.log(1)");
        File.WriteAllText(Path.Combine(site, "robots.txt"), "User-agent: *");
        var host = await StartAsync(o => o.StaticDirectory = site);
        using var http = new HttpClient();

        async Task<string?> CacheControl(HttpRequestMessage m)
        {
            m.Headers.TryAddWithoutValidation("X-Test-Key", "alice");
            using var r = await http.SendAsync(m, HttpCompletionOption.ResponseHeadersRead);
            return r.Headers.CacheControl?.ToString();
        }
        HttpRequestMessage Get(string path) => new(HttpMethod.Get, host.BaseUrl + path);

        // every dynamic endpoint, success and failure alike
        Assert.Equal("no-store", await CacheControl(Get("/health")));
        Assert.Equal("no-store", await CacheControl(Get("/v1/games")));
        Assert.Equal("no-store", await CacheControl(Get("/v1/games/does-not-exist")));
        Assert.Equal("no-store", await CacheControl(Get("/v1/games/does-not-exist/download")));
        Assert.Equal("no-store", await CacheControl(Get("/admin/status")));
        Assert.Equal("no-store", await CacheControl(Get("/api/winrate")));     // not a GET route: 405, still not cacheable
        Assert.Equal("no-store", await CacheControl(Get("/nothing-here")));
        Assert.Equal("no-store", await CacheControl(new HttpRequestMessage(HttpMethod.Post, host.BaseUrl + "/ai_move") { Content = new StringContent("{", Encoding.UTF8, "application/json") }));
        Assert.Equal("no-store", await CacheControl(new HttpRequestMessage(HttpMethod.Post, host.BaseUrl + "/ai_move") { Content = new StringContent(new string(' ', 2 << 20), Encoding.UTF8, "application/json") })); // 413
        Assert.Equal("no-store", await CacheControl(new HttpRequestMessage(HttpMethod.Post, host.BaseUrl + "/api/recommend") { Content = new StringContent("{}", Encoding.UTF8, "application/json") }));

        // static files: hashed assets for a year, immutable; the page itself is revalidated
        Assert.Equal("public, max-age=31536000, immutable", await CacheControl(Get("/assets/app.1a2b3c4d.js")));
        Assert.Equal("no-cache", await CacheControl(Get("/index.html")));
        Assert.Equal("no-cache", await CacheControl(Get("/")));
        Assert.Equal("no-cache", await CacheControl(Get("/robots.txt")));

        // the event stream
        using var live = new HttpRequestMessage(HttpMethod.Get, host.BaseUrl + "/v1/live");
        live.Headers.Add("X-Test-Key", "alice");
        using var liveResponse = await http.SendAsync(live, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("no-store, no-cache", liveResponse.Headers.CacheControl?.ToString());
    }

    // ============================================================================================================================ the event stream behind a proxy

    private static LiveHub Hub(int replay = 16, int capacity = 64) => new(capacity, replay, TimeSpan.FromMinutes(15));

    private static void Emit(LiveHub hub, string key, int count, string? game = null)
    {
        for (int i = 1; i <= count; i++) hub.Publish(key, "tick", game, () => new Dictionary<string, object?> { ["n"] = i });
    }

    private static List<long> Drain(LiveSubscription sub)
    {
        var ids = new List<long>();
        while (sub.Reader.TryRead(out var e)) ids.Add(e.Id);
        return ids;
    }

    [Fact]
    public void Events_are_numbered_per_key_and_a_returning_client_gets_what_it_missed()
    {
        var hub = Hub();
        Emit(hub, "alice", 5);
        Emit(hub, "bob", 2);
        Assert.Equal(5, hub.LastEventId("alice"));
        Assert.Equal(2, hub.LastEventId("bob"));

        using var back = hub.Subscribe("alice", lastEventId: 2);   // saw 1 and 2, then the connection broke
        Assert.Equal(new ResumeInfo("resumed", 0, 3), back.Resume);
        Assert.Equal(new long[] { 3, 4, 5 }, Drain(back));
        Emit(hub, "alice", 1);                                       // and the live events follow without a gap or a duplicate
        Assert.Equal(new long[] { 6 }, Drain(back));

        using var current = hub.Subscribe("alice", lastEventId: 6);  // nothing missed
        Assert.Equal(new ResumeInfo("resumed", 0, 0), current.Resume);
        Assert.Empty(Drain(current));

        using var fresh = hub.Subscribe("alice");                     // a first connection replays nothing
        Assert.Equal(ResumeInfo.None, fresh.Resume);
        Assert.Empty(Drain(fresh));
    }

    [Fact]
    public void A_client_that_was_away_too_long_is_told_how_many_events_are_gone()
    {
        var hub = Hub(replay: 4);
        Emit(hub, "alice", 10);                                      // only 7, 8, 9, 10 are kept
        using var late = hub.Subscribe("alice", lastEventId: 2);
        Assert.Equal(new ResumeInfo("gap", 4, 4), late.Resume);       // 3, 4, 5, 6 are lost
        Assert.Equal(new long[] { 7, 8, 9, 10 }, Drain(late));

        using var edge = hub.Subscribe("alice", lastEventId: 6);      // 7 is the oldest kept: nothing is lost
        Assert.Equal(new ResumeInfo("resumed", 0, 4), edge.Resume);
        using var edge2 = hub.Subscribe("alice", lastEventId: 5);     // 6 is gone
        Assert.Equal(new ResumeInfo("gap", 1, 4), edge2.Resume);
    }

    [Fact]
    public void An_id_from_before_a_restart_is_recognised_as_not_belonging_to_this_run()
    {
        var hub = Hub();
        Emit(hub, "alice", 3);
        using var stale = hub.Subscribe("alice", lastEventId: 900);
        Assert.Equal("reset", stale.Resume.Kind);
        Assert.Empty(Drain(stale));
        using var unknownKey = hub.Subscribe("nobody", lastEventId: 5);
        Assert.Equal("reset", unknownKey.Resume.Kind);
    }

    [Fact]
    public void The_game_filter_applies_to_the_replay_and_keys_never_see_each_others_events()
    {
        var hub = Hub();
        Emit(hub, "alice", 2, "g1");
        Emit(hub, "alice", 2, "g2");
        Emit(hub, "bob", 3, "g1");
        using var g2 = hub.Subscribe("alice", "g2", lastEventId: 0);
        Assert.Equal(new long[] { 3, 4 }, Drain(g2));
        using var bob = hub.Subscribe("bob", lastEventId: 0);
        Assert.Equal(new long[] { 1, 2, 3 }, Drain(bob));
    }

    [Fact]
    public void The_kept_events_are_bounded_by_count_and_by_size_and_forgotten_when_idle()
    {
        var clock = new ManualTimeProvider(DateTimeOffset.UtcNow);
        var hub = new LiveHub(64, replayEvents: 100, replayIdle: TimeSpan.FromMinutes(15), time: clock, replayBytes: 4096);
        for (int i = 1; i <= 50; i++) hub.Publish("alice", "big", null, () => new Dictionary<string, object?> { ["pad"] = new string('x', 500) });
        using (var sub = hub.Subscribe("alice", lastEventId: 0))
        {
            var kept = Drain(sub);
            Assert.InRange(kept.Count, 1, 4096 / 2 / 500 + 1);        // the size limit (4096 bytes = 2048 characters) holds about four events
            Assert.Equal(50, kept[^1]);
            Assert.Equal("gap", sub.Resume.Kind);
        }

        clock.Advance(TimeSpan.FromMinutes(16));
        hub.Sweep();
        using var after = hub.Subscribe("alice", lastEventId: 49);
        Assert.Equal("reset", after.Resume.Kind);                     // forgotten: a new run of the numbering
    }

    [Fact]
    public async Task The_stream_sends_a_heartbeat_comment_at_the_configured_interval_and_a_retry_hint()
    {
        Assert.Equal(20, new LiveOptions().HeartbeatSeconds);          // well below Cloudflare's 100 s idle timeout
        Assert.Throws<ArgumentException>(() => new LiveOptions { HeartbeatSeconds = 100 }.Validate());
        Assert.Throws<ArgumentException>(() => new LiveOptions { HeartbeatSeconds = 0 }.Validate());
        var host = await StartAsync(o => o.Live.HeartbeatSeconds = 0.1);

        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        using var request = new HttpRequestMessage(HttpMethod.Get, host.BaseUrl + "/v1/live");
        request.Headers.Add("X-Test-Key", "alice");
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(200, (int)response.StatusCode);
        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream, Encoding.UTF8);
        using var guard = new CancellationTokenSource(TimeSpan.FromSeconds(60)); // a hang detector only
        var lines = new List<string>();
        while (lines.Count(l => l == ": heartbeat") < 3)
            lines.Add((await reader.ReadLineAsync(guard.Token))!);
        Assert.Contains("retry: 3000", lines);
        Assert.Contains(": connected", lines);
        _output.WriteLine(string.Join(" | ", lines));
    }

    [Fact]
    public async Task A_client_that_reconnects_with_Last_Event_ID_receives_the_missed_events_over_HTTP()
    {
        var host = await StartAsync(o => o.Live.ReplayEvents = 3);
        Emit(host.Hub, "alice", 8);
        // ids 6, 7, 8 are kept; the client saw 4: events 5 is gone
        await using var client = await SseClient.ConnectAsync(host.BaseUrl, "alice", lastEventId: "4");
        Assert.Equal(200, client.Status);
        var got = await client.ReadUntilAsync((type, data) => type == "tick" && data.GetProperty("n").GetInt32() == 8);
        Assert.Equal("resync", got[0].Event);
        Assert.Equal("events_lost", got[0].Data.GetProperty("reason").GetString());
        Assert.Equal(1, got[0].Data.GetProperty("missed").GetInt64());
        Assert.Equal(new[] { "6", "7", "8" }, got.Where(e => e.Event == "tick").Select(e => e.Id));

        await using var all = await SseClient.ConnectAsync(host.BaseUrl, "alice", lastEventId: "5");
        var full = await all.ReadUntilAsync((type, data) => type == "tick" && data.GetProperty("n").GetInt32() == 8);
        Assert.DoesNotContain(full, e => e.Event == "resync");
        Assert.Equal(new[] { "6", "7", "8" }, full.Where(e => e.Event == "tick").Select(e => e.Id));

        // the query form for clients that cannot set a header
        await using var viaQuery = await SseClient.ConnectAsync(host.BaseUrl, "alice", "?last_event_id=7");
        var tail = await viaQuery.ReadUntilAsync((type, data) => type == "tick" && data.GetProperty("n").GetInt32() == 8);
        Assert.Equal(new[] { "8" }, tail.Where(e => e.Event == "tick").Select(e => e.Id));

        // an id from a run of the server that no longer exists
        await using var stale = await SseClient.ConnectAsync(host.BaseUrl, "alice", lastEventId: "5000");
        var first = await stale.ReadUntilAsync((type, _) => type == "resync");
        Assert.Equal("stream_restarted", first[^1].Data.GetProperty("reason").GetString());
    }

    // ============================================================================================================================ origin timeout and the cap per request

    [Fact]
    public void The_default_settings_fit_into_the_origin_timeout_and_a_setting_that_does_not_fit_stops_the_start()
    {
        var defaults = new ServerOptions();
        defaults.Validate();
        Assert.Equal(95, defaults.OriginTimeoutSeconds);
        Assert.True(defaults.MaxSearchSeconds + defaults.QueueWaitMs / 1000.0 + defaults.OriginMarginSeconds < defaults.OriginTimeoutSeconds);

        var tooLong = new ServerOptions { MaxSearchSeconds = 40 };       // 40 + 60 + 10 = 110 >= 95
        var e = Assert.Throws<ArgumentException>(() => tooLong.Validate());
        Assert.Contains("110", e.Message);
        Assert.Contains("OriginTimeoutSeconds", e.Message);
        Assert.Throws<ArgumentException>(() => new ServerOptions { QueueWaitMs = 80_000 }.Validate()); // 20 + 80 + 10
        new ServerOptions { MaxSearchSeconds = 40, OriginTimeoutSeconds = 120 }.Validate();             // a proxy that waits longer
        new ServerOptions { MaxSearchSeconds = 300, OriginTimeoutSeconds = 0 }.Validate();             // no proxy timeout: nothing to fit into
        Assert.Throws<ArgumentException>(() => new ServerOptions { MaxSearchSeconds = 0 }.Validate()); // a proxy timeout needs a cap
        Assert.Throws<ArgumentException>(() => new ServerOptions { MaxSearchSeconds = -1 }.Validate());
        new ServerOptions { MaxSearchSeconds = 0, OriginTimeoutSeconds = 0 }.Validate();
        // the boundary: the sum must stay BELOW the timeout
        new ServerOptions { MaxSearchSeconds = 24.9 }.Validate();
        Assert.Throws<ArgumentException>(() => new ServerOptions { MaxSearchSeconds = 25 }.Validate());
    }

    [Fact]
    public async Task A_request_cannot_ask_for_more_search_time_than_the_cap_and_is_told_so()
    {
        var host = await StartAsync(o =>
        {
            o.MaxSearchSeconds = 0.4;
            o.OriginTimeoutSeconds = 0;
        });
        using var http = new HttpClient();
        http.DefaultRequestHeaders.Add("X-Test-Key", "alice");
        // an opening with the depth of the plugin's default and an hour of time: without the cap this would run for many seconds
        string body = "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4},{\"numU\":5,\"numL\":4,\"numD\":6,\"numR\":5},{\"numU\":2,\"numL\":2,\"numD\":7,\"numR\":2}," +
                      "{\"numU\":4,\"numL\":3,\"numD\":3,\"numR\":3},{\"numU\":1,\"numL\":3,\"numD\":7,\"numR\":2}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}," +
                      "{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}],\"myOwner\":2,\"currentPlayer\":2,\"rules\":\"加算,同数\",\"max_depth\":10,\"max_time\":3600}";
        using var response = await http.PostAsync(host.BaseUrl + "/ai_move", new StringContent(body, Encoding.UTF8, "application/json"));
        string answer = await response.Content.ReadAsStringAsync();
        Assert.True(response.StatusCode == HttpStatusCode.OK, answer);
        using var doc = JsonDocument.Parse(answer);
        var limits = doc.RootElement.GetProperty("engine_info").GetProperty("limits");
        Assert.Equal(0.4, limits.GetProperty("max_time").GetDouble());
        Assert.Equal(3600, limits.GetProperty("requested_max_time").GetInt32());
        Assert.True(doc.RootElement.GetProperty("performance_stats").GetProperty("search_depth").GetInt32() < 10, "a depth-10 opening cannot be finished in 0.4 s");

        // a request within the cap is not touched, and says nothing about limits
        string small = body.Replace("\"max_time\":3600", "\"max_time\":0.2").Replace("\"max_depth\":10", "\"max_depth\":3");
        using var ok = await http.PostAsync(host.BaseUrl + "/ai_move", new StringContent(small, Encoding.UTF8, "application/json"));
        using var okDoc = JsonDocument.Parse(await ok.Content.ReadAsStringAsync());
        Assert.False(okDoc.RootElement.GetProperty("engine_info").TryGetProperty("limits", out _));
    }
}
