using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Review L8-1: the test seed switch is refused unless the server listens on a loopback address, logs a warning when it is on, and is not believed for a request that a proxy handled.
/// </summary>
public sealed class TestSeedSwitchTests : ServerTestBase
{
    public TestSeedSwitchTests(ITestOutputHelper output) : base(output)
    {
    }

    [Theory]
    [InlineData("0.0.0.0", false)]
    [InlineData("192.168.1.20", false)]
    [InlineData("::", false)]
    [InlineData("0.0.0.0", true)]   // behind a TLS proxy it is just as forbidden
    public void The_test_seed_switch_is_refused_unless_the_listener_is_loopback(string host, bool behindProxy)
    {
        var options = new ServerOptions { Host = host, TestSeedHeader = true, Security = new SecurityOptions { BehindTlsProxy = behindProxy } };
        var e = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("TestSeedHeader", e.Message);
    }

    [Theory]
    [InlineData("127.0.0.1")]
    [InlineData("::1")]
    public void The_test_seed_switch_is_accepted_on_a_loopback_listener(string host)
    {
        new ServerOptions { Host = host, TestSeedHeader = true }.Validate();
    }

    [Fact]
    public void The_test_seed_switch_is_refused_behind_a_proxy_even_on_loopback()
    {
        // a local tunnel (cloudflared) delivers requests of the whole internet to a loopback listener
        var viaTunnel = new ServerOptions { Host = "127.0.0.1", TestSeedHeader = true, Security = new SecurityOptions { TrustedProxies = { TrustLoopback = true } } };
        Assert.Contains("proxy", Assert.Throws<ArgumentException>(() => viaTunnel.Validate()).Message);
        var tls = new ServerOptions { Host = "127.0.0.1", TestSeedHeader = true, Security = new SecurityOptions { BehindTlsProxy = true } };
        Assert.Throws<ArgumentException>(() => tls.Validate());
    }

    [Fact]
    public async Task The_test_seed_switch_logs_a_warning_when_it_is_on_and_the_header_is_ignored_when_a_proxy_handled_the_request()
    {
        var capture = new CapturingLoggerProvider();
        var host = await StartAsync(o => o.TestSeedHeader = true, capture);
        lock (capture.Messages) Assert.Contains(capture.Messages, m => m.Contains("TestSeedHeader is ON") && m.Contains("Warning"));

        var seeds = new List<ulong>();
        host.RequestCompleted += o => seeds.Add(o.Seed);
        using var http = new HttpClient();
        using (var r = await http.SendAsync(Post(host.BaseUrl, Opening, seed: 4242))) Assert.Equal(200, (int)r.StatusCode);
        // the same header, but the request came through a proxy: it is not believed
        using (var r = await http.SendAsync(Post(host.BaseUrl, Opening.Replace("fixes-1", "fixes-2"), seed: 4242, ("CF-Connecting-IP", "203.0.113.9")))) Assert.Equal(200, (int)r.StatusCode);
        using (var r = await http.SendAsync(Post(host.BaseUrl, Opening.Replace("fixes-1", "fixes-3"), seed: 4242, ("X-Forwarded-For", "203.0.113.9")))) Assert.Equal(200, (int)r.StatusCode);
        Assert.Equal(4242UL, seeds[0]);
        Assert.NotEqual(4242UL, seeds[1]);
        Assert.NotEqual(4242UL, seeds[2]);
    }

    [Fact]
    public async Task Without_the_switch_there_is_no_warning()
    {
        var capture = new CapturingLoggerProvider();
        await StartAsync(null, capture);
        lock (capture.Messages) Assert.DoesNotContain(capture.Messages, m => m.Contains("TestSeedHeader"));
    }
}
