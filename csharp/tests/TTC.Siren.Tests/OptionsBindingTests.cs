using TTC.Siren.Server;

namespace TTC.Siren.Tests;

/// <summary>The settings of Phase 8B are read from the places an operator uses (command line, environment, appsettings.json) into the right properties, and nonsense is refused with a message.</summary>
public sealed class OptionsBindingTests
{
    [Fact]
    public void The_new_settings_can_be_given_on_the_command_line_and_in_the_environment()
    {
        string[] args =
        {
            "--Server:Access:PluginReservedShare", "0.5", "--Server:Access:AnonymousMaxShare", "0.125", "--Server:Access:AnonymousQueueLimit", "7",
            "--Server:Access:AnonymousMaxDepth", "5", "--Server:Access:AnonymousMaxSeconds", "1.5", "--Server:Access:AnonymousRatePerMinute", "12",
            "--Server:Security:Sessions:LifetimeHours", "2", "--Server:Security:Sessions:MaxPerKey", "3", "--Server:Security:Sessions:LoginBurst", "4",
            "--Server:Security:Sessions:AuditLogFile", "D:\\audit\\logins.jsonl",
            "--Server:Security:TrustedProxies:TrustLoopback", "true", "--Server:Security:TrustedProxies:Cidrs:0", "10.0.0.0/8", "--Server:Security:TrustedProxies:ClientIpHeader", "X-Real-IP",
            "--Server:Live:HeartbeatSeconds", "5", "--Server:Live:ReplayEvents", "50", "--Server:Live:ReplayMinutes", "3",
            "--Server:MaxSearchSeconds", "30", "--Server:OriginTimeoutSeconds", "120", "--Server:OriginMarginSeconds", "5", "--Server:WarmUp", "false",
        };
        var o = ServerOptions.Parse(args);
        Assert.Equal(0.5, o.Access.PluginReservedShare);
        Assert.Equal(0.125, o.Access.AnonymousMaxShare);
        Assert.Equal(7, o.Access.AnonymousQueueLimit);
        Assert.Equal(5, o.Access.AnonymousMaxDepth);
        Assert.Equal(1.5, o.Access.AnonymousMaxSeconds);
        Assert.Equal(12, o.Access.AnonymousRatePerMinute);
        Assert.Equal(2, o.Security.Sessions.LifetimeHours);
        Assert.Equal(3, o.Security.Sessions.MaxPerKey);
        Assert.Equal(4, o.Security.Sessions.LoginBurst);
        Assert.Equal("D:\\audit\\logins.jsonl", o.Security.Sessions.AuditLogFile);
        Assert.True(o.Security.TrustedProxies.TrustLoopback);
        Assert.Equal(new[] { "10.0.0.0/8" }, o.Security.TrustedProxies.Cidrs);
        Assert.Equal("X-Real-IP", o.Security.TrustedProxies.ClientIpHeader);
        Assert.Equal(5, o.Live.HeartbeatSeconds);
        Assert.Equal(50, o.Live.ReplayEvents);
        Assert.Equal(3, o.Live.ReplayMinutes);
        Assert.Equal(30, o.MaxSearchSeconds);
        Assert.Equal(120, o.OriginTimeoutSeconds);
        Assert.Equal(5, o.OriginMarginSeconds);
        Assert.False(o.WarmUp);

        // and the environment, the way the service is configured on the server
        Environment.SetEnvironmentVariable("TTC_Server__Access__AnonymousMaxShare", "0.5");
        Environment.SetEnvironmentVariable("TTC_Server__Security__Sessions__LifetimeHours", "6");
        try
        {
            var e = ServerOptions.Parse(Array.Empty<string>());
            Assert.Equal(0.5, e.Access.AnonymousMaxShare);
            Assert.Equal(6, e.Security.Sessions.LifetimeHours);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TTC_Server__Access__AnonymousMaxShare", null);
            Environment.SetEnvironmentVariable("TTC_Server__Security__Sessions__LifetimeHours", null);
        }
    }

    [Fact]
    public void The_defaults_are_the_documented_ones()
    {
        var o = new ServerOptions();
        Assert.Equal(20, o.Live.HeartbeatSeconds);
        Assert.Equal(256, o.Live.ReplayEvents);
        Assert.Equal(20, o.MaxSearchSeconds);
        Assert.Equal(95, o.OriginTimeoutSeconds);
        Assert.True(o.WarmUp);
        Assert.Equal(12, o.Security.Sessions.LifetimeHours);
        Assert.Equal(20, o.Security.Sessions.MaxPerKey);
        Assert.Equal(5, o.Security.Sessions.LoginBurst);
        Assert.Equal(6, o.Security.Sessions.LoginAttemptsPerMinute);
        Assert.False(o.Security.TrustedProxies.AnyTrusted);
        Assert.Equal("CF-Connecting-IP", o.Security.TrustedProxies.ClientIpHeader);
        Assert.Equal(16, o.Access.AnonymousQueueLimit);
        Assert.Equal(3, o.Access.AnonymousMaxSeconds);
        Assert.Equal(8, o.Access.AnonymousMaxDepth);
        o.Validate();
    }

    [Theory]
    [InlineData("Security:Sessions:LifetimeHours", "0")]
    [InlineData("Security:Sessions:LifetimeHours", "1000")]
    [InlineData("Security:Sessions:MaxPerKey", "0")]
    [InlineData("Security:Sessions:LoginBurst", "0")]
    [InlineData("Access:PluginReservedShare", "1")]
    [InlineData("Access:AnonymousMaxShare", "0")]
    [InlineData("Live:HeartbeatSeconds", "100")]
    [InlineData("Live:HeartbeatSeconds", "0")]
    public void A_nonsense_setting_stops_the_start_with_a_message(string key, string value)
    {
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(new[] { "--Server:" + key, value }));
    }
}
