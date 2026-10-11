using System.Net;
using Microsoft.Extensions.Configuration;

namespace TTC.Siren.Server;

/// <summary>
/// Server configuration. Sources, later ones win: defaults, <c>appsettings.json</c> (section "Server"), environment variables
/// (<c>TTC_Server__Port</c> ...), the command line (<c>--Server:Port 5000</c> or the short forms below).
/// </summary>
public sealed class ServerOptions
{
    /// <summary>Address to listen on. The default is loopback only; the cloud topology listens on all addresses (see docs/DEPLOY_WINDOWS.md).</summary>
    public string Host { get; set; } = "127.0.0.1";

    /// <summary>TCP port; 0 picks a free port (tests).</summary>
    public int Port { get; set; } = 5000;

    /// <summary>Number of search workers = searches that run at the same time.</summary>
    public int Workers { get; set; } = Environment.ProcessorCount;

    public int QueueLimit { get; set; } = 256;
    public int QueueWaitMs { get; set; } = 60_000;

    /// <summary>Order the wait queue by estimated cost (cheap endgame requests overtake openings; review O8-1).</summary>
    public bool PriorityScheduling { get; set; } = true;

    /// <summary>A waiting request moves up one cost class for every this many milliseconds, so an opening is overtaken for at most twice this long.</summary>
    public int AgingMs { get; set; } = 2000;

    /// <summary>The card database CSV. Null: <c>data/幻卡数据库.csv</c> found above the application directory.</summary>
    public string? CardDbPath { get; set; }

    public int TtEntries { get; set; } = 1 << 22;
    public int MemoEntries { get; set; } = 1 << 15;

    /// <summary>
    /// TEST ONLY. When set, a request header <c>X-Ttc-Test-Seed: &lt;integer&gt;</c> seeds the unknown-card sampling like <c>random.seed(n)</c>, so a
    /// response can be compared with the Python server's. NEVER enable it in production: in production every request gets a fresh
    /// cryptographic seed, which is written to the game log.
    /// </summary>
    public bool TestSeedHeader { get; set; }

    /// <summary>Game logs (<c>Server:GameLog</c>): directory, retention, game-identity gap.</summary>
    public GameLog.GameLogOptions GameLog { get; set; } = new();

    /// <summary>How many live events a subscriber of <c>/v1/live</c> may fall behind before the oldest are dropped (a <c>lagged</c> notice follows).</summary>
    public int LiveBuffer { get; set; } = Live.LiveHub.DefaultCapacity;

    /// <summary>Serve the plain HTTP listener (<see cref="Port"/>). Off with HTTPS only.</summary>
    public bool HttpEnabled { get; set; } = true;

    /// <summary>Keys, limits, HTTPS, CORS (<c>Server:Security</c>).</summary>
    public Security.SecurityOptions Security { get; set; } = new();

    /// <summary>The proxy to the Python app for the three <c>/api/*</c> endpoints (<c>Server:Proxy</c>).</summary>
    public Proxy.ProxyOptions Proxy { get; set; } = new();

    /// <summary>A directory of static files served at <c>/</c> (the web UI of Phase 8B). Null: the placeholder page.</summary>
    public string? StaticDirectory { get; set; }

    /// <summary>Directory with the explanation template file <c>explain.zh-CN.json</c> to use instead of the built-in one (null: built-in).</summary>
    public string? TemplatesDirectory { get; set; }

    /// <summary>Only the security settings (appsettings.json and environment): what the <c>keys</c> commands need, without validating the rest.</summary>
    public static Security.SecurityOptions LoadSecurity()
    {
        var config = new ConfigurationBuilder().SetBasePath(AppContext.BaseDirectory).AddJsonFile("appsettings.json", optional: true).AddEnvironmentVariables("TTC_").Build();
        var security = new Security.SecurityOptions();
        config.GetSection("Server:Security").Bind(security);
        return security;
    }

    public static ServerOptions Parse(string[] args)
    {
        var mappings = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["--host"] = "Server:Host",
            ["--port"] = "Server:Port",
            ["--workers"] = "Server:Workers",
            ["--queue"] = "Server:QueueLimit",
            ["--queue-wait-ms"] = "Server:QueueWaitMs",
            ["--aging-ms"] = "Server:AgingMs",
            ["--card-db"] = "Server:CardDbPath",
            ["--tt-entries"] = "Server:TtEntries",
            ["--memo-entries"] = "Server:MemoEntries",
        };
        var config = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true)
            .AddEnvironmentVariables("TTC_")
            .AddCommandLine(args, mappings)
            .Build();
        var options = new ServerOptions();
        config.GetSection("Server").Bind(options);
        options.Validate();
        return options;
    }

    public void Validate()
    {
        if (!IPAddress.TryParse(Host, out _)) throw new ArgumentException($"Host '{Host}' is not an IP address");
        if (Port is < 0 or > 65535) throw new ArgumentException($"Port {Port} out of range");

        // review L8-1: the test seed header fixes the sampling seed of a request. It exists for the parity tests, on a machine nobody else can reach.
        if (TestSeedHeader)
        {
            if (!IPAddress.IsLoopback(IPAddress.Parse(Host))) throw new ArgumentException($"TestSeedHeader is a test switch and is refused on {Host}: it is allowed only when the server listens on a loopback address");
            if (Security.BehindTlsProxy) throw new ArgumentException("TestSeedHeader is a test switch and is refused when the server sits behind a proxy (BehindTlsProxy): the proxy would carry the header from anybody");
        }

        if (Workers < 1) throw new ArgumentException("Workers must be at least 1");
        if (QueueLimit < 0) throw new ArgumentException("QueueLimit must not be negative");
        if (QueueWaitMs < 0) throw new ArgumentException("QueueWaitMs must not be negative");
        if (!System.Numerics.BitOperations.IsPow2(TtEntries) || TtEntries < 4) throw new ArgumentException("TtEntries must be a power of two >= 4");
        if (!System.Numerics.BitOperations.IsPow2(MemoEntries) || MemoEntries < 4) throw new ArgumentException("MemoEntries must be a power of two >= 4");
        if (AgingMs < 1) throw new ArgumentException("AgingMs must be at least 1");
        GameLog.Validate();
        Proxy.Validate();
        if (!HttpEnabled && !Security.Https.Enabled) throw new ArgumentException("HttpEnabled=false needs Security:Https:Enabled");
        Security.Validate(Host, IPAddress.IsLoopback(IPAddress.Parse(Host)));
        if (LiveBuffer < 1) throw new ArgumentException("LiveBuffer must be at least 1");
    }

    public IPAddress Address => IPAddress.Parse(Host);
}
