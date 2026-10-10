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

    public static ServerOptions Parse(string[] args)
    {
        var mappings = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["--host"] = "Server:Host",
            ["--port"] = "Server:Port",
            ["--workers"] = "Server:Workers",
            ["--queue"] = "Server:QueueLimit",
            ["--queue-wait-ms"] = "Server:QueueWaitMs",
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
        if (Workers < 1) throw new ArgumentException("Workers must be at least 1");
        if (QueueLimit < 0) throw new ArgumentException("QueueLimit must not be negative");
        if (QueueWaitMs < 0) throw new ArgumentException("QueueWaitMs must not be negative");
        if (!System.Numerics.BitOperations.IsPow2(TtEntries) || TtEntries < 4) throw new ArgumentException("TtEntries must be a power of two >= 4");
        if (!System.Numerics.BitOperations.IsPow2(MemoEntries) || MemoEntries < 4) throw new ArgumentException("MemoEntries must be a power of two >= 4");
    }

    public IPAddress Address => IPAddress.Parse(Host);
}
