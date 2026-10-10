using System.Net;
using Microsoft.Extensions.Configuration;

namespace TTC.Siren.Engine;

/// <summary>
/// Engine configuration. Sources, later ones win: defaults, <c>appsettings.json</c> (section "Engine"), the command line.
/// Command line: <c>--port 5300 --workers 12 --queue 256 --queue-wait-ms 60000 --card-db path --tt-entries 4194304
/// --memo-entries 32768 --host 127.0.0.1</c>, and the flag <c>--allow-remote</c>.
/// </summary>
public sealed class EngineOptions
{
    public const int DefaultPort = 5300;

    /// <summary>Address to listen on. Must be a loopback address unless <see cref="AllowRemote"/> is set.</summary>
    public string Host { get; set; } = "127.0.0.1";

    /// <summary>TCP port; 0 picks a free port (tests).</summary>
    public int Port { get; set; } = DefaultPort;

    /// <summary>Number of worker contexts = searches that run at the same time.</summary>
    public int Workers { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Requests that may wait for a worker; further requests are answered "busy" (HTTP 503) at once. The Python server no longer falls
    /// back to its own solver (A-20), so a request waits for the engine instead: the queue is deep.
    /// </summary>
    public int QueueLimit { get; set; } = 256;

    /// <summary>
    /// How long a queued request waits for a worker before it is answered "busy". Reported by <c>/v1/health</c>: the Python client's
    /// read timeout is <c>max_time</c> + this + a margin, so a queued request is always answered by the engine, never by a client timeout.
    /// </summary>
    public int QueueWaitMs { get; set; } = 60_000;

    /// <summary>The card database CSV. Null: <c>data/幻卡数据库.csv</c> found above the application directory.</summary>
    public string? CardDbPath { get; set; }

    /// <summary>Entries of each worker's Fast transposition table (power of two; 24 bytes per entry; 4 Mi entries = 96 MiB).</summary>
    public int TtEntries { get; set; } = 1 << 22;

    /// <summary>Entries of each worker's exact-endgame memo (power of two; 24 bytes per entry).</summary>
    public int MemoEntries { get; set; } = 1 << 15;

    /// <summary>Explicit opt-in to listen on a non-loopback address.</summary>
    public bool AllowRemote { get; set; }

    public static EngineOptions Parse(string[] args)
    {
        bool allowRemote = args.Contains("--allow-remote", StringComparer.Ordinal);
        var rest = args.Where(a => !string.Equals(a, "--allow-remote", StringComparison.Ordinal)).ToArray();

        var mappings = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["--host"] = "Engine:Host",
            ["--port"] = "Engine:Port",
            ["--workers"] = "Engine:Workers",
            ["--queue"] = "Engine:QueueLimit",
            ["--queue-wait-ms"] = "Engine:QueueWaitMs",
            ["--card-db"] = "Engine:CardDbPath",
            ["--tt-entries"] = "Engine:TtEntries",
            ["--memo-entries"] = "Engine:MemoEntries",
        };
        var config = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true)
            .AddCommandLine(rest, mappings)
            .Build();

        var options = new EngineOptions();
        config.GetSection("Engine").Bind(options);
        if (allowRemote) options.AllowRemote = true;
        options.Validate();
        return options;
    }

    /// <summary>Throws <see cref="ArgumentException"/> with a readable message for a configuration that must not start.</summary>
    public void Validate()
    {
        if (!IPAddress.TryParse(Host, out var address))
            throw new ArgumentException($"Host '{Host}' is not an IP address");
        if (!IPAddress.IsLoopback(address) && !AllowRemote)
            throw new ArgumentException(
                $"Refusing to listen on non-loopback address {Host}: the engine has no authentication. " +
                "Pass --allow-remote to do it anyway.");
        if (Port is < 0 or > 65535) throw new ArgumentException($"Port {Port} out of range");
        if (Workers < 1) throw new ArgumentException("Workers must be at least 1");
        if (QueueLimit < 0) throw new ArgumentException("QueueLimit must not be negative");
        if (QueueWaitMs < 0) throw new ArgumentException("QueueWaitMs must not be negative");
        if (!System.Numerics.BitOperations.IsPow2(TtEntries) || TtEntries < 4)
            throw new ArgumentException("TtEntries must be a power of two >= 4");
        if (!System.Numerics.BitOperations.IsPow2(MemoEntries) || MemoEntries < 4)
            throw new ArgumentException("MemoEntries must be a power of two >= 4");
    }

    public IPAddress Address => IPAddress.Parse(Host);
}
