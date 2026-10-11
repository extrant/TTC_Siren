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
/// A real server on a loopback socket for the tests of the Phase 8A review findings that 8B fixes (a temporary log folder per test, every started server
/// stopped at the end): the same code path as production, with the key check switched off.
/// </summary>
public abstract class ServerTestBase : IDisposable
{
    protected readonly ITestOutputHelper _output;
    protected readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-fixes-" + Guid.NewGuid().ToString("N"));
    private readonly List<IAsyncDisposable> _hosts = new();

    protected ServerTestBase(ITestOutputHelper output)
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

    protected static ServerOptions Options(string logDir, Action<ServerOptions>? tweak = null)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries,
            GameLog = new GameLogOptions { Directory = logDir },
            Security = new SecurityOptions { RequireKey = false, DefaultBurst = 1000, DefaultRatePerMinute = 100000, DefaultMaxInFlight = 64 },
        };
        tweak?.Invoke(options);
        return options;
    }

    protected async Task<ServerHost> StartAsync(Action<ServerOptions>? tweak = null, Microsoft.Extensions.Logging.ILoggerProvider? logger = null, string? sub = null)
    {
        var host = ServerHost.Create(Options(Path.Combine(_dir, sub ?? "logs-" + Guid.NewGuid().ToString("N")), tweak), loggerProvider: logger);
        await host.StartAsync();
        _hosts.Add(new HostDisposer(host));
        return host;
    }

    protected static HttpRequestMessage Post(string url, string json, ulong? seed = null, params (string Name, string Value)[] headers)
    {
        var m = new HttpRequestMessage(HttpMethod.Post, url + "/ai_move") { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        if (seed is not null) m.Headers.Add("X-Ttc-Test-Seed", seed.Value.ToString());
        foreach (var (n, v) in headers) m.Headers.TryAddWithoutValidation(n, v);
        return m;
    }

    protected const string Opening = """
        {"board":[],"myHand":[{"numU":4,"numL":2,"numD":3,"numR":4},{"numU":5,"numL":4,"numD":6,"numR":5},{"numU":2,"numL":2,"numD":7,"numR":2}],
        "oppHand":[{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0}],
        "myOwner":1,"currentPlayer":1,"rules":"加算,同数","max_depth":3,"max_time":5.5,"gameId":"fixes-1"}
        """;
}
