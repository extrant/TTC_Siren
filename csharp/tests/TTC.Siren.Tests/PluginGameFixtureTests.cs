using System.Net.Http;
using System.Text;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// The sample game of the web UI. A complete game is played by the engine on both sides through the real server; its log is the DEMO REPLAY of the landing page
/// (<c>web/src/demo/demo-game.json</c>) and the requests the plugin sent are the fixture of the browser tests (<c>web/e2e/fixtures/plugin-game.json</c>), which post them to a
/// real server and watch the game appear. The game is a pure function of a seed: this test replays it and checks that the committed requests are what the engine produces today.
/// To make new files (after a change of the rules, the engine or the log format): <c>TTC_UPDATE_FIXTURES=1</c> once; it also searches for a seed whose game shows every cause of a flip.
/// </summary>
public sealed class PluginGameFixtureTests : IDisposable
{
    private const string Rules = "加算,同数,王牌杀手"; // not 逆转: under it every plain comparison is a Reverse flip, so one game cannot show both
    private const string GameId = "demo-game";
    private const int Seed = 185; // found by the search below (TTC_UPDATE_FIXTURES=1): every cause but Reverse, a win for the user, four user moves

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-fixture-" + Guid.NewGuid().ToString("N"));

    public PluginGameFixtureTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private static string DemoPath => Path.Combine(GoldenFiles.RepoRoot, "web", "src", "demo", "demo-game.json");
    private static string FixturePath => Path.Combine(GoldenFiles.RepoRoot, "web", "e2e", "fixtures", "plugin-game.json");

    private sealed record Played(List<string> Requests, List<string> LogLines, HashSet<string> Causes, int UserMoves, string Winner);

    private async Task<Played> PlayAsync(int seed)
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        var clock = new ManualTimeProvider(new DateTimeOffset(2026, 10, 11, 9, 0, 0, TimeSpan.Zero));
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries, TestSeedHeader = true,
            GameLog = new GameLogOptions { Directory = Path.Combine(_dir, "logs-" + seed + "-" + Guid.NewGuid().ToString("N")) },
            Security = new TTC.Siren.Server.Security.SecurityOptions { RequireKey = false, DefaultBurst = 1000, DefaultRatePerMinute = 100000, DefaultMaxInFlight = 64 },
        };
        await using var host = ServerHost.Create(options, time: clock);
        await host.StartAsync();
        using var http = new HttpClient();
        var sim = new GameSimulator(db, opponentAi, new Random(seed), Rules, user: seed % 2, first: (seed / 2) % 2);
        var requests = new List<string>();
        int turn = 0;
        while (!sim.Finished)
        {
            if (sim.UserToMove)
            {
                string body = sim.UserRequest(4, $",\"gameId\":\"{GameId}\"");
                requests.Add(body);
                var (status, text) = await SimHttp.PostAsync(http, host.BaseUrl, body, seed: (ulong)(9000 + turn++));
                Assert.Equal(200, status);
                using var answer = JsonDocument.Parse(text);
                sim.PlayUser(answer.RootElement);
                clock.Advance(TimeSpan.FromSeconds(4));
            }
            else
            {
                await sim.PlayOpponentAsync();
                clock.Advance(TimeSpan.FromSeconds(3));
            }
        }
        if (sim.UserToMove) // the opponent's last move is seen by one more request
        {
            string body = sim.UserRequest(4, $",\"gameId\":\"{GameId}\"");
            requests.Add(body);
            var (status, _) = await SimHttp.PostAsync(http, host.BaseUrl, body, seed: (ulong)(9000 + turn));
            Assert.Equal(200, status);
        }
        var lines = host.GameLog.Store.ReadLines("local", GameId);
        var causes = sim.Truth.SelectMany(t => t.Flips.Select(f => f.Cause)).ToHashSet();
        var (mine, theirs) = sim.FinalCount();
        return new Played(requests, lines, causes, sim.Truth.Count(t => t.Mover == sim.User), mine > theirs ? "mine" : mine < theirs ? "theirs" : "draw");
    }

    private static readonly string[] AllCauses = { "basic", "same", "plus", "fallen_ace", "combo" };

    [Fact]
    public async Task The_sample_game_is_what_the_engine_plays_and_its_files_are_current()
    {
        bool update = Environment.GetEnvironmentVariable("TTC_UPDATE_FIXTURES") == "1";
        int seed = Seed;
        if (update)
        {
            // the best seed whose game shows every cause of a flip: one that the user wins, and short
            Played? best = null;
            int bestSeed = -1;
            var seen = new SortedDictionary<string, int>();
            for (int s = 1; s <= 3000; s++)
            {
                var p = await PlayAsync(s);
                foreach (var c in p.Causes) seen[c] = seen.GetValueOrDefault(c) + 1;
                if (!AllCauses.All(p.Causes.Contains)) continue;
                int Rank(Played x) => (x.Winner == "mine" ? 0 : 100) + x.UserMoves; // a game the user wins, and a short one
                if (best is null || Rank(p) < Rank(best)) { best = p; bestSeed = s; }
                if (Rank(best) <= 5) break;
            }
            Assert.True(best is not null, "no seed with every cause in 3000 games; games with each cause: " + string.Join(", ", seen.Select(kv => $"{kv.Key} {kv.Value}")));
            seed = bestSeed;
            _output.WriteLine($"seed {seed}: {best!.UserMoves} user moves, causes {string.Join(",", best.Causes.OrderBy(c => c))}, winner {best.Winner}");
        }

        var game = await PlayAsync(seed);
        string demo = "{\"game_id\":\"" + GameId + "\",\"note\":\"a sample game played by the engine on both sides; made by PluginGameFixtureTests\",\"records\":[" + string.Join(",", game.LogLines) + "]}\n";
        string fixture = JsonSerializer.Serialize(new { game_id = GameId, rules = Rules, seed, requests = game.Requests }, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }) + "\n";
        if (update)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(DemoPath)!);
            Directory.CreateDirectory(Path.GetDirectoryName(FixturePath)!);
            File.WriteAllText(DemoPath, demo, new UTF8Encoding(false));
            File.WriteAllText(FixturePath, fixture, new UTF8Encoding(false));
            Assert.Equal(Seed, seed); // put the seed into this file
        }

        // the committed files: the plugin's requests are exactly what this seed produces, and the demo log is a well-formed log of that game
        Assert.True(File.Exists(FixturePath) && File.Exists(DemoPath), "fixtures missing: run once with TTC_UPDATE_FIXTURES=1");
        using var committed = JsonDocument.Parse(File.ReadAllText(FixturePath));
        Assert.Equal(game.Requests, committed.RootElement.GetProperty("requests").EnumerateArray().Select(e => e.GetString()!).ToList());
        using var demoDoc = JsonDocument.Parse(File.ReadAllText(DemoPath));
        var records = demoDoc.RootElement.GetProperty("records").EnumerateArray().ToList();
        Assert.Equal(game.LogLines.Count, records.Count);
        Assert.Equal(Enumerable.Range(0, records.Count), records.Select(r => r.GetProperty("seq").GetInt32()));
        Assert.All(records, r => Assert.Equal(GameLogService.SchemaVersion, r.GetProperty("v").GetInt32()));
        Assert.Equal("game_start", records[0].GetProperty("type").GetString());
        Assert.Equal("game_end", records[^1].GetProperty("type").GetString());
        // both languages are in every explanation, and the game shows every cause of a flip (so the demo shows every effect)
        foreach (var r in records.Where(r => r.GetProperty("type").GetString() == "request" && r.TryGetProperty("explanation", out _)))
        {
            var texts = r.GetProperty("explanation").GetProperty("texts");
            Assert.True(texts.TryGetProperty("zh-CN", out _) && texts.TryGetProperty("en", out _));
        }
        Assert.True(AllCauses.All(game.Causes.Contains), "the sample game does not show every cause: " + string.Join(",", game.Causes));
        _output.WriteLine($"seed {seed}: {records.Count} records, {game.Requests.Count} plugin requests, causes {string.Join(",", game.Causes.OrderBy(c => c))}");
    }
}
