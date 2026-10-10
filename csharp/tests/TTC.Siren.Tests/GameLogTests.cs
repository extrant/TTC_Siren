using System.Net.Http;
using System.Text;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Explain;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

internal sealed class ManualTimeProvider : TimeProvider
{
    private DateTimeOffset _now;

    public ManualTimeProvider(DateTimeOffset start) => _now = start;

    public override DateTimeOffset GetUtcNow() => _now;

    public void Advance(TimeSpan by) => _now += by;
}

/// <summary>A test authenticator: the header <c>X-Test-Key: name</c> selects the key id "name" (a request without it is refused with 401).</summary>
internal sealed class HeaderAuthenticator : IApiKeyAuthenticator
{
    public AuthResult Authenticate(Microsoft.AspNetCore.Http.HttpRequest request) =>
        request.Headers.TryGetValue("X-Test-Key", out var v) && v.ToString().Length > 0 ? AuthResult.Allowed(v.ToString()) : AuthResult.Denied(401, "missing key");
}

/// <summary>
/// Game logs (goal section C2): complete games played by the engine on both sides are reconstructed exactly from the log — the moves of the opponent
/// that the log infers between requests, with the cause of every flip, the end of the game — and every logged decision is reproduced from its record.
/// Also: game identity (client id, inference, the time gap), retention and deletion, and that a key sees only its own games.
/// </summary>
public sealed class GameLogTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-gamelog-" + Guid.NewGuid().ToString("N"));

    public GameLogTests(ITestOutputHelper output) => _output = output;

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

    private async Task<ServerHost> StartAsync(IApiKeyAuthenticator? auth = null, TimeProvider? time = null, int retentionDays = 30)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries,
            GameLog = new GameLogOptions { Directory = _dir, RetentionDays = retentionDays, GapMinutes = 15 },
            Security = new TTC.Siren.Server.Security.SecurityOptions { RequireKey = false, DefaultBurst = 1000, DefaultRatePerMinute = 100000, DefaultMaxInFlight = 64 },
        };
        var host = ServerHost.Create(options, authenticator: auth, time: time);
        await host.StartAsync();
        return host;
    }

    private static readonly string[] RuleSets =
    {
        "", "加算", "同数", "加算,同数", "逆转", "王牌杀手", "同类强化", "同类弱化", "加算,同数,逆转,王牌杀手", "同数,同类强化", "全明牌,加算", "逆转,同类弱化",
    };

    private static List<JsonElement> Records(GameLogStore store, string key, string gameId, List<JsonDocument> keep)
    {
        var list = new List<JsonElement>();
        foreach (var line in store.ReadLines(key, gameId))
        {
            var doc = JsonDocument.Parse(line);
            keep.Add(doc);
            list.Add(doc.RootElement);
        }
        return list;
    }

    /// <summary>Plays one complete game against the server and returns the simulator (with the truth) and the id of the logged game.</summary>
    private static async Task<GameSimulator> PlayAsync(HttpClient client, string url, CardDb db, AiMoveService opponentAi, int index, bool sendFinalRequest, string? gameId)
    {
        var rng = new Random(900 + index);
        var sim = new GameSimulator(db, opponentAi, rng, RuleSets[index % RuleSets.Length], user: index % 2, first: (index / 2) % 2);
        while (!sim.Finished)
        {
            if (sim.UserToMove)
            {
                string extra = gameId is null ? "" : $",\"gameId\":\"{gameId}\"";
                var (status, text) = await SimHttp.PostAsync(client, url, sim.UserRequest(6, extra));
                Assert.Equal(200, status);
                using var answer = JsonDocument.Parse(text);
                sim.PlayUser(answer.RootElement);
            }
            else await sim.PlayOpponentAsync();
        }
        if (sendFinalRequest && sim.UserToMove)
        {
            string extra = gameId is null ? "" : $",\"gameId\":\"{gameId}\"";
            var (status, _) = await SimHttp.PostAsync(client, url, sim.UserRequest(6, extra));
            Assert.Equal(200, status);
        }
        return sim;
    }

    [Fact]
    public async Task Thirty_six_complete_games_are_reconstructed_exactly_and_every_decision_replays()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        var replayService = AiMoveParityTests.CreateService(); // same table size as the server: node counts and lines match
        await using var host = await StartAsync();
        using var client = new HttpClient();
        var keep = new List<JsonDocument>();
        int games = 0, opponentMoves = 0, flips = 0, requests = 0, ends = 0, replayed = 0;
        var causes = new SortedDictionary<string, int>();
        try
        {
            for (int i = 0; i < 36; i++)
            {
                bool finalRequest = i % 3 != 2;
                string? clientId = i % 3 == 0 ? $"sim-{i}" : null;
                var sim = await PlayAsync(client, host.BaseUrl, db, opponentAi, i, finalRequest, clientId);

                var summaries = host.GameLog.Store.List("local", 0, 200).Items;
                string gameId = clientId ?? summaries.First().GameId;
                var records = Records(host.GameLog.Store, "local", gameId, keep);

                // ---- structure
                Assert.Equal(Enumerable.Range(0, records.Count), records.Select(r => r.GetProperty("seq").GetInt32()));
                Assert.Equal("game_start", records[0].GetProperty("type").GetString());
                Assert.Equal(1, records.Count(r => r.GetProperty("type").GetString() == "game_start"));
                Assert.Equal(clientId is null ? "new" : "client", records[0].GetProperty("id_source").GetString());
                Assert.Equal(sim.User == 0 ? "red" : "blue", records[0].GetProperty("my_owner").GetString());
                var times = records.Select(r => DateTimeOffset.Parse(r.GetProperty("t").GetString()!)).ToList();
                Assert.True(times.Zip(times.Skip(1), (a, b) => a <= b).All(x => x));
                Assert.All(records, r => Assert.Equal(1, r.GetProperty("v").GetInt32()));

                // ---- the requests: one per turn of the user
                var requestRecords = records.Where(r => r.GetProperty("type").GetString() == "request").ToList();
                int userMoves = sim.Truth.Count(t => t.Mover == sim.User);
                int expectedRequests = userMoves + (finalRequest && sim.Truth[^1].Mover != sim.User ? 1 : 0);
                Assert.Equal(expectedRequests, requestRecords.Count);
                Assert.Equal(Enumerable.Range(1, requestRecords.Count), requestRecords.Select(r => r.GetProperty("request_no").GetInt32()));

                // ---- the opponent's moves, reconstructed from the requests alone
                int firstUserTurn = sim.Truth.FindIndex(t => t.Mover == sim.User);
                var expected = new List<TrueMove>();
                for (int t = firstUserTurn + 1; t < sim.Truth.Count; t++)
                {
                    if (sim.Truth[t].Mover == sim.User) continue;
                    bool seen = sim.Truth.Skip(t + 1).Any(x => x.Mover == sim.User) || finalRequest; // some later request shows it
                    if (seen) expected.Add(sim.Truth[t]);
                }
                var inferred = records.Where(r => r.GetProperty("type").GetString() == "opponent_move").ToList();
                Assert.True(expected.Count == inferred.Count, $"game {i}: {inferred.Count} opponent moves logged, {expected.Count} expected");
                for (int k = 0; k < expected.Count; k++)
                {
                    var truth = expected[k];
                    var rec = inferred[k];
                    string where = $"game {i} opponent move {k}";
                    Assert.True(rec.GetProperty("consistent").GetBoolean(), $"{where}: not consistent: {rec.GetProperty("note")}");
                    Assert.Equal(truth.CardId, rec.GetProperty("card_id").GetInt32());
                    Assert.Equal(truth.Cell / 3, rec.GetProperty("row").GetInt32());
                    Assert.Equal(truth.Cell % 3, rec.GetProperty("col").GetInt32());
                    Assert.Equal(truth.Mover == 0 ? "red" : "blue", rec.GetProperty("owner").GetString());
                    var logged = rec.GetProperty("flips").EnumerateArray()
                        .Select(f => (f.GetProperty("row").GetInt32() * 3 + f.GetProperty("col").GetInt32(), f.GetProperty("card_id").GetInt32(), f.GetProperty("cause").GetString()!)).ToList();
                    Assert.True(logged.SequenceEqual(truth.Flips), $"{where}: flips {string.Join(' ', logged)} expected {string.Join(' ', truth.Flips)}");
                    Assert.True(rec.GetProperty("my_move").GetProperty("as_recommended").GetBoolean(), $"{where}: the user's move was the recommended one");
                    foreach (var f in truth.Flips) causes[f.Cause] = causes.GetValueOrDefault(f.Cause) + 1;
                    flips += truth.Flips.Count;
                    opponentMoves++;
                }

                // ---- the end of the game
                var end = records.Where(r => r.GetProperty("type").GetString() == "game_end").ToList();
                bool userPlaysLast = sim.Truth[^1].Mover == sim.User;
                if (userPlaysLast || finalRequest)
                {
                    var e = Assert.Single(end);
                    var (mine, theirs) = sim.FinalCount();
                    Assert.Equal(mine, e.GetProperty("final").GetProperty("mine").GetInt32());
                    Assert.Equal(theirs, e.GetProperty("final").GetProperty("theirs").GetInt32());
                    Assert.Equal(mine > theirs ? "mine" : mine < theirs ? "theirs" : "draw", e.GetProperty("winner").GetString());
                    Assert.Equal(userPlaysLast ? "own_last_move" : "observed", e.GetProperty("source").GetString());
                    ends++;
                }
                else Assert.Empty(end); // the opponent's last card was never shown to the server

                // ---- every decision replays: same move, same scores at every depth, same facts and text
                foreach (var rr in requestRecords.Where(r => r.GetProperty("http_status").GetInt32() == 200 && r.GetProperty("result").GetProperty("move").ValueKind == JsonValueKind.Object))
                {
                    var outcome = await GameReplay.ReplayAsync(replayService, rr);
                    var differences = ReplayDifferences(rr, outcome);
                    Assert.True(differences.Count == 0, $"game {i} request {rr.GetProperty("request_no")}: replay differs:\n  " + string.Join("\n  ", differences.Take(10)));
                    replayed++;
                }
                requests += requestRecords.Count;
                games++;
            }
        }
        finally
        {
            foreach (var d in keep) d.Dispose();
        }
        Assert.Equal(36, games);
        Assert.True(opponentMoves >= 120 && flips >= 60 && ends >= 30 && replayed >= 150 && causes.Count >= 5, $"{opponentMoves} opponent moves, {flips} flips, {ends} ends, {replayed} replays, causes {string.Join(",", causes.Keys)}");
        _output.WriteLine($"{games} complete games reconstructed exactly: {requests} requests, {opponentMoves} inferred opponent moves with {flips} flips (causes: " +
                          string.Join(", ", causes.Select(p => $"{p.Key} {p.Value}")) + $"), {ends} game ends; {replayed} decisions replayed identically");
    }

    internal static List<string> ReplayDifferences(JsonElement record, AiMoveOutcome replay)
    {
        var diffs = new List<string>();
        if (replay.Status != 200 || replay.Solve is null) return new List<string> { "replay status " + replay.Status };
        var logged = record.GetProperty("result");
        var solve = replay.Solve;
        MoveOutLike? Logged(JsonElement e) => e.ValueKind == JsonValueKind.Object
            ? new MoveOutLike(e.GetProperty("slot").GetInt32(), e.GetProperty("row").GetInt32(), e.GetProperty("col").GetInt32(), e.GetProperty("card_id").GetInt32()) : null;
        MoveOutLike? FromSolve(TTC.Siren.Engine.MoveOut? m) => m is { } v ? new MoveOutLike(v.Slot, v.Row, v.Col, v.CardId) : null;
        if (!Equals(Logged(logged.GetProperty("move")), FromSolve(solve.Final))) diffs.Add("move");
        if (!Equals(Logged(logged.GetProperty("standard_move")), FromSolve(solve.Standard))) diffs.Add("standard move");
        if (logged.GetProperty("depth_reached").GetInt32() != solve.DepthCompleted) diffs.Add("depth reached");
        var depths = logged.GetProperty("depths").EnumerateArray().ToList();
        if (depths.Count != solve.Depths.Count) diffs.Add("number of depths");
        else
            for (int d = 0; d < depths.Count; d++)
            {
                if (depths[d].GetProperty("score_bits").GetString() != BitConverter.DoubleToInt64Bits(solve.Depths[d].Score).ToString("X16")) diffs.Add($"score of depth {d + 1}");
                if (!Equals(Logged(depths[d].GetProperty("move")), FromSolve(solve.Depths[d].Move))) diffs.Add($"move of depth {d + 1}");
                if (depths[d].GetProperty("nodes_total").GetInt64() != solve.Depths[d].NodesTotal) diffs.Add($"nodes of depth {d + 1}");
            }
        if ((logged.GetProperty("robust").ValueKind == JsonValueKind.Object) != (solve.Robust is not null)) diffs.Add("robust layer");
        else if (solve.Robust is { } rb && (logged.GetProperty("robust").GetProperty("scenarios").GetInt32() != rb.Scenarios || logged.GetProperty("robust").GetProperty("overridden").GetBoolean() != rb.Overridden))
            diffs.Add("robust result");

        if (record.TryGetProperty("explanation", out var explanation) && explanation.ValueKind == JsonValueKind.Object)
        {
            using var replayed = JsonDocument.Parse(PyJson.Serialize(replay.Facts!.ToJson()));
            // search time always differs; after an interrupted search the node count and the requested depth do too
            bool interrupted = GameReplay.WasInterrupted(record, out _);
            var compare = new JsonCompare(path => path.EndsWith(".search_ms", StringComparison.Ordinal) ||
                                                  (interrupted && (path.EndsWith(".nodes", StringComparison.Ordinal) || path.EndsWith(".depth_requested", StringComparison.Ordinal) ||
                                                                   path.EndsWith(".stopped_early", StringComparison.Ordinal))));
            compare.Compare(explanation.GetProperty("facts"), replayed.RootElement);
            diffs.AddRange(compare.Differences.Select(x => "facts " + x));
            if (explanation.GetProperty("text").GetProperty("brief").GetString() != replay.Text!.Brief) diffs.Add("brief text");
            if (explanation.GetProperty("text").GetProperty("full").GetString() != replay.Text!.Full) diffs.Add("full text");
        }
        return diffs;
    }

    private sealed record MoveOutLike(int Slot, int Row, int Col, int CardId);

    // =============================================================================================================== identity

    [Fact]
    public async Task Game_identity_follows_the_client_id_then_inference_then_the_time_gap()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        var time = new ManualTimeProvider(new DateTimeOffset(2026, 10, 11, 12, 0, 0, TimeSpan.Zero));
        await using var host = await StartAsync(time: time);
        using var client = new HttpClient();

        // a game without a client id: two requests of one game are one game, because the second is a consistent successor of the first
        var sim = new GameSimulator(db, opponentAi, new Random(5), "加算,同数", user: 0, first: 0);
        var (_, t1) = await SimHttp.PostAsync(client, host.BaseUrl, sim.UserRequest(4));
        using (var a1 = JsonDocument.Parse(t1)) sim.PlayUser(a1.RootElement);
        await sim.PlayOpponentAsync();
        time.Advance(TimeSpan.FromMinutes(14));
        var (_, t2) = await SimHttp.PostAsync(client, host.BaseUrl, sim.UserRequest(4));
        using var r1 = JsonDocument.Parse(t1);
        using var r2 = JsonDocument.Parse(t2);
        Assert.Equal(r1.RootElement.GetProperty("game_id").GetString(), r2.RootElement.GetProperty("game_id").GetString());
        string game = r1.RootElement.GetProperty("game_id").GetString()!;
        var lines = host.GameLog.Store.ReadLines("local", game);
        Assert.Contains("\"id_source\":\"new\"", lines[0]);
        Assert.Contains(lines, l => l.Contains("\"type\":\"opponent_move\""));
        Assert.Contains(lines, l => l.Contains("consistent successor"));

        // the same request again 16 minutes later is a new game (beyond the gap), the old game stays as it was
        using (var again = JsonDocument.Parse(t2)) sim.PlayUser(again.RootElement);
        await sim.PlayOpponentAsync();
        time.Advance(TimeSpan.FromMinutes(16));
        var (_, t3) = await SimHttp.PostAsync(client, host.BaseUrl, sim.UserRequest(4));
        using var r3 = JsonDocument.Parse(t3);
        Assert.NotEqual(game, r3.RootElement.GetProperty("game_id").GetString());
        Assert.Equal(2, host.GameLog.Store.List("local", 0, 50).Total);

        // a request of the other colour never continues a game
        var other = new GameSimulator(db, opponentAi, new Random(6), "加算,同数", user: 1, first: 1);
        var (_, t4) = await SimHttp.PostAsync(client, host.BaseUrl, other.UserRequest(4));
        using var r4 = JsonDocument.Parse(t4);
        Assert.NotEqual(r3.RootElement.GetProperty("game_id").GetString(), r4.RootElement.GetProperty("game_id").GetString());

        // a client id wins over inference, and a later request with the same id continues the same game (also across a restart of the in-memory index)
        var (_, t5) = await SimHttp.PostAsync(client, host.BaseUrl, sim.UserRequest(4, ",\"gameId\":\"my-own-id\""));
        using var r5 = JsonDocument.Parse(t5);
        Assert.Equal("my-own-id", r5.RootElement.GetProperty("game_id").GetString());
        await using var restarted = await StartAsync(time: time);
        var (_, t6) = await SimHttp.PostAsync(client, restarted.BaseUrl, sim.UserRequest(4, ",\"gameId\":\"my-own-id\""));
        using var r6 = JsonDocument.Parse(t6);
        Assert.Equal("my-own-id", r6.RootElement.GetProperty("game_id").GetString());
        var own = host.GameLog.Store.ReadLines("local", "my-own-id");
        Assert.Equal(2, own.Count(l => l.Contains("\"type\":\"request\"")));
        Assert.Equal(1, own.Count(l => l.Contains("\"type\":\"game_start\"")));

        // an invalid client id is ignored (it never reaches a file path)
        var (_, t7) = await SimHttp.PostAsync(client, host.BaseUrl, sim.UserRequest(4, ",\"gameId\":\"../../evil\""));
        using var r7 = JsonDocument.Parse(t7);
        Assert.Matches("^g-[0-9]{14}-[0-9a-f]{8}$", r7.RootElement.GetProperty("game_id").GetString()!);
        Assert.False(File.Exists(Path.Combine(_dir, "..", "evil.jsonl")));
    }

    // =============================================================================================================== API, isolation, retention

    [Fact]
    public async Task The_games_api_lists_reads_downloads_and_deletes_and_a_key_sees_only_its_own_games()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        await using var host = await StartAsync(new HeaderAuthenticator());
        using var client = new HttpClient();
        var simA = new GameSimulator(db, opponentAi, new Random(11), "加算", user: 0, first: 0);
        var simB = new GameSimulator(db, opponentAi, new Random(12), "同数", user: 1, first: 0);
        var (sa, ta) = await SimHttp.PostAsync(client, host.BaseUrl, simA.UserRequest(4), key: "alice");
        var (sb, tb) = await SimHttp.PostAsync(client, host.BaseUrl, simB.UserRequest(4), key: "bob");
        Assert.Equal(200, sa);
        Assert.Equal(200, sb);
        var (s0, _) = await SimHttp.PostAsync(client, host.BaseUrl, simA.UserRequest(4));
        Assert.Equal(401, s0);
        string gameA = JsonDocument.Parse(ta).RootElement.GetProperty("game_id").GetString()!;
        string gameB = JsonDocument.Parse(tb).RootElement.GetProperty("game_id").GetString()!;

        async Task<(int, string)> Call(HttpMethod method, string path, string? key)
        {
            var message = new HttpRequestMessage(method, host.BaseUrl + path);
            if (key is not null) message.Headers.Add("X-Test-Key", key);
            using var response = await client.SendAsync(message);
            return ((int)response.StatusCode, await response.Content.ReadAsStringAsync());
        }

        // each key lists exactly its own game
        var (ls, listA) = await Call(HttpMethod.Get, "/v1/games", "alice");
        Assert.Equal(200, ls);
        using (var doc = JsonDocument.Parse(listA))
        {
            Assert.Equal(1, doc.RootElement.GetProperty("total").GetInt32());
            Assert.Equal(gameA, doc.RootElement.GetProperty("games")[0].GetProperty("game_id").GetString());
        }
        using (var doc = JsonDocument.Parse((await Call(HttpMethod.Get, "/v1/games", "bob")).Item2))
            Assert.Equal(gameB, doc.RootElement.GetProperty("games")[0].GetProperty("game_id").GetString());

        // reading: own game yes, the other key's game is "not found" (not "forbidden": it is not revealed that it exists)
        Assert.Equal(200, (await Call(HttpMethod.Get, "/v1/games/" + gameA, "alice")).Item1);
        Assert.Equal(404, (await Call(HttpMethod.Get, "/v1/games/" + gameA, "bob")).Item1);
        Assert.Equal(404, (await Call(HttpMethod.Get, "/v1/games/" + gameA + "/download", "bob")).Item1);
        Assert.Equal(404, (await Call(HttpMethod.Delete, "/v1/games/" + gameA, "bob")).Item1);
        Assert.Equal(401, (await Call(HttpMethod.Get, "/v1/games", null)).Item1);
        Assert.Equal(401, (await Call(HttpMethod.Get, "/v1/live", null)).Item1);
        Assert.Equal(400, (await Call(HttpMethod.Get, "/v1/games/a..b%2Fc", "alice")).Item1 is 400 or 404 ? 400 : 0);

        // the full game as JSON and as a JSONL download
        var (_, json) = await Call(HttpMethod.Get, "/v1/games/" + gameA, "alice");
        using (var doc = JsonDocument.Parse(json))
        {
            Assert.Equal(1, doc.RootElement.GetProperty("schema").GetInt32());
            Assert.Equal("game_start", doc.RootElement.GetProperty("records")[0].GetProperty("type").GetString());
        }
        var (ds, jsonl) = await Call(HttpMethod.Get, "/v1/games/" + gameA + "/download", "alice");
        Assert.Equal(200, ds);
        Assert.Equal(host.GameLog.Store.ReadLines("alice", gameA), jsonl.Split('\n', StringSplitOptions.RemoveEmptyEntries).ToList());

        // deletion removes the file; the other key's game is untouched
        Assert.Equal(200, (await Call(HttpMethod.Delete, "/v1/games/" + gameA, "alice")).Item1);
        Assert.Equal(404, (await Call(HttpMethod.Get, "/v1/games/" + gameA, "alice")).Item1);
        Assert.Equal(404, (await Call(HttpMethod.Delete, "/v1/games/" + gameA, "alice")).Item1);
        Assert.Equal(200, (await Call(HttpMethod.Get, "/v1/games/" + gameB, "bob")).Item1);
        Assert.True(Directory.Exists(Path.Combine(_dir, "bob")));
    }

    [Fact]
    public async Task Retention_removes_old_games_keeps_recent_ones_and_the_cleanup_job_is_the_same_code()
    {
        var time = new ManualTimeProvider(new DateTimeOffset(2026, 10, 11, 12, 0, 0, TimeSpan.Zero));
        var store = new GameLogStore(new GameLogOptions { Directory = _dir, RetentionDays = 30 }, time);
        store.Append("alice", "old-game", "{\"type\":\"game_start\"}");
        store.Append("alice", "recent-game", "{\"type\":\"game_start\"}");
        store.Append("bob", "old-bob", "{\"type\":\"game_start\"}");
        File.SetLastWriteTimeUtc(Path.Combine(_dir, "alice", "old-game.jsonl"), time.GetUtcNow().UtcDateTime - TimeSpan.FromDays(31));
        File.SetLastWriteTimeUtc(Path.Combine(_dir, "alice", "recent-game.jsonl"), time.GetUtcNow().UtcDateTime - TimeSpan.FromDays(29));
        File.SetLastWriteTimeUtc(Path.Combine(_dir, "bob", "old-bob.jsonl"), time.GetUtcNow().UtcDateTime - TimeSpan.FromDays(45));

        Assert.Equal(2, store.Cleanup());
        Assert.False(store.Exists("alice", "old-game"));
        Assert.True(store.Exists("alice", "recent-game"));
        Assert.False(store.Exists("bob", "old-bob"));
        Assert.False(Directory.Exists(Path.Combine(_dir, "bob"))); // a key directory left empty is removed

        // the retention is configurable: a shorter period removes the "recent" game as well
        var shorter = new GameLogStore(new GameLogOptions { Directory = _dir, RetentionDays = 7 }, time);
        Assert.Equal(1, shorter.Cleanup());
        Assert.False(shorter.Exists("alice", "recent-game"));

        // ids that could reach another path are refused
        Assert.Throws<ArgumentException>(() => store.Append("alice", "../x", "{}"));
        Assert.Throws<ArgumentException>(() => store.Append("..", "x", "{}"));
        Assert.Throws<ArgumentException>(() => store.Exists("a/b", "x"));
        await Task.CompletedTask;
    }

    [Fact]
    public void Switching_the_log_off_writes_nothing()
    {
        var store = new GameLogStore(new GameLogOptions { Directory = _dir, Enabled = false });
        Assert.False(store.Options.Enabled);
        Assert.Empty(store.Files("alice"));
    }
}
