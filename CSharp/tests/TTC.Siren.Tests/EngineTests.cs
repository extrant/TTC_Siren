using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Phase 7: the engine protocol, strict validation, edge cases of maxDepth/maxTime, duplicate ids, robust round trips.</summary>
public sealed class EngineTests : IClassFixture<EngineFixture>
{
    private readonly EngineFixture _fx;
    private readonly ITestOutputHelper _output;

    public EngineTests(EngineFixture fx, ITestOutputHelper output)
    {
        _fx = fx;
        _output = output;
    }

    private static CardDb Db => SearchGolden.Db;

    private static StateSnapshot Position(string name) => ReferenceScenarios.Create(name, Db).ToSnapshot();

    private static long Bits(JsonNode node) => Convert.ToInt64(node["scoreBits"]!.GetValue<string>(), 16);

    // ------------------------------------------------------------------ health

    [Fact]
    public async Task Health_reports_protocol_card_database_hash_and_worker_state()
    {
        var response = await _fx.Client.GetAsync("/v1/health");
        Assert.Equal(200, (int)response.StatusCode);
        var body = JsonNode.Parse(await response.Content.ReadAsStringAsync())!;
        Assert.Equal(Protocol.Version, body["protocol"]!.GetValue<int>());
        Assert.Equal("ok", body["status"]!.GetValue<string>());
        string path = CardDb.LocateDataFile(AppContext.BaseDirectory);
        string sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
        Assert.Equal(sha, body["cardDbSha256"]!.GetValue<string>());
        Assert.Equal(475, body["cardCount"]!.GetValue<int>());
        Assert.Equal(4, body["workers"]!.GetValue<int>());
        Assert.True(body["idleWorkers"]!.GetValue<int>() <= 4);
        Assert.Equal(256, body["queueLimit"]!.GetValue<int>());
        Assert.False(string.IsNullOrEmpty(body["version"]!.GetValue<string>()));
        // the Python client derives its read timeout from the engine's queue wait (A-20 / M7-1), so health must report it
        Assert.Equal(120_000, body["queueWaitMs"]!.GetValue<int>());
        Assert.Equal(0, body["abandonedSkipped"]!.GetValue<long>());
        Assert.True(body["served"]!.GetValue<long>() >= 0);
        Assert.Equal(System.Runtime.GCSettings.IsServerGC, body["serverGc"]!.GetValue<bool>());
    }

    [Fact]
    public void The_engine_runtime_config_states_the_chosen_gc_mode_explicitly()
    {
        // Review L7-1: the project used to set a property MSBuild ignores (ServerGarbageCollector). The measured choice (Server GC,
        // non-concurrent; docs/migration/reports/phase7/gc) must be what the build writes into the runtimeconfig.json.
        string engineBin = Path.Combine(GoldenFiles.RepoRoot, "..", "CSharp", "src", "TTC.Siren.Engine", "bin");
        var configs = Directory.GetFiles(engineBin, "TTC.Siren.Engine.runtimeconfig.json", SearchOption.AllDirectories);
        Assert.NotEmpty(configs);
        foreach (string config in configs)
        {
            var props = JsonNode.Parse(File.ReadAllText(config))!["runtimeOptions"]!["configProperties"]!;
            Assert.True(props["System.GC.Server"]!.GetValue<bool>(), config);
            Assert.False(props["System.GC.Concurrent"]!.GetValue<bool>(), config);
        }
    }

    [Fact]
    public void The_default_queue_settings_fit_an_engine_that_has_no_fallback()
    {
        // A-20: a request waits for the engine instead of being answered by Python, so the queue is deep and the wait long.
        // docs/DEPLOY_WINDOWS.md section 3 documents these defaults.
        var defaults = new EngineOptions();
        Assert.Equal(256, defaults.QueueLimit);
        Assert.Equal(60_000, defaults.QueueWaitMs);
        var parsed = EngineOptions.Parse(new[] { "--port", "0" });
        Assert.Equal(256, parsed.QueueLimit);
        Assert.Equal(60_000, parsed.QueueWaitMs);
    }

    // ------------------------------------------------------------------ round trip against the oracle

    [Theory]
    [InlineData("open_none", false)]
    [InlineData("open_same_plus", true)]
    [InlineData("mid3_ascension", false)]
    [InlineData("mid4_rev_ace", true)]
    [InlineData("end5_same_plus", false)]
    [InlineData("open_all_rules", false)]
    public async Task Every_level_score_equals_the_oracle_and_the_move_is_in_the_exact_tie_set(string name, bool aggressive)
    {
        var snapshot = Position(name);
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, aggressive, maxDepth: "4"));
        Assert.Equal(200, status);
        Assert.Equal("ok", body["status"]!.GetValue<string>());

        var depths = body["depths"]!.AsArray();
        Assert.True(depths.Count >= 1);
        foreach (var level in depths)
        {
            int depth = level!["depth"]!.GetValue<int>();
            var oracle = Oracle.Exact(GameState.FromSnapshot(snapshot), depth, aggressive);
            Assert.Equal(BitConverter.DoubleToInt64Bits(oracle.Score), Bits(level));
        }

        // final move: hand slot of the request, cell, card id; in the tie set of the last completed depth
        int last = depths[^1]!["depth"]!.GetValue<int>();
        var move = body["move"]!;
        int slot = move["slot"]!.GetValue<int>(), row = move["row"]!.GetValue<int>(), col = move["col"]!.GetValue<int>();
        Assert.Equal(snapshot.Hands[snapshot.Current][slot].Id, move["cardId"]!.GetValue<int>());
        var g = GameState.FromSnapshot(snapshot);
        int inst = g.HandInstance(g.CurrentPlayer, slot); // before RootMoves: make/undo reorders the hand (D-001)
        var tie = Oracle.RootMoves(g, last, aggressive);
        var oracleLast = Oracle.Exact(GameState.FromSnapshot(snapshot), last, aggressive);
        Assert.True(Oracle.InTieSet(tie, oracleLast.Score, inst, row * 3 + col), $"{name}: engine move {slot}@{row},{col} not in the tie set");
        Assert.Equal(body["standard"]!["scoreBits"]!.GetValue<string>(), depths[^1]!["scoreBits"]!.GetValue<string>());
        _output.WriteLine($"{name}: depth {last}, nodes {body["totals"]!["nodes"]}, tie size {Oracle.TieSize(tie, oracleLast.Score)}");
    }

    // ------------------------------------------------------------------ strict validation

    public static IEnumerable<object[]> InvalidMutations() => new[]
    {
        new object[] { "protocol 2", (Action<JsonNode>)(n => n["protocol"] = 2) },
        new object[] { "protocol missing", (Action<JsonNode>)(n => n.AsObject().Remove("protocol")) },
        new object[] { "rules missing", (Action<JsonNode>)(n => n.AsObject().Remove("rules")) },
        new object[] { "unknown rule name", (Action<JsonNode>)(n => n["rules"] = new JsonArray("同数", "闪电")) },
        new object[] { "rules not an array", (Action<JsonNode>)(n => n["rules"] = "同数") },
        new object[] { "cur 2", (Action<JsonNode>)(n => n["state"]!["cur"] = 2) },
        new object[] { "board of 8 cells", (Action<JsonNode>)(n => n["state"]!["board"]!.AsArray().RemoveAt(0)) },
        new object[] { "three hands", (Action<JsonNode>)(n => n["state"]!["hands"]!.AsArray().Add(new JsonArray())) },
        new object[] { "card with 12 values", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]!.AsArray().RemoveAt(12)) },
        new object[] { "unknown owner letter", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![2] = "x") },
        new object[] { "owner is a number", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![2] = 1) },
        new object[] { "base edge 11", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![4] = 11) },
        new object[] { "base edge negative", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![5] = -1) },
        new object[] { "display edge 11", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![8] = 11) },
        new object[] { "edge is a fraction", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![6] = 3.5) },
        new object[] { "unknown card type", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![1] = "魔法") },
        new object[] { "modifier 200", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![3] = 200) },
        new object[] { "canUse 2", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![12] = 2) },
        new object[] { "id negative", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![0] = -5) },
        new object[] { "id too large", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![0] = 2_000_000_000) },
        new object[] { "id is a string", (Action<JsonNode>)(n => n["state"]!["hands"]![0]![0]![0] = "7") },
        new object[] { "six cards in a hand", (Action<JsonNode>)(n => { var h = n["state"]!["hands"]![0]!.AsArray(); h.Add(JsonNode.Parse(h[0]!.ToJsonString())); }) },
        new object[] { "side to move has no cards", (Action<JsonNode>)(n => n["state"]!["hands"]![0] = new JsonArray()) },
        new object[] { "maxDepth is a string", (Action<JsonNode>)(n => n["maxDepth"] = "4") },
        new object[] { "maxTime is a string", (Action<JsonNode>)(n => n["maxTime"] = "10") },
        new object[] { "aggressive is a string", (Action<JsonNode>)(n => n["aggressive"] = "yes") },
        new object[] { "state missing", (Action<JsonNode>)(n => n.AsObject().Remove("state")) },
    };

    [Theory]
    [MemberData(nameof(InvalidMutations))]
    public async Task Invalid_requests_are_rejected_with_a_message(string name, Action<JsonNode> mutate)
    {
        var snapshot = Position("open_none"); // red to move, five cards in each hand
        var node = JsonNode.Parse(EngineJson.Request(snapshot))!;
        mutate(node);
        var (status, body) = await EngineJson.Post(_fx.Client, node.ToJsonString());
        Assert.True(status == 400, $"{name}: HTTP {status}: {body}");
        Assert.Equal("invalid_request", body["status"]!.GetValue<string>());
        Assert.False(string.IsNullOrWhiteSpace(body["message"]!.GetValue<string>()), name);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    [InlineData("[1,2,3]")]
    [InlineData("{\"protocol\":1,")]
    public async Task Malformed_bodies_are_rejected(string text)
    {
        var (status, body) = await EngineJson.Post(_fx.Client, text);
        Assert.Equal(400, status);
        Assert.Equal("invalid_request", body["status"]!.GetValue<string>());
    }

    [Fact]
    public async Task A_body_over_the_limit_is_not_served()
    {
        using var content = new StringContent(new string(' ', 2 * Protocol.MaxBodyBytes) + "{}", System.Text.Encoding.UTF8, "application/json");
        using var response = await _fx.Client.PostAsync("/v1/solve", content);
        Assert.True((int)response.StatusCode is 400 or 413, $"HTTP {(int)response.StatusCode}");
    }

    public static IEnumerable<object[]> InvalidRobust() => new[]
    {
        new object[] { "aiPlayer is not the side to move", (Func<string, string>)(r => r.Replace("\"aiPlayer\":0", "\"aiPlayer\":1")) },
        new object[] { "opponent 2", (Func<string, string>)(r => r.Replace("\"opponent\":1", "\"opponent\":2")) },
        new object[] { "sampleCount 0", (Func<string, string>)(r => r.Replace("\"sampleCount\":16", "\"sampleCount\":0")) },
        new object[] { "generated has the wrong length", (Func<string, string>)(r => r.Replace("\"generated\":[[0,0,0,0,0]", "\"generated\":[[0,0,0,0]")) },
        new object[] { "negative npc id", (Func<string, string>)(r => r.Replace("\"npcIds\":[]", "\"npcIds\":[-3]")) },
        new object[] { "malformed fallback candidates", (Func<string, string>)(r => r.Replace("\"fallbackCandidates\":null", "\"fallbackCandidates\":[[1,2,3]]")) },
    };

    [Theory]
    [MemberData(nameof(InvalidRobust))]
    public async Task Invalid_robust_sections_are_rejected(string name, Func<string, string> mutate)
    {
        var snapshot = Position("end5_same_plus"); // blue to move
        snapshot.Current = 0;
        string robust = EngineJson.Robust(0, 1, new[] { new bool[5], new bool[snapshot.Hands[1].Count] }, 16);
        string request = EngineJson.Request(snapshot, robust: robust);
        string mutated = mutate(request);
        Assert.NotEqual(request, mutated);
        var (status, body) = await EngineJson.Post(_fx.Client, mutated);
        Assert.True(status == 400, $"{name}: HTTP {status}: {body}");
        Assert.Equal("invalid_request", body["status"]!.GetValue<string>());
    }

    // ------------------------------------------------------------------ maxDepth / maxTime like Python

    [Theory]
    [InlineData("0")]
    [InlineData("-3")]
    [InlineData("0.5")]
    public async Task A_max_depth_below_one_searches_nothing(string maxDepth)
    {
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(Position("mid3_ascension"), maxDepth: maxDepth));
        Assert.Equal(200, status);
        Assert.Null(body["move"]);
        Assert.Empty(body["depths"]!.AsArray());
        Assert.Equal(0, body["totals"]!["depthCompleted"]!.GetValue<int>());
    }

    [Fact]
    public async Task A_fractional_max_depth_means_its_floor()
    {
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(Position("mid3_ascension"), maxDepth: "2.7"));
        Assert.Equal(200, status);
        Assert.Equal(2, body["totals"]!["depthCompleted"]!.GetValue<int>());
    }

    [Theory]
    [InlineData("1000")]
    [InlineData("1e30")]
    [InlineData("100")]
    public async Task A_huge_max_depth_is_clamped_and_stops_when_the_game_tree_is_searched(string maxDepth)
    {
        var snapshot = Position("end5_same_plus"); // 4 empty cells
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, maxDepth: maxDepth));
        Assert.Equal(200, status);
        Assert.Equal(4, body["totals"]!["depthCompleted"]!.GetValue<int>()); // Fast stops at depth = empty cells (D-014)
        Assert.NotNull(body["move"]!["cardId"]);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("-1e300")]
    public async Task A_max_time_of_zero_or_less_is_already_over(string maxTime)
    {
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(Position("mid3_ascension"), maxDepth: "10", maxTime: maxTime));
        Assert.Equal(200, status);
        Assert.Null(body["move"]);
        Assert.Empty(body["depths"]!.AsArray());
    }

    [Theory]
    [InlineData("null")]
    [InlineData("1e300")]
    [InlineData("1000000")]
    public async Task No_limit_or_a_huge_limit_searches_to_the_requested_depth(string maxTime)
    {
        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(Position("mid3_ascension"), maxDepth: "3", maxTime: maxTime));
        Assert.Equal(200, status);
        Assert.Equal(3, body["totals"]!["depthCompleted"]!.GetValue<int>());
    }

    // ------------------------------------------------------------------ duplicate card ids (D-004 / D-019)

    [Fact]
    public async Task Duplicate_card_ids_in_a_hand_keep_their_own_slots()
    {
        // two cards with id 7 but different edges, and two None placeholders, in the hand of the side to move
        var s = new StateSnapshot { Current = 0, Rules = RuleSet.Same | RuleSet.Plus };
        s.Board[4] = new CardSnapshot(5, 5, 5, 5, OwnerId.Blue, 40);
        s.Hands[0].Add(new CardSnapshot(9, 2, 4, 6, OwnerId.Red, 7));
        s.Hands[0].Add(new CardSnapshot(1, 8, 3, 3, OwnerId.Red, 7));
        s.Hands[0].Add(new CardSnapshot(0, 0, 0, 0, OwnerId.Red, -1));
        s.Hands[0].Add(new CardSnapshot(6, 6, 2, 2, OwnerId.Red, -1));
        s.Hands[1].Add(new CardSnapshot(4, 4, 4, 4, OwnerId.Blue, 21));
        s.Hands[1].Add(new CardSnapshot(4, 4, 4, 4, OwnerId.Blue, 21));
        s.Hands[1].Add(new CardSnapshot(3, 7, 2, 5, OwnerId.Blue, 23));

        var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(s, maxDepth: "5"));
        Assert.Equal(200, status);
        var depths = body["depths"]!.AsArray();
        foreach (var level in depths)
        {
            var oracle = Oracle.Exact(GameState.FromSnapshot(s), level!["depth"]!.GetValue<int>(), false);
            Assert.Equal(BitConverter.DoubleToInt64Bits(oracle.Score), Bits(level));
        }
        var move = body["move"]!;
        int slot = move["slot"]!.GetValue<int>();
        Assert.Equal(s.Hands[0][slot].Id, move["cardId"]!.GetValue<int>());
        var g = GameState.FromSnapshot(s);
        int inst = g.HandInstance(0, slot); // before RootMoves: make/undo reorders the hand (D-001)
        int last = depths[^1]!["depth"]!.GetValue<int>();
        var tie = Oracle.RootMoves(g, last, false);
        var oracleLast = Oracle.Exact(GameState.FromSnapshot(s), last, false);
        Assert.True(Oracle.InTieSet(tie, oracleLast.Score, inst, move["row"]!.GetValue<int>() * 3 + move["col"]!.GetValue<int>()));
    }

    // ------------------------------------------------------------------ robust layer through the protocol

    private static (StateSnapshot State, bool[][] Generated, int Opponent, int Sample, List<int>? Npc, bool Aggressive, string Label)? Robust(
        System.Text.Json.JsonElement rec)
    {
        var rules = EndgameGolden.Rules(rec);
        var snapshot = EndgameGolden.Snapshot(rec.GetProperty("state"), rules);
        if (snapshot.Hands[snapshot.Current].Count == 0) return null;
        return (snapshot, EndgameGolden.Generated(rec), rec.GetProperty("opp").GetInt32(), rec.GetProperty("sample").GetInt32(),
            EndgameGolden.NpcIds(rec), false, $"robust record {rec.GetProperty("i").GetInt32()}");
    }

    [Fact]
    public async Task The_robust_table_and_the_override_equal_the_in_process_computation()
    {
        int checkedRecords = 0, withUnknown = 0, overridden = 0;
        foreach (var doc in EndgameGolden.Records("robust"))
        {
            using (doc)
            {
                var r = Robust(doc.RootElement);
                if (r is null || checkedRecords >= 60) continue;
                var (snapshot, generated, opp, sample, npc, _, label) = r.Value;
                foreach (bool aggressive in new[] { false, true })
                {
                    string robust = EngineJson.Robust(snapshot.Current, opp, generated, sample, npc);
                    var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, aggressive, maxDepth: "3", robust: robust));
                    Assert.True(status == 200, $"{label}: HTTP {status}: {body}");
                    Assert.Equal("ok", body["status"]!.GetValue<string>());

                    // the same computation, in process, on a fresh request-order state
                    var baseState = GameState.FromSnapshot(snapshot);
                    var build = EndgameScenarios.Build(Db, baseState, opp, generated, sample, npc);
                    Assert.False(build.RequiresFallback, label);
                    var sel = EndgameRobust.Select(Db, baseState, build.Scenarios!, baseState.CurrentPlayer, aggressive);
                    var table = body["robust"]!["candidates"]!.AsArray();
                    Assert.Equal(sel.Scored.Count, table.Count);
                    for (int i = 0; i < table.Count; i++)
                    {
                        Assert.Equal(BitConverter.DoubleToInt64Bits(sel.Scored[i].FinalScore), Convert.ToInt64(table[i]!["finalBits"]!.GetValue<string>(), 16));
                        Assert.Equal(BitConverter.DoubleToInt64Bits(sel.Scored[i].SafetyRatio), Convert.ToInt64(table[i]!["safetyBits"]!.GetValue<string>(), 16));
                        Assert.Equal(baseState.HandSlotOf(baseState.CurrentPlayer, sel.Scored[i].Move.Inst), table[i]!["move"]!["slot"]!.GetValue<int>());
                        Assert.Equal(sel.Scored[i].Move.Cell, table[i]!["move"]!["row"]!.GetValue<int>() * 3 + table[i]!["move"]!["col"]!.GetValue<int>());
                    }
                    Assert.Equal(build.Scenarios!.Count, body["robust"]!["scenarios"]!.GetValue<int>());

                    // the final move: the standard move unless the override block replaces it
                    var standard = body["standard"]!["move"]!;
                    var searchState = GameState.FromSnapshot(snapshot);
                    int sInst = searchState.HandInstance(searchState.CurrentPlayer, standard["slot"]!.GetValue<int>());
                    var standardKey = new MoveKey(searchState.InstanceId(sInst), standard["row"]!.GetValue<int>(), standard["col"]!.GetValue<int>());
                    var items = sel.Scored.Select(s => new OverrideItem(s.Key, s.FinalScore, s.SafetyRatio)).ToList();
                    var decided = EndgameRobust.DecideOverride(standardKey, sel.HasBest ? sel.BestKey : null, items, aggressive)!.Value;
                    var final = body["move"]!;
                    Assert.Equal(decided, new MoveKey(final["cardId"]!.GetValue<int>(), final["row"]!.GetValue<int>(), final["col"]!.GetValue<int>()));
                    Assert.Equal(decided != standardKey, body["robust"]!["overridden"]!.GetValue<bool>());
                    if (decided != standardKey) overridden++;
                    if (generated[opp].Any(x => x)) withUnknown++;
                }
                checkedRecords++;
            }
        }
        _output.WriteLine($"robust records checked={checkedRecords} (x2 modes), with unknown cards={withUnknown}, overridden={overridden}");
        Assert.True(checkedRecords >= 40);
        Assert.True(withUnknown >= 10);
    }

    [Fact]
    public async Task A_blocked_pool_asks_for_fallback_candidates_before_searching_and_accepts_them()
    {
        // find a golden robust position with unknown cards and block every id of the NPC pool
        foreach (var doc in EndgameGolden.Records("robust"))
        {
            using (doc)
            {
                var r = Robust(doc.RootElement);
                if (r is null) continue;
                var (snapshot, generated, opp, sample, _, _, label) = r.Value;
                if (!generated[opp].Any(x => x)) continue;
                var known = new List<int>();
                foreach (var c in snapshot.Board) if (c is not null && c.Id > 0) known.Add(EndgameScenarios.BaseCardId(c.Id));
                for (int p = 0; p < 2; p++) // the placeholders of unknown cards are not "known": their ids stay candidates
                    for (int k = 0; k < snapshot.Hands[p].Count; k++)
                        if (!generated[p][k] && snapshot.Hands[p][k].Id > 0) known.Add(EndgameScenarios.BaseCardId(snapshot.Hands[p][k].Id));
                var blocked = known.Distinct().ToList();
                if (blocked.Count == 0) continue;

                string robust = EngineJson.Robust(snapshot.Current, opp, generated, sample, blocked);
                var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, maxDepth: "3", robust: robust));
                Assert.Equal(200, status);
                Assert.Equal("needs_fallback_candidates", body["status"]!.GetValue<string>());
                Assert.Equal(generated[opp].Count(x => x), body["unknownCount"]!.GetValue<int>());
                Assert.Null(body["move"]); // nothing was searched

                // three candidate cards that are not on the table
                var present = new HashSet<int>(known);
                var cards = Db.Cards.Where(d => !present.Contains(d.Id) && d.Up + d.Right + d.Down + d.Left > 0).Take(3)
                    .Select(d => new CardSnapshot(d.Up, d.Right, d.Down, d.Left, OwnerId.None, d.Id, CardTypes.NameOf(d.TypeId))).ToList();
                string fallback = "[" + string.Join(",", cards.Select(EngineJson.Card)) + "]";
                string withCandidates = EngineJson.Robust(snapshot.Current, opp, generated, sample, blocked, fallback: fallback);
                (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, maxDepth: "3", robust: withCandidates));
                Assert.True(status == 200, $"{label}: {body}");
                Assert.Equal("ok", body["status"]!.GetValue<string>());

                // identical to the in-process computation with the same candidates
                var baseState = GameState.FromSnapshot(snapshot);
                var candidates = cards.Select(c => new UnknownCandidate(c.Id, c.Base[0], c.Base[1], c.Base[2], c.Base[3], CardTypes.IdOf(c.Type))).ToList();
                var build = EndgameScenarios.Build(Db, baseState, opp, generated, sample, blocked, candidates);
                Assert.False(build.RequiresFallback);
                Assert.Equal(build.Scenarios!.Count, body["robust"]!["scenarios"]!.GetValue<int>());
                var sel = EndgameRobust.Select(Db, baseState, build.Scenarios, baseState.CurrentPlayer, false);
                var table = body["robust"]!["candidates"]!.AsArray();
                Assert.Equal(sel.Scored.Count, table.Count);
                for (int i = 0; i < table.Count; i++)
                    Assert.Equal(BitConverter.DoubleToInt64Bits(sel.Scored[i].FinalScore), Convert.ToInt64(table[i]!["finalBits"]!.GetValue<string>(), 16));

                // an empty list from the handler is valid too: Python falls back to the base state
                string empty = EngineJson.Robust(snapshot.Current, opp, generated, sample, blocked, fallback: "[]");
                (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, maxDepth: "3", robust: empty));
                Assert.Equal(200, status);
                Assert.Equal("ok", body["status"]!.GetValue<string>());
                Assert.Equal(1, body["robust"]!["scenarios"]!.GetValue<int>());
                return;
            }
        }
        Assert.Fail("no golden robust record with unknown cards found");
    }

    [Fact]
    public async Task Without_a_handler_the_base_state_is_the_only_scenario()
    {
        foreach (var doc in EndgameGolden.Records("robust"))
        {
            using (doc)
            {
                var r = Robust(doc.RootElement);
                if (r is null) continue;
                var (snapshot, generated, opp, sample, npc, _, _) = r.Value;
                if (!generated[opp].Any(x => x)) continue;
                string robust = EngineJson.Robust(snapshot.Current, opp, generated, sample, npc, handler: false);
                var (status, body) = await EngineJson.Post(_fx.Client, EngineJson.Request(snapshot, maxDepth: "3", robust: robust));
                Assert.Equal(200, status);
                Assert.Equal(1, body["robust"]!["scenarios"]!.GetValue<int>());
                return;
            }
        }
        Assert.Fail("no golden robust record with unknown cards found");
    }

    // ------------------------------------------------------------------ options and the review findings folded into Phase 7

    [Fact]
    public void The_engine_refuses_a_non_loopback_address_unless_explicitly_allowed()
    {
        Assert.Throws<ArgumentException>(() => EngineOptions.Parse(new[] { "--host", "0.0.0.0" }));
        Assert.Throws<ArgumentException>(() => EngineOptions.Parse(new[] { "--host", "192.168.1.5", "--port", "5300" }));
        Assert.Throws<ArgumentException>(() => EngineOptions.Parse(new[] { "--host", "not-an-address" }));
        var allowed = EngineOptions.Parse(new[] { "--host", "0.0.0.0", "--allow-remote", "--workers", "3", "--port", "5400" });
        Assert.Equal("0.0.0.0", allowed.Host);
        Assert.Equal(3, allowed.Workers);
        Assert.Equal(5400, allowed.Port);
        var defaults = EngineOptions.Parse(Array.Empty<string>());
        Assert.Equal("127.0.0.1", defaults.Host);
        Assert.Equal(EngineOptions.DefaultPort, defaults.Port);
        Assert.Throws<ArgumentException>(() => EngineOptions.Parse(new[] { "--workers", "0" }));
        Assert.Throws<ArgumentException>(() => EngineOptions.Parse(new[] { "--tt-entries", "1000" }));
    }
}
