using System.IO.Compression;
using System.Numerics;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Helpers for the Phase 5 golden data (tools/parity/py_endgame_trace.py).</summary>
internal static class EndgameGolden
{
    public static IEnumerable<JsonDocument> Records(string kind)
    {
        string path = GoldenFiles.EnsureEndgameGolden(kind);
        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        using (var header = JsonDocument.Parse(reader.ReadLine()!))
        {
            var h = header.RootElement.GetProperty("header");
            Assert.Equal(GoldenFiles.EndgameGeneratorVersion, h.GetProperty("generator").GetString());
            Assert.Equal(kind, h.GetProperty("kind").GetString());
        }
        string? line;
        while ((line = reader.ReadLine()) is not null) yield return JsonDocument.Parse(line);
    }

    public static RuleSet Rules(JsonElement rec) =>
        RuleParser.FromPythonList(rec.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));

    public static StateSnapshot Snapshot(JsonElement state, RuleSet rules) => StateSnapshot.FromJson(state, rules);

    public static GameState State(JsonElement rec, string property = "state") =>
        GameState.FromSnapshot(StateSnapshot.FromJson(rec.GetProperty(property), Rules(rec)));

    public static bool[][] Generated(JsonElement rec) =>
        rec.GetProperty("gen").EnumerateArray().Select(h => h.EnumerateArray().Select(x => x.GetInt32() != 0).ToArray()).ToArray();

    public static List<int>? NpcIds(JsonElement rec) =>
        rec.GetProperty("npc").ValueKind == JsonValueKind.Null ? null : rec.GetProperty("npc").EnumerateArray().Select(x => x.GetInt32()).ToList();

    public static int Occupied(GameState g) => BitOperations.PopCount(g.OccupiedMask);

    public static void AssertBits(string expectedHex, double actual, string context) =>
        SearchGolden.AssertSameDouble(expectedHex, actual, context);

    public static MoveKey? Key(JsonElement e) =>
        e.ValueKind == JsonValueKind.Null ? null : new MoveKey(e[0].ValueKind == JsonValueKind.Null ? -1 : e[0].GetInt32(), e[1].GetInt32(), e[2].GetInt32());
}

/// <summary>Phase 5 differential tests against Python's ai_server (CSHARP_MIGRATION.md section 7, Phase 5).</summary>
public sealed class EndgameDifferentialTests
{
    private readonly ITestOutputHelper _output;

    public EndgameDifferentialTests(ITestOutputHelper output) => _output = output;

    private static CardDb Db => SearchGolden.Db;

    // ------------------------------------------------------------------ scenario construction

    [Fact]
    public void Legal_candidates_threat_scores_and_scenarios_match_python_in_order()
    {
        int records = 0, withUnknown = 0, scenariosTotal = 0, scoredCandidates = 0, truncated = 0, withNpc = 0, sameObject = 0;
        var unknownHistogram = new int[5];
        var scenarioCounts = new SortedDictionary<int, int>();

        foreach (var doc in EndgameGolden.Records("scenario"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var rules = EndgameGolden.Rules(rec);
                var g = EndgameGolden.State(rec);
                var gen = EndgameGolden.Generated(rec);
                int opp = rec.GetProperty("opp").GetInt32();
                var npc = EndgameGolden.NpcIds(rec);
                int sample = rec.GetProperty("sample").GetInt32();
                int unknown = rec.GetProperty("unknown").GetInt32();
                string ctx = $"scenario record {rec.GetProperty("i").GetInt32()} (golden seed {GoldenFiles.EndgameSeed})";

                Assert.Equal(unknown, EndgameScenarios.UnknownSlots(gen[opp]).Count);

                var legal = EndgameScenarios.LegalCandidates(Db, g, opp, gen, npc);
                var expectedLegal = rec.GetProperty("legal").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                Assert.True(expectedLegal.SequenceEqual(legal.Select(c => c.Id)),
                    $"{ctx}: legal candidate ids differ\n  python: [{string.Join(",", expectedLegal.Take(40))}] ({expectedLegal.Length})\n  c#:     [{string.Join(",", legal.Select(c => c.Id).Take(40))}] ({legal.Count})");

                if (rec.TryGetProperty("scores", out var scores))
                {
                    foreach (var pair in scores.EnumerateArray())
                    {
                        var def = Db.Get(pair[0].GetInt32());
                        var candidate = new UnknownCandidate(def.Id, def.Up, def.Right, def.Down, def.Left, def.TypeId);
                        EndgameGolden.AssertBits(pair[1].GetString()!, EndgameScenarios.ThreatScore(candidate, g, rules, (byte)opp),
                            $"{ctx} threat score of card {def.Id}");
                        scoredCandidates++;
                    }
                }

                var build = EndgameScenarios.Build(Db, g, opp, gen, sample, npc);
                Assert.False(build.RequiresFallback, ctx);
                var expected = rec.GetProperty("scen").EnumerateArray().ToList();
                Assert.True(expected.Count == build.Scenarios!.Count, $"{ctx}: {expected.Count} scenarios in Python, {build.Scenarios.Count} in C#");
                for (int i = 0; i < expected.Count; i++)
                {
                    string py = EndgameGolden.Snapshot(expected[i], rules).ToCanonicalString();
                    string cs = build.Scenarios[i].ToSnapshot().ToCanonicalString();
                    Assert.True(py == cs, $"{ctx}: scenario {i} differs\n  python: {py}\n  c#:     {cs}");
                }
                if (rec.GetProperty("same_object").GetBoolean())
                {
                    Assert.True(ReferenceEquals(build.Scenarios[0], g), $"{ctx}: Python returns the base state itself");
                    sameObject++;
                }

                records++;
                if (unknown > 0) withUnknown++;
                unknownHistogram[Math.Min(unknown, 4)]++;
                scenariosTotal += expected.Count;
                scenarioCounts[expected.Count] = scenarioCounts.GetValueOrDefault(expected.Count) + 1;
                if (unknown > 0 && expected.Count == Math.Max(1, sample)) truncated++;
                if (npc is not null) withNpc++;
            }
        }

        _output.WriteLine($"records={records} (with unknown cards {withUnknown}; unknown 0/1/2/3/4+: {string.Join("/", unknownHistogram)}), scenarios compared={scenariosTotal}, " +
                          $"threat scores compared={scoredCandidates}, hit the scenario limit={truncated}, NPC pool restricted={withNpc}, base state returned={sameObject}; " +
                          $"scenario counts: {string.Join(", ", scenarioCounts.Select(kv => $"{kv.Key}:{kv.Value}"))}");
        Assert.True(records >= 500);
        Assert.True(withUnknown >= 300);
        Assert.True(unknownHistogram[1] >= 50 && unknownHistogram[2] >= 50 && unknownHistogram[3] >= 20, "unknown-card counts insufficiently covered");
        Assert.True(truncated >= 100, "scenario truncation insufficiently covered");
        Assert.True(withNpc >= 50);
        Assert.True(scoredCandidates >= 5_000);
    }

    [Fact]
    public void Unique_candidates_match_python()
    {
        int records = 0, removed = 0;
        foreach (var doc in EndgameGolden.Records("unique"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                static UnknownCandidate Parse(JsonElement a) => new(
                    a[0].ValueKind == JsonValueKind.Null ? -1 : a[0].GetInt32(),
                    (byte)a[2].GetInt32(), (byte)a[3].GetInt32(), (byte)a[4].GetInt32(), (byte)a[5].GetInt32(),
                    CardTypes.IdOf(a[1].ValueKind == JsonValueKind.Null ? null : a[1].GetString()));
                var input = rec.GetProperty("in").EnumerateArray().Select(Parse).ToList();
                var expected = rec.GetProperty("out").EnumerateArray().Select(Parse).ToList();
                var actual = EndgameScenarios.Unique(input);
                Assert.True(expected.SequenceEqual(actual), $"unique record {rec.GetProperty("i").GetInt32()}: order or content differs");
                removed += input.Count - actual.Count;
                records++;
            }
        }
        _output.WriteLine($"records={records}, duplicates removed={removed}");
        Assert.True(records >= 800);
        Assert.True(removed >= 500, "deduplication insufficiently covered");
    }

    // ------------------------------------------------------------------ corner risk, reply score, exact solver

    [Fact]
    public void Corner_safety_risk_matches_python_bit_for_bit()
    {
        int records = 0, checks = 0, nonZero = 0;
        var occupancies = new int[10];
        foreach (var doc in EndgameGolden.Records("corner"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var g = EndgameGolden.State(rec);
                occupancies[EndgameGolden.Occupied(g)]++;
                foreach (var check in rec.GetProperty("checks").EnumerateArray())
                {
                    var move = g.MoveFromSlot(check[0].GetInt32(), check[1].GetInt32());
                    double risk = EndgameRobust.CornerSafetyRisk(g, move);
                    EndgameGolden.AssertBits(check[2].GetString()!, risk, $"corner record {rec.GetProperty("i").GetInt32()} slot {check[0]} cell {check[1]}");
                    checks++;
                    if (risk != 0.0) nonZero++;
                }
                records++;
            }
        }
        _output.WriteLine($"records={records}, checks={checks} (non-zero {nonZero}), occupancy 0..9: {string.Join("/", occupancies)}");
        Assert.True(checks >= 3_000);
        Assert.True(nonZero >= 1_000);
        Assert.True(occupancies[5..9].All(c => c > 50), "occupancy 5..8 insufficiently covered");
    }

    [Fact]
    public void Best_immediate_reply_score_matches_python_bit_for_bit()
    {
        int records = 0, checks = 0;
        foreach (var doc in EndgameGolden.Records("reply"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var expected = rec.GetProperty("reply");
                for (int ai = 0; ai < 2; ai++)
                {
                    var g = EndgameGolden.State(rec);
                    var evaluator = new Evaluator(g, Db);
                    double value = EndgameRobust.BestImmediateReplyScore(g, evaluator, ai);
                    EndgameGolden.AssertBits(expected[ai].GetString()!, value, $"reply record {rec.GetProperty("i").GetInt32()} ai={ai}");
                    checks++;
                }
                records++;
            }
        }
        _output.WriteLine($"records={records}, checks={checks}");
        Assert.True(checks >= 3_000);
    }

    [Fact]
    public void Exact_endgame_values_match_python_and_pythons_shared_memo_is_sound()
    {
        int records = 0, rootValues = 0, moveValues = 0, sharedDiffers = 0, memoHits = 0;
        long nodes = 0, makes = 0;
        var emptiesSeen = new int[10];
        var valueRange = new SortedSet<int>();

        foreach (var doc in EndgameGolden.Records("exact"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var probe = EndgameGolden.State(rec);
                emptiesSeen[probe.AvailableCellCount]++;
                foreach (var item in rec.GetProperty("ai").EnumerateArray())
                {
                    int ai = item.GetProperty("ai").GetInt32();
                    string ctx = $"exact record {rec.GetProperty("i").GetInt32()} ai={ai}";

                    var g = EndgameGolden.State(rec);
                    var moves = item.GetProperty("moves").EnumerateArray()
                        .Select(m => (Move: g.MoveFromSlot(m[0].GetInt32() / 9, m[0].GetInt32() % 9), Fresh: m[1].GetInt32(), Shared: m[2].GetInt32())).ToList();
                    var memo = ExactEndgameSolver.CreateMemo();
                    var solver = new ExactEndgameSolver(g, memo);
                    solver.Bind(ai);

                    int root = solver.Value(ai);
                    Assert.True(item.GetProperty("value").GetInt32() == root, $"{ctx}: root value Python {item.GetProperty("value").GetInt32()}, C# {root}");
                    valueRange.Add(root);
                    rootValues++;

                    // One memo for all first moves of the position, as _evaluate_endgame_move_robustly shares its cache.
                    foreach (var (move, fresh, shared) in moves)
                    {
                        g.MakeMove(move);
                        int value = solver.Value(ai);
                        g.UndoMove();
                        Assert.True(fresh == value, $"{ctx} first move {move}: Python (fresh cache) {fresh}, C# {value}");
                        if (shared != fresh) sharedDiffers++; // Python's shared cache returned a different value: a Python memo bug
                        moveValues++;
                    }
                    nodes += solver.Nodes;
                    makes += solver.MakeMoves;
                    memoHits += (int)solver.MemoHits;
                }
                records++;
            }
        }

        _output.WriteLine($"records={records}, root values={rootValues}, first-move values={moveValues}, Python shared-cache values that differ from fresh-cache values={sharedDiffers}, " +
                          $"C# nodes={nodes:N0}, make_move calls={makes:N0}, memo hits={memoHits:N0}, empty cells 0..9: {string.Join("/", emptiesSeen)}, root values seen: {valueRange.Min}..{valueRange.Max}");
        Assert.Equal(0, sharedDiffers);
        Assert.True(rootValues >= 1_000);
        Assert.True(moveValues >= 4_000);
        Assert.True(emptiesSeen[0] > 10 && emptiesSeen[1] > 10 && emptiesSeen[4] > 10, "end-of-game and 4-empty positions insufficiently covered");
        Assert.Equal(0, emptiesSeen[5..].Sum()); // the generator stays at five or more occupied cells (Python needs seconds per record below that)
        Assert.True(valueRange.Count >= 8, "exact values insufficiently varied");
    }

    // ------------------------------------------------------------------ robust selection

    private sealed class RobustCoverage
    {
        public int Records, Moves, Exact, Heuristic, Aggressive, NegInf, SortedByAlias, SafePath;
        public int[] Unknown = new int[4];
        public int ScenarioMin = int.MaxValue, ScenarioMax;
    }

    private static void CompareSelection(JsonElement mode, RobustSelection actual, GameState g, string ctx, RobustCoverage cov, bool exactPath)
    {
        var expected = mode.GetProperty("scored").EnumerateArray().ToList();
        Assert.True(expected.Count == actual.Scored.Count, $"{ctx}: Python scored {expected.Count} moves, C# {actual.Scored.Count}");
        for (int i = 0; i < expected.Count; i++)
        {
            var e = expected[i];
            var a = actual.Scored[i];
            int code = e[0].GetInt32();
            int actualCode = g.HandSlotOf(g.CurrentPlayer, a.Move.Inst) * 9 + a.Move.Cell;
            Assert.True(code == actualCode, $"{ctx}: returned order differs at position {i}: Python move code {code}, C# {actualCode}");
            EndgameGolden.AssertBits(e[1].GetString()!, a.FinalScore, $"{ctx} move {code} final score");
            EndgameGolden.AssertBits(e[2].GetString()!, a.SafetyRatio, $"{ctx} move {code} safety ratio");
            EndgameGolden.AssertBits(e[3].GetString()!, a.RobustScore, $"{ctx} move {code} robust score");
            EndgameGolden.AssertBits(e[4].GetString()!, a.CornerRisk, $"{ctx} move {code} corner risk");
            cov.Moves++;
            if (double.IsNegativeInfinity(a.FinalScore)) cov.NegInf++;
            // PY-QUIRK (unreachable code): the corner early return needs "not exact" (fewer than 5 occupied cells) AND a
            // corner risk >= 5.0, but the risk is 0.0 below 5 occupied cells. Both read the same board, so the branch can
            // never run; it is ported for fidelity and this invariant documents why no golden record can reach it.
            if (!exactPath) Assert.True(a.CornerRisk == 0.0, $"{ctx}: corner risk {a.CornerRisk} below five occupied cells");
        }

        if (mode.GetProperty("best").ValueKind == JsonValueKind.Null)
        {
            Assert.False(actual.HasBest, ctx);
        }
        else
        {
            Assert.True(actual.HasBest, ctx);
            int best = mode.GetProperty("best").GetInt32();
            int actualBest = g.HandSlotOf(g.CurrentPlayer, actual.Best.Inst) * 9 + actual.Best.Cell;
            Assert.True(best == actualBest, $"{ctx}: selected move Python {best}, C# {actualBest}");
        }
    }

    private void RunRobust(bool pythonScenarios)
    {
        var cov = new RobustCoverage();
        long exactNodes = 0;
        foreach (var doc in EndgameGolden.Records("robust"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var rules = EndgameGolden.Rules(rec);
                var g = EndgameGolden.State(rec);
                var gen = EndgameGolden.Generated(rec);
                int opp = rec.GetProperty("opp").GetInt32();
                int unknown = rec.GetProperty("unknown").GetInt32();
                int occupied = rec.GetProperty("occupancy").GetInt32();
                string ctx = $"robust record {rec.GetProperty("i").GetInt32()} (golden seed {GoldenFiles.EndgameSeed}, {(pythonScenarios ? "Python" : "C#")} scenarios)";

                Assert.Equal(occupied, EndgameGolden.Occupied(g));
                Assert.Equal(rec.GetProperty("use_robust").GetBoolean(), EndgameRobust.ShouldUse(occupied, unknown));

                List<GameState> scenarios;
                if (pythonScenarios)
                {
                    scenarios = rec.GetProperty("scen").EnumerateArray().Select(s => GameState.FromSnapshot(EndgameGolden.Snapshot(s, rules))).ToList();
                }
                else
                {
                    var build = EndgameScenarios.Build(Db, g, opp, gen, rec.GetProperty("sample").GetInt32(), EndgameGolden.NpcIds(rec));
                    Assert.False(build.RequiresFallback, ctx);
                    scenarios = build.Scenarios!;
                    Assert.Equal(rec.GetProperty("scen").GetArrayLength(), scenarios.Count);
                }

                string before = g.ToSnapshot().ToCanonicalString();
                foreach (var mode in rec.GetProperty("modes").EnumerateArray())
                {
                    bool aggressive = mode.GetProperty("aggr").GetBoolean();
                    var actual = EndgameRobust.Select(Db, g, scenarios, g.CurrentPlayer, aggressive);
                    CompareSelection(mode, actual, g, $"{ctx} aggressive={aggressive}", cov, occupied >= 5);
                    Assert.True(before == g.ToSnapshot().ToCanonicalString(), $"{ctx}: Select modified the base state");
                    exactNodes += actual.ExactNodes;
                    if (aggressive) cov.Aggressive++;
                    if (occupied >= 5) cov.Exact++; else cov.Heuristic++;
                    if (!aggressive && actual.Scored.Count > 0)
                    {
                        // the non-aggressive path reorders scored_moves only when the ranking aliases it
                        if (actual.Scored.Any(s => s.SafetyRatio >= 1.0) == false) cov.SortedByAlias++;
                        else cov.SafePath++;
                    }
                }
                cov.Records++;
                cov.Unknown[Math.Min(unknown, 3)]++;
                cov.ScenarioMin = Math.Min(cov.ScenarioMin, scenarios.Count);
                cov.ScenarioMax = Math.Max(cov.ScenarioMax, scenarios.Count);
            }
        }

        _output.WriteLine($"[{(pythonScenarios ? "Python" : "C#")} scenarios] records={cov.Records}, move score sets compared={cov.Moves}, exact-solver selections={cov.Exact}, heuristic selections={cov.Heuristic}, " +
                          $"aggressive selections={cov.Aggressive}, -inf scores={cov.NegInf}, non-aggressive with a fully safe move={cov.SafePath}, without={cov.SortedByAlias}, " +
                          $"unknown 0/1/2/3+: {string.Join("/", cov.Unknown)}, scenarios per record {cov.ScenarioMin}..{cov.ScenarioMax}, exact solver nodes={exactNodes:N0}");
        Assert.True(cov.Records >= 400);
        Assert.True(cov.Exact >= 300 && cov.Heuristic >= 100, "exact and heuristic paths insufficiently covered");
        Assert.True(cov.Unknown[0] >= 30 && cov.Unknown[1] >= 50 && cov.Unknown[2] >= 50, "unknown-card counts insufficiently covered");
        Assert.True(cov.SafePath >= 50 && cov.SortedByAlias >= 5, "ranking branches insufficiently covered");
    }

    [Fact]
    public void Robust_selection_matches_python_on_python_built_scenarios() => RunRobust(pythonScenarios: true);

    [Fact]
    public void Robust_selection_matches_python_on_csharp_built_scenarios() => RunRobust(pythonScenarios: false);

    // ------------------------------------------------------------------ override logic

    [Fact]
    public void Endgame_override_logic_matches_the_block_extracted_from_ai_move()
    {
        int records = 0, overrides = 0, kept = 0, missing = 0, aggressiveOverrides = 0, standardOnly = 0;
        foreach (var doc in EndgameGolden.Records("override"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                var standard = EndgameGolden.Key(rec.GetProperty("standard"));
                var robust = EndgameGolden.Key(rec.GetProperty("robust"));
                var items = rec.GetProperty("items").EnumerateArray().Select(it => new OverrideItem(
                    new MoveKey(it[0].ValueKind == JsonValueKind.Null ? -1 : it[0].GetInt32(), it[1].GetInt32(), it[2].GetInt32()),
                    BitConverter.Int64BitsToDouble(SearchGolden.HexBits(it[3].GetString()!)),
                    BitConverter.Int64BitsToDouble(SearchGolden.HexBits(it[4].GetString()!)))).ToList();
                var expected = EndgameGolden.Key(rec.GetProperty("chosen"));

                var actual = EndgameRobust.DecideOverride(standard, robust, items, aggressive);
                Assert.True(expected == actual, $"override record {rec.GetProperty("i").GetInt32()} aggressive={aggressive}: Python chose {expected}, C# {actual}");

                records++;
                if (standard is null) standardOnly++;
                else if (robust is not null && actual == robust && robust != standard) { overrides++; if (aggressive) aggressiveOverrides++; }
                else kept++;
                if (standard is not null && robust is not null && (!items.Any(i => i.Key == standard) || !items.Any(i => i.Key == robust))) missing++;
            }
        }
        _output.WriteLine($"records={records}, overridden={overrides} (aggressive {aggressiveOverrides}), kept={kept}, no standard move={standardOnly}, move missing from the table={missing}");
        Assert.True(records >= 6_000);
        Assert.True(overrides >= 500 && aggressiveOverrides >= 30 && kept >= 1_000, "override branches insufficiently covered");
        Assert.True(missing >= 100);
    }
}
