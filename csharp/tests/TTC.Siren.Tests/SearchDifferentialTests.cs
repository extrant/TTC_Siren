using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Phase 3 differential tests against Python's ai/ai.py (CSHARP_MIGRATION.md section 7, Phase 3).</summary>
public sealed class SearchDifferentialTests
{
    private readonly ITestOutputHelper _output;

    public SearchDifferentialTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Evaluate_state_and_evaluate_move_match_python_bit_for_bit()
    {
        string path = GoldenFiles.EnsureSearchGolden("eval");
        int positions = 0, stateChecks = 0, moveChecks = 0, withHistory = 0;
        var phases = new int[3];

        foreach (var doc in SearchGolden.Records(path, "eval"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var g = SearchGolden.BuildState(rec, out string label);
                var eval = new Evaluator(g, SearchGolden.Db);
                var history = new HistoryTable(g);
                foreach (var h in rec.GetProperty("hist").EnumerateArray())
                    history.SetById(h[0].ValueKind == JsonValueKind.Null ? -1 : h[0].GetInt32(), h[1].GetInt32(), h[2].GetInt32(), h[3].GetInt32());
                if (rec.GetProperty("hist").GetArrayLength() > 0) withHistory++;

                int occupied = System.Numerics.BitOperations.PopCount(g.OccupiedMask);
                phases[occupied <= 2 ? 0 : occupied <= 5 ? 1 : 2]++;

                var es = rec.GetProperty("es");
                for (int ai = 0; ai < 2; ai++)
                {
                    SearchGolden.AssertSameDouble(es[ai][0].GetString()!, eval.EvaluateState(ai, false), $"{label} evaluate_state ai={ai} normal");
                    SearchGolden.AssertSameDouble(es[ai][1].GetString()!, eval.EvaluateState(ai, true), $"{label} evaluate_state ai={ai} aggressive");
                    stateChecks += 2;
                }

                foreach (var em in rec.GetProperty("em").EnumerateArray())
                {
                    int code = em[0].GetInt32();
                    var move = g.MoveFromSlot(code / 9, code % 9); // slot resolved against the hand of this exact position
                    SearchGolden.AssertSameDouble(em[1].GetString()!, eval.EvaluateMove(move, history, false), $"{label} evaluate_move {move} normal");
                    SearchGolden.AssertSameDouble(em[2].GetString()!, eval.EvaluateMove(move, history, true), $"{label} evaluate_move {move} aggressive");
                    moveChecks += 2;
                }
                positions++;
            }
        }

        _output.WriteLine($"positions={positions} evaluate_state checks={stateChecks} evaluate_move checks={moveChecks} " +
                          $"positions with history={withHistory} occupancy phases (<=2, 3..5, >=6)={phases[0]}/{phases[1]}/{phases[2]}");
        Assert.True(positions >= 100_000, $"only {positions} positions");
        Assert.True(phases.All(p => p > 5_000), "aggressive phases insufficiently covered");
    }

    [Fact]
    public void Fixed_depth_minimax_matches_python_node_for_node()
    {
        string path = GoldenFiles.EnsureSearchGolden("search");
        var tt = new CompatTranspositionTable();
        int positions = 0, runs = 0, benchItems = 0, aggressiveRuns = 0, endgameToTerminal = 0;
        long nodes = 0;
        var depthSeen = new int[8];

        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                bool isBench = rec.TryGetProperty("name", out _);
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    var g = SearchGolden.BuildState(rec, out string label);
                    int depth = run.GetProperty("d").GetInt32();
                    int empties = 9 - System.Numerics.BitOperations.PopCount(g.OccupiedMask);
                    string ctx = $"{label} depth={depth} aggressive={aggressive}";

                    var searcher = new CompatSearcher(g, SearchGolden.Db, tt);
                    string stateBefore = SearchGolden.NormalizedState(g);
                    var result = searcher.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    Assert.True(stateBefore == SearchGolden.NormalizedState(g), $"{ctx}: search did not restore the state (beyond D-001 hand order / D-003 display)");
                    Assert.Equal(0, g.UndoDepth);

                    SearchGolden.AssertSameDouble(run.GetProperty("score").GetString()!, result.Score, ctx + " root score");
                    Assert.Equal(SearchGolden.MoveText(run.GetProperty("move")), SearchGolden.MoveText(g, result.BestInstance, result.BestCell));
                    var s = searcher.Stats;
                    Assert.True(run.GetProperty("nodes").GetInt64() == s.NodesSearched, $"{ctx} nodes: expected {run.GetProperty("nodes").GetInt64()} got {s.NodesSearched}");
                    Assert.True(run.GetProperty("cut").GetInt64() == s.AlphaBetaCutoffs, $"{ctx} cutoffs: expected {run.GetProperty("cut").GetInt64()} got {s.AlphaBetaCutoffs}");
                    Assert.True(run.GetProperty("tt").GetInt64() == s.TtHits, $"{ctx} tt_hits: expected {run.GetProperty("tt").GetInt64()} got {s.TtHits}");
                    Assert.True(run.GetProperty("ttc").GetInt64() == s.TtCutoffs, $"{ctx} tt_cutoffs: expected {run.GetProperty("ttc").GetInt64()} got {s.TtCutoffs}");
                    Assert.True(run.GetProperty("me").GetInt64() == s.MoveEvaluations, $"{ctx} move_evaluations: expected {run.GetProperty("me").GetInt64()} got {s.MoveEvaluations}");
                    Assert.Equal(SearchGolden.HistoryText(run.GetProperty("hist")), SearchGolden.HistoryText(searcher.History));

                    runs++;
                    nodes += s.NodesSearched;
                    if (aggressive) aggressiveRuns++;
                    if (depth <= 7) depthSeen[depth]++;
                    if (depth >= empties) endgameToTerminal++;
                }
                if (isBench) benchItems += rec.GetProperty("runs").GetArrayLength(); else positions++;
            }
        }

        _output.WriteLine($"positions={positions} runs={runs} (aggressive {aggressiveRuns}, searched to terminal {endgameToTerminal}) " +
                          $"nodes={nodes:N0} benchmark items={benchItems} runs per depth 1..5={string.Join('/', depthSeen[1..6])}");
        Assert.True(positions >= 500, $"only {positions} positions");
        Assert.Equal(22, benchItems); // section 3.4 of CSHARP_MIGRATION.md
        for (int d = 1; d <= 5; d++) Assert.True(depthSeen[d] >= 500, $"depth {d}: only {depthSeen[d]} runs");
        Assert.True(endgameToTerminal >= 300);
    }

    [Fact]
    public void Iterative_deepening_and_time_up_control_flow_match_python()
    {
        string path = GoldenFiles.EnsureSearchGolden("id");
        var tt = new CompatTranspositionTable();
        int positions = 0, unlimitedRuns = 0, budgetRuns = 0, directBudget = 0, discardedLevels = 0, rootNone = 0;

        foreach (var doc in SearchGolden.Records(path, "id"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    var g = SearchGolden.BuildState(rec, out string label);
                    int maxDepth = run.GetProperty("maxd").GetInt32();
                    bool hasBudget = run.GetProperty("budget").ValueKind != JsonValueKind.Null;
                    var limits = hasBudget ? SearchLimits.WithNodeBudget(run.GetProperty("budget").GetInt64()) : SearchLimits.Unlimited;
                    string ctx = $"{label} ID maxDepth={maxDepth} budget={(hasBudget ? run.GetProperty("budget").GetInt64().ToString() : "none")} aggressive={aggressive}";

                    var searcher = new CompatSearcher(g, SearchGolden.Db, tt);
                    string stateBefore = SearchGolden.NormalizedState(g);
                    var result = searcher.RunIterativeDeepening(maxDepth, aggressive, limits);
                    Assert.True(stateBefore == SearchGolden.NormalizedState(g), $"{ctx}: iterative deepening did not restore the state");
                    Assert.Equal(0, g.UndoDepth);
                    var s = searcher.Stats;

                    var npd = run.GetProperty("npd");
                    string levelsText = $"expected levels [{string.Join(",", npd.EnumerateArray().Select(x => x.GetInt64()))}] got [{string.Join(",", Enumerable.Range(0, s.Levels).Select(l => s.NodesPerDepth[l]))}]";
                    Assert.True(npd.GetArrayLength() == s.Levels, $"{ctx} completed levels: {levelsText}");
                    for (int level = 0; level < s.Levels; level++)
                    {
                        Assert.True(npd[level].GetInt64() == s.NodesPerDepth[level], $"{ctx} nodes at level {level + 1}: {levelsText}");
                        SearchGolden.AssertSameDouble(run.GetProperty("scores")[level].GetString()!, s.BestScoreHistory[level], $"{ctx} score at level {level + 1}");
                        Assert.Equal(SearchGolden.MoveText(run.GetProperty("moves")[level]),
                                     SearchGolden.MoveText(g, s.BestInstanceHistory[level], s.BestCellHistory[level]));
                    }
                    Assert.Equal(SearchGolden.MoveText(run.GetProperty("move")), SearchGolden.MoveText(g, result.BestInstance, result.BestCell));
                    Assert.True(run.GetProperty("nodes").GetInt64() == s.NodesSearched, $"{ctx} nodes: expected {run.GetProperty("nodes").GetInt64()} got {s.NodesSearched}");
                    Assert.True(run.GetProperty("cut").GetInt64() == s.AlphaBetaCutoffs, $"{ctx} cutoffs");
                    Assert.True(run.GetProperty("tt").GetInt64() == s.TtHits, $"{ctx} tt_hits");
                    Assert.True(run.GetProperty("ttc").GetInt64() == s.TtCutoffs, $"{ctx} tt_cutoffs");
                    Assert.True(run.GetProperty("me").GetInt64() == s.MoveEvaluations, $"{ctx} move_evaluations");
                    Assert.Equal(run.GetProperty("done").GetInt32(), s.DepthCompleted);

                    if (hasBudget) budgetRuns++; else unlimitedRuns++;
                    if (s.Levels < maxDepth && hasBudget) discardedLevels++;
                    if (!result.HasMove) rootNone++;
                }

                if (rec.TryGetProperty("direct", out var direct) && direct.ValueKind == JsonValueKind.Object)
                {
                    var g = SearchGolden.BuildState(rec, out string label);
                    int depth = direct.GetProperty("d").GetInt32();
                    long budget = direct.GetProperty("budget").GetInt64();
                    string ctx = $"{label} minimax depth={depth} node budget={budget} aggressive={aggressive}";
                    var searcher = new CompatSearcher(g, SearchGolden.Db, tt);
                    var result = searcher.RunFixedDepth(depth, aggressive, SearchLimits.WithNodeBudget(budget));
                    SearchGolden.AssertSameDouble(direct.GetProperty("score").GetString()!, result.Score, ctx + " root score");
                    Assert.Equal(SearchGolden.MoveText(direct.GetProperty("move")), SearchGolden.MoveText(g, result.BestInstance, result.BestCell));
                    Assert.True(direct.GetProperty("nodes").GetInt64() == searcher.Stats.NodesSearched, $"{ctx} nodes");
                    Assert.True(direct.GetProperty("cut").GetInt64() == searcher.Stats.AlphaBetaCutoffs, $"{ctx} cutoffs");
                    Assert.True(direct.GetProperty("tt").GetInt64() == searcher.Stats.TtHits, $"{ctx} tt_hits");
                    Assert.True(direct.GetProperty("ttc").GetInt64() == searcher.Stats.TtCutoffs, $"{ctx} tt_cutoffs");
                    Assert.Equal(SearchGolden.HistoryText(direct.GetProperty("hist")), SearchGolden.HistoryText(searcher.History));
                    directBudget++;
                }
                if (!rec.TryGetProperty("name", out _)) positions++;
            }
        }

        _output.WriteLine($"positions={positions} unlimited ID runs={unlimitedRuns} node-budget ID runs={budgetRuns} (interrupted before maxDepth: {discardedLevels}) " +
                          $"direct minimax with budget={directBudget} root move None={rootNone}");
        Assert.True(positions >= 200);
        Assert.True(unlimitedRuns >= 200 + 3); // 200 positions + mid3_ascension, mid4_rev_ace, end5_same_plus to depth 10
        Assert.True(discardedLevels >= 50, "time-up path insufficiently covered");
    }

    [Fact]
    public void Transposition_key_partition_equals_pythons_get_state_hash()
    {
        string path = GoldenFiles.EnsureSearchGolden("tt");
        var pyToCs = new Dictionary<string, (ulong, ulong)>();
        var csToPy = new Dictionary<(ulong, ulong), string>();
        int states = 0, families = 0;

        foreach (var doc in SearchGolden.Records(path, "tt"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var rules = RuleParser.FromPythonList(rec.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
                foreach (var member in rec.GetProperty("members").EnumerateArray())
                {
                    var g = GameState.FromSnapshot(StateSnapshot.FromJson(member.GetProperty("state"), rules));
                    PythonStateKey.Compute(g, out ulong h1, out ulong h2);
                    string py = member.GetProperty("key").GetString()!;
                    string where = $"family {rec.GetProperty("i").GetInt32()}";

                    if (pyToCs.TryGetValue(py, out var known))
                        Assert.True(known == (h1, h2), $"{where}: Python keys are equal but the C# digests differ");
                    else
                        pyToCs[py] = (h1, h2);

                    if (csToPy.TryGetValue((h1, h2), out var knownPy))
                        Assert.True(knownPy == py, $"{where}: the C# digests are equal but the Python keys differ");
                    else
                        csToPy[(h1, h2)] = py;
                    states++;
                }
                families++;
            }
        }

        _output.WriteLine($"families={families} states={states} distinct Python keys={pyToCs.Count} distinct C# digests={csToPy.Count}");
        Assert.Equal(pyToCs.Count, csToPy.Count);
        Assert.True(states > 3 * pyToCs.Count / 2, "collision variants must produce repeated keys");
    }
}
