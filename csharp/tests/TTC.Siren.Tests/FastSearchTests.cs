using System.Numerics;
using System.Text;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// The oracle of Phase 4: ReferenceCompat with its transposition table switched off (plain stable-ordered alpha-beta;
/// root score = exact depth-limited minimax value), plus the exact full-window value of every root move, which gives
/// the tie set required by the Q-5 acceptance rule.
/// </summary>
internal static class Oracle
{
    public readonly struct RootMoveScore
    {
        public readonly Move Move;
        public readonly double Score;
        public RootMoveScore(Move move, double score) { Move = move; Score = score; }
    }

    /// <summary>ReferenceCompat with the table disabled (the table object is a never-used 4-slot placeholder).</summary>
    public static CompatSearcher Searcher(GameState g) =>
        new(g, SearchGolden.Db, new CompatTranspositionTable(initialCapacity: 4, maxCapacity: 4)) { UseTranspositionTable = false };

    /// <summary>Oracle root score and best move (compat, table off).</summary>
    public static SearchResult Exact(GameState g, int depth, bool aggressive) =>
        Searcher(g).RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);

    /// <summary>Exact depth-limited value of every legal root move, each searched with the full window (generation order).</summary>
    public static List<RootMoveScore> RootMoves(GameState g, int depth, bool aggressive)
    {
        var searcher = Searcher(g);
        Span<Move> buffer = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(buffer);
        var moves = buffer[..n].ToArray();
        int root = g.CurrentPlayer;
        var list = new List<RootMoveScore>(n);
        foreach (var move in moves)
        {
            g.MakeMove(move);
            double s = searcher.RunFixedDepthAs(depth - 1, false, root, aggressive, SearchLimits.Unlimited).Score;
            g.UndoMove();
            list.Add(new RootMoveScore(move, s));
        }
        return list;
    }

    public static bool InTieSet(List<RootMoveScore> rootMoves, double rootScore, int inst, int cell)
    {
        long bits = BitConverter.DoubleToInt64Bits(rootScore);
        foreach (var m in rootMoves)
        {
            if (m.Move.Inst == inst && m.Move.Cell == cell) return BitConverter.DoubleToInt64Bits(m.Score) == bits;
        }
        return false;
    }

    public static int TieSize(List<RootMoveScore> rootMoves, double rootScore)
    {
        long bits = BitConverter.DoubleToInt64Bits(rootScore);
        return rootMoves.Count(m => BitConverter.DoubleToInt64Bits(m.Score) == bits);
    }

    /// <summary>Full-width minimax without alpha-beta, ordering or table: an independent check of the oracle itself.</summary>
    public static double Brute(GameState g, Evaluator eval, int depth, bool maximizing, int ai, bool aggressive, ref long leaves)
    {
        if (depth == 0 || g.IsGameOver)
        {
            leaves++;
            return eval.EvaluateState(ai, aggressive);
        }
        var moves = new Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        if (n == 0)
        {
            leaves++;
            return eval.EvaluateState(ai, aggressive);
        }
        double best = maximizing ? double.NegativeInfinity : double.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            g.MakeMove(moves[i]);
            double s = Brute(g, eval, depth - 1, !maximizing, ai, aggressive, ref leaves);
            g.UndoMove();
            if (maximizing ? s > best : s < best) best = s;
        }
        return best;
    }

    /// <summary>Upper estimate of the leaves of the full tree, used to keep the brute-force check affordable.</summary>
    public static double EstimateLeaves(GameState g, int depth)
    {
        double leaves = 1;
        int empties = g.AvailableCellCount;
        int[] hand = { g.HandCount(0), g.HandCount(1) };
        int p = g.CurrentPlayer;
        for (int k = 0; k < depth && empties - k > 0; k++)
        {
            leaves *= Math.Max(1, hand[p]) * (empties - k);
            hand[p] = Math.Max(0, hand[p] - 1);
            p = 1 - p;
        }
        return leaves;
    }
}

/// <summary>Phase 4 acceptance tests for the Fast search (CSHARP_MIGRATION.md section 7, Phase 4; Q-5 = option b).</summary>
public sealed class FastSearchTests
{
    private readonly ITestOutputHelper _output;

    public FastSearchTests(ITestOutputHelper output) => _output = output;

    private static FastTranspositionTable SmallTable() => new(1 << 4); // brutal replacement pressure

    [Fact]
    public void Oracle_equals_an_independent_brute_force_minimax()
    {
        string path = GoldenFiles.EnsureSearchGolden("search");
        int checkedRuns = 0, skipped = 0, terminal = 0;
        long leaves = 0;
        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    int depth = run.GetProperty("d").GetInt32();
                    var probe = SearchGolden.BuildState(rec, out string label);
                    if (Oracle.EstimateLeaves(probe, depth) > 150_000) { skipped++; continue; }

                    var g1 = SearchGolden.BuildState(rec, out _);
                    var oracle = Oracle.Exact(g1, depth, aggressive);
                    var g2 = SearchGolden.BuildState(rec, out _);
                    var eval = new Evaluator(g2, SearchGolden.Db);
                    double brute = Oracle.Brute(g2, eval, depth, true, g2.CurrentPlayer, aggressive, ref leaves);
                    Assert.True(BitConverter.DoubleToInt64Bits(brute) == BitConverter.DoubleToInt64Bits(oracle.Score),
                        $"{label} depth={depth} aggressive={aggressive}: oracle {SearchGolden.Describe(oracle.Score)} != brute force {SearchGolden.Describe(brute)}");
                    checkedRuns++;
                    if (depth >= 9 - BitOperations.PopCount(probe.OccupiedMask)) terminal++;
                }
            }
        }
        _output.WriteLine($"oracle == brute force minimax on {checkedRuns} runs ({terminal} to the end of the game), {leaves:N0} leaves; skipped as too large: {skipped}");
        Assert.True(checkedRuns >= 1000, $"only {checkedRuns} runs verified");
    }

    [Fact]
    public void Fast_fixed_depth_matches_the_oracle_and_every_tie_difference_is_reported()
    {
        string path = GoldenFiles.EnsureSearchGolden("search");
        var report = new StringBuilder();
        report.AppendLine("# Phase 4 / Q-5 (option b) tie report. Generated by FastSearchTests.Fast_fixed_depth_matches_the_oracle_and_every_tie_difference_is_reported");
        report.AppendLine("# Each line: a run where Fast's root move differs from the oracle's (ReferenceCompat, table off) or from Python's golden move.");
        report.AppendLine("# Every listed Fast move was verified to have a full-window score bit-equal to the root score (member of the exact tie set).");
        report.AppendLine("# label | depth | aggressive | root score | tie size | oracle move | Fast move | Python move | notes");

        int runs = 0, scoreMismatchesVsPythonTtOn = 0, moveDiffOracle = 0, moveDiffPython = 0, pythonMoveOutsideTieSet = 0;
        int smallTableMoveDiff = 0, withTie = 0, noMoveRuns = 0, sameTypeRuns = 0, toTerminal = 0, aggressiveRuns = 0;
        long nodesFast = 0, nodesOracle = 0, nodesPythonTtOn = 0;
        var tieSizes = new SortedDictionary<int, int>();

        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                var rules = RuleParser.FromPythonList(rec.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    int depth = run.GetProperty("d").GetInt32();

                    // Oracle (also fixes the label) and the exact value of each root move.
                    var gOracle = SearchGolden.BuildState(rec, out string label);
                    var oracle = Oracle.Searcher(gOracle);
                    var oracleResult = oracle.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    var gTies = SearchGolden.BuildState(rec, out _);
                    var rootMoves = Oracle.RootMoves(gTies, depth, aggressive);
                    string ctx = $"{label} depth={depth} aggressive={aggressive}";

                    if (rootMoves.Count > 0)
                    {
                        double maxChild = rootMoves.Max(m => m.Score);
                        Assert.True(BitConverter.DoubleToInt64Bits(maxChild) == BitConverter.DoubleToInt64Bits(oracleResult.Score),
                            $"{ctx}: oracle root score {SearchGolden.Describe(oracleResult.Score)} != best child {SearchGolden.Describe(maxChild)}");
                    }

                    // Fast, default table size.
                    var gFast = SearchGolden.BuildState(rec, out _);
                    var fast = new FastSearcher(gFast, SearchGolden.Db, new FastTranspositionTable(1 << 16));
                    string before = SearchGolden.NormalizedState(gFast);
                    ulong hashBefore = gFast.ZobristHash;
                    var fastResult = fast.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    Assert.True(before == SearchGolden.NormalizedState(gFast), $"{ctx}: Fast search did not restore the state");
                    Assert.Equal(0, gFast.UndoDepth);
                    Assert.Equal(hashBefore, gFast.ZobristHash);
                    Assert.Equal(gFast.ComputeZobristHash(), gFast.ZobristHash);

                    // Acceptance 1: root score bit-identical to the oracle.
                    Assert.True(BitConverter.DoubleToInt64Bits(oracleResult.Score) == BitConverter.DoubleToInt64Bits(fastResult.Score),
                        $"{ctx}: Fast root score {SearchGolden.Describe(fastResult.Score)} != oracle {SearchGolden.Describe(oracleResult.Score)}");
                    Assert.Equal(oracleResult.HasMove, fastResult.HasMove);

                    // Acceptance 2: the root move belongs to the exact tie set.
                    string oracleMove = SearchGolden.MoveText(gOracle, oracleResult.BestInstance, oracleResult.BestCell);
                    string fastMove = SearchGolden.MoveText(gFast, fastResult.BestInstance, fastResult.BestCell);
                    string pyMove = SearchGolden.MoveText(run.GetProperty("move"));
                    int tie = 0;
                    if (fastResult.HasMove)
                    {
                        Assert.True(Oracle.InTieSet(rootMoves, oracleResult.Score, fastResult.BestInstance, fastResult.BestCell),
                            $"{ctx}: Fast move {fastMove} is NOT in the exact tie set (BLOCKED, not a tie)");
                        tie = Oracle.TieSize(rootMoves, oracleResult.Score);
                        Assert.True(Oracle.InTieSet(rootMoves, oracleResult.Score, oracleResult.BestInstance, oracleResult.BestCell),
                            $"{ctx}: the oracle's own move is not in its tie set");
                    }
                    else
                    {
                        noMoveRuns++;
                    }
                    tieSizes[tie] = tieSizes.GetValueOrDefault(tie) + 1;
                    if (tie > 1) withTie++;

                    // Acceptance 3: deterministic.
                    var gAgain = SearchGolden.BuildState(rec, out _);
                    var again = new FastSearcher(gAgain, SearchGolden.Db, new FastTranspositionTable(1 << 16));
                    var againResult = again.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    Assert.True(BitConverter.DoubleToInt64Bits(fastResult.Score) == BitConverter.DoubleToInt64Bits(againResult.Score) &&
                                fastResult.BestInstance == againResult.BestInstance && fastResult.BestCell == againResult.BestCell &&
                                fast.Stats.NodesSearched == again.Stats.NodesSearched, $"{ctx}: Fast search is not deterministic");

                    // Replacement pressure: a 16-entry table must give the same score and a tie-set move.
                    var gSmall = SearchGolden.BuildState(rec, out _);
                    var small = new FastSearcher(gSmall, SearchGolden.Db, SmallTable());
                    var smallResult = small.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    Assert.True(BitConverter.DoubleToInt64Bits(oracleResult.Score) == BitConverter.DoubleToInt64Bits(smallResult.Score),
                        $"{ctx}: Fast (16-entry table) root score {SearchGolden.Describe(smallResult.Score)} != oracle {SearchGolden.Describe(oracleResult.Score)}");
                    string smallMove = SearchGolden.MoveText(gSmall, smallResult.BestInstance, smallResult.BestCell);
                    if (smallResult.HasMove)
                        Assert.True(Oracle.InTieSet(rootMoves, oracleResult.Score, smallResult.BestInstance, smallResult.BestCell), $"{ctx}: small-table move not in tie set");
                    if (smallMove != oracleMove)
                    {
                        // The only place where a move different from the oracle's actually occurs: an equally scored move.
                        smallTableMoveDiff++;
                        report.AppendLine($"{label} | {depth} | {aggressive} | {SearchGolden.Describe(oracleResult.Score)} | {tie} | {oracleMove} | {smallMove} | {pyMove} | 16-entry-table-run (exact tie-set member)");
                    }

                    // Reporting.
                    bool pyScoreEqual = SearchGolden.HexBits(run.GetProperty("score").GetString()!) == BitConverter.DoubleToInt64Bits(oracleResult.Score);
                    if (!pyScoreEqual) scoreMismatchesVsPythonTtOn++;
                    bool pyInTie = false;
                    foreach (var m in rootMoves)
                        if (SearchGolden.MoveText(gTies, m.Move.Inst, m.Move.Cell) == pyMove && BitConverter.DoubleToInt64Bits(m.Score) == BitConverter.DoubleToInt64Bits(oracleResult.Score))
                            pyInTie = true;
                    if (!pyInTie && pyMove != "none") pythonMoveOutsideTieSet++;

                    string notes = "";
                    if (!pyScoreEqual) notes += "python-TT-on-score-differs(D-006) ";
                    if (!pyInTie && pyMove != "none") notes += "python-move-not-in-exact-tie-set ";
                    bool diffOracle = fastMove != oracleMove, diffPython = fastMove != pyMove;
                    if (diffOracle) moveDiffOracle++;
                    if (diffPython) moveDiffPython++;
                    if (diffOracle || diffPython)
                        report.AppendLine($"{label} | {depth} | {aggressive} | {SearchGolden.Describe(oracleResult.Score)} | {tie} | {oracleMove} | {fastMove} | {pyMove} | {notes.Trim()}");

                    runs++;
                    nodesFast += fast.Stats.NodesSearched;
                    nodesOracle += oracle.Stats.NodesSearched;
                    nodesPythonTtOn += run.GetProperty("nodes").GetInt64();
                    if (aggressive) aggressiveRuns++;
                    if ((rules & (RuleSet.Ascension | RuleSet.Descension)) != 0) sameTypeRuns++;
                    if (depth >= 9 - BitOperations.PopCount(gOracle.OccupiedMask) || depth >= gFast.AvailableCellCount) toTerminal++;
                }
            }
        }

        string summary =
            $"runs={runs} (aggressive {aggressiveRuns}, with 同类 rules {sameTypeRuns}, searched to the end of the game {toTerminal}, root side without move {noMoveRuns}); " +
            $"root score == oracle bits and move in the exact tie set on all of them (also with a 16-entry table); " +
            $"Fast move differs from the oracle's: {moveDiffOracle} (16-entry-table runs: {smallTableMoveDiff}); from Python's golden move: {moveDiffPython} " +
            $"(of which Python's move is outside the exact tie set: {pythonMoveOutsideTieSet}); runs whose Python (TT on) score != exact value (D-006): {scoreMismatchesVsPythonTtOn}; " +
            $"runs with a tie of >=2 best moves: {withTie}; tie-size histogram: {string.Join(", ", tieSizes.Select(kv => $"{kv.Key}:{kv.Value}"))}; " +
            $"nodes: Fast {nodesFast:N0}, oracle (compat TT off) {nodesOracle:N0}, Python TT on {nodesPythonTtOn:N0}";
        report.AppendLine("# SUMMARY " + summary);
        Directory.CreateDirectory(GoldenFiles.GoldenDir);
        File.WriteAllText(Path.Combine(GoldenFiles.GoldenDir, "phase4_q5_tie_report.txt"), report.ToString());
        _output.WriteLine(summary);

        Assert.True(runs >= 4_000, $"only {runs} runs");
        Assert.True(sameTypeRuns >= 500, "same-type rules insufficiently covered");
        Assert.True(toTerminal >= 300);
        Assert.True(withTie > 0, "no tied positions: the Q-5 path was not exercised");
    }

    [Fact]
    public void Fast_iterative_deepening_levels_match_the_oracle()
    {
        string path = GoldenFiles.EnsureSearchGolden("id");
        int runs = 0, levelsChecked = 0, stoppedEarly = 0, moveDiffs = 0, levelsWithTie = 0;
        long nodesFast = 0;
        var report = new StringBuilder();
        report.AppendLine("# Phase 4 / Q-5 (option b), iterative deepening: levels where Fast's move differs from the oracle's fixed-depth move.");
        report.AppendLine("# Each listed Fast move was verified to have a full-window score bit-equal to the level's root score (member of the exact tie set).");
        report.AppendLine("# label | level | aggressive | level score | tie size | oracle move | Fast move");
        foreach (var doc in SearchGolden.Records(path, "id"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    if (run.GetProperty("budget").ValueKind != JsonValueKind.Null) continue; // interrupted runs: separate test
                    int maxDepth = run.GetProperty("maxd").GetInt32();
                    var g = SearchGolden.BuildState(rec, out string label);
                    int empties = g.AvailableCellCount;
                    string ctx = $"{label} ID maxDepth={maxDepth} aggressive={aggressive}";

                    var fast = new FastSearcher(g, SearchGolden.Db, new FastTranspositionTable(1 << 16));
                    string before = SearchGolden.NormalizedState(g);
                    var result = fast.RunIterativeDeepening(maxDepth, aggressive, SearchLimits.Unlimited);
                    Assert.True(before == SearchGolden.NormalizedState(g), $"{ctx}: state not restored");
                    var s = fast.Stats;

                    int expectedLevels = Math.Min(maxDepth, empties);
                    Assert.True(s.Levels <= expectedLevels && s.Levels >= 1 && s.DepthCompleted == s.Levels, $"{ctx}: levels={s.Levels}, expected up to {expectedLevels}");
                    if (s.Levels < expectedLevels) Assert.False(result.HasMove, $"{ctx}: stopped early with a move available");
                    if (maxDepth > empties) stoppedEarly++;

                    for (int level = 1; level <= s.Levels; level++)
                    {
                        var gl = SearchGolden.BuildState(rec, out _);
                        var oracle = Oracle.Exact(gl, level, aggressive);
                        Assert.True(BitConverter.DoubleToInt64Bits(oracle.Score) == BitConverter.DoubleToInt64Bits(s.BestScoreHistory[level - 1]),
                            $"{ctx} level {level}: Fast {SearchGolden.Describe(s.BestScoreHistory[level - 1])} != oracle {SearchGolden.Describe(oracle.Score)}");
                        if (s.BestInstanceHistory[level - 1] >= 0)
                        {
                            var gt = SearchGolden.BuildState(rec, out _);
                            var rootMoves = Oracle.RootMoves(gt, level, aggressive);
                            Assert.True(Oracle.InTieSet(rootMoves, oracle.Score, s.BestInstanceHistory[level - 1], s.BestCellHistory[level - 1]),
                                $"{ctx} level {level}: move not in the exact tie set");
                            int tie = Oracle.TieSize(rootMoves, oracle.Score);
                            if (tie > 1) levelsWithTie++;
                            string oracleMove = SearchGolden.MoveText(gl, oracle.BestInstance, oracle.BestCell);
                            string fastMove = SearchGolden.MoveText(g, s.BestInstanceHistory[level - 1], s.BestCellHistory[level - 1]);
                            if (oracleMove != fastMove)
                            {
                                moveDiffs++;
                                report.AppendLine($"{label} | {level} | {aggressive} | {SearchGolden.Describe(oracle.Score)} | {tie} | {oracleMove} | {fastMove}");
                            }
                        }
                        levelsChecked++;
                    }
                    runs++;
                    nodesFast += s.NodesSearched;
                }
            }
        }
        string summary = $"unlimited ID runs={runs}, levels compared with the oracle={levelsChecked}, levels with a tie of >=2 best moves={levelsWithTie}, " +
                         $"levels where Fast's move differs from the oracle's fixed-depth move (all exact tie-set members)={moveDiffs}, " +
                         $"runs where maxDepth exceeded the empty cells (Fast stops, D-014)={stoppedEarly}, Fast nodes={nodesFast:N0}";
        report.AppendLine("# SUMMARY " + summary);
        Directory.CreateDirectory(GoldenFiles.GoldenDir);
        File.WriteAllText(Path.Combine(GoldenFiles.GoldenDir, "phase4_q5_id_tie_report.txt"), report.ToString());
        _output.WriteLine(summary);
        Assert.True(runs >= 203);
        Assert.True(stoppedEarly >= 3);
    }

    [Fact]
    public void An_interrupted_search_stores_nothing_that_later_searches_could_reuse()
    {
        string path = GoldenFiles.EnsureSearchGolden("id");
        int runs = 0, interrupted = 0, reused = 0;
        foreach (var doc in SearchGolden.Records(path, "id"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool aggressive = rec.GetProperty("aggr").GetBoolean();
                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    if (run.GetProperty("budget").ValueKind == JsonValueKind.Null) continue;
                    int maxDepth = run.GetProperty("maxd").GetInt32();
                    long budget = run.GetProperty("budget").GetInt64();
                    var g = SearchGolden.BuildState(rec, out string label);
                    string ctx = $"{label} ID maxDepth={maxDepth} budget={budget} aggressive={aggressive}";

                    var table = new FastTranspositionTable(1 << 16);
                    var fast = new FastSearcher(g, SearchGolden.Db, table);
                    var result = fast.RunIterativeDeepening(maxDepth, aggressive, SearchLimits.WithNodeBudget(budget));
                    runs++;
                    int empties = g.AvailableCellCount;
                    int next = fast.Stats.Levels + 1;
                    if (next > Math.Min(maxDepth, empties)) continue; // it finished: nothing was interrupted
                    interrupted++;

                    // Levels that were accepted must be exact.
                    for (int level = 1; level <= fast.Stats.Levels; level++)
                    {
                        if (level == 1 && fast.Stats.Levels == 1) break; // level 1 may be a partial result by design
                        var gl = SearchGolden.BuildState(rec, out _);
                        Assert.True(BitConverter.DoubleToInt64Bits(Oracle.Exact(gl, level, aggressive).Score) == BitConverter.DoubleToInt64Bits(fast.Stats.BestScoreHistory[level - 1]),
                            $"{ctx}: accepted level {level} is not the exact value");
                    }

                    // Continue on the same table with an unlimited search of the level that was interrupted. If an aborted
                    // node had stored its partial value, equal-depth entries would now return it.
                    var cont = fast.RunFixedDepth(next, aggressive, SearchLimits.Unlimited, clearTable: false);
                    reused += (int)fast.Stats.TtHits > 0 ? 1 : 0;
                    var go = SearchGolden.BuildState(rec, out _);
                    var oracle = Oracle.Exact(go, next, aggressive);
                    Assert.True(BitConverter.DoubleToInt64Bits(oracle.Score) == BitConverter.DoubleToInt64Bits(cont.Score),
                        $"{ctx}: continuing after an interruption gave {SearchGolden.Describe(cont.Score)}, exact value {SearchGolden.Describe(oracle.Score)} (table polluted by an aborted node?)");
                }
            }
        }
        _output.WriteLine($"node-budget runs={runs}, interrupted before the end={interrupted}, of which the table was consulted on the continuation={reused}");
        Assert.True(interrupted >= 50, $"only {interrupted} interrupted runs");
        Assert.True(reused >= 25);
    }

    [Fact]
    public void Fast_search_hot_path_allocates_nothing_in_steady_state()
    {
        var db = SearchGolden.Db;
        var table = new FastTranspositionTable(1 << 16);
        var searchers = ReferenceScenarios.Names
            .Select(name => new FastSearcher(ReferenceScenarios.Create(name, db), db, table))
            .ToArray();
        foreach (var s in searchers)
        {
            s.RunFixedDepth(4, false, SearchLimits.Unlimited);
            s.RunFixedDepth(4, true, SearchLimits.Unlimited);
            s.RunIterativeDeepening(5, false, SearchLimits.Unlimited);
        }

        long nodes = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int round = 0; round < 5; round++)
        {
            foreach (var s in searchers)
            {
                s.RunFixedDepth(4, round % 2 == 1, SearchLimits.Unlimited);
                nodes += s.Stats.NodesSearched;
                s.RunIterativeDeepening(5, false, SearchLimits.WithNodeBudget(20_000));
                nodes += s.Stats.NodesSearched;
                s.RunIterativeDeepening(5, false, SearchLimits.WithTime(0.05));
                nodes += s.Stats.NodesSearched;
            }
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        _output.WriteLine($"{nodes:N0} nodes searched, allocated {allocated} bytes");
        Assert.True(nodes > 100_000);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void Fast_search_with_a_wall_clock_limit_stops_and_restores_the_state()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        var fast = new FastSearcher(g, db, new FastTranspositionTable(1 << 18));
        string before = SearchGolden.NormalizedState(g);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = fast.RunIterativeDeepening(9, false, SearchLimits.WithTime(0.3));
        sw.Stop();
        Assert.True(sw.Elapsed.TotalSeconds < 5, $"took {sw.Elapsed.TotalSeconds:F2} s for a 0.3 s limit");
        Assert.True(result.HasMove);
        Assert.True(fast.Stats.DepthCompleted >= 3);
        Assert.Equal(before, SearchGolden.NormalizedState(g));
        Assert.Equal(0, g.UndoDepth);
        Assert.Equal(g.ComputeZobristHash(), g.ZobristHash);
        _output.WriteLine($"completed depth {fast.Stats.DepthCompleted}, {fast.Stats.NodesSearched:N0} nodes in {sw.Elapsed.TotalSeconds:F2} s");
    }

    [Fact]
    public void Fast_search_rejects_invalid_depths_without_touching_the_state()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        var fast = new FastSearcher(g, db, new FastTranspositionTable(1 << 10));
        string before = SearchGolden.NormalizedState(g);
        Assert.Throws<ArgumentOutOfRangeException>(() => fast.RunFixedDepth(-1, false, SearchLimits.Unlimited));
        Assert.Throws<ArgumentOutOfRangeException>(() => fast.RunFixedDepth(FastSearcher.MaxDepth + 1, false, SearchLimits.Unlimited));
        Assert.Throws<ArgumentOutOfRangeException>(() => fast.RunIterativeDeepening(FastSearcher.MaxDepth + 1, false, SearchLimits.Unlimited));
        Assert.Equal(before, SearchGolden.NormalizedState(g));
        Assert.Equal(0, g.UndoDepth);
    }

    [Fact]
    public void Benchmark_scenarios_to_greater_depth_match_the_oracle()
    {
        var db = SearchGolden.Db;
        (string Name, int Depth)[] items =
        {
            ("open_none", 5), ("open_same_plus", 5), ("mid3_ascension", 8), ("mid4_rev_ace", 8), ("end5_same_plus", 6), ("open_all_rules", 4),
        };
        long nodesFast = 0, nodesOracle = 0;
        foreach (var (name, depth) in items)
        {
            for (int d = 1; d <= depth; d++)
            {
                var gOracle = ReferenceScenarios.Create(name, db);
                var oracle = Oracle.Searcher(gOracle);
                var o = oracle.RunFixedDepth(d, false, SearchLimits.Unlimited);
                var gFast = ReferenceScenarios.Create(name, db);
                var fast = new FastSearcher(gFast, db, new FastTranspositionTable(1 << 18));
                var f = fast.RunFixedDepth(d, false, SearchLimits.Unlimited);
                Assert.True(BitConverter.DoubleToInt64Bits(o.Score) == BitConverter.DoubleToInt64Bits(f.Score),
                    $"{name} depth {d}: Fast {SearchGolden.Describe(f.Score)} != oracle {SearchGolden.Describe(o.Score)}");
                var gTies = ReferenceScenarios.Create(name, db);
                var rootMoves = Oracle.RootMoves(gTies, d, false);
                Assert.True(Oracle.InTieSet(rootMoves, o.Score, f.BestInstance, f.BestCell), $"{name} depth {d}: Fast move not in tie set");
                nodesFast += fast.Stats.NodesSearched;
                nodesOracle += oracle.Stats.NodesSearched;
            }
        }
        _output.WriteLine($"nodes: Fast {nodesFast:N0} vs oracle (compat, table off) {nodesOracle:N0}");
    }
}

public sealed class FastTranspositionTableTests
{
    [Fact]
    public void Entries_round_trip_and_generations_hide_old_entries()
    {
        var tt = new FastTranspositionTable(1 << 6);
        Assert.False(tt.TryGet(123, out _));
        tt.Store(123, 4, -1.5, Bound.Lower, 7, 3);
        Assert.True(tt.TryGet(123, out var e));
        Assert.Equal(-1.5, e.Score);
        Assert.Equal(4, e.Depth);
        Assert.Equal(Bound.Lower, e.Bound);
        Assert.Equal(7, e.BestInstance);
        Assert.Equal(3, e.BestCell);

        tt.Store(123, 5, 2.25, Bound.Exact, -1, 0); // same key: overwritten
        Assert.True(tt.TryGet(123, out e));
        Assert.Equal(2.25, e.Score);
        Assert.Equal(Bound.Exact, e.Bound);

        tt.NewGeneration();
        Assert.False(tt.TryGet(123, out _));
        // Survives the generation counter wrapping (the array is cleared on wrap, never exposing a stale entry).
        for (int i = 0; i < 600; i++)
        {
            tt.Store(999, 1, i, Bound.Exact, -1, 0);
            tt.NewGeneration();
            Assert.False(tt.TryGet(999, out _), $"stale entry visible after {i + 1} generations");
        }
    }

    [Fact]
    public void A_full_bucket_replaces_the_shallower_entry_and_always_stores_the_new_one()
    {
        var tt = new FastTranspositionTable(4); // two buckets of two slots; keys 0, 2, 4 ... all map to bucket 0
        tt.Store(0, 6, 1.0, Bound.Exact, -1, 0);
        tt.Store(4, 2, 2.0, Bound.Exact, -1, 0);
        tt.Store(8, 3, 3.0, Bound.Exact, -1, 0); // evicts depth 2, keeps depth 6
        Assert.True(tt.TryGet(0, out var deep));
        Assert.Equal(1.0, deep.Score);
        Assert.False(tt.TryGet(4, out _));
        Assert.True(tt.TryGet(8, out var fresh));
        Assert.Equal(3.0, fresh.Score);
    }

    [Fact]
    public void Capacity_must_be_a_power_of_two()
    {
        Assert.Throws<ArgumentException>(() => new FastTranspositionTable(100));
        Assert.Throws<ArgumentException>(() => new FastTranspositionTable(2));
    }
}
