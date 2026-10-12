using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Review H4-1: Fast transposition-table entries are only valid for one (GameState, AI side, evaluation mode).
/// Reuse under another context must be rejected before anything is changed; valid reuse must stay exact.
/// </summary>
public sealed class FastTableContextTests
{
    private readonly ITestOutputHelper _output;

    public FastTableContextTests(ITestOutputHelper output) => _output = output;

    /// <summary>Everything a rejected call must leave alone.</summary>
    private readonly record struct Probe(
        int UndoDepth, ulong Hash, string State, long Nodes, long TtHits, long TtCutoffs, long Cutoffs, long MoveEvals,
        int Done, int Levels, long Stores, bool Bound, string History);

    private static Probe Capture(GameState g, FastSearcher f, FastTranspositionTable table) => new(
        g.UndoDepth, g.ZobristHash, g.ToSnapshot().ToCanonicalString(), f.Stats.NodesSearched, f.Stats.TtHits, f.Stats.TtCutoffs,
        f.Stats.AlphaBetaCutoffs, f.Stats.MoveEvaluations, f.Stats.DepthCompleted, f.Stats.Levels, table.Stores, table.IsBound,
        SearchGolden.HistoryText(f.History));

    private static void AssertUnchanged(Probe before, GameState g, FastSearcher f, FastTranspositionTable table, string ctx)
    {
        Assert.Equal(g.ComputeZobristHash(), g.ZobristHash);
        Assert.True(before == Capture(g, f, table), $"{ctx}: a rejected call changed state, hash, statistics, history or table");
    }

    /// <summary>The generated move at <paramref name="which"/> (0 = first, 1 = middle), or null if there is none.</summary>
    private static Move? PickMove(GameState g, int which)
    {
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        if (n == 0) return null;
        return moves[which == 0 ? 0 : n / 2];
    }

    private static string Reject(Action call)
    {
        var ex = Assert.Throws<InvalidOperationException>(call);
        return ex.Message;
    }

    private static void RunKind(FastSearcher f, bool iterative, int depth, bool aggressive, bool clearTable)
    {
        if (iterative) f.RunIterativeDeepening(depth, aggressive, SearchLimits.Unlimited, clearTable);
        else f.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited, clearTable);
    }

    private static void AssertEqualsOracle(FastSearcher f, GameState g, int depth, bool aggressive, string ctx)
    {
        var result = f.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
        var oracle = Oracle.Exact(g, depth, aggressive);
        Assert.True(BitConverter.DoubleToInt64Bits(result.Score) == BitConverter.DoubleToInt64Bits(oracle.Score),
            $"{ctx}: clearTable search {SearchGolden.Describe(result.Score)} != oracle {SearchGolden.Describe(oracle.Score)}");
    }

    [Fact]
    public void Table_reuse_with_a_different_ai_side_is_rejected()
    {
        var db = SearchGolden.Db;
        int cases = 0;
        foreach (string name in ReferenceScenarios.Names)
        {
            foreach (bool iterative in new[] { false, true })
            {
                var g = ReferenceScenarios.Create(name, db);
                var table = new FastTranspositionTable(1 << 14);
                var fast = new FastSearcher(g, db, table);
                int depth = Math.Min(3, g.AvailableCellCount);
                string ctx = $"{name} {(iterative ? "ID" : "fixed")} depth={depth}";

                fast.RunFixedDepth(depth, false, SearchLimits.Unlimited);
                g.MakeMove(PickMove(g, 0)!.Value); // the caller plays: the side to move, and with it the AI side, flips
                var before = Capture(g, fast, table);

                string message = Reject(() => RunKind(fast, iterative, depth - 1, false, clearTable: false));
                Assert.Contains("AI side", message);
                AssertUnchanged(before, g, fast, table, ctx);

                AssertEqualsOracle(fast, g, depth - 1, false, ctx + " after the rejection");
                cases++;
            }
        }
        _output.WriteLine($"rejected cases: {cases}");
        Assert.Equal(12, cases);
    }

    [Fact]
    public void Table_reuse_with_a_different_evaluation_mode_is_rejected()
    {
        var db = SearchGolden.Db;
        int cases = 0;
        foreach (string name in ReferenceScenarios.Names)
        {
            foreach (bool iterative in new[] { false, true })
            {
                foreach (bool firstAggressive in new[] { false, true })
                {
                    var g = ReferenceScenarios.Create(name, db);
                    var table = new FastTranspositionTable(1 << 14);
                    var fast = new FastSearcher(g, db, table);
                    int depth = Math.Min(3, g.AvailableCellCount);
                    string ctx = $"{name} {(iterative ? "ID" : "fixed")} depth={depth} first aggressive={firstAggressive}";

                    fast.RunFixedDepth(depth, firstAggressive, SearchLimits.Unlimited);
                    var before = Capture(g, fast, table);

                    string message = Reject(() => RunKind(fast, iterative, depth, !firstAggressive, clearTable: false));
                    Assert.Contains("evaluation mode", message);
                    AssertUnchanged(before, g, fast, table, ctx);

                    // Same root, same depth, same mode, clearTable:false is valid reuse and still exact.
                    var again = fast.RunFixedDepth(depth, firstAggressive, SearchLimits.Unlimited, clearTable: false);
                    var oracle = Oracle.Exact(g, depth, firstAggressive);
                    Assert.True(BitConverter.DoubleToInt64Bits(again.Score) == BitConverter.DoubleToInt64Bits(oracle.Score), $"{ctx}: valid reuse not exact");

                    AssertEqualsOracle(fast, g, depth, !firstAggressive, ctx + " after the rejection");
                    cases++;
                }
            }
        }
        _output.WriteLine($"rejected cases: {cases}");
        Assert.Equal(24, cases);
    }

    [Fact]
    public void Table_shared_between_game_states_is_rejected()
    {
        var db = SearchGolden.Db;
        int cases = 0;
        foreach (var (first, second) in new[] { ("open_none", "open_same_plus"), ("open_none", "open_all_rules") })
        {
            foreach (bool iterative in new[] { false, true })
            {
                var table = new FastTranspositionTable(1 << 14);
                var gA = ReferenceScenarios.Create(first, db);
                var gB = ReferenceScenarios.Create(second, db);
                var fastA = new FastSearcher(gA, db, table);
                var fastB = new FastSearcher(gB, db, table);
                string ctx = $"{first} -> {second} {(iterative ? "ID" : "fixed")}";

                fastA.RunFixedDepth(3, false, SearchLimits.Unlimited);
                var beforeB = Capture(gB, fastB, table);
                var beforeA = Capture(gA, fastA, table);

                string message = Reject(() => RunKind(fastB, iterative, 3, false, clearTable: false));
                Assert.Contains("different GameState", message);
                AssertUnchanged(beforeB, gB, fastB, table, ctx);

                // The table still belongs to A: A may continue on it (valid reuse), B can start fresh.
                Assert.True(ReferenceEquals(table.Context.State, gA), $"{ctx}: the rejection changed the table's context");
                AssertUnchanged(beforeA, gA, fastA, table, ctx + " (state A)");
                var continued = fastA.RunFixedDepth(3, false, SearchLimits.Unlimited, clearTable: false);
                Assert.True(BitConverter.DoubleToInt64Bits(continued.Score) == BitConverter.DoubleToInt64Bits(Oracle.Exact(gA, 3, false).Score), $"{ctx}: A's own reuse not exact");

                AssertEqualsOracle(fastB, gB, 3, false, ctx + " fresh start for B");
                Assert.True(ReferenceEquals(table.Context.State, gB));
                cases++;
            }
        }

        // A table that was never bound (new, or emptied without a context) is rejected too.
        foreach (bool iterative in new[] { false, true })
        {
            var g = ReferenceScenarios.Create("end5_same_plus", db);
            var table = new FastTranspositionTable(1 << 10);
            var fast = new FastSearcher(g, db, table);
            Assert.False(table.IsBound);
            var before = Capture(g, fast, table);
            Assert.Contains("not bound", Reject(() => RunKind(fast, iterative, 2, false, clearTable: false)));
            AssertUnchanged(before, g, fast, table, "unbound new table");
            Assert.False(table.IsBound);

            fast.RunFixedDepth(2, false, SearchLimits.Unlimited);
            Assert.True(table.IsBound);
            table.NewGeneration(); // context-free overload: empties and unbinds
            Assert.False(table.IsBound);
            before = Capture(g, fast, table);
            Assert.Contains("not bound", Reject(() => RunKind(fast, iterative, 2, false, clearTable: false)));
            AssertUnchanged(before, g, fast, table, "unbound after a context-free generation");
            cases++;
        }
        _output.WriteLine($"rejected cases: {cases}");
        Assert.Equal(6, cases);
    }

    [Fact]
    public void Valid_table_reuse_matches_the_oracle()
    {
        var db = SearchGolden.Db;
        var table = new FastTranspositionTable(1 << 16); // one table for everything: each first search rebinds it
        int scenarioPositions = 0, goldenPositions = 0, runs = 0, reuseHits = 0, tieMembers = 0;

        void Check(Func<GameState> make, string label)
        {
            foreach (bool aggressive in new[] { false, true })
            {
                var g = make();
                int empties = g.AvailableCellCount;
                int depth = Math.Min(5, empties); // empties >= 4 so depth - 2 >= 2
                string ctx = $"{label} aggressive={aggressive} D={depth} (golden seed {GoldenFiles.SearchSeed})";
                var fast = new FastSearcher(g, db, table);

                var first = fast.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                if (!first.HasMove) continue;
                g.MakeMove(new Move(first.BestInstance, first.BestCell));
                var reply = PickMove(g, 1); // fixed choice: the middle reply in generated order
                if (reply is null) { g.UndoMove(); continue; }
                g.MakeMove(reply.Value);

                // The root advanced by two plies: same state, same AI side, same mode.
                CheckReuse(fast, g, depth - 2, aggressive, ctx + " root advanced by two plies");
                g.UndoMove();
                g.UndoMove();
                // The original root again, at another depth.
                CheckReuse(fast, g, depth - 1, aggressive, ctx + " original root at D-1");
            }
        }

        void CheckReuse(FastSearcher fast, GameState g, int depth, bool aggressive, string ctx)
        {
            var result = fast.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited, clearTable: false);
            if (fast.Stats.TtHits > 0) reuseHits++;
            var gOracle = g; // the oracle restores the state it searches
            var oracle = Oracle.Exact(gOracle, depth, aggressive);
            Assert.True(BitConverter.DoubleToInt64Bits(result.Score) == BitConverter.DoubleToInt64Bits(oracle.Score),
                $"{ctx}: reused-table score {SearchGolden.Describe(result.Score)} != oracle {SearchGolden.Describe(oracle.Score)}");
            Assert.Equal(oracle.HasMove, result.HasMove);
            if (result.HasMove)
            {
                var rootMoves = Oracle.RootMoves(g, depth, aggressive);
                Assert.True(Oracle.InTieSet(rootMoves, oracle.Score, result.BestInstance, result.BestCell), $"{ctx}: move not in the exact tie set");
                tieMembers++;
            }
            runs++;
        }

        foreach (string name in ReferenceScenarios.Names)
        {
            if (ReferenceScenarios.Create(name, db).AvailableCellCount < 4) continue;
            Check(() => ReferenceScenarios.Create(name, db), $"scenario {name}");
            scenarioPositions++;
        }

        foreach (var doc in SearchGolden.Records(GoldenFiles.EnsureSearchGolden("id"), "id"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var probe = SearchGolden.BuildState(rec, out string label);
                if (probe.AvailableCellCount < 4) continue;
                Check(() => SearchGolden.BuildState(rec, out _), $"golden {label}");
                goldenPositions++;
            }
        }

        _output.WriteLine($"scenario positions={scenarioPositions}, golden positions={goldenPositions}, reuse searches={runs}, " +
                          $"of which the table was consulted={reuseHits}, moves in the exact tie set={tieMembers}");
        Assert.True(scenarioPositions >= 6, $"scenarios: {scenarioPositions}");
        Assert.True(goldenPositions >= 100, $"only {goldenPositions} golden positions with >= 4 empty cells");
        Assert.True(reuseHits > runs / 2, "the reuse path was barely exercised");
    }
}
