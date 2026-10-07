using System.Diagnostics;
using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Tests;

/// <summary>Behaviour that cannot be compared with Python bit for bit because it depends on the wall clock.</summary>
public sealed class SearchBehaviourTests
{
    [Fact]
    public void Wall_clock_limit_stops_iterative_deepening_with_a_usable_move()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        var searcher = new CompatSearcher(g, db, new CompatTranspositionTable());

        long t0 = Stopwatch.GetTimestamp();
        var result = searcher.RunIterativeDeepening(9, false, SearchLimits.WithTime(0.5));
        double seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds;

        Assert.True(seconds < 3.0, $"search took {seconds:F2}s with a 0.5s limit");
        Assert.True(searcher.Stats.Levels >= 3, $"only {searcher.Stats.Levels} levels completed");
        Assert.True(searcher.Stats.Levels < 9, "an empty-board search to depth 9 cannot finish in 0.5s");
        Assert.True(result.HasMove);
        Assert.Equal(searcher.Stats.Levels, searcher.Stats.DepthCompleted);
    }

    [Fact]
    public void Unlimited_search_is_deterministic_across_runs()
    {
        var db = SearchGolden.Db;
        var tt = new CompatTranspositionTable();
        long[] nodes = new long[3];
        long[] scoreBits = new long[3];
        for (int i = 0; i < 3; i++)
        {
            var g = ReferenceScenarios.Create("mid3_ascension", db);
            var searcher = new CompatSearcher(g, db, tt);
            var r = searcher.RunFixedDepth(5, false, SearchLimits.Unlimited);
            nodes[i] = searcher.Stats.NodesSearched;
            scoreBits[i] = BitConverter.DoubleToInt64Bits(r.Score);
        }
        Assert.Equal(nodes[0], nodes[1]);
        Assert.Equal(nodes[0], nodes[2]);
        Assert.Equal(scoreBits[0], scoreBits[2]);
    }

    [Fact]
    public void Move_order_is_recomputed_by_card_identity_after_hand_reordering()
    {
        // Regression: a Move holds a hand slot, but make+undo reorders the hand (D-001). The searcher must make each
        // ordered move with the card it was ordered for. A depth-2 search of the empty board exercises this on every node.
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        var searcher = new CompatSearcher(g, db, new CompatTranspositionTable());
        var r = searcher.RunFixedDepth(2, false, SearchLimits.Unlimited);

        // CSHARP_MIGRATION.md section 3.4 (Python, open_none depth 2): score -0x1.23d70a3d70a3ep+2 (bit pattern below),
        // move [64,2,1], 214 nodes, 42 cutoffs.
        Assert.Equal(unchecked((long)0xC0123D70A3D70A3EUL), BitConverter.DoubleToInt64Bits(r.Score));
        Assert.Equal(64, g.InstanceId(r.BestInstance));
        Assert.Equal(7, r.BestCell); // row 2, col 1
        Assert.Equal(214, searcher.Stats.NodesSearched);
        Assert.Equal(42, searcher.Stats.AlphaBetaCutoffs);
    }
}
