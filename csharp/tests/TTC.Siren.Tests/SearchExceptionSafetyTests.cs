using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Tests;

/// <summary>Review finding H-1: an exception escaping the recursive search must leave the shared GameState restored.</summary>
public sealed class SearchExceptionSafetyTests
{
    private const string Scenario = "open_none";

    private static CompatTranspositionTable TinyTable() => new(64, maxCapacity: 64); // hard cap: throws once 48 entries are in use

    private static void AssertSameSearch(GameState a, GameState b, string label)
    {
        var db = SearchGolden.Db;
        var sa = new CompatSearcher(a, db, new CompatTranspositionTable());
        var sb = new CompatSearcher(b, db, new CompatTranspositionTable());
        var ra = sa.RunFixedDepth(3, false, SearchLimits.Unlimited);
        var rb = sb.RunFixedDepth(3, false, SearchLimits.Unlimited);

        Assert.True(BitConverter.DoubleToInt64Bits(ra.Score) == BitConverter.DoubleToInt64Bits(rb.Score),
            $"{label}: score {SearchGolden.Describe(ra.Score)} vs {SearchGolden.Describe(rb.Score)}");
        Assert.Equal(SearchGolden.MoveText(a, ra.BestInstance, ra.BestCell), SearchGolden.MoveText(b, rb.BestInstance, rb.BestCell));
        Assert.True(sa.Stats.NodesSearched == sb.Stats.NodesSearched, $"{label}: nodes {sa.Stats.NodesSearched} vs {sb.Stats.NodesSearched}");
        Assert.True(sa.Stats.AlphaBetaCutoffs == sb.Stats.AlphaBetaCutoffs, $"{label}: cutoffs {sa.Stats.AlphaBetaCutoffs} vs {sb.Stats.AlphaBetaCutoffs}");
    }

    private static void RunScenario(string label, Action<CompatSearcher> search)
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create(Scenario, db);
        string before = SearchGolden.NormalizedState(g);
        var searcher = new CompatSearcher(g, db, TinyTable());

        var ex = Assert.ThrowsAny<Exception>(() => search(searcher));

        Assert.True(ex is TranspositionTableFullException, $"{label}: expected the table-full exception, got {ex.GetType().Name}: {ex.Message}");
        Assert.True(g.UndoDepth == 0, $"{label}: UndoDepth left at {g.UndoDepth} (scenario {Scenario})");
        Assert.True(before == SearchGolden.NormalizedState(g), $"{label}: state not restored after the exception (scenario {Scenario})");

        // No hidden undo/scratch state may survive: a fresh copy of the state must search identically.
        var g2 = GameState.FromSnapshot(g.ToSnapshot());
        AssertSameSearch(g, g2, label + " (after exception vs snapshot copy)");
    }

    [Fact]
    public void Search_restores_state_when_an_exception_escapes()
    {
        RunScenario("RunFixedDepth(4)", s => s.RunFixedDepth(4, false, SearchLimits.Unlimited));
        RunScenario("RunIterativeDeepening(6)", s => s.RunIterativeDeepening(6, false, SearchLimits.Unlimited));
        RunScenario("RunFixedDepth(4, aggressive)", s => s.RunFixedDepth(4, true, SearchLimits.Unlimited));
    }
}
