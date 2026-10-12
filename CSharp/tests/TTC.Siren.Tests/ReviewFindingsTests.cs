using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Tests;

/// <summary>Findings of the Phase 3 review that the Phase 7 goal folds in (L-1, L-2, L-9).</summary>
public sealed class ReviewFindingsTests
{
    private static CardDb Db => SearchGolden.Db;

    [Fact]
    public void L9_an_unknown_owner_in_a_golden_card_is_rejected()
    {
        using var doc = System.Text.Json.JsonDocument.Parse("[7,null,\"x\",0,1,2,3,4,1,2,3,4,1]");
        Assert.Throws<InvalidGameStateException>(() => CardSnapshot.FromJson(doc.RootElement));
        using var ok = System.Text.Json.JsonDocument.Parse("[7,null,\"b\",0,1,2,3,4,1,2,3,4,1]");
        Assert.Equal(OwnerId.Blue, CardSnapshot.FromJson(ok.RootElement).Owner);
        using var number = System.Text.Json.JsonDocument.Parse("[7,null,1,0,1,2,3,4,1,2,3,4,1]");
        Assert.Throws<InvalidGameStateException>(() => CardSnapshot.FromJson(number.RootElement));
    }

    [Fact]
    public void L1_iterative_deepening_with_a_huge_max_depth_is_clamped_to_pythons_100()
    {
        var g = ReferenceScenarios.Create("end5_same_plus", Db);
        var searcher = new CompatSearcher(g, Db, new CompatTranspositionTable(1 << 10));
        var result = searcher.RunIterativeDeepening(100_000, false, SearchLimits.Unlimited); // used to throw IndexOutOfRange past 128 levels
        Assert.True(result.HasMove);
        Assert.Equal(CompatSearcher.PythonMaxDepth, searcher.Stats.Levels);
    }

    [Fact]
    public void L2_search_limits_follow_pythons_comparison()
    {
        Assert.True(double.IsPositiveInfinity(new SearchLimits(double.NaN).TimeLimitSeconds)); // elapsed > NaN is never true
        foreach (double limit in new[] { 0.0, -1.0, double.NegativeInfinity, -1e300 })
        {
            var g = ReferenceScenarios.Create("mid3_ascension", Db);
            var compat = new CompatSearcher(g, Db, new CompatTranspositionTable(1 << 10));
            Assert.False(compat.RunIterativeDeepening(5, false, new SearchLimits(limit)).HasMove, $"compat, limit {limit}");
            var fast = new FastSearcher(ReferenceScenarios.Create("mid3_ascension", Db), Db, new FastTranspositionTable(1 << 10));
            Assert.False(fast.RunIterativeDeepening(5, false, new SearchLimits(limit)).HasMove, $"fast, limit {limit}");
        }
        var unlimited = new FastSearcher(ReferenceScenarios.Create("mid3_ascension", Db), Db, new FastTranspositionTable(1 << 10));
        Assert.True(unlimited.RunIterativeDeepening(3, false, new SearchLimits(double.NaN)).HasMove);
        var huge = new FastSearcher(ReferenceScenarios.Create("mid3_ascension", Db), Db, new FastTranspositionTable(1 << 10));
        Assert.True(huge.RunIterativeDeepening(3, false, new SearchLimits(1e300)).HasMove);
    }
}
