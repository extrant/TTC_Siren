using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 8A: the only change to Search is the per-level callback of <c>FastSearcher.RunIterativeDeepening</c> (live progress). Proof that it
/// observes only: with and without a callback the result, every per-level statistic and the node counts are identical, on all reference
/// scenarios, in both evaluation modes, with a time-unlimited search; the callback fires exactly once per completed level, in order, with the
/// numbers the statistics hold; and a steady-state search with a (cached) callback still allocates nothing.
/// </summary>
public sealed class LevelCallbackTests
{
    private readonly ITestOutputHelper _output;

    public LevelCallbackTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void A_callback_changes_nothing_and_fires_once_per_completed_level()
    {
        int compared = 0;
        foreach (string name in ReferenceScenarios.Names)
            foreach (bool aggressive in new[] { false, true })
            {
                var plain = new FastSearcher(ReferenceScenarios.Create(name, SearchGolden.Db), SearchGolden.Db, new FastTranspositionTable(1 << 16));
                var observed = new FastSearcher(ReferenceScenarios.Create(name, SearchGolden.Db), SearchGolden.Db, new FastTranspositionTable(1 << 16));
                var a = plain.RunIterativeDeepening(6, aggressive, SearchLimits.Unlimited);
                var seen = new List<LevelProgress>();
                var b = observed.RunIterativeDeepening(6, aggressive, SearchLimits.Unlimited, onLevel: seen.Add);

                string where = $"{name} aggressive={aggressive}";
                Assert.True(BitConverter.DoubleToInt64Bits(a.Score) == BitConverter.DoubleToInt64Bits(b.Score), $"{where}: score");
                Assert.True(a.BestInstance == b.BestInstance && a.BestCell == b.BestCell && a.HasMove == b.HasMove, $"{where}: move");
                var s1 = plain.Stats;
                var s2 = observed.Stats;
                Assert.Equal(s1.Levels, s2.Levels);
                Assert.Equal(s1.NodesSearched, s2.NodesSearched);
                Assert.Equal(s1.TtHits, s2.TtHits);
                Assert.Equal(s1.AlphaBetaCutoffs, s2.AlphaBetaCutoffs);
                Assert.Equal(s1.DepthCompleted, s2.DepthCompleted);
                for (int i = 0; i < s1.Levels; i++)
                {
                    Assert.Equal(s1.NodesPerDepth[i], s2.NodesPerDepth[i]);
                    Assert.Equal(BitConverter.DoubleToInt64Bits(s1.BestScoreHistory[i]), BitConverter.DoubleToInt64Bits(s2.BestScoreHistory[i]));
                    Assert.Equal(s1.BestInstanceHistory[i], s2.BestInstanceHistory[i]);
                    Assert.Equal(s1.BestCellHistory[i], s2.BestCellHistory[i]);
                    Assert.Equal(s1.LevelTtHitsTotal[i], s2.LevelTtHitsTotal[i]);
                    Assert.Equal(s1.LevelCutoffsTotal[i], s2.LevelCutoffsTotal[i]);
                }

                // once per completed level, in order, carrying the statistics of that level
                Assert.Equal(s2.Levels, seen.Count);
                for (int i = 0; i < seen.Count; i++)
                {
                    Assert.Equal(i + 1, seen[i].Depth);
                    Assert.Equal(BitConverter.DoubleToInt64Bits(s2.BestScoreHistory[i]), BitConverter.DoubleToInt64Bits(seen[i].Score));
                    Assert.Equal((int)s2.BestInstanceHistory[i], seen[i].BestInstance);
                    Assert.Equal((int)s2.BestCellHistory[i], seen[i].BestCell);
                    Assert.Equal(s2.LevelTtHitsTotal[i], seen[i].TtHitsTotal);
                    Assert.Equal(s2.LevelCutoffsTotal[i], seen[i].CutoffsTotal);
                    Assert.True(seen[i].NodesTotal >= (i == 0 ? 1 : seen[i - 1].NodesTotal));
                    Assert.True(seen[i].ElapsedTimestampTicks >= 0);
                }
                Assert.Equal(s2.NodesSearched, seen.Count == 0 ? 0 : seen[^1].NodesTotal);
                compared++;
            }
        Assert.True(compared >= 2 * ReferenceScenarios.Names.Length);
    }

    [Fact]
    public void A_time_limited_search_reports_only_completed_levels()
    {
        var g = ReferenceScenarios.Create(ReferenceScenarios.Names[0], SearchGolden.Db);
        var searcher = new FastSearcher(g, SearchGolden.Db, new FastTranspositionTable(1 << 16));
        var seen = new List<LevelProgress>();
        searcher.RunIterativeDeepening(40, false, SearchLimits.WithNodeBudget(5_000), onLevel: seen.Add);
        Assert.Equal(searcher.Stats.Levels, seen.Count);
        Assert.All(seen, p => Assert.True(p.Depth >= 1 && p.Depth <= searcher.Stats.DepthCompleted));
    }

    private static long _sink;
    private static readonly Action<LevelProgress> CachedCallback = p => _sink += p.Depth + p.NodesTotal;

    [Fact]
    public void The_search_with_a_cached_callback_still_allocates_nothing_in_steady_state()
    {
        var searchers = ReferenceScenarios.Names
            .Select(n => new FastSearcher(ReferenceScenarios.Create(n, SearchGolden.Db), SearchGolden.Db, new FastTranspositionTable(1 << 16)))
            .ToArray();
        foreach (var s in searchers) s.RunIterativeDeepening(5, false, SearchLimits.Unlimited, onLevel: CachedCallback); // warm-up
        long nodes = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int round = 0; round < 4; round++)
            foreach (var s in searchers)
            {
                s.RunIterativeDeepening(5, round % 2 == 1, SearchLimits.Unlimited, onLevel: CachedCallback);
                nodes += s.Stats.NodesSearched;
            }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        _output.WriteLine($"{nodes:N0} nodes, allocated {allocated} bytes, callback sink {_sink}");
        Assert.True(nodes > 50_000);
        Assert.Equal(0, allocated);
    }
}
