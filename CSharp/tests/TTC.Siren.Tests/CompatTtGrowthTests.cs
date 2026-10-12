using System.Diagnostics;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Review finding H-2: the compat table must grow on demand without changing any result.</summary>
public sealed class CompatTtGrowthTests
{
    private readonly ITestOutputHelper _output;

    public CompatTtGrowthTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Compat_TT_grows_without_changing_results()
    {
        const int minPositions = 50;
        string path = GoldenFiles.EnsureSearchGolden("search");
        int benchItems = 0, positions = 0, runs = 0, runsThatGrew = 0;
        long totalGrows = 0, nodes = 0;

        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                bool isBench = rec.TryGetProperty("name", out _);
                if (!isBench && positions >= minPositions) break; // bench items come first, then the positions in file order
                bool aggressive = rec.GetProperty("aggr").GetBoolean();

                foreach (var run in rec.GetProperty("runs").EnumerateArray())
                {
                    var g = SearchGolden.BuildState(rec, out string label);
                    int depth = run.GetProperty("d").GetInt32();
                    string ctx = $"{label} depth={depth} aggressive={aggressive} initialCapacity=64";

                    var tt = new CompatTranspositionTable(1 << 6); // fresh tiny table per run: grows many times
                    var searcher = new CompatSearcher(g, SearchGolden.Db, tt);
                    var result = searcher.RunFixedDepth(depth, aggressive, SearchLimits.Unlimited);
                    var s = searcher.Stats;

                    SearchGolden.AssertSameDouble(run.GetProperty("score").GetString()!, result.Score, ctx + " root score");
                    Assert.Equal(SearchGolden.MoveText(run.GetProperty("move")), SearchGolden.MoveText(g, result.BestInstance, result.BestCell));
                    Assert.True(run.GetProperty("nodes").GetInt64() == s.NodesSearched, $"{ctx} nodes: expected {run.GetProperty("nodes").GetInt64()} got {s.NodesSearched}");
                    Assert.True(run.GetProperty("cut").GetInt64() == s.AlphaBetaCutoffs, $"{ctx} cutoffs");
                    Assert.True(run.GetProperty("tt").GetInt64() == s.TtHits, $"{ctx} tt_hits");
                    Assert.True(run.GetProperty("ttc").GetInt64() == s.TtCutoffs, $"{ctx} tt_cutoffs");
                    Assert.True(run.GetProperty("me").GetInt64() == s.MoveEvaluations, $"{ctx} move_evaluations");
                    Assert.Equal(SearchGolden.HistoryText(run.GetProperty("hist")), SearchGolden.HistoryText(searcher.History));

                    // A table of 64 slots holds at most 48 entries: any run with more stores than that must have grown.
                    if (tt.PeakUsed > 48) Assert.True(tt.Grows > 0, $"{ctx}: {tt.PeakUsed} entries but no growth");
                    if (tt.Grows > 0) runsThatGrew++;
                    totalGrows += tt.Grows;
                    nodes += s.NodesSearched;
                    runs++;
                }
                if (isBench) benchItems += rec.GetProperty("runs").GetArrayLength(); else positions++;
            }
        }

        _output.WriteLine($"bench items={benchItems} positions={positions} runs={runs} (grew in {runsThatGrew}) total grows={totalGrows} nodes={nodes:N0}");
        Assert.Equal(22, benchItems);
        Assert.Equal(minPositions, positions);
        Assert.True(totalGrows > 0 && runsThatGrew > 20, $"growth insufficiently exercised: {runsThatGrew} runs grew");
    }

    [Fact]
    public void Compat_TT_one_shared_table_keeps_results_across_generations_and_growth()
    {
        // The same table object is reused for many searches (NewGeneration between them) while it grows in between.
        var db = SearchGolden.Db;
        var tt = new CompatTranspositionTable(1 << 6);
        long[] reference = new long[6];
        for (int round = 0; round < 2; round++)
        {
            for (int k = 0; k < ReferenceScenarios.Names.Length; k++)
            {
                var g = ReferenceScenarios.Create(ReferenceScenarios.Names[k], db);
                var searcher = new CompatSearcher(g, db, tt);
                var r = searcher.RunFixedDepth(3, false, SearchLimits.Unlimited);
                long bits = BitConverter.DoubleToInt64Bits(r.Score) ^ (searcher.Stats.NodesSearched << 40) ^ searcher.Stats.TtHits;
                if (round == 0) reference[k] = bits;
                else Assert.True(reference[k] == bits, $"scenario {ReferenceScenarios.Names[k]}: result changed between rounds (capacity now {tt.Capacity})");
            }
        }
        Assert.True(tt.Grows > 0 && tt.Capacity > 64);
    }

    [Fact]
    public void Compat_TT_hard_cap_throws_dedicated_exception_and_state_is_restored()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        string before = SearchGolden.NormalizedState(g);
        var tt = new CompatTranspositionTable(64, maxCapacity: 256); // can hold at most 192 entries
        var searcher = new CompatSearcher(g, db, tt);

        var ex = Assert.Throws<TranspositionTableFullException>(() => searcher.RunFixedDepth(4, false, SearchLimits.Unlimited));

        Assert.Equal(256, ex.MaxCapacity);
        Assert.Equal(192, ex.Entries);
        Assert.Contains("256", ex.Message);
        Assert.Contains("192", ex.Message);
        Assert.Equal(256, tt.Capacity);
        Assert.True(tt.Grows == 2, $"grows={tt.Grows}");
        Assert.Equal(0, g.UndoDepth);
        Assert.True(before == SearchGolden.NormalizedState(g), "state not restored after the hard cap was reached");
    }

    [Fact]
    public void Server_equivalent_iterative_deepening_completes()
    {
        // find_best_move_parallel(max_depth=10, max_time=10) as called by ai_server. Wall-clock dependent, so the
        // assertions are behavioural; the measurements are printed (and recorded in CSHARP_MIGRATION.md section 10).
        var db = SearchGolden.Db;
        foreach (string name in new[] { "open_none", "open_same_plus", "mid3_ascension", "open_all_rules" })
        {
            var tt = new CompatTranspositionTable(); // fresh table per scenario: per-scenario growth and memory
            var g = ReferenceScenarios.Create(name, db);
            string before = SearchGolden.NormalizedState(g);
            var searcher = new CompatSearcher(g, db, tt);

            long t0 = Stopwatch.GetTimestamp();
            var result = searcher.RunIterativeDeepening(10, false, SearchLimits.WithTime(10));
            double seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            var s = searcher.Stats;

            _output.WriteLine($"{name,-16} levels={s.Levels} nodes={s.NodesSearched:N0} ttEntries={tt.Used:N0} grows={tt.Grows} " +
                              $"capacity={tt.Capacity:N0} ttMemory={tt.MemoryBytes / (1024.0 * 1024.0):F0} MiB wall={seconds:F2}s");

            Assert.Equal(0, g.UndoDepth);
            Assert.True(before == SearchGolden.NormalizedState(g), $"{name}: state not restored");
            Assert.True(result.HasMove, $"{name}: no move");
            Assert.Equal(s.Levels, s.DepthCompleted);
            Assert.True(s.Levels >= 3, $"{name}: only {s.Levels} levels completed in 10 s");
            Assert.True(seconds < 20, $"{name}: {seconds:F1}s for a 10 s limit");
            Assert.True(tt.MemoryBytes <= 1L << 30, $"{name}: TT needs {tt.MemoryBytes >> 20} MiB (> 1 GiB): stop and ask the user (review H-2)");
        }
    }
}
