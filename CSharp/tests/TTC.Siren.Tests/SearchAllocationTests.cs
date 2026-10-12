using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

public sealed class SearchAllocationTests
{
    private readonly ITestOutputHelper _output;

    public SearchAllocationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Search_hot_path_allocates_nothing_in_steady_state()
    {
        var db = SearchGolden.Db;
        var tt = new CompatTranspositionTable();
        var searchers = ReferenceScenarios.Names
            .Select(name => new CompatSearcher(ReferenceScenarios.Create(name, db), db, tt))
            .ToArray();

        // Warm up every code path (JIT tiering, stackalloc frames) before measuring.
        foreach (var s in searchers)
        {
            s.RunFixedDepth(3, false, SearchLimits.Unlimited);
            s.RunFixedDepth(3, true, SearchLimits.Unlimited);
            s.RunIterativeDeepening(4, false, SearchLimits.Unlimited);
        }

        long nodes = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int round = 0; round < 5; round++)
        {
            foreach (var s in searchers)
            {
                s.RunFixedDepth(3, round % 2 == 1, SearchLimits.Unlimited);
                nodes += s.Stats.NodesSearched;
                s.RunIterativeDeepening(4, false, SearchLimits.WithNodeBudget(20_000));
                nodes += s.Stats.NodesSearched;
            }
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        _output.WriteLine($"{nodes:N0} nodes searched, allocated {allocated} bytes");
        Assert.True(nodes > 100_000);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void Evaluation_and_ordering_inputs_allocate_nothing()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("mid4_rev_ace", db);
        var eval = new Evaluator(g, db);
        var history = new HistoryTable(g);
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);

        double sink = 0;
        for (int i = 0; i < 2_000; i++) // warm-up
            for (int j = 0; j < n; j++) sink += eval.EvaluateMove(moves[j], history, i % 2 == 0) + eval.EvaluateState(i & 1, i % 2 == 0);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20_000; i++)
            for (int j = 0; j < n; j++) sink += eval.EvaluateMove(moves[j], history, i % 2 == 0) + eval.EvaluateState(i & 1, i % 2 == 0);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.NotEqual(double.NaN, sink);
        Assert.Equal(0, allocated);
    }
}
