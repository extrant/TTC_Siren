using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

public sealed class ExactEndgameSolverTests
{
    private readonly ITestOutputHelper _output;

    public ExactEndgameSolverTests(ITestOutputHelper output) => _output = output;

    /// <summary>Plain recursion with no memo at all: an oracle for the memoised solver that shares nothing with it.</summary>
    private static int Brute(GameState g, int ai, ref long nodes)
    {
        nodes++;
        static int Score(GameState s, int ai)
        {
            s.CountCards(out int red, out int blue);
            return ai == 0 ? red - blue : blue - red;
        }
        if (g.IsGameOver) return Score(g, ai);
        var moves = new Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        if (n == 0) return Score(g, ai);
        bool maximizing = g.CurrentPlayer == ai;
        int best = maximizing ? int.MinValue : int.MaxValue;
        for (int i = 0; i < n; i++)
        {
            g.MakeMove(moves[i]);
            int v = Brute(g, ai, ref nodes);
            g.UndoMove();
            if (maximizing ? v > best : v < best) best = v;
        }
        return best;
    }

    [Fact]
    public void Memoised_exact_solver_equals_memo_free_minimax()
    {
        int positions = 0, bothSides = 0;
        long bruteNodes = 0, solverNodes = 0;

        void Check(GameState g, string label)
        {
            for (int ai = 0; ai < 2; ai++)
            {
                long n = 0;
                int expected = Brute(g, ai, ref n);
                var solver = new ExactEndgameSolver(g, ExactEndgameSolver.CreateMemo());
                solver.Bind(ai);
                int actual = solver.Value(ai);
                Assert.True(expected == actual, $"{label} ai={ai}: memo-free {expected}, memoised {actual}");
                Assert.Equal(0, g.UndoDepth);
                bruteNodes += n;
                solverNodes += solver.Nodes;
                bothSides++;
            }
            positions++;
        }

        foreach (string name in new[] { "mid4_rev_ace", "end5_same_plus", "mid3_ascension" })
            Check(ReferenceScenarios.Create(name, SearchGolden.Db), name);

        int golden = 0;
        foreach (var doc in EndgameGolden.Records("exact"))
        {
            using (doc)
            {
                var g = EndgameGolden.State(doc.RootElement);
                if (g.AvailableCellCount > 5) continue;
                Check(g, $"exact golden record {doc.RootElement.GetProperty("i").GetInt32()}");
                if (++golden >= 250) break;
            }
        }
        _output.WriteLine($"positions={positions}, ai-sides={bothSides}, memo-free nodes={bruteNodes:N0}, memoised nodes={solverNodes:N0}");
        Assert.True(positions >= 200);
    }

    [Fact]
    public void Exact_memo_is_bound_to_state_ai_side_and_its_own_evaluation_mode()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("end5_same_plus", db);
        var table = ExactEndgameSolver.CreateMemo();
        var solver = new ExactEndgameSolver(g, table);

        // Not bound yet.
        string before = SearchGolden.NormalizedState(g);
        Assert.Contains("not bound", Assert.Throws<InvalidOperationException>(() => solver.Value(0)).Message);

        solver.Bind(0);
        Assert.Equal(EvalMode.ExactCardDifference, table.Context.Mode);
        solver.Value(0);
        // The other AI side would reuse values of the wrong perspective.
        Assert.Contains("AI side", Assert.Throws<InvalidOperationException>(() => solver.Value(1)).Message);

        // A second state (a different scenario) cannot continue on this generation.
        var other = ReferenceScenarios.Create("end5_same_plus", db);
        var otherSolver = new ExactEndgameSolver(other, table);
        Assert.Contains("different GameState", Assert.Throws<InvalidOperationException>(() => otherSolver.Value(0)).Message);
        otherSolver.Bind(0); // a new generation for the new state is fine ...
        Assert.True(ReferenceEquals(table.Context.State, other));
        Assert.Contains("different GameState", Assert.Throws<InvalidOperationException>(() => solver.Value(0)).Message); // ... and the first state lost it

        // A heuristic search generation can never be mistaken for an exact one, and vice versa.
        var fast = new FastSearcher(g, db, table);
        fast.RunFixedDepth(2, false, SearchLimits.Unlimited); // rebinds the table to (g, side, Heuristic)
        Assert.Contains("evaluation mode", Assert.Throws<InvalidOperationException>(() => solver.Value(g.CurrentPlayer)).Message);
        solver.Bind(g.CurrentPlayer);
        Assert.Contains("evaluation mode", Assert.Throws<InvalidOperationException>(() => fast.RunFixedDepth(2, false, SearchLimits.Unlimited, clearTable: false)).Message);

        Assert.Equal(before, SearchGolden.NormalizedState(g)); // none of this changed the position (hand order and display edges aside)
        Assert.Equal(0, g.UndoDepth);
    }

    [Fact]
    public void Exact_solver_hot_path_allocates_nothing_in_steady_state()
    {
        var db = SearchGolden.Db;
        var states = new[] { "mid4_rev_ace", "end5_same_plus", "mid3_ascension" }.Select(n => ReferenceScenarios.Create(n, db)).ToArray();
        var solvers = states.Select(s => new ExactEndgameSolver(s, ExactEndgameSolver.CreateMemo())).ToArray();

        for (int i = 0; i < solvers.Length; i++) // warm-up
            for (int ai = 0; ai < 2; ai++)
            {
                solvers[i].Bind(ai);
                solvers[i].Value(ai);
            }

        long nodes = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int round = 0; round < 30; round++)
        {
            for (int i = 0; i < solvers.Length; i++)
            {
                int ai = round & 1;
                long n0 = solvers[i].Nodes;
                solvers[i].Bind(ai);
                solvers[i].Value(ai);
                nodes += solvers[i].Nodes - n0;
            }
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        _output.WriteLine($"{nodes:N0} exact-solver nodes, allocated {allocated} bytes");
        Assert.True(nodes > 100_000);
        Assert.Equal(0, allocated);
    }
}
