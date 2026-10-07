using System.Diagnostics;
using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>Python <c>SEARCH_STATS</c> (the fields that influence or describe control flow).</summary>
public sealed class SearchStats
{
    public const int MaxLevels = 128;

    public long NodesSearched;
    public long TtHits;
    public long TtCutoffs;
    public long AlphaBetaCutoffs;
    public long MoveEvaluations;
    public int DepthCompleted;

    // Per completed iterative-deepening level (Python: nodes_per_depth, best_score_history, best_move_history).
    public int Levels;
    public readonly long[] NodesPerDepth = new long[MaxLevels];
    public readonly double[] BestScoreHistory = new double[MaxLevels];
    public readonly sbyte[] BestInstanceHistory = new sbyte[MaxLevels]; // -1 = None
    public readonly byte[] BestCellHistory = new byte[MaxLevels];

    public void Reset()
    {
        NodesSearched = TtHits = TtCutoffs = AlphaBetaCutoffs = MoveEvaluations = 0;
        DepthCompleted = 0;
        Levels = 0;
    }
}

/// <summary>Stop conditions. Both default to "never": the fixed-depth mode required for deterministic parity tests.</summary>
public readonly struct SearchLimits
{
    /// <summary>Wall-clock limit in seconds (Python <c>TIME_LIMIT</c>); infinity disables the clock.</summary>
    public readonly double TimeLimitSeconds;

    /// <summary>
    /// Deterministic stand-in for the clock, used by tests: time is "up" once <c>NodesSearched &gt; NodeBudget</c>
    /// (mirrors the patched Python predicate <c>SEARCH_STATS.nodes_searched &gt; N</c>).
    /// </summary>
    public readonly long NodeBudget;

    public SearchLimits(double timeLimitSeconds, long nodeBudget = long.MaxValue)
    {
        TimeLimitSeconds = timeLimitSeconds;
        NodeBudget = nodeBudget;
    }

    public static SearchLimits Unlimited => new(double.PositiveInfinity);
    public static SearchLimits WithTime(double seconds) => new(seconds);
    public static SearchLimits WithNodeBudget(long nodes) => new(double.PositiveInfinity, nodes);
}

/// <summary>Result of one root search. <c>BestInstance</c> is -1 when Python's best_move would be <c>None</c>.</summary>
public readonly struct SearchResult
{
    public readonly double Score;
    public readonly int BestInstance;
    public readonly int BestCell;

    public SearchResult(double score, int bestInstance, int bestCell)
    {
        Score = score;
        BestInstance = bestInstance;
        BestCell = bestCell;
    }

    public bool HasMove => BestInstance >= 0;
}

/// <summary>
/// ReferenceCompat search: a node-for-node port of <c>ai.minimax</c> and <c>iterative_deepening_search</c>.
/// The same control flow (clock checks, TT probe/store timing, history updates, fail-hard alpha-beta, no bound types)
/// is reproduced so that scores, best moves, node counts and all counters equal Python's (D-005, D-006, D-007, D-014
/// are emulated on purpose). Not for production strength: the Fast mode of Phase 4 will not inherit these quirks.
/// Hot path: GenerateMoves -> order -> MakeMove -> recursion -> UndoMove with no managed allocation.
/// </summary>
public sealed class CompatSearcher
{
    private readonly GameState _g;
    private readonly Evaluator _eval;
    private readonly CompatTranspositionTable _tt;

    private bool _aggressive;
    private long _nodeBudget = long.MaxValue;
    private bool _timed;
    private long _deadlineTimestamp;

    /// <summary>
    /// Phase 4 oracle switch. <c>true</c> (default) is Python's behaviour. <c>false</c> disables the transposition
    /// table completely (no probe, no store; D-005/D-006 are then irrelevant): the search becomes plain stable-ordered
    /// alpha-beta whose root score is the exact depth-limited minimax value, the reference the Fast mode is compared
    /// against. Everything else (ordering, history, clock, statistics) is unchanged.
    /// </summary>
    public bool UseTranspositionTable { get; set; } = true;

    public SearchStats Stats { get; } = new();
    public HistoryTable History { get; }
    public Evaluator Evaluator => _eval;

    public CompatSearcher(GameState state, CardDb db, CompatTranspositionTable tt)
    {
        _g = state;
        _tt = tt;
        _eval = new Evaluator(state, db);
        History = new HistoryTable(state);
    }

    // ------------------------------------------------------------------ entry points

    /// <summary>
    /// A direct root call of Python <c>minimax(state, depth, -inf, +inf, True, current_player, ..., is_root=True, {})</c>
    /// with a fresh transposition table, fresh statistics and an empty history table.
    /// </summary>
    public SearchResult RunFixedDepth(int depth, bool aggressive, SearchLimits limits) =>
        RunFixedDepthAs(depth, true, _g.CurrentPlayer, aggressive, limits);

    /// <summary>
    /// The same direct call with an explicit node role: <paramref name="maximizing"/> and the AI perspective
    /// <paramref name="aiPlayer"/> are passed instead of "side to move, maximising". Used (with the table off) to
    /// compute the exact full-window value of each child of a root, i.e. the tie set of the root moves (Phase 4, Q-5).
    /// <c>RunFixedDepthAs(d, true, state.CurrentPlayer, ...)</c> is exactly <see cref="RunFixedDepth"/>.
    /// </summary>
    public SearchResult RunFixedDepthAs(int depth, bool maximizing, int aiPlayer, bool aggressive, SearchLimits limits)
    {
        _tt.NewGeneration();
        Stats.Reset();
        History.Clear();
        StartClock(limits, aggressive);
        int rootUndo = _g.UndoDepth;
        try
        {
            return Node(depth, double.NegativeInfinity, double.PositiveInfinity, maximizing, aiPlayer);
        }
        finally
        {
            UnwindTo(rootUndo); // review H-1: an exception escaping the recursion must not leave moves applied
        }
    }

    /// <summary>
    /// Python <c>find_best_move_parallel</c> / <c>iterative_deepening_search</c>: depth 1..maxDepth with one shared
    /// history table and one transposition table that is only cleared at the start. Per-level data is left in
    /// <see cref="Stats"/>; the returned result is the last accepted level's best move.
    /// </summary>
    public SearchResult RunIterativeDeepening(int maxDepth, bool aggressive, SearchLimits limits)
    {
        _tt.NewGeneration();
        Stats.Reset();
        History.Clear();
        StartClock(limits, aggressive);

        int bestInst = -1, bestCell = 0;
        double bestScore = 0;
        int depth = 1;
        int rootUndo = _g.UndoDepth;
        try
        {
            while (!IsTimeUp() && depth <= maxDepth)
            {
                long nodesBefore = Stats.NodesSearched;
                var result = Node(depth, double.NegativeInfinity, double.PositiveInfinity, true, _g.CurrentPlayer);
                long nodesThisDepth = Stats.NodesSearched - nodesBefore;

                if (IsTimeUp() && depth > 1) break; // the unfinished iteration is discarded

                bestInst = result.BestInstance;
                bestCell = result.BestCell;
                bestScore = result.Score;

                int level = Stats.Levels++;
                Stats.DepthCompleted = depth;
                Stats.NodesPerDepth[level] = nodesThisDepth;
                Stats.BestScoreHistory[level] = result.Score;
                Stats.BestInstanceHistory[level] = (sbyte)result.BestInstance;
                Stats.BestCellHistory[level] = (byte)result.BestCell;

                depth++;
                if (depth > maxDepth) break;
                // PY-QUIRK(D-014): Python's "predict next depth time" block only prints; it never stops the loop, and the
                // loop keeps deepening past the number of empty cells.
            }
        }
        finally
        {
            UnwindTo(rootUndo); // review H-1
        }
        return new SearchResult(bestScore, bestInst, bestCell);
    }

    /// <summary>
    /// Review H-1. Python wraps every child in <c>try/finally: undo_move</c>; here the unwind happens once at the entry
    /// point (the per-node path stays untouched). Undoing in LIFO order performs exactly the mutations a normal
    /// recursive return would have performed, including the D-001 hand appends, so after an exception the state is
    /// the same as after an early return. The exception itself keeps propagating.
    /// </summary>
    private void UnwindTo(int undoDepth)
    {
        while (_g.UndoDepth > undoDepth) _g.UndoMove();
    }

    // ------------------------------------------------------------------ clock

    private void StartClock(SearchLimits limits, bool aggressive)
    {
        _aggressive = aggressive;
        _nodeBudget = limits.NodeBudget;
        _timed = !double.IsPositiveInfinity(limits.TimeLimitSeconds);
        if (_timed)
            _deadlineTimestamp = Stopwatch.GetTimestamp() + (long)(limits.TimeLimitSeconds * Stopwatch.Frequency);
    }

    private bool IsTimeUp() =>
        Stats.NodesSearched > _nodeBudget || (_timed && Stopwatch.GetTimestamp() > _deadlineTimestamp);

    // ------------------------------------------------------------------ minimax

    private SearchResult Node(int depth, double alpha, double beta, bool maximizing, int aiPlayer)
    {
        Stats.NodesSearched++;
        var g = _g;

        if (IsTimeUp()) return Leaf(aiPlayer);
        if (depth == 0 || g.IsGameOver) return Leaf(aiPlayer);

        ulong h1 = 0, h2 = 0;
        bool useTt = UseTranspositionTable;
        if (useTt)
        {
            PythonStateKey.Compute(g, out h1, out h2);
            if (_tt.TryGet(h1, h2, out var hit))
            {
                Stats.TtHits++;
                if (hit.Depth >= depth)
                {
                    Stats.TtCutoffs++;
                    return new SearchResult(hit.Score, hit.BestInstance, hit.BestCell);
                }
            }
        }

        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        if (n == 0) return Leaf(aiPlayer);

        Span<double> scores = stackalloc double[GameState.MaxMoves];
        OrderMoves(moves, scores, n);

        int bestInst = -1, bestCell = 0;
        double best = maximizing ? double.NegativeInfinity : double.PositiveInfinity;

        for (int i = 0; i < n; i++)
        {
            if (IsTimeUp()) break;
            Stats.MoveEvaluations++;

            Move move = moves[i]; // names a card instance: valid however D-001 has reordered the hand since generation
            int inst = move.Inst;
            g.MakeMove(move);
            var child = Node(depth - 1, alpha, beta, !maximizing, aiPlayer);
            g.UndoMove();

            double s = child.Score;
            if (maximizing)
            {
                if (s > best)
                {
                    best = s;
                    bestInst = inst;
                    bestCell = move.Cell;
                }
                if (s > alpha) alpha = s;
            }
            else
            {
                if (s < best)
                {
                    best = s;
                    bestInst = inst;
                    bestCell = move.Cell;
                }
                if (s < beta) beta = s;
            }

            if (beta <= alpha)
            {
                Stats.AlphaBetaCutoffs++;
                if (bestInst >= 0) History.Add(bestInst, bestCell, depth);
                break;
            }
        }

        // Written on every exit of the loop, including time-up and cutoffs, and always overwriting (D-006).
        if (useTt) _tt.Store(h1, h2, depth, best, bestInst, bestCell);
        return new SearchResult(best, bestInst, bestCell);
    }

    private SearchResult Leaf(int aiPlayer) => new(_eval.EvaluateState(aiPlayer, _aggressive), -1, 0);

    /// <summary>
    /// Python <c>order_moves</c>: <c>sorted(..., key=evaluate_move, reverse=True)</c>. Scores are computed once in
    /// generation order and the sort is stable (equal scores keep generation order), done as an insertion sort.
    /// </summary>
    private void OrderMoves(Span<Move> moves, Span<double> scores, int n)
    {
        for (int i = 0; i < n; i++) scores[i] = _eval.EvaluateMove(moves[i], History, _aggressive);
        for (int i = 1; i < n; i++)
        {
            double key = scores[i];
            Move move = moves[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < key)
            {
                scores[j + 1] = scores[j];
                moves[j + 1] = moves[j];
                j--;
            }
            scores[j + 1] = key;
            moves[j + 1] = move;
        }
    }
}
