using System.Diagnostics;
using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>What the per-level callback of <see cref="FastSearcher.RunIterativeDeepening"/> reports after a completed level.</summary>
/// <param name="Depth">The completed depth (1-based).</param>
/// <param name="Score">The root score of this level.</param>
/// <param name="BestInstance">Card instance of the best move, -1 if the side to move has no move.</param>
/// <param name="BestCell">Row-major cell (0..8) of the best move.</param>
/// <param name="NodesTotal">Nodes searched so far, all levels.</param>
/// <param name="TtHitsTotal">Transposition-table hits so far.</param>
/// <param name="CutoffsTotal">Alpha-beta cutoffs so far.</param>
/// <param name="ElapsedTimestampTicks">Stopwatch ticks from the start of the search to the end of this level.</param>
public readonly record struct LevelProgress(int Depth, double Score, int BestInstance, int BestCell, long NodesTotal, long TtHitsTotal,
    long CutoffsTotal, long ElapsedTimestampTicks);

/// <summary>
/// Fast search (Phase 4): alpha-beta with an incremental Zobrist key and a bound-aware, fixed-size transposition
/// table. It is NOT a node-for-node port of Python. Its contract (CSHARP_MIGRATION.md, Phase 4 and Q-5 = option b):
/// <list type="bullet">
/// <item>At a fixed depth with no time limit, the root score is bit-identical to the exact depth-limited minimax value,
/// i.e. to <see cref="CompatSearcher"/> with its transposition table switched off (the oracle).</item>
/// <item>The root move is a move whose exact value equals the root score. It may differ from Python's choice only among
/// moves with exactly equal scores (move ordering decides ties; here the order uses the previous iteration's best move
/// first).</item>
/// <item>It is deterministic: same state, same depth, same table size, same limits give the same result.</item>
/// </list>
/// What is different from ReferenceCompat on purpose: bound types in the table (D-006), Zobrist keys instead of the
/// Python string key (D-005), the stored best move is tried first, entries of an interrupted search are never stored,
/// and iterative deepening stops once the whole game tree has been searched (D-014). What is the same: leaf evaluation
/// (<c>evaluate_state</c> also at the end of the game, D-007), move generation order, history-heuristic ordering with
/// the stable sort, fail-soft alpha-beta, and the rules (Make/Undo including D-001, D-003).
/// Hot path: no managed allocation in steady state.
/// </summary>
public sealed class FastSearcher
{
    /// <summary>Largest accepted search depth (table depths are stored as <see cref="sbyte"/>). A game has at most 9 plies.</summary>
    public const int MaxDepth = 100;

    private readonly GameState _g;
    private readonly Evaluator _eval;
    private readonly FastTranspositionTable _tt;

    private bool _aggressive;
    private long _nodeBudget = long.MaxValue;
    private bool _timed;
    private long _deadlineTimestamp;
    private bool _stopped;
    private int _rootInst = -1, _rootCell;

    public SearchStats Stats { get; } = new();
    public HistoryTable History { get; }
    public Evaluator Evaluator => _eval;
    public FastTranspositionTable Table => _tt;

    public FastSearcher(GameState state, CardDb db, FastTranspositionTable tt)
    {
        _g = state;
        _tt = tt;
        _eval = new Evaluator(state, db);
        History = new HistoryTable(state);
    }

    // ------------------------------------------------------------------ entry points

    /// <summary>
    /// Searches exactly <paramref name="depth"/> plies from the side to move (maximising, window -inf..+inf).
    /// <paramref name="clearTable"/> = true (default) starts from an empty table bound to this search's
    /// <see cref="SearchContext"/>. <paramref name="clearTable"/> = false continues on the entries of an earlier search
    /// and is only allowed if that search had the same context: the same <see cref="GameState"/> object, the same AI
    /// side (the side to move at the root) and the same evaluation mode (<paramref name="aggressive"/>). The root may
    /// be any position of that state (the caller can have played moves in between); scores are reused at equal depth
    /// only. Under any other context (another state, the AI side flipped, the other evaluation mode, or a table that
    /// was never used) it throws <see cref="InvalidOperationException"/> before anything is changed (review H4-1).
    /// </summary>
    public SearchResult RunFixedDepth(int depth, bool aggressive, SearchLimits limits, bool clearTable = true)
    {
        if ((uint)depth > MaxDepth) throw new ArgumentOutOfRangeException(nameof(depth), depth, $"depth must be 0..{MaxDepth}");
        Begin(limits, aggressive, clearTable);
        int rootUndo = _g.UndoDepth;
        try
        {
            double score = Node(depth, double.NegativeInfinity, double.PositiveInfinity, true, _g.CurrentPlayer, true);
            return new SearchResult(score, _rootInst, _rootCell);
        }
        finally
        {
            UnwindTo(rootUndo); // same exception-safety contract as CompatSearcher (review H-1)
        }
    }

    /// <summary>
    /// Iterative deepening from depth 1 to <paramref name="maxDepth"/> with one history table and one transposition
    /// table shared by all levels. An interrupted level (time or node budget) is discarded except for level 1, as in
    /// Python. The loop ends early when a level has searched the whole game tree (depth >= empty cells) or the root
    /// side has no legal move: deeper levels cannot change anything (D-014). Per-level data is left in <see cref="Stats"/>.
    /// <paramref name="clearTable"/> has the meaning and the context rule documented on <see cref="RunFixedDepth"/>:
    /// reuse needs the same GameState, the same AI side and the same evaluation mode, otherwise it throws
    /// <see cref="InvalidOperationException"/> without changing anything.
    /// <paramref name="onLevel"/> (Phase 8A, live progress): called ONCE per completed level, never per node, after the level's
    /// statistics are recorded. It observes; it cannot change the search. A null callback costs one comparison per level.
    /// </summary>
    public SearchResult RunIterativeDeepening(int maxDepth, bool aggressive, SearchLimits limits, bool clearTable = true,
        Action<LevelProgress>? onLevel = null)
    {
        if ((uint)maxDepth > MaxDepth) throw new ArgumentOutOfRangeException(nameof(maxDepth), maxDepth, $"maxDepth must be 0..{MaxDepth}");
        Begin(limits, aggressive, clearTable);
        Stats.StartTimestamp = Stopwatch.GetTimestamp();

        int bestInst = -1, bestCell = 0;
        double bestScore = 0;
        int empties = _g.AvailableCellCount;
        int rootUndo = _g.UndoDepth;
        try
        {
            for (int depth = 1; depth <= maxDepth && !TimeExpired(); depth++)
            {
                long nodesBefore = Stats.NodesSearched;
                double score = Node(depth, double.NegativeInfinity, double.PositiveInfinity, true, _g.CurrentPlayer, true);
                if (_stopped && depth > 1) break; // the unfinished level is discarded

                bestScore = score;
                bestInst = _rootInst;
                bestCell = _rootCell;

                int level = Stats.Levels++;
                Stats.DepthCompleted = depth;
                Stats.NodesPerDepth[level] = Stats.NodesSearched - nodesBefore;
                Stats.BestScoreHistory[level] = score;
                Stats.BestInstanceHistory[level] = (sbyte)_rootInst;
                Stats.BestCellHistory[level] = (byte)_rootCell;
                Stats.LevelEndTimestamp[level] = Stopwatch.GetTimestamp(); // statistics only: once per level
                Stats.LevelTtHitsTotal[level] = Stats.TtHits;
                Stats.LevelCutoffsTotal[level] = Stats.AlphaBetaCutoffs;
                onLevel?.Invoke(new LevelProgress(depth, score, _rootInst, _rootCell, Stats.NodesSearched, Stats.TtHits, Stats.AlphaBetaCutoffs,
                    Stats.LevelEndTimestamp[level] - Stats.StartTimestamp));

                // Nothing deeper can change the result once the whole tree is covered, or if the root side cannot move.
                if (_stopped || depth >= empties || _rootInst < 0) break;
            }
        }
        finally
        {
            UnwindTo(rootUndo);
        }
        return new SearchResult(bestScore, bestInst, bestCell);
    }

    private void Begin(SearchLimits limits, bool aggressive, bool clearTable)
    {
        // Review H4-1: the table is only valid for one (GameState, AI side, evaluation mode). Checked here, once per
        // search, before anything is modified, so a rejected call leaves state, statistics, history, table and clock alone.
        var context = new SearchContext(_g, _g.CurrentPlayer, aggressive ? EvalMode.HeuristicAggressive : EvalMode.Heuristic);
        if (clearTable) _tt.NewGeneration(context);
        else _tt.RequireContext(context);
        Stats.Reset();
        History.Clear();
        _aggressive = aggressive;
        _nodeBudget = limits.NodeBudget;
        _timed = !double.IsPositiveInfinity(limits.TimeLimitSeconds);
        if (_timed) _deadlineTimestamp = limits.DeadlineFrom(Stopwatch.GetTimestamp());
        _stopped = false;
        _rootInst = -1;
        _rootCell = 0;
    }

    private void UnwindTo(int undoDepth)
    {
        while (_g.UndoDepth > undoDepth) _g.UndoMove();
    }

    // ------------------------------------------------------------------ clock

    /// <summary>Full check (also reads the clock); used between iterative-deepening levels.</summary>
    private bool TimeExpired()
    {
        if (_stopped) return true;
        if (Stats.NodesSearched > _nodeBudget || (_timed && Stopwatch.GetTimestamp() > _deadlineTimestamp)) _stopped = true;
        return _stopped;
    }

    /// <summary>Per-node check. The wall clock is read only every 1024 nodes; the node budget (tests) every node.</summary>
    private bool CheckStop()
    {
        if (_stopped) return true;
        long n = Stats.NodesSearched;
        if (n > _nodeBudget || (_timed && (n & 1023) == 0 && Stopwatch.GetTimestamp() > _deadlineTimestamp)) _stopped = true;
        return _stopped;
    }

    // ------------------------------------------------------------------ alpha-beta

    private double Node(int depth, double alpha, double beta, bool maximizing, int aiPlayer, bool isRoot)
    {
        Stats.NodesSearched++;
        var g = _g;

        if (CheckStop()) return Leaf(aiPlayer);
        if (depth == 0 || g.IsGameOver) return Leaf(aiPlayer);

        ulong key = g.ZobristHash;
        int ttInst = -1, ttCell = 0;
        if (_tt.TryGet(key, out var hit))
        {
            Stats.TtHits++;
            ttInst = hit.BestInstance;
            ttCell = hit.BestCell;
            // The root is always searched (its best move is needed). Values are only reused at EQUAL depth, not at "at
            // least this depth": a position is reached at a single ply, so within one fixed-depth search (or one level of
            // iterative deepening) equal depth is the only case that occurs, and the root score stays exactly the
            // depth-limited minimax value even if the table still holds entries of an earlier, deeper search
            // (clearTable = false). Shallower and deeper entries still supply their best move for ordering.
            if (!isRoot && hit.Depth == depth)
            {
                double s = hit.Score;
                if (hit.Bound == Bound.Exact ||
                    (hit.Bound == Bound.Lower && s >= beta) ||
                    (hit.Bound == Bound.Upper && s <= alpha))
                {
                    Stats.TtCutoffs++;
                    return s;
                }
            }
        }

        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        if (n == 0) return Leaf(aiPlayer);

        // The stored move goes first (if it is legal here: it is verified, so a hash collision cannot inject a move).
        int first = 0;
        if (ttInst >= 0)
        {
            for (int t = 0; t < n; t++)
            {
                if (moves[t].Inst != ttInst || moves[t].Cell != ttCell) continue;
                Move m = moves[t];
                for (int j = t; j > 0; j--) moves[j] = moves[j - 1];
                moves[0] = m;
                first = 1;
                break;
            }
        }
        Span<double> scores = stackalloc double[GameState.MaxMoves];
        // Depth 1 (non-root): the children are leaves, so the order cannot change the node's value; sorting by
        // evaluate_move would only cost n evaluations. Generation order is used (the stored move is still first).
        if (depth > 1 || isRoot) OrderMoves(moves.Slice(first), scores, n - first);

        double alphaIn = alpha, betaIn = beta;
        int bestInst = -1, bestCell = 0;
        double best = maximizing ? double.NegativeInfinity : double.PositiveInfinity;

        for (int i = 0; i < n; i++)
        {
            if (_stopped) break;
            Stats.MoveEvaluations++;

            Move move = moves[i];
            g.MakeMove(move);
            double s = Node(depth - 1, alpha, beta, !maximizing, aiPlayer, false);
            g.UndoMove();

            if (maximizing)
            {
                if (s > best)
                {
                    best = s;
                    bestInst = move.Inst;
                    bestCell = move.Cell;
                }
                if (s > alpha) alpha = s;
            }
            else
            {
                if (s < best)
                {
                    best = s;
                    bestInst = move.Inst;
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

        if (isRoot)
        {
            _rootInst = bestInst;
            _rootCell = bestCell;
        }

        // An interrupted node holds a partial result: never store it (D-006 stored these as if exact).
        if (!_stopped)
        {
            Bound bound = best <= alphaIn ? Bound.Upper : best >= betaIn ? Bound.Lower : Bound.Exact;
            _tt.Store(key, depth, best, bound, bestInst, bestCell);
        }
        return best;
    }

    private double Leaf(int aiPlayer) => _eval.EvaluateState(aiPlayer, _aggressive);

    /// <summary>Stable sort by descending <c>evaluate_move</c> score (equal scores keep generation order).</summary>
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
