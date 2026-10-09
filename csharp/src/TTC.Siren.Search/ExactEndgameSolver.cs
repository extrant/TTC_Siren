using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// Port of <c>ai_server._solve_endgame_exact</c> (Phase 5): complete minimax to the end of the game, no depth
/// limit, no heuristic, no pruning. The value is the AI side's card count minus the opponent's (by <c>card.owner</c>,
/// board and both hands) when the board is full, or when the side to move has no playable card (秩序/混乱 can
/// filter every card out): that position is scored where it stands, as in Python. Positive is better for the AI.
///
/// Memoisation. Python memoises on a structural key (side, board, both hands in hand order, rules) that is shared
/// by all scenarios of one candidate move. That key is only an implementation detail: the value of a position is a
/// pure function of the position, so any sound memo gives the same value (verified against Python on every
/// golden position, and Python's own shared-cache values equal its fresh-cache values on all of them). Here the memo
/// is the Zobrist-keyed <see cref="FastTranspositionTable"/> (entries always Exact) under the context
/// (GameState object, AI player, <see cref="EvalMode.ExactCardDifference"/>): a solver is bound to ONE GameState, the
/// memo may be kept for any number of positions of that state (for example after each first move of a scenario), and a
/// different scenario GameState needs its own generation (<see cref="Bind"/>), which
/// <see cref="FastTranspositionTable.RequireContext"/> enforces. Hot path: no managed allocation in steady state.
/// </summary>
public sealed class ExactEndgameSolver
{
    private readonly GameState _g;
    private readonly FastTranspositionTable _memo;
    private int _ai;

    /// <summary>Positions visited (calls of the recursion, memo hits included).</summary>
    public long Nodes { get; private set; }

    /// <summary>Moves made (Python: calls of <c>make_move</c> in the exact solver).</summary>
    public long MakeMoves { get; private set; }

    public long MemoHits { get; private set; }

    public ExactEndgameSolver(GameState state, FastTranspositionTable memo)
    {
        _g = state;
        _memo = memo;
    }

    /// <summary>A memo table of a size that suits one scenario (a few thousand positions at most).</summary>
    public static FastTranspositionTable CreateMemo(int capacity = 1 << 15) => new(capacity);

    /// <summary>Starts a new, empty memo generation bound to (this state, <paramref name="aiPlayer"/>, exact mode).</summary>
    public void Bind(int aiPlayer)
    {
        _memo.NewGeneration(new SearchContext(_g, aiPlayer, EvalMode.ExactCardDifference));
        _ai = aiPlayer;
    }

    /// <summary>
    /// The exact value of the current position for <paramref name="aiPlayer"/>. The memo must have been bound with
    /// <see cref="Bind"/> for the same AI side (and this state); otherwise <see cref="InvalidOperationException"/>
    /// is thrown before anything is touched. On return the state is as before the call except for the documented
    /// make/undo side effects (hand order, D-001; display edges, D-003).
    /// </summary>
    public int Value(int aiPlayer)
    {
        _memo.RequireContext(new SearchContext(_g, aiPlayer, EvalMode.ExactCardDifference));
        _ai = aiPlayer;
        int rootUndo = _g.UndoDepth;
        try
        {
            return Node();
        }
        finally
        {
            while (_g.UndoDepth > rootUndo) _g.UndoMove();
        }
    }

    /// <summary>Python <c>_endgame_score</c>: card difference from the AI side's view.</summary>
    private int Score()
    {
        _g.CountCards(out int red, out int blue);
        return _ai == 0 ? red - blue : blue - red;
    }

    private int Node()
    {
        var g = _g;
        Nodes++;
        if (g.IsGameOver) return Score();

        ulong key = g.ZobristHash;
        if (_memo.TryGet(key, out var hit))
        {
            MemoHits++;
            return (int)hit.Score;
        }

        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        int depth = Math.Min(sbyte.MaxValue, g.AvailableCellCount);
        if (n == 0)
        {
            int stuck = Score(); // PY-QUIRK: no pass rule; the position is scored as it stands
            _memo.Store(key, depth, stuck, Bound.Exact, -1, 0);
            return stuck;
        }

        bool maximizing = g.CurrentPlayer == _ai;
        int best = maximizing ? int.MinValue : int.MaxValue;
        for (int i = 0; i < n; i++)
        {
            g.MakeMove(moves[i]);
            MakeMoves++;
            int v = Node();
            g.UndoMove();
            if (maximizing ? v > best : v < best) best = v;
        }
        _memo.Store(key, depth, best, Bound.Exact, -1, 0);
        return best;
    }
}
