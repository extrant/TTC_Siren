using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// How a leaf (and therefore every stored score) is valued. A transposition-table entry is only meaningful under the
/// evaluation mode that produced it, so the mode is part of the <see cref="SearchContext"/>. A future exact
/// endgame scoring (card difference) must add its own value here instead of reusing a heuristic one.
/// </summary>
public enum EvalMode : byte
{
    /// <summary><c>evaluate_state(..., aggressive=False)</c>.</summary>
    Heuristic = 0,

    /// <summary><c>evaluate_state(..., aggressive=True)</c>.</summary>
    HeuristicAggressive = 1,
}

/// <summary>
/// The conditions under which Fast transposition-table entries are valid (review H4-1). The Zobrist key only
/// separates positions inside ONE <see cref="GameState"/> (instance indices are reused between states, and card
/// contents, rules and <c>can_use</c> are not hashed), and stored scores are from the fixed AI perspective under one
/// evaluation mode. Entries may therefore be reused only if all three components are equal:
/// the same <see cref="GameState"/> object (its cards, rules and can_use never change during its lifetime), the same
/// AI player (the side to move at the root of the search that stored them) and the same <see cref="EvalMode"/>.
/// The root position itself does not have to be the same: any position reachable in the same state is fine.
/// </summary>
public readonly struct SearchContext
{
    public readonly GameState State;
    public readonly int AiPlayer;
    public readonly EvalMode Mode;

    public SearchContext(GameState state, int aiPlayer, EvalMode mode)
    {
        State = state;
        AiPlayer = aiPlayer;
        Mode = mode;
    }

    /// <summary>
    /// Null if <paramref name="other"/> is the same context; otherwise a sentence naming the first component that
    /// differs (state, AI side, evaluation mode).
    /// </summary>
    public string? DescribeMismatch(in SearchContext other)
    {
        if (!ReferenceEquals(State, other.State))
            return "the table was filled for a different GameState object (card contents, rules and can_use are not part of the hash)";
        if (AiPlayer != other.AiPlayer)
            return $"the table was filled for AI side {AiPlayer} but this search is for AI side {other.AiPlayer} (stored scores are from a fixed AI perspective)";
        if (Mode != other.Mode)
            return $"the table was filled under evaluation mode {Mode} but this search uses {other.Mode}";
        return null;
    }
}
