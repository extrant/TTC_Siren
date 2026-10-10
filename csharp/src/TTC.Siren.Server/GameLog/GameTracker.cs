using TTC.Siren.Core;
using TTC.Siren.Explain;
using TTC.Siren.Search;

namespace TTC.Siren.Server.GameLog;

/// <summary>What the tracker keeps of one request to relate it to the next one of the same game.</summary>
/// <param name="State">The request state before the search (hands in request order).</param>
/// <param name="MyOwner">"red" or "blue": the plugin user's colour.</param>
/// <param name="Move">The move the server recommended (slot, row, col), null when it found none.</param>
public sealed record RequestInfo(StateSnapshot State, string MyOwner, (int Slot, int Row, int Col)? Move, DateTimeOffset At, bool[][] Generated, int RequestNo);

/// <summary>One flip of the inferred opponent move, with the cause recomputed from the rules (null when the position is inconsistent with the rules).</summary>
public sealed record InferredFlip(int Row, int Col, int CardId, string? Cause);

/// <summary>The opponent's move reconstructed from two consecutive requests of a game.</summary>
/// <param name="Consistent">True when the board of the new request equals the board the rules predict after the recommended/played move and the opponent's move.</param>
/// <param name="MyCell">The cell the plugin user played in between (-1 when it cannot be told).</param>
/// <param name="OpponentCell">The cell of the opponent's card (-1 when no single new card could be identified).</param>
/// <param name="AsRecommended">True when the user played the move the server recommended.</param>
public sealed record OpponentInference(bool Consistent, int MyCell, int OpponentCell, int OpponentCardId, int[] OpponentEdges, string OpponentOwner,
    List<InferredFlip> Flips, bool AsRecommended, string? Note, List<int> NewCells);

/// <summary>
/// Game identity and opponent-move inference (decision A-21, goal section C2). Pure functions on states: the same two requests always give the
/// same answer, which is what the replay and the tests rely on.
/// </summary>
public static class GameTracker
{
    /// <summary>Why a request does or does not continue a game; <see cref="Reason"/> is recorded in the log.</summary>
    public sealed record SuccessorResult(bool IsSuccessor, string Reason, int NewCells);

    /// <summary>
    /// Whether <paramref name="cur"/> can be the next request of the game that <paramref name="prev"/> belongs to: same rules and colour, the same side
    /// to move, within the allowed gap, every card of the old board still on its cell (owners may have changed), at most four new cards, and no card
    /// in the mover's known hand that was not in it before.
    /// </summary>
    public static SuccessorResult CheckSuccessor(RequestInfo prev, RequestInfo cur, TimeSpan maxGap)
    {
        if (prev.State.Rules != cur.State.Rules) return new(false, "rules differ", 0);
        if (prev.MyOwner != cur.MyOwner) return new(false, "colour differs", 0);
        if (cur.State.Current != prev.State.Current) return new(false, "another side is to move", 0);
        var gap = cur.At - prev.At;
        if (gap < TimeSpan.Zero || gap > maxGap) return new(false, "gap beyond the limit", 0);

        int newCells = 0;
        for (int c = 0; c < 9; c++)
        {
            var before = prev.State.Board[c];
            var now = cur.State.Board[c];
            if (before is not null)
            {
                if (now is null) return new(false, "a card left the board", 0);
                if (before.Id != now.Id || !before.Base.SequenceEqual(now.Base)) return new(false, "a card on the board changed", 0);
            }
            else if (now is not null) newCells++;
        }
        if (newCells > 4) return new(false, "too many new cards", newCells);

        // the mover's known hand can only shrink
        int mover = prev.State.Current;
        var oldIds = KnownIds(prev.State.Hands[mover], prev.Generated[mover]);
        foreach (int id in KnownIds(cur.State.Hands[mover], cur.Generated[mover]))
            if (!oldIds.Remove(id)) return new(false, "a card appeared in the hand", newCells);
        return new(true, newCells == 0 ? "same position (repeated request)" : "consistent successor", newCells);
    }

    private static List<int> KnownIds(IReadOnlyList<CardSnapshot> hand, bool[] generated)
    {
        var ids = new List<int>();
        for (int i = 0; i < hand.Count; i++)
            if (!(i < generated.Length && generated[i]) && hand[i].Id > 0 && hand[i].Id < 1000) ids.Add(hand[i].Id);
        return ids;
    }

    /// <summary>
    /// Reconstructs what happened between two requests of one game: the card the plugin user played, the card the opponent played, and the flips of
    /// the opponent's move with their causes, all recomputed by the rules and checked against the board the new request reports.
    /// </summary>
    public static OpponentInference? InferOpponentMove(CardDb db, RequestInfo prev, RequestInfo cur)
    {
        var newCells = Enumerable.Range(0, 9).Where(c => prev.State.Board[c] is null && cur.State.Board[c] is not null).ToList();
        if (newCells.Count == 0) return null;

        int mover = prev.State.Current, opponent = 1 - mover;
        string opponentOwner = opponent == 0 ? "red" : "blue";
        var myHandIds = KnownIds(prev.State.Hands[mover], prev.Generated[mover]);

        // which new card is the user's: the one whose id was in the user's known hand (the recommendation first)
        int myCell = -1, oppCell = -1;
        string? note = null;
        bool asRecommended = false;
        var mine = newCells.Where(c => myHandIds.Contains(cur.State.Board[c]!.Id)).ToList();
        if (prev.Move is { } rec && mine.Contains(rec.Row * 3 + rec.Col) && cur.State.Board[rec.Row * 3 + rec.Col]!.Id == prev.State.Hands[mover][rec.Slot].Id)
        {
            myCell = rec.Row * 3 + rec.Col;
            asRecommended = true;
        }
        else if (mine.Count == 1) myCell = mine[0];
        else if (mine.Count > 1) note = "several new cards could be the user's";
        var others = newCells.Where(c => c != myCell).ToList();

        if (myCell >= 0 && others.Count == 1) oppCell = others[0];
        else if (myCell < 0 && newCells.Count == 1) oppCell = newCells[0];
        else if (note is null) note = newCells.Count > 2 ? "more than two new cards: requests were missed" : "the new cards cannot be assigned to the two sides";

        if (oppCell < 0)
            return new OpponentInference(false, myCell, -1, 0, Array.Empty<int>(), opponentOwner, new List<InferredFlip>(), asRecommended, note, newCells);

        var oppCard = cur.State.Board[oppCell]!;
        var edges = new[] { (int)oppCard.Base[0], oppCard.Base[1], oppCard.Base[2], oppCard.Base[3] };

        // replay the user's move on the old state, then let the opponent's card be placed
        var s = GameState.FromSnapshot(prev.State);
        if (myCell >= 0)
        {
            int myId = cur.State.Board[myCell]!.Id;
            int inst = -1;
            for (int k = 0; k < s.HandCount(mover); k++)
                if (s.InstanceId(s.HandInstance(mover, k)) == myId) { inst = s.HandInstance(mover, k); break; }
            if (inst < 0)
                return new OpponentInference(false, myCell, oppCell, oppCard.Id, edges, opponentOwner, DiffFlips(prev.State, cur.State, myCell, oppCell), asRecommended,
                    "the user's card is not in the known old hand", newCells);
            s.MakeMove(new Move(inst, myCell));
        }
        else if (s.CurrentPlayer == mover)
        {
            // the user's move is unknown: the opponent's move cannot be replayed on a consistent position
            return new OpponentInference(false, -1, oppCell, oppCard.Id, edges, opponentOwner, DiffFlips(prev.State, cur.State, myCell, oppCell), false,
                note ?? "the user's own move was not seen", newCells);
        }

        var view = FlipAnalyzer.ViewOf(s);
        byte opp = (byte)opponent;
        var placed = OpponentView(s, oppCard);
        var flips = FlipAnalyzer.Analyze(view, oppCell, placed, opp, s.Rules);

        // the board the rules predict must equal the reported one (owners)
        var predictedOwner = new byte[9];
        for (int c = 0; c < 9; c++) predictedOwner[c] = view[c] is { } b ? b.Owner : OwnerId.None;
        foreach (var f in flips) predictedOwner[f.Cell] = opp;
        predictedOwner[oppCell] = opp;
        bool consistent = true;
        for (int c = 0; c < 9; c++)
        {
            var reported = cur.State.Board[c];
            byte want = reported?.Owner ?? OwnerId.None;
            if (predictedOwner[c] != want) consistent = false;
        }
        var list = flips.Select(f => new InferredFlip(f.Cell / 3, f.Cell % 3, f.CardId, Names.Wire(f.Cause))).ToList();
        return new OpponentInference(consistent, myCell, oppCell, oppCard.Id, edges, opponentOwner, consistent ? list : DiffFlips(prev.State, cur.State, myCell, oppCell),
            asRecommended, consistent ? note : (note ?? "the reported board differs from the one the rules predict"), newCells);
    }

    /// <summary>The opponent's card as it was in its hand: base values plus the type modifier the board gave its type.</summary>
    private static BoardCard OpponentView(GameState afterMyMove, CardSnapshot card)
    {
        int mod = 0;
        byte typeId = card.Type is null ? (byte)0 : CardTypes.IdOf(card.Type);
        if (typeId != 0 && (afterMyMove.Rules & (RuleSet.Ascension | RuleSet.Descension)) != 0)
        {
            int onBoard = 0;
            for (int c = 0; c < 9; c++)
            {
                int inst = afterMyMove.BoardInstance(c);
                if (inst >= 0 && afterMyMove.InstanceType(inst) == typeId) onBoard++;
            }
            mod = (afterMyMove.Rules & RuleSet.Ascension) != 0 ? onBoard : -onBoard;
        }
        int E(int v) => Math.Max(1, Math.Min(10, v + mod));
        return new BoardCard(card.Id, 0, typeId, E(card.Base[0]), E(card.Base[1]), E(card.Base[2]), E(card.Base[3]));
    }

    /// <summary>When the rules do not explain the new board: the cells whose owner differs, without causes.</summary>
    private static List<InferredFlip> DiffFlips(StateSnapshot prev, StateSnapshot cur, int myCell, int oppCell)
    {
        var list = new List<InferredFlip>();
        for (int c = 0; c < 9; c++)
        {
            if (prev.Board[c] is { } before && cur.Board[c] is { } now && before.Owner != now.Owner) list.Add(new InferredFlip(c / 3, c % 3, now.Id, null));
        }
        return list;
    }

    /// <summary>The final count of a finished game: cards of each side on the board plus the cards left in hands, as the rules count them.</summary>
    public static (int Red, int Blue) FinalCount(StateSnapshot finished)
    {
        var g = GameState.FromSnapshot(finished);
        g.CountCards(out int red, out int blue);
        return (red, blue);
    }
}
