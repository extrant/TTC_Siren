namespace TTC.Siren.Compat;

/// <summary>Edge/direction indices of the Python model: the order of <c>up, right, down, left</c>.</summary>
public static class Dir
{
    public const int Up = 0, Right = 1, Down = 2, Left = 3;
}

/// <summary>
/// Python's <c>core.card.Card</c> as an ordinary mutable object. This legacy object model exists ONLY for the request-time compatibility
/// layer of <c>/ai_move</c> (request parsing, unknown-card sampling, the analysis texts of the old response fields): it is built once
/// per request, never in a search. The search itself runs on the data-oriented <c>TTC.Siren.Core.GameState</c> (CLAUDE.md sections 3 and 5).
/// It deliberately keeps Python's quirks, because the old response fields are derived from them (D-001 hand reordering by undo,
/// D-003 display values that undo does not restore, D-010 copies replacing hand cards).
/// </summary>
public sealed class LCard
{
    public readonly int[] Base = new int[4];
    /// <summary>Python <c>card.up/right/down/left</c>: the display values, equal to the base values until a type modifier is applied.</summary>
    public readonly int[] Disp = new int[4];
    public string? Owner;
    public int? CardId;
    public string? CardType;
    public bool CanUse;
    public int TypeModifier;
    public bool IsGenerated;
    public bool IsPrediction;

    public LCard(int up, int right, int down, int left, string? owner = null, int? cardId = null, string? cardType = null, bool canUse = true)
    {
        Base[0] = Disp[0] = up;
        Base[1] = Disp[1] = right;
        Base[2] = Disp[2] = down;
        Base[3] = Disp[3] = left;
        Owner = owner;
        CardId = cardId;
        CardType = cardType;
        CanUse = canUse;
    }

    public int Up => Disp[0];
    public int Right => Disp[1];
    public int Down => Disp[2];
    public int Left => Disp[3];

    /// <summary><c>get_modified_value</c> (also <c>get_effective_value</c>): base + type modifier, clamped to 1..10.</summary>
    public int Modified(int dir) => Math.Max(1, Math.Min(10, Base[dir] + TypeModifier));

    /// <summary>Python <c>compare_values</c>: 1 = this card wins, -1 = the other wins, 0 = tie.</summary>
    public int Compare(int myDir, LCard other, int otherDir, IReadOnlyList<string> rules)
    {
        int mine = Modified(myDir), theirs = other.Modified(otherDir);
        if (rules.Contains("王牌杀手"))
        {
            bool myAce = mine is 1 or 10, otherAce = theirs is 1 or 10;
            if (myAce && otherAce)
            {
                if (mine == theirs) return 0;
                if (mine == 1 && theirs == 10) return !rules.Contains("逆转") ? 1 : -1;
                if (mine == 10 && theirs == 1) return !rules.Contains("逆转") ? -1 : 1;
            }
        }
        if (rules.Contains("逆转")) return mine < theirs ? 1 : mine > theirs ? -1 : 0;
        return mine > theirs ? 1 : mine < theirs ? -1 : 0;
    }

    /// <summary><c>apply_type_modifier</c>: sets the modifier and refreshes the display values.</summary>
    public void ApplyTypeModifier(int modifier)
    {
        TypeModifier = modifier;
        for (int i = 0; i < 4; i++) Disp[i] = Modified(i);
    }

    /// <summary>
    /// <c>copy.deepcopy(card)</c>: every attribute as it is (display values, flags, owner, modifier), unlike <see cref="Copy"/>, which recomputes
    /// the display values and drops the flags.
    /// </summary>
    public LCard DeepCopy()
    {
        var c = new LCard(Base[0], Base[1], Base[2], Base[3], Owner, CardId, CardType, CanUse)
        {
            TypeModifier = TypeModifier, IsGenerated = IsGenerated, IsPrediction = IsPrediction,
        };
        for (int i = 0; i < 4; i++) c.Disp[i] = Disp[i];
        return c;
    }

    /// <summary><c>Card.copy()</c>: a new object with the same values (the generated/prediction flags are not copied, as in Python).</summary>
    public LCard Copy()
    {
        var c = new LCard(Base[0], Base[1], Base[2], Base[3], Owner, CardId, CardType, CanUse) { TypeModifier = TypeModifier };
        for (int i = 0; i < 4; i++) c.Disp[i] = c.Modified(i);
        return c;
    }
}

public sealed class LBoard
{
    public readonly LCard?[,] Grid = new LCard?[3, 3];

    public LCard? Get(int row, int col) => Grid[row, col];

    public bool IsEmpty(int row, int col) => Grid[row, col] is null;

    public bool Place(int row, int col, LCard card)
    {
        if (Grid[row, col] is not null) return false;
        Grid[row, col] = card;
        return true;
    }

    public LCard? Remove(int row, int col)
    {
        var c = Grid[row, col];
        Grid[row, col] = null;
        return c;
    }

    /// <summary>Row-major list of the empty cells (<c>available_positions</c>).</summary>
    public List<(int Row, int Col)> AvailablePositions()
    {
        var list = new List<(int, int)>(9);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                if (Grid[r, c] is null) list.Add((r, c));
        return list;
    }

    public int Occupancy()
    {
        int n = 0;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                if (Grid[r, c] is not null) n++;
        return n;
    }
}

public sealed class LPlayer
{
    public string Name;
    public List<LCard> Hand;
    public readonly List<LCard> UsedCards = new();

    public LPlayer(string name, List<LCard> hand)
    {
        Name = name;
        Hand = hand;
    }

    /// <summary><c>play_card</c>: removes the first hand entry that IS this card (Python <c>in</c>/<c>remove</c> compare by identity here).</summary>
    public void PlayCard(LCard card)
    {
        int index = Hand.FindIndex(c => ReferenceEquals(c, card));
        if (index < 0) return;
        Hand.RemoveAt(index);
        UsedCards.Add(card);
    }

    /// <summary><c>get_playable_cards</c>: a new list; with 秩序/混乱 only the cards whose <c>can_use</c> is set.</summary>
    public List<LCard> PlayableCards(IReadOnlyList<string> rules) =>
        rules.Count > 0 && (rules.Contains("秩序") || rules.Contains("混乱")) ? Hand.Where(c => c.CanUse).ToList() : new List<LCard>(Hand);
}

public sealed class LMoveRecord
{
    public int Row, Col, PlayerIdx;
    public LCard Card = null!;
    public List<(int Row, int Col, string? Owner)> Flipped = new();
    public List<(LCard Card, int Modifier)> TypeModifiers = new();
    public string? CardOriginalOwner;
}

/// <summary>Python's <c>core.game_state.GameState</c> restricted to what the compatibility layer calls (make/undo, flips, type modifiers).</summary>
public sealed class LGameState
{
    public LBoard Board;
    public LPlayer[] Players;
    public int CurrentPlayerIdx;
    public List<string> Rules;
    private readonly List<LMoveRecord> _history = new();

    public LGameState(LBoard board, LPlayer[] players, int currentPlayerIdx, List<string> rules)
    {
        Board = board;
        Players = players;
        CurrentPlayerIdx = currentPlayerIdx;
        Rules = rules;
    }

    public LPlayer CurrentPlayer => Players[CurrentPlayerIdx];

    /// <summary><c>GameState.copy()</c>: a deep copy of the board and of both players (hands and used cards).</summary>
    public LGameState Copy()
    {
        var board = new LBoard();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                board.Grid[r, c] = Board.Grid[r, c]?.DeepCopy();
        var players = Players.Select(p =>
        {
            var copy = new LPlayer(p.Name, p.Hand.Select(c => c.DeepCopy()).ToList());
            copy.UsedCards.AddRange(p.UsedCards.Select(c => c.DeepCopy()));
            return copy;
        }).ToArray();
        return new LGameState(board, players, CurrentPlayerIdx, new List<string>(Rules));
    }

    /// <summary><c>play_move</c>: make_move without keeping the record. Note that the card object that is passed in is the one that is placed.</summary>
    public bool PlayMove(int row, int col, LCard card) => MakeMove(row, col, card) is not null;

    private bool HasTypeRule => Rules.Contains("同类强化") || Rules.Contains("同类弱化");

    /// <summary>Counts the cards of both sides on the board and in the hands: (red, blue).</summary>
    public (int Red, int Blue) CountCards()
    {
        int red = 0, blue = 0;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var card = Board.Get(r, c);
                if (card is null) continue;
                if (card.Owner == "red") red++;
                else if (card.Owner == "blue") blue++;
            }
        foreach (var p in Players)
            foreach (var card in p.Hand)
            {
                if (card.Owner == "red") red++;
                else if (card.Owner == "blue") blue++;
            }
        return (red, blue);
    }

    /// <summary>Python <c>make_move</c> (with the D-004 fix: the hand card is found by identity first, then by card id). Null when illegal.</summary>
    public LMoveRecord? MakeMove(int row, int col, LCard card)
    {
        var hand = CurrentPlayer.Hand;
        LCard? handCard = hand.FirstOrDefault(c => ReferenceEquals(c, card));
        handCard ??= hand.FirstOrDefault(c => c.CardId == card.CardId);
        if (!Board.IsEmpty(row, col) || handCard is null) return null;

        var originalModifiers = new List<(LCard, int)>();
        if (HasTypeRule)
        {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var b = Board.Get(r, c);
                    if (b is not null) originalModifiers.Add((b, b.TypeModifier));
                }
            foreach (var p in Players)
                foreach (var h in p.Hand) originalModifiers.Add((h, h.TypeModifier));
        }

        string? originalOwner = card.Owner;
        card.Owner = CurrentPlayerIdx == 0 ? "red" : "blue";
        Board.Place(row, col, card);
        CurrentPlayer.PlayCard(handCard);

        var flipped = new List<(int, int, string?)>();
        ResolveFlip(row, col, card, flipped, new HashSet<(int, int)>(), chainOnly: false);

        if (HasTypeRule && !string.IsNullOrEmpty(card.CardType)) RecalculateTypeModifiers();

        var record = new LMoveRecord
        {
            Row = row, Col = col, Card = card, PlayerIdx = CurrentPlayerIdx, Flipped = flipped,
            TypeModifiers = originalModifiers, CardOriginalOwner = originalOwner,
        };
        CurrentPlayerIdx = 1 - CurrentPlayerIdx;
        _history.Add(record);
        return record;
    }

    /// <summary>Python <c>undo_move</c>: the card goes to the END of the hand (D-001) and display values are not restored (D-003).</summary>
    public void UndoMove(LMoveRecord record)
    {
        _history.Remove(record);
        CurrentPlayerIdx = record.PlayerIdx;
        foreach (var (r, c, owner) in record.Flipped)
        {
            var flippedCard = Board.Get(r, c);
            if (flippedCard is not null) flippedCard.Owner = owner;
        }
        Board.Remove(record.Row, record.Col);
        CurrentPlayer.Hand.Add(record.Card);
        record.Card.Owner = record.CardOriginalOwner;
        foreach (var (tracked, modifier) in record.TypeModifiers) tracked.TypeModifier = modifier;
    }

    private static readonly (int Dr, int Dc, int My, int Opp)[] Directions =
    {
        (-1, 0, Dir.Up, Dir.Down), (1, 0, Dir.Down, Dir.Up), (0, -1, Dir.Left, Dir.Right), (0, 1, Dir.Right, Dir.Left),
    };

    private void ResolveFlip(int row, int col, LCard card, List<(int, int, string?)> flipped, HashSet<(int, int)> flippedSet, bool chainOnly)
    {
        string? owner = card.Owner;
        var toFlip = new List<((int R, int C) Cell, string Reason)>(); // insertion-ordered dict

        void AddFlip(int r, int c, string reason)
        {
            int index = toFlip.FindIndex(t => t.Cell == (r, c));
            if (index >= 0)
            {
                if (toFlip[index].Reason is "same" or "plus") return;
                toFlip[index] = ((r, c), reason);
                return;
            }
            toFlip.Add(((r, c), reason));
        }

        foreach (var (dr, dc, my, opp) in Directions)
        {
            int nr = row + dr, nc = col + dc;
            if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
            var oppCard = Board.Get(nr, nc);
            if (oppCard is not null && oppCard.Owner != owner && card.Compare(my, oppCard, opp, Rules) == 1) AddFlip(nr, nc, "base");
        }

        if (!chainOnly)
        {
            if (Rules.Contains("加算"))
            {
                var sumMap = new List<(int Sum, List<(int R, int C, LCard Card)> Items)>();
                foreach (var (dr, dc, my, opp) in Directions)
                {
                    int nr = row + dr, nc = col + dc;
                    if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
                    var oppCard = Board.Get(nr, nc);
                    if (oppCard is null) continue;
                    int s = card.Modified(my) + oppCard.Modified(opp);
                    int idx = sumMap.FindIndex(e => e.Sum == s);
                    if (idx < 0)
                    {
                        sumMap.Add((s, new List<(int, int, LCard)>()));
                        idx = sumMap.Count - 1;
                    }
                    sumMap[idx].Items.Add((nr, nc, oppCard));
                }
                foreach (var (_, items) in sumMap)
                {
                    if (items.Count < 2) continue;
                    if (items.Any(i => i.Card.Owner != owner))
                        foreach (var (nr, nc, oppCard) in items)
                            if (oppCard.Owner != owner) AddFlip(nr, nc, "plus");
                }
            }

            if (Rules.Contains("同数"))
            {
                var sameList = new List<(int R, int C, LCard Card)>();
                foreach (var (dr, dc, my, opp) in Directions)
                {
                    int nr = row + dr, nc = col + dc;
                    if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
                    var oppCard = Board.Get(nr, nc);
                    if (oppCard is not null && card.Modified(my) == oppCard.Modified(opp)) sameList.Add((nr, nc, oppCard));
                }
                if (sameList.Count >= 2 && sameList.Any(i => i.Card.Owner != owner))
                    foreach (var (nr, nc, oppCard) in sameList)
                        if (oppCard.Owner != owner) AddFlip(nr, nc, "same");
            }
        }

        foreach (var ((nr, nc), reason) in toFlip)
        {
            if (flippedSet.Contains((nr, nc))) continue;
            var oppCard = Board.Get(nr, nc);
            if (oppCard is not null && oppCard.Owner != owner)
            {
                flipped.Add((nr, nc, oppCard.Owner));
                oppCard.Owner = owner;
                flippedSet.Add((nr, nc));
                if (reason is "same" or "plus") ResolveFlip(nr, nc, oppCard, flipped, flippedSet, chainOnly: true);
            }
        }
    }

    /// <summary>Python <c>recalculate_type_modifiers</c>: board and hand cards of each type that is on the board get ±count.</summary>
    public void RecalculateTypeModifiers()
    {
        if (!HasTypeRule) return;
        var typeCounts = new List<(string Type, int Count)>(); // insertion-ordered dict
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var b = Board.Get(r, c);
                if (b is null || string.IsNullOrEmpty(b.CardType)) continue;
                int idx = typeCounts.FindIndex(t => t.Type == b.CardType);
                if (idx < 0) typeCounts.Add((b.CardType, 1));
                else typeCounts[idx] = (b.CardType, typeCounts[idx].Count + 1);
            }
        foreach (var (type, count) in typeCounts)
        {
            int modifier = Rules.Contains("同类强化") ? count : Rules.Contains("同类弱化") ? -count : 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var b = Board.Get(r, c);
                    if (b is not null && b.CardType == type) b.ApplyTypeModifier(modifier);
                }
            foreach (var p in Players)
                foreach (var h in p.Hand)
                    if (h.CardType == type) h.ApplyTypeModifier(modifier);
        }
    }
}
