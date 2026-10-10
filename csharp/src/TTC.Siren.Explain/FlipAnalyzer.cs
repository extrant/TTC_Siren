using TTC.Siren.Core;

namespace TTC.Siren.Explain;

/// <summary>Why a card changed sides (explanation facts v2). The names are the wire values of the facts.</summary>
public enum FlipCause
{
    /// <summary>The placed card's edge is higher than the neighbour's facing edge (basic comparison).</summary>
    Basic,
    /// <summary>The basic comparison was won by the lower value because 逆转 is active.</summary>
    Reverse,
    /// <summary>The basic comparison was decided by 王牌杀手 (1 against 10).</summary>
    FallenAce,
    /// <summary>同数: two or more neighbouring edges are equal.</summary>
    Same,
    /// <summary>加算: two or more neighbours have the same sum with the placed card.</summary>
    Plus,
    /// <summary>连携: a card flipped by 同数/加算 flips a further neighbour by basic comparison (one level only, D-013).</summary>
    Combo,
}

public static class FlipCauseNames
{
    public static string Wire(FlipCause cause) => cause switch
    {
        FlipCause.Basic => "basic",
        FlipCause.Reverse => "reverse",
        FlipCause.FallenAce => "fallen_ace",
        FlipCause.Same => "same",
        FlipCause.Plus => "plus",
        FlipCause.Combo => "combo",
        _ => throw new ArgumentOutOfRangeException(nameof(cause)),
    };
}

/// <summary>One card that changes sides because of a placement.</summary>
/// <param name="Cell">Row-major cell (0..8) of the flipped card.</param>
/// <param name="CardId">Database id of the flipped card.</param>
/// <param name="Cause">Why it flips.</param>
/// <param name="MySide">The side of the flipping card that touches it (0 up, 1 right, 2 down, 3 left).</param>
/// <param name="MyValue">Effective value of that side.</param>
/// <param name="TheirValue">Effective value of the flipped card's facing side.</param>
/// <param name="Via">For <see cref="FlipCause.Combo"/>: the cell of the card whose flip triggered this one; otherwise -1.</param>
/// <param name="ComboKind">For <see cref="FlipCause.Combo"/>: how the basic comparison was decided (Basic, Reverse or FallenAce).</param>
public readonly record struct FlipRecord(int Cell, int CardId, FlipCause Cause, int MySide, int MyValue, int TheirValue, int Via, FlipCause ComboKind);

/// <summary>A card on the board as the analyzer sees it: owner, id, type and the four effective edge values.</summary>
public readonly record struct BoardCard(int Id, byte Owner, byte TypeId, int Up, int Right, int Down, int Left)
{
    public int Edge(int side) => side switch { 0 => Up, 1 => Right, 2 => Down, 3 => Left, _ => throw new ArgumentOutOfRangeException(nameof(side)) };
}

/// <summary>
/// Re-derives, from the board alone, which cards flip when a card is placed and WHY (explanation facts v2). The flipping algorithm is
/// that of <c>GameState.ResolveFlips</c> (Python <c>resolve_flip_with_record</c>): basic comparison (incl. 逆转 / 王牌杀手), then 加算, then
/// 同数, with 同数/加算 flips triggering one level of basic-comparison 连携. The set of flipped cards is checked against the engine's own
/// <c>MakeMove</c> over thousands of random positions (ExplainTests), so the causes attached here are the causes of the real flips.
/// The analyzer works on a plain board view, so it can also evaluate a card that is not in any hand (a possible unknown opponent card).
/// </summary>
public static class FlipAnalyzer
{
    /// <summary>Reads the board of <paramref name="g"/> (effective values, i.e. base + type modifier clamped to 1..10).</summary>
    public static BoardCard?[] ViewOf(GameState g)
    {
        var view = new BoardCard?[9];
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = g.BoardInstance(cell);
            if (inst >= 0) view[cell] = ViewOfInstance(g, inst);
        }
        return view;
    }

    public static BoardCard ViewOfInstance(GameState g, int inst) =>
        new(g.InstanceId(inst), g.InstanceOwner(inst), g.InstanceType(inst),
            g.InstanceEffective(inst, 0), g.InstanceEffective(inst, 1), g.InstanceEffective(inst, 2), g.InstanceEffective(inst, 3));

    /// <summary>
    /// The flips caused by placing <paramref name="placed"/> (owned by <paramref name="mover"/>) on the empty cell
    /// <paramref name="cell"/> of <paramref name="board"/>, in the order they happen. The board is not modified.
    /// </summary>
    public static List<FlipRecord> Analyze(BoardCard?[] board, int cell, BoardCard placed, byte mover, RuleSet rules)
    {
        var owners = new byte[9];
        for (int i = 0; i < 9; i++) owners[i] = board[i]?.Owner ?? OwnerId.None;
        var flips = new List<FlipRecord>();
        var flipped = new bool[9];
        Resolve(board, owners, cell, placed, mover, rules, flips, flipped, chainOnly: false, via: -1);
        return flips;
    }

    private static FlipCause BasicKind(int info) => info switch
    {
        5 => FlipCause.Reverse,
        1 or 4 => FlipCause.FallenAce, // 王牌杀手 decided in favour of the mover (1 beats 10; with 逆转: 10 beats 1)
        _ => FlipCause.Basic,
    };

    private static void Resolve(BoardCard?[] board, byte[] owners, int cell, BoardCard card, byte mover, RuleSet rules, List<FlipRecord> flips,
        bool[] flipped, bool chainOnly, int via)
    {
        int n = GameState.NeighborCount(cell);
        // reason per neighbour slot: 0 none, 1 base, 2 plus, 3 same; insertion order as Python's dict
        var reason = new int[4];
        var kind = new FlipCause[4];
        var order = new List<int>();

        void Add(int k, int r)
        {
            if (reason[k] is 2 or 3) return; // 同数/加算 are never overwritten
            if (reason[k] == 0) order.Add(k);
            reason[k] = r;
        }

        for (int k = 0; k < n; k++)
        {
            int nc = GameState.NeighborCell(cell, k);
            if (board[nc] is not { } other || owners[nc] == mover) continue;
            int mine = card.Edge(GameState.NeighborMySide(cell, k)), theirs = other.Edge(GameState.NeighborOppSide(cell, k));
            if (GameState.CompareValues(mine, theirs, rules, out int info) == 1)
            {
                Add(k, 1);
                kind[k] = BasicKind(info);
            }
        }

        if (!chainOnly)
        {
            if ((rules & RuleSet.Plus) != 0)
            {
                var sums = new List<(int Sum, List<int> Slots)>();
                for (int k = 0; k < n; k++)
                {
                    int nc = GameState.NeighborCell(cell, k);
                    if (board[nc] is not { } other) continue;
                    int s = card.Edge(GameState.NeighborMySide(cell, k)) + other.Edge(GameState.NeighborOppSide(cell, k));
                    int idx = sums.FindIndex(e => e.Sum == s);
                    if (idx < 0)
                    {
                        sums.Add((s, new List<int>()));
                        idx = sums.Count - 1;
                    }
                    sums[idx].Slots.Add(k);
                }
                foreach (var (_, slots) in sums)
                {
                    if (slots.Count < 2 || !slots.Any(k => owners[GameState.NeighborCell(cell, k)] != mover)) continue;
                    foreach (int k in slots)
                        if (owners[GameState.NeighborCell(cell, k)] != mover) Add(k, 2);
                }
            }

            if ((rules & RuleSet.Same) != 0)
            {
                var same = new List<int>();
                for (int k = 0; k < n; k++)
                {
                    int nc = GameState.NeighborCell(cell, k);
                    if (board[nc] is not { } other) continue;
                    if (card.Edge(GameState.NeighborMySide(cell, k)) == other.Edge(GameState.NeighborOppSide(cell, k))) same.Add(k);
                }
                if (same.Count >= 2 && same.Any(k => owners[GameState.NeighborCell(cell, k)] != mover))
                    foreach (int k in same)
                        if (owners[GameState.NeighborCell(cell, k)] != mover) Add(k, 3);
            }
        }

        foreach (int k in order)
        {
            int nc = GameState.NeighborCell(cell, k);
            if (flipped[nc]) continue;
            if (board[nc] is not { } other || owners[nc] == mover) continue;
            int mySide = GameState.NeighborMySide(cell, k);
            int mine = card.Edge(mySide), theirs = other.Edge(GameState.NeighborOppSide(cell, k));
            FlipCause cause = chainOnly ? FlipCause.Combo
                : reason[k] == 2 ? FlipCause.Plus
                : reason[k] == 3 ? FlipCause.Same
                : kind[k];
            flips.Add(new FlipRecord(nc, other.Id, cause, mySide, mine, theirs, chainOnly ? via : -1, chainOnly ? kind[k] : FlipCause.Basic));
            owners[nc] = mover;
            flipped[nc] = true;
            if (reason[k] is 2 or 3) Resolve(board, owners, nc, other, mover, rules, flips, flipped, chainOnly: true, via: nc);
        }
    }
}
