using TTC.Siren.Core;

namespace TTC.Siren.Tests;

/// <summary>Ports of the rule-related tests in tests/test_game_rules.py (Phase 2 scope: no evaluation or search).</summary>
public sealed class PortedRuleTests
{
    private const byte R = OwnerId.Red;
    private const byte B = OwnerId.Blue;

    private static CardSnapshot C(int up, int right, int down, int left, byte owner, int id, string? type = null) =>
        new(up, right, down, left, owner, id, type);

    private static int Cell(int row, int col) => row * 3 + col;

    private static GameState State(int current, RuleSet rules, (int Row, int Col, CardSnapshot Card)[] board,
        CardSnapshot[] red, CardSnapshot[] blue)
    {
        var s = new StateSnapshot { Current = current, Rules = rules };
        foreach (var (row, col, card) in board) s.Board[Cell(row, col)] = card;
        s.Hands[0].AddRange(red);
        s.Hands[1].AddRange(blue);
        return GameState.FromSnapshot(s);
    }

    private static byte OwnerAt(GameState g, int row, int col) => g.InstanceOwner(g.BoardInstance(Cell(row, col)));

    [Fact]
    public void Ace_killer_only_special_cases_one_and_a()
    {
        Assert.Equal(1, GameState.CompareValues(1, 10, RuleSet.FallenAce));
        Assert.Equal(-1, GameState.CompareValues(10, 1, RuleSet.FallenAce));
        Assert.Equal(1, GameState.CompareValues(1, 2, RuleSet.Reverse | RuleSet.FallenAce));
    }

    [Fact]
    public void Same_number_flips_only_when_two_sides_match()
    {
        var player = C(5, 4, 2, 3, R, 10);
        var g = State(0, RuleSet.Same,
            new[] { (0, 1, C(2, 2, 5, 2, B, 1)), (1, 0, C(2, 3, 2, 3, R, 2)) },
            new[] { player }, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(0, Cell(1, 1)));

        Assert.Equal(R, OwnerAt(g, 0, 1));
        Assert.Equal(R, OwnerAt(g, 1, 0));
    }

    [Fact]
    public void Same_number_does_not_flip_with_only_one_match()
    {
        var g = State(0, RuleSet.Same,
            new[] { (0, 1, C(2, 2, 5, 5, B, 1)) },
            new[] { C(2, 5, 2, 3, R, 10) }, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(0, Cell(0, 0)));

        Assert.Equal(B, OwnerAt(g, 0, 1));
    }

    [Fact]
    public void Base_capture_does_not_trigger_combo_chain()
    {
        var g = State(1, RuleSet.None,
            new[] { (1, 0, C(1, 9, 1, 1, R, 1)), (1, 1, C(1, 1, 1, 1, R, 2)) },
            Array.Empty<CardSnapshot>(), new[] { C(1, 1, 5, 1, B, 10) });

        g.MakeMove(g.MoveFromSlot(0, Cell(0, 0)));

        Assert.Equal(B, OwnerAt(g, 1, 0));
        Assert.Equal(R, OwnerAt(g, 1, 1));
    }

    [Fact]
    public void Same_number_capture_still_triggers_combo_chain()
    {
        var g = State(0, RuleSet.Same,
            new[] { (0, 1, C(1, 9, 5, 1, B, 1)), (1, 0, C(1, 3, 1, 1, B, 2)), (0, 2, C(1, 1, 1, 1, B, 3)) },
            new[] { C(5, 1, 1, 3, R, 10) }, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(0, Cell(1, 1)));

        Assert.Equal(R, OwnerAt(g, 0, 1));
        Assert.Equal(R, OwnerAt(g, 1, 0));
        Assert.Equal(R, OwnerAt(g, 0, 2));
    }

    [Fact]
    public void Plus_uses_current_type_modified_values()
    {
        var top = C(1, 1, 4, 1, B, 1);
        top.WithAppliedModifier(1);
        var g = State(0, RuleSet.Plus,
            new[] { (0, 1, top), (1, 0, C(1, 4, 1, 1, B, 2)) },
            new[] { C(2, 1, 1, 3, R, 10) }, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(0, Cell(1, 1)));

        Assert.Equal(R, OwnerAt(g, 0, 1));
        Assert.Equal(R, OwnerAt(g, 1, 0));
    }

    [Fact]
    public void Same_type_strengthen_counts_each_placed_type_card()
    {
        // Python places the second card on the same board and recalculates again; here the two positions are built separately.
        var first = C(1, 1, 1, 1, R, 1, "拂晓");
        var hand = C(3, 3, 3, 3, B, 3, "拂晓");
        var one = State(1, RuleSet.Ascension, new[] { (0, 0, first) }, Array.Empty<CardSnapshot>(), new[] { hand });
        one.RecalculateTypeModifiers();
        Assert.Equal(1, one.InstanceModifier(one.BoardInstance(Cell(0, 0))));
        Assert.Equal(1, one.InstanceModifier(one.HandInstance(1, 0)));

        var two = State(1, RuleSet.Ascension,
            new[] { (0, 0, C(1, 1, 1, 1, R, 1, "拂晓")), (0, 1, C(2, 2, 2, 2, B, 2, "拂晓")) },
            Array.Empty<CardSnapshot>(), new[] { C(3, 3, 3, 3, B, 3, "拂晓") });
        two.RecalculateTypeModifiers();
        Assert.Equal(2, two.InstanceModifier(two.BoardInstance(Cell(0, 0))));
        Assert.Equal(2, two.InstanceModifier(two.BoardInstance(Cell(0, 1))));
        Assert.Equal(2, two.InstanceModifier(two.HandInstance(1, 0)));
    }

    [Fact]
    public void Same_type_weaken_counts_each_placed_type_card()
    {
        var g = State(1, RuleSet.Descension,
            new[] { (0, 0, C(5, 5, 5, 5, R, 1, "拂晓")), (0, 1, C(6, 6, 6, 6, B, 2, "拂晓")) },
            Array.Empty<CardSnapshot>(), Array.Empty<CardSnapshot>());
        g.RecalculateTypeModifiers();

        int a = g.BoardInstance(Cell(0, 0)), b = g.BoardInstance(Cell(0, 1));
        Assert.Equal(-2, g.InstanceModifier(a));
        Assert.Equal(-2, g.InstanceModifier(b));
        Assert.Equal(3, g.InstanceEffective(a, Side.Up));
        Assert.Equal(4, g.InstanceEffective(b, Side.Up));
    }

    [Fact]
    public void Undo_move_restores_hand_card_owner()
    {
        var g = State(1, RuleSet.None, Array.Empty<(int, int, CardSnapshot)>(),
            Array.Empty<CardSnapshot>(), new[] { C(5, 4, 2, 3, B, 10) });
        int inst = g.HandInstance(1, 0);

        g.MakeMove(g.MoveFromSlot(0, Cell(1, 1)));
        Assert.Equal(B, g.InstanceOwner(inst));

        Assert.True(g.UndoMove());

        Assert.Equal(B, g.InstanceOwner(inst));
        Assert.Equal(B, g.InstanceOwner(g.HandInstance(1, 0)));
    }
}
