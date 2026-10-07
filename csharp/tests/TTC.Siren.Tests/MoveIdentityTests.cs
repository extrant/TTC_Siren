using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Tests;

/// <summary>Review finding H-3: a Move names a card instance, which stays valid however D-001 reorders the hand.</summary>
public sealed class MoveIdentityTests
{
    private static int[] HandIds(GameState g, int p) =>
        Enumerable.Range(0, g.HandCount(p)).Select(i => g.InstanceId(g.HandInstance(p, i))).ToArray();

    [Fact]
    public void Move_identity_survives_hand_reordering()
    {
        // open_none: red hand [56, 64, 27, 46, 7]; blue hand [59, 51, 35, 48, 8]; red to move; empty board.
        var g = ReferenceScenarios.Create("open_none", SearchGolden.Db);
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        Assert.Equal(45, n);

        Move first = moves[2 * 9 + 3];  // card at slot 2 (id 27) on cell 3
        Move second = moves[1 * 9 + 5]; // card at slot 1 (id 64) on cell 5
        Assert.Equal(27, g.InstanceId(first.Inst));
        Assert.Equal(64, g.InstanceId(second.Inst));
        Assert.Equal(new[] { 56, 64, 27, 46, 7 }, HandIds(g, 0));

        g.MakeMove(second);
        g.UndoMove();

        // D-001: undo appended card 64 to the end, so slot 2 now holds card 46. The old slot-based Move would have played it.
        Assert.Equal(new[] { 56, 27, 46, 7, 64 }, HandIds(g, 0));
        Assert.Equal(27, g.InstanceId(g.HandInstance(0, 1)));
        Assert.Equal(46, g.InstanceId(g.HandInstance(0, 2)));

        g.MakeMove(first); // generated BEFORE the make/undo of `second`

        Assert.Equal(first.Inst, g.BoardInstance(3));
        Assert.Equal(27, g.InstanceId(g.BoardInstance(3)));
        Assert.Equal(new[] { 56, 46, 7, 64 }, HandIds(g, 0)); // 27 removed, order otherwise as left by D-001
        Assert.Equal(1, g.CurrentPlayer);
    }

    [Fact]
    public void EvaluateMove_depends_on_the_card_not_on_the_current_hand_order()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        var eval = new Evaluator(g, db);
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        var before = new long[n];
        for (int i = 0; i < n; i++) before[i] = BitConverter.DoubleToInt64Bits(eval.EvaluateMove(moves[i], null, false));

        g.MakeMove(moves[1 * 9 + 5]);
        g.UndoMove(); // hand reordered, board identical, no type rules (display edges unchanged)

        for (int i = 0; i < n; i++)
            Assert.True(before[i] == BitConverter.DoubleToInt64Bits(eval.EvaluateMove(moves[i], null, false)),
                $"EvaluateMove changed for move {moves[i]} after the hand was reordered");
    }

    [Fact]
    public void MakeMove_rejects_instance_not_in_hand()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("mid4_rev_ace", db); // red to move, 4 cards on the board, 3 + 3 cards in hand
        Assert.Equal(0, g.CurrentPlayer);

        int opponentCard = g.HandInstance(1, 0);
        int boardCard = g.BoardInstance(1); // occupied cell 1
        int ownCard = g.HandInstance(0, 0);
        int freeCell = Enumerable.Range(0, 9).First(g.IsEmpty);
        int occupiedCell = Enumerable.Range(0, 9).First(c => !g.IsEmpty(c));
        Assert.NotEqual(-1, g.HandSlotOf(1, opponentCard));
        Assert.Equal(-1, g.HandSlotOf(0, opponentCard));

        string before = g.ToSnapshot().ToCanonicalString(); // strict: includes hand order and display edges

        void AssertRejected(string what, Move move)
        {
            var ex = Assert.Throws<InvalidOperationException>(() => g.MakeMove(move));
            Assert.True(before == g.ToSnapshot().ToCanonicalString(), $"{what}: state changed by a rejected move ({ex.Message})");
            Assert.Equal(0, g.UndoDepth);
            Assert.Equal(0, g.CurrentPlayer);
        }

        AssertRejected("opponent's hand card", new Move(opponentCard, freeCell));
        AssertRejected("card already on the board", new Move(boardCard, freeCell));
        AssertRejected("occupied cell", new Move(ownCard, occupiedCell));
        AssertRejected("cell out of range", new Move(ownCard, 9));
        AssertRejected("instance index out of range", new Move(200, freeCell));

        // And the same state still accepts a legal move afterwards.
        g.MakeMove(new Move(ownCard, freeCell));
        Assert.Equal(ownCard, g.BoardInstance(freeCell));
    }

    [Fact]
    public void MakeMove_rejects_a_full_undo_stack_without_changing_state()
    {
        var g = ReferenceScenarios.Create("open_none", SearchGolden.Db);
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        for (int i = 0; i < 9; i++) // 9 plies fill the board
        {
            int n = g.GenerateMoves(moves);
            Assert.True(n > 0, $"ply {i}: no moves");
            g.MakeMove(moves[0]);
        }
        Assert.True(g.IsGameOver);
        string before = g.ToSnapshot().ToCanonicalString();
        Assert.Throws<InvalidOperationException>(() => g.MakeMove(new Move(g.HandInstance(g.CurrentPlayer, 0), 0))); // cell occupied
        Assert.Equal(before, g.ToSnapshot().ToCanonicalString());
    }

    [Fact]
    public void Generated_moves_keep_hand_order_times_row_major_cells_with_can_use_filter()
    {
        // Instance-based moves must come out in exactly the order the slot-based ones did (CLAUDE.md section 4).
        var s = new StateSnapshot { Current = 0, Rules = RuleSet.Order };
        s.Hands[0].Add(new CardSnapshot(1, 1, 1, 1, OwnerId.Red, 11, canUse: true));
        s.Hands[0].Add(new CardSnapshot(2, 2, 2, 2, OwnerId.Red, 12, canUse: false));
        s.Hands[0].Add(new CardSnapshot(3, 3, 3, 3, OwnerId.Red, 13, canUse: true));
        s.Board[4] = new CardSnapshot(5, 5, 5, 5, OwnerId.Blue, 99);
        var g = GameState.FromSnapshot(s);

        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);

        var expected = new List<(int Id, int Cell)>();
        foreach (int id in new[] { 11, 13 }) // hand order, card 12 filtered out by can_use
            foreach (int cell in new[] { 0, 1, 2, 3, 5, 6, 7, 8 }) // row-major empties
                expected.Add((id, cell));
        var actual = new List<(int Id, int Cell)>();
        for (int i = 0; i < n; i++) actual.Add((g.InstanceId(moves[i].Inst), moves[i].Cell));
        Assert.Equal(expected, actual);
    }
}
