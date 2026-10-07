using System.Text.Json;
using TTC.Siren.Core;

namespace TTC.Siren.Tests;

public sealed class GameStateBehaviourTests
{
    private const byte R = OwnerId.Red;
    private const byte B = OwnerId.Blue;

    private static CardSnapshot C(int up, int right, int down, int left, byte owner, int id, string? type = null,
        bool canUse = true) => new(up, right, down, left, owner, id, type, canUse);

    private static GameState Make(int current, RuleSet rules, CardSnapshot?[] board, CardSnapshot[] red, CardSnapshot[] blue)
    {
        var s = new StateSnapshot { Current = current, Rules = rules };
        for (int i = 0; i < board.Length; i++) s.Board[i] = board[i];
        s.Hands[0].AddRange(red);
        s.Hands[1].AddRange(blue);
        return GameState.FromSnapshot(s);
    }

    private static int[] HandIds(GameState g, int p) =>
        Enumerable.Range(0, g.HandCount(p)).Select(i => g.InstanceId(g.HandInstance(p, i))).ToArray();

    [Fact]
    public void D001_undo_appends_card_to_end_of_hand()
    {
        var hand = new[] { C(1, 1, 1, 1, R, 56), C(2, 2, 2, 2, R, 64), C(3, 3, 3, 3, R, 27), C(4, 4, 4, 4, R, 46), C(5, 5, 5, 5, R, 7) };
        var g = Make(0, RuleSet.None, new CardSnapshot?[9], hand, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(1, 4));
        Assert.Equal(new[] { 56, 27, 46, 7 }, HandIds(g, 0));
        g.UndoMove();

        // Measured in Phase 1: [56,64,27,46,7] -> play 64 -> undo -> [56,27,46,7,64]
        Assert.Equal(new[] { 56, 27, 46, 7, 64 }, HandIds(g, 0));
    }

    [Fact]
    public void D002_fixed_undo_restores_modifiers_by_card_identity()
    {
        // Hand [A(mod 1), B(mod 0), C(mod 2)]; B is played -> after undo the hand is [A, C, B] (D-001 still applies),
        // and every card keeps its own modifier: A=1, C=2, B=0. Before the fix (Python and C#) the old-index snapshot
        // [1,0,2] was written onto [A, C, B], giving C=0 and B=2. Mirrors tests/test_undo_type_modifiers.py.
        var board = new CardSnapshot?[9];
        board[0] = C(5, 5, 5, 5, B, 40, "拂晓");
        var red = new[]
        {
            C(5, 5, 5, 5, R, 1).WithAppliedModifier(1),
            C(5, 5, 5, 5, R, 2),
            C(5, 5, 5, 5, R, 3).WithAppliedModifier(2),
        };
        var g = Make(0, RuleSet.Ascension, board, red, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(1, 8));
        g.UndoMove();

        Assert.Equal(new[] { 1, 3, 2 }, HandIds(g, 0));
        Assert.Equal(new[] { 1, 2, 0 }, Enumerable.Range(0, 3).Select(i => g.InstanceModifier(g.HandInstance(0, i))).ToArray());
    }

    [Fact]
    public void D003_display_edges_are_not_restored_by_undo()
    {
        var board = new CardSnapshot?[9];
        board[0] = C(5, 5, 5, 5, B, 40, "拂晓");
        var red = new[] { C(4, 4, 4, 4, R, 46, "拂晓"), C(1, 1, 1, 1, R, 47, "拂晓") };
        var g = Make(0, RuleSet.Ascension, board, red, Array.Empty<CardSnapshot>());

        int inst = g.HandInstance(0, 0);
        Assert.Equal(4, g.InstanceDisplay(inst, Side.Up));
        g.MakeMove(g.MoveFromSlot(1, 8)); // two 拂晓 on board -> +2 for everything of that type, display refreshed
        Assert.Equal(6, g.InstanceDisplay(inst, Side.Up));
        g.UndoMove();

        Assert.Equal(6, g.InstanceDisplay(inst, Side.Up)); // stale on purpose (D-003)
        Assert.Equal(0, g.InstanceModifier(inst));
    }

    [Fact]
    public void D004_duplicate_hand_ids_are_rejected()
    {
        var s = new StateSnapshot { Current = 0, Rules = RuleSet.None };
        s.Hands[0].Add(C(1, 1, 1, 1, R, 7));
        s.Hands[0].Add(C(2, 2, 2, 2, R, 7));
        Assert.Throws<InvalidGameStateException>(() => GameState.FromSnapshot(s));

        var none = new StateSnapshot { Current = 0, Rules = RuleSet.None };
        none.Hands[1].Add(C(0, 0, 0, 0, B, -1));
        none.Hands[1].Add(C(0, 0, 0, 0, B, -1));
        Assert.Throws<InvalidGameStateException>(() => GameState.FromSnapshot(none));
    }

    [Fact]
    public void Hand_over_capacity_is_rejected()
    {
        var s = new StateSnapshot { Current = 0, Rules = RuleSet.None };
        for (int i = 0; i < 6; i++) s.Hands[0].Add(C(1, 1, 1, 1, R, i + 1));
        Assert.Throws<InvalidGameStateException>(() => GameState.FromSnapshot(s));
    }

    [Fact]
    public void Order_and_chaos_filter_moves_by_can_use()
    {
        var red = new[] { C(1, 1, 1, 1, R, 1, canUse: false), C(2, 2, 2, 2, R, 2), C(3, 3, 3, 3, R, 3, canUse: false) };
        Span<Move> buf = stackalloc Move[GameState.MaxMoves];

        var plain = Make(0, RuleSet.None, new CardSnapshot?[9], red, Array.Empty<CardSnapshot>());
        Assert.Equal(27, plain.GenerateMoves(buf));

        foreach (var rule in new[] { RuleSet.Order, RuleSet.Chaos })
        {
            var g = Make(0, rule, new CardSnapshot?[9], red, Array.Empty<CardSnapshot>());
            int n = g.GenerateMoves(buf);
            Assert.Equal(9, n);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(1, g.HandSlotOf(0, buf[i].Inst)); // slot 1 is the only card with can_use
                Assert.Equal(i, buf[i].Cell); // row-major
            }
        }
    }

    [Fact]
    public void Make_then_undo_restores_everything_except_documented_quirks()
    {
        // No type rule: no quirk applies except D-001 (hand order), so only a single-card hand is compared exactly.
        var board = new CardSnapshot?[9];
        board[1] = C(1, 9, 5, 1, B, 1);
        board[3] = C(1, 3, 1, 1, B, 2);
        board[2] = C(1, 1, 1, 1, B, 3);
        var g = Make(0, RuleSet.Same | RuleSet.Plus | RuleSet.Reverse, board, new[] { C(5, 1, 1, 3, R, 10) }, Array.Empty<CardSnapshot>());
        string before = g.ToSnapshot().ToCanonicalString();

        g.MakeMove(g.MoveFromSlot(0, 4));
        Assert.NotEqual(before, g.ToSnapshot().ToCanonicalString());
        g.UndoMove();

        Assert.Equal(before, g.ToSnapshot().ToCanonicalString());
        Assert.Equal(0, g.UndoDepth);
        Assert.False(g.UndoMove());
    }

    [Fact]
    public void Flip_from_none_owner_is_restored_to_none()
    {
        var board = new CardSnapshot?[9];
        board[1] = C(1, 1, 1, 1, OwnerId.None, 1);
        var g = Make(0, RuleSet.None, board, new[] { C(9, 9, 9, 9, R, 10) }, Array.Empty<CardSnapshot>());

        g.MakeMove(g.MoveFromSlot(0, 4));
        Assert.Equal(R, g.InstanceOwner(g.BoardInstance(1)));
        g.UndoMove();
        Assert.Equal(OwnerId.None, g.InstanceOwner(g.BoardInstance(1)));
    }

    [Fact]
    public void Winner_and_card_counts_follow_card_owner()
    {
        var board = new CardSnapshot?[9];
        for (int i = 0; i < 9; i++) board[i] = C(1, 1, 1, 1, i < 5 ? R : B, i + 1);
        var g = Make(0, RuleSet.None, board, new[] { C(1, 1, 1, 1, B, 20) }, Array.Empty<CardSnapshot>());
        g.CountCards(out int red, out int blue);
        Assert.Equal((5, 5), (red, blue));
        Assert.True(g.IsGameOver);
        Assert.Equal(GameState.NoWinner, g.GetWinner());

        board[8] = C(1, 1, 1, 1, R, 9);
        g = Make(0, RuleSet.None, board, Array.Empty<CardSnapshot>(), Array.Empty<CardSnapshot>());
        Assert.Equal(0, g.GetWinner());
    }

    [Fact]
    public void Invalid_move_throws()
    {
        var g = Make(0, RuleSet.None, new CardSnapshot?[9], new[] { C(1, 1, 1, 1, R, 1) }, Array.Empty<CardSnapshot>());
        g.MakeMove(g.MoveFromSlot(0, 0));
        Assert.Throws<InvalidOperationException>(() => g.MakeMove(new Move(0, 1))); // the card is on the board; the opponent has no cards
    }

    [Fact]
    public void Rule_parser_matches_python_substring_semantics()
    {
        var (rules, open) = RuleParser.ParsePythonRuleString("同数,加算,逆转,王牌杀手,同类强化,同类弱化,秩序,混乱,选拔,全明牌");
        Assert.Equal(RuleSet.Same | RuleSet.Plus | RuleSet.Reverse | RuleSet.FallenAce | RuleSet.Ascension
                     | RuleSet.Descension | RuleSet.Order | RuleSet.Chaos | RuleSet.Draft, rules);
        Assert.Equal(OpenMode.All, open);

        Assert.Equal(OpenMode.Three, RuleParser.ParsePythonRuleString("三明牌").OpenMode);
        Assert.Equal(OpenMode.All, RuleParser.ParsePythonRuleString("三明牌 全明牌").OpenMode); // 全明牌 wins
        Assert.Equal((RuleSet.None, OpenMode.None), RuleParser.ParsePythonRuleString("无"));
        Assert.Equal(RuleSet.Same | RuleSet.Reverse, RuleParser.ParsePythonRuleString("xx同数yy逆转").Rules);
    }

    [Fact]
    public void Card_database_matches_python_reference()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        using var doc = JsonDocument.Parse(File.ReadAllText(GoldenFiles.EnsureCardDb()));
        var cards = doc.RootElement.GetProperty("cards");
        Assert.Equal(cards.GetArrayLength(), db.Count);
        Assert.Equal(475, db.Count);
        foreach (var c in cards.EnumerateArray())
        {
            var def = db.Get(c[0].GetInt32());
            Assert.Equal(c[1].GetInt32(), def.Up);
            Assert.Equal(c[2].GetInt32(), def.Right);
            Assert.Equal(c[3].GetInt32(), def.Down);
            Assert.Equal(c[4].GetInt32(), def.Left);
            Assert.Equal(c[5].GetInt32(), def.Star);
            Assert.Equal(c[6].ValueKind == JsonValueKind.Null ? null : c[6].GetString(), CardTypes.NameOf(def.TypeId));
        }
        foreach (var l in doc.RootElement.GetProperty("lookup").EnumerateArray())
            Assert.Equal(l[4].GetInt32(), db.FindFirstIdByEdges(l[0].GetInt32(), l[1].GetInt32(), l[2].GetInt32(), l[3].GetInt32()));

        // Facts recorded in CSHARP_MIGRATION.md §1.1.
        Assert.Equal(12, db.FindFirstIdByEdges(2, 5, 5, 2)); // 12 and 369 share edges; first wins
        foreach (int id in new[] { 474, 475 })
        {
            var def = db.Get(id);
            Assert.Equal((0, 0, 0, 0, 0, 0), (def.Up, def.Right, def.Down, def.Left, def.Star, (int)def.TypeId));
        }
        Assert.Equal(new[] { 2, 53, 98, 170, 83, 69 }, Enumerable.Range(0, 6).Select(s => db.Cards.Count(c => c.Star == s)).ToArray());
        Assert.Equal(new[] { 360, 35, 22, 28, 30 }, Enumerable.Range(0, 5).Select(t => db.Cards.Count(c => c.TypeId == t)).ToArray());
    }
}
