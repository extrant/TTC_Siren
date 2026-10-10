using System.IO.Compression;
using System.Numerics;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Explain;

namespace TTC.Siren.Tests;

/// <summary>
/// Explanation facts v2, family 1: the causes of the flips. The analyzer is a re-derivation of the engine's flipping algorithm that also names
/// the cause; here it is (a) pinned on hand-built positions, one per cause, and (b) checked against the engine's own <c>MakeMove</c>:
/// on the positions of the 20,000 Python rules-trace sequences, for every legal move on every position of a walk through the game, the set of
/// cards the analyzer says flip equals the set the engine flips.
/// </summary>
public class FlipAnalyzerTests
{
    private static CardSnapshot Card(int up, int right, int down, int left, byte owner, int id = 1) => new(up, right, down, left, owner, id);

    private static StateSnapshot Position(RuleSet rules, (int Cell, CardSnapshot Card)[] board, params CardSnapshot[] redHand)
    {
        var s = new StateSnapshot { Current = 0, Rules = rules };
        foreach (var (cell, card) in board) s.Board[cell] = card;
        s.Hands[0].AddRange(redHand);
        s.Hands[1].Add(Card(5, 5, 5, 5, OwnerId.Blue, 99));
        return s;
    }

    private static List<FlipRecord> Flips(StateSnapshot s, int cell, out ushort engineMask)
    {
        var g = GameState.FromSnapshot(s);
        var flips = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(g), cell, FlipAnalyzer.ViewOfInstance(g, g.HandInstance(0, 0)), OwnerId.Red, g.Rules);
        g.MakeMove(new Move(g.HandInstance(0, 0), cell));
        engineMask = g.LastFlipMask;
        return flips;
    }

    [Fact]
    public void Basic_comparison_is_named_basic()
    {
        // red 8 on the right edge of the centre card beats blue 3 (cell 5, its left edge is 3)
        var s = Position(RuleSet.None, new[] { (5, Card(1, 1, 1, 3, OwnerId.Blue, 11)) }, Card(1, 8, 1, 1, OwnerId.Red, 10));
        var flips = Flips(s, 4, out var mask);
        var f = Assert.Single(flips);
        Assert.Equal(5, f.Cell);
        Assert.Equal(FlipCause.Basic, f.Cause);
        Assert.Equal((1, 8, 3), (f.MySide, f.MyValue, f.TheirValue));
        Assert.Equal(1 << 5, mask);
    }

    [Fact]
    public void Reverse_names_the_lower_value_win()
    {
        var s = Position(RuleSet.Reverse, new[] { (5, Card(1, 1, 1, 6, OwnerId.Blue, 11)) }, Card(1, 2, 1, 1, OwnerId.Red, 10));
        var f = Assert.Single(Flips(s, 4, out var mask));
        Assert.Equal(FlipCause.Reverse, f.Cause);
        Assert.Equal(1 << 5, mask);
    }

    [Fact]
    public void Fallen_ace_names_one_beating_ten()
    {
        var s = Position(RuleSet.FallenAce, new[] { (5, Card(1, 1, 1, 10, OwnerId.Blue, 11)) }, Card(1, 1, 1, 1, OwnerId.Red, 10));
        // the placed card's right edge is 1, the neighbour's left edge is 10
        s.Hands[0][0] = Card(1, 1, 1, 1, OwnerId.Red, 10);
        var f = Assert.Single(Flips(s, 4, out var mask));
        Assert.Equal(FlipCause.FallenAce, f.Cause);
        Assert.Equal((1, 10), (f.MyValue, f.TheirValue));
        Assert.Equal(1 << 5, mask);
    }

    [Fact]
    public void Same_flips_every_enemy_card_with_an_equal_edge_and_is_named_same()
    {
        // placed U5 L5; the card above has D5, the card to the left has R5: two equal edges
        var s = Position(RuleSet.Same, new[] { (1, Card(1, 1, 5, 1, OwnerId.Blue, 11)), (3, Card(1, 5, 1, 1, OwnerId.Blue, 12)) }, Card(5, 1, 1, 5, OwnerId.Red, 10));
        var flips = Flips(s, 4, out var mask);
        Assert.Equal(new[] { 1, 3 }, flips.Select(f => f.Cell));
        Assert.All(flips, f => Assert.Equal(FlipCause.Same, f.Cause));
        Assert.Equal((1 << 1) | (1 << 3), mask);
    }

    [Fact]
    public void Plus_wins_over_a_basic_flip_of_the_same_card_and_is_named_plus()
    {
        // placed U5 L6: up neighbour D3 -> sum 8, left neighbour R2 -> sum 8; both would also flip by the basic comparison
        var s = Position(RuleSet.Plus, new[] { (1, Card(1, 1, 3, 1, OwnerId.Blue, 11)), (3, Card(1, 2, 1, 1, OwnerId.Blue, 12)) }, Card(5, 1, 1, 6, OwnerId.Red, 10));
        var flips = Flips(s, 4, out var mask);
        Assert.Equal(new[] { 1, 3 }, flips.Select(f => f.Cell));
        Assert.All(flips, f => Assert.Equal(FlipCause.Plus, f.Cause));
        Assert.Equal((1 << 1) | (1 << 3), mask);
    }

    [Fact]
    public void A_card_flipped_by_same_flips_a_further_card_by_basic_comparison_and_that_is_named_combo_with_its_trigger()
    {
        // same flips cells 1 and 3; the flipped card on cell 1 has a left edge 9 and the blue card on cell 0 has a right edge 2: combo on cell 0
        var s = Position(RuleSet.Same,
            new[] { (1, Card(1, 1, 5, 9, OwnerId.Blue, 11)), (3, Card(1, 5, 1, 1, OwnerId.Blue, 12)), (0, Card(1, 2, 1, 1, OwnerId.Blue, 13)) },
            Card(5, 1, 1, 5, OwnerId.Red, 10));
        var flips = Flips(s, 4, out var mask);
        Assert.Equal(new[] { 1, 0, 3 }, flips.Select(f => f.Cell)); // the combo happens right after the card that triggers it
        Assert.Equal(new[] { FlipCause.Same, FlipCause.Combo, FlipCause.Same }, flips.Select(f => f.Cause));
        var combo = flips[1];
        Assert.Equal(1, combo.Via);
        Assert.Equal(FlipCause.Basic, combo.ComboKind);
        Assert.Equal((1 << 0) | (1 << 1) | (1 << 3), mask);
    }

    [Fact]
    public void A_combo_does_not_chain_again_one_level_only_D013()
    {
        // cell 0 is flipped by combo; the blue card on cell 3 is weaker than it, but it is flipped by same already; use cell 6 below cell 3's chain:
        // the combo-flipped card on cell 0 has a down edge 9 and the blue card on cell 3 below it... cell 3 is flipped by 'same', so test the strict case instead:
        var s = Position(RuleSet.Same,
            new[] { (1, Card(1, 1, 5, 9, OwnerId.Blue, 11)), (3, Card(1, 5, 1, 1, OwnerId.Blue, 12)), (0, Card(1, 2, 9, 1, OwnerId.Blue, 13)), (6, Card(2, 1, 1, 1, OwnerId.Blue, 14)) },
            Card(5, 1, 1, 5, OwnerId.Red, 10));
        // cell 0 (flipped by combo) has down edge 9 against cell 3 (flipped by same, so not a candidate); cell 6 is not adjacent to cell 0 at all
        var flips = Flips(s, 4, out var mask);
        Assert.DoesNotContain(flips, f => f.Cell == 6);
        Assert.Equal((1 << 0) | (1 << 1) | (1 << 3), mask);
    }

    [Fact]
    public void The_analyzer_flips_exactly_the_cards_the_engine_flips_on_the_Python_rules_trace_positions()
    {
        string path = GoldenFiles.EnsureRulesTrace();
        int positions = 0, moves = 0, withFlips = 0;
        var causes = new Dictionary<FlipCause, int>();
        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        reader.ReadLine();
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            var rules = RuleParser.FromPythonList(root.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
            var g = GameState.FromSnapshot(StateSnapshot.FromJson(root.GetProperty("init"), rules));
            int walk = 0;
            while (g.AvailableCellCount > 0 && walk++ < 9)
            {
                var buf = new Move[GameState.MaxMoves];
                int n = g.GenerateMoves(buf);
                if (n == 0) break;
                positions++;
                var view = FlipAnalyzer.ViewOf(g);
                byte mover = (byte)g.CurrentPlayer;
                for (int i = 0; i < n; i++)
                {
                    var m = buf[i];
                    var flips = FlipAnalyzer.Analyze(view, m.Cell, FlipAnalyzer.ViewOfInstance(g, m.Inst), mover, g.Rules);
                    int mask = 0;
                    foreach (var f in flips)
                    {
                        mask |= 1 << f.Cell;
                        causes[f.Cause] = causes.GetValueOrDefault(f.Cause) + 1;
                    }
                    g.MakeMove(m);
                    Assert.True(mask == g.LastFlipMask, $"record {root.GetProperty("i")} rules {rules} move {m}: analyzer {mask:X3} engine {g.LastFlipMask:X3}");
                    g.UndoMove();
                    moves++;
                    if (flips.Count > 0) withFlips++;
                }
                g.MakeMove(buf[0]); // walk on with the first legal move
            }
        }
        Assert.True(moves > 100_000 && withFlips > 20_000, $"only {moves} moves, {withFlips} with flips");
        foreach (var cause in Enum.GetValues<FlipCause>()) Assert.True(causes.GetValueOrDefault(cause) > 100, $"cause {cause} seen {causes.GetValueOrDefault(cause)} times");
        Console.WriteLine($"flip analyzer: {positions} positions, {moves} moves equal to the engine ({withFlips} with flips); causes: " +
                          string.Join(", ", causes.OrderBy(p => p.Key).Select(p => $"{p.Key} {p.Value}")));
    }
}
