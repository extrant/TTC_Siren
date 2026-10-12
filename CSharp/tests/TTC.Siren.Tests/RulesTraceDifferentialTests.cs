using System.IO.Compression;
using System.Text.Json;
using TTC.Siren.Core;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 2 differential test: replays Python-generated make/undo sequences and demands that the complete
/// state after every step (cards, owners, modifiers, display edges, ordered hands, side to move, legal moves,
/// card counts) equals what the Python reference exported.
/// </summary>
public sealed class RulesTraceDifferentialTests
{
    private readonly ITestOutputHelper _output;

    public RulesTraceDifferentialTests(ITestOutputHelper output) => _output = output;

    private static string Fail(int seed, int index, string where, string expected, string actual) =>
        $"PARITY MISMATCH seed={seed} sequence={index} {where}\n  expected: {expected}\n  actual:   {actual}\n" +
        $"  regenerate: python tools/parity/py_rules_trace.py --out x.jsonl.gz --seed {seed} --count {index + 1}";

    private static string Moves(IEnumerable<int> m) => string.Join(",", m);

    private static void CheckMoves(GameState g, JsonElement expected, int seed, int index, string where)
    {
        Span<Move> buf = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(buf);
        var actual = new int[n];
        for (int i = 0; i < n; i++) actual[i] = g.HandSlotOf(g.CurrentPlayer, buf[i].Inst) * 9 + buf[i].Cell; // golden codes are slot * 9 + cell
        var exp = expected.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Assert.True(exp.AsSpan().SequenceEqual(actual), Fail(seed, index, where + " moves", Moves(exp), Moves(actual)));
    }

    private static void CheckCounts(GameState g, JsonElement expected, int seed, int index, string where)
    {
        g.CountCards(out int red, out int blue);
        int winner = g.GetWinner();
        string exp = $"{expected[0].GetInt32()},{expected[1].GetInt32()},{expected[2].GetInt32()}";
        string act = $"{red},{blue},{winner}";
        Assert.True(exp == act, Fail(seed, index, where + " counts/winner", exp, act));
    }

    [Fact]
    public void Python_golden_sequences_match_after_every_make_and_undo()
    {
        string path = GoldenFiles.EnsureRulesTrace();
        var cov = new CoverageCounters();
        var maskSeen = new int[256];
        int sequences = 0, steps = 0, makes = 0, undos = 0, seed = 0;
        int dupSequences = 0, dupSteps = 0, dupMakes = 0;

        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);

        using (var header = JsonDocument.Parse(reader.ReadLine()!))
        {
            var h = header.RootElement.GetProperty("header");
            seed = h.GetProperty("seed").GetInt32();
            Assert.Equal(GoldenFiles.GeneratorVersion, h.GetProperty("generator").GetString());
        }

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            int index = root.GetProperty("i").GetInt32();
            var ruleNames = root.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!).ToList();
            RuleSet rules = RuleParser.FromPythonList(ruleNames);
            maskSeen[(int)rules & 0xFF]++;

            var initSnap = StateSnapshot.FromJson(root.GetProperty("init"), rules);
            var g = GameState.FromSnapshot(initSnap);
            string expInit = initSnap.ToCanonicalString();
            string actInit = g.ToSnapshot().ToCanonicalString();
            Assert.True(expInit == actInit, Fail(seed, index, "init round-trip", expInit, actInit));
            CheckMoves(g, root.GetProperty("init_moves"), seed, index, "init");
            CheckCounts(g, root.GetProperty("init_cw"), seed, index, "init");

            g.Coverage = cov;
            bool dupSequence = SearchGolden.HasDuplicateHandIds(g); // D-004: a hand with two cards of one id
            if (dupSequence) dupSequences++;
            int stepNo = 0;
            foreach (var step in root.GetProperty("steps").EnumerateArray())
            {
                string where = $"step {stepNo}";
                string op = step.GetProperty("op").GetString()!;
                if (op == "make")
                {
                    int cell = step.GetProperty("cell").GetInt32();
                    int slot = step.GetProperty("slot").GetInt32();
                    var played = g.MoveFromSlot(slot, cell); // slot resolved against the hand at this exact step
                    if (SearchGolden.SameIdInHand(g, g.CurrentPlayer, played.Inst) >= 2) dupMakes++;
                    g.MakeMove(played);
                    int expFlips = 0;
                    foreach (var f in step.GetProperty("flips").EnumerateArray()) expFlips |= 1 << f.GetInt32();
                    Assert.True(expFlips == g.LastFlipMask,
                        Fail(seed, index, where + " make flipped cells", expFlips.ToString("X"), g.LastFlipMask.ToString("X")));
                    makes++;
                }
                else
                {
                    Assert.True(g.UndoMove(), Fail(seed, index, where, "undo ok", "undo returned false"));
                    undos++;
                }

                var expState = StateSnapshot.FromJson(step.GetProperty("after"), rules);
                string exp = expState.ToCanonicalString();
                string act = g.ToSnapshot().ToCanonicalString();
                Assert.True(exp == act, Fail(seed, index, where + $" after {op}", exp, act));
                CheckMoves(g, step.GetProperty("moves"), seed, index, where);
                CheckCounts(g, step.GetProperty("cw"), seed, index, where);
                stepNo++;
                steps++;
                if (dupSequence) dupSteps++;
            }
            sequences++;
        }

        _output.WriteLine($"seed={seed} sequences={sequences} steps={steps} (makes={makes}, undos={undos})");
        _output.WriteLine("coverage: " + cov);
        _output.WriteLine($"D-004 duplicate hand ids: sequences={dupSequences} ({100.0 * dupSequences / sequences:F1}%), steps in them={dupSteps}, makes of a card whose id is duplicated in the hand={dupMakes}");
        Assert.True(dupSequences * 20 >= sequences, "less than 5% of the sequences have duplicate hand ids");
        Assert.True(dupMakes >= 200, $"only {dupMakes} makes of a duplicated-id card");

        Assert.True(sequences >= 20000, $"expected >= 20000 sequences, replayed {sequences}");

        // Rule coverage of the generated population: all 256 subsets of the 8 settlement rules (>= 1,000 random subsets).
        int subsetsSeen = maskSeen.Count(c => c > 0);
        _output.WriteLine($"distinct rule subsets: {subsetsSeen}/256, min sequences per subset: {maskSeen.Min()}");
        Assert.Equal(256, subsetsSeen);
        Assert.True(maskSeen.Sum() >= 1000);
        RuleSet[] singles =
        {
            RuleSet.Same, RuleSet.Plus, RuleSet.Reverse, RuleSet.FallenAce,
            RuleSet.Ascension, RuleSet.Descension, RuleSet.Order, RuleSet.Chaos,
        };
        foreach (var single in singles) Assert.True(maskSeen[(int)single] > 0, $"no sequence with only {single}");
        for (int a = 0; a < singles.Length; a++)
            for (int b = a + 1; b < singles.Length; b++)
                Assert.True(maskSeen[(int)(singles[a] | singles[b])] > 0, $"no sequence with only {singles[a]}+{singles[b]}");

        // Event coverage: every rule path must have been triggered at least 100 times.
        void AtLeast100(string name, long value) => Assert.True(value >= 100, $"coverage '{name}' = {value} (< 100)\n{cov}");
        AtLeast100("base flip", cov.BaseFlip);
        AtLeast100("same flip", cov.SameFlip);
        AtLeast100("plus flip", cov.PlusFlip);
        AtLeast100("combo flip", cov.ComboFlip);
        AtLeast100("reverse win", cov.ReverseWin);
        AtLeast100("ace 1 vs 10", cov.Ace1v10);
        AtLeast100("ace 10 vs 1", cov.Ace10v1);
        AtLeast100("ace 1 vs 10 + reverse", cov.Ace1v10Reverse);
        AtLeast100("ace 10 vs 1 + reverse", cov.Ace10v1Reverse);
        AtLeast100("ascension recalculation", cov.AscensionRecalc);
        AtLeast100("descension recalculation", cov.DescensionRecalc);
        AtLeast100("D-002 modifier restore by identity", cov.UndoModifierChanged);
    }
}
