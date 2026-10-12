using System.Numerics;
using TTC.Siren.Core;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Review M4-1: hash sufficiency is checked independently of the hash's own component list. The Zobrist tests of
/// Phase 4 compare the hash with a text key that mirrors the hash, so a value-relevant field missing from both would
/// go unnoticed. These tests walk complete game trees (Make/Undo, no search code involved) and require, within one
/// <see cref="GameState"/>, that
///   (1) two positions with the same hash have the same <c>ToSnapshot()</c> state apart from the documented
///       exclusions (hand order, D-001; display edges, D-003), and
///   (2) two positions with the same hash have the same exact depth-limited minimax value (full-width brute force,
///       no pruning, no table), in both evaluation modes.
/// (1) breaks if a field that appears in the snapshot is missing from the hash. (2) breaks if ANY information that
/// matters for a value is missing, snapshot field or not. When <see cref="GameState"/> gains a field, extend the
/// snapshot (so that (1) covers it) and keep (2).
/// </summary>
public sealed class ZobristSufficiencyTests
{
    private readonly ITestOutputHelper _output;

    public ZobristSufficiencyTests(ITestOutputHelper output) => _output = output;

    private static void Walk(GameState g, int depth, int ply, Action<int> visit)
    {
        visit(ply);
        if (depth == 0 || g.IsGameOver) return;
        var moves = new Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        for (int i = 0; i < n; i++)
        {
            g.MakeMove(moves[i]);
            Walk(g, depth - 1, ply + 1, visit);
            g.UndoMove();
        }
    }

    [Fact]
    public void Equal_hashes_imply_equal_snapshots_outside_the_documented_exclusions()
    {
        string path = GoldenFiles.EnsureSearchGolden("search");
        int positions = 0, nodes = 0, repeated = 0, distinct = 0, typed = 0;

        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var g = SearchGolden.BuildState(rec, out string label);
                int depth = Math.Min(5, g.AvailableCellCount);
                if (Oracle.EstimateLeaves(g, depth) > 60_000) continue;

                var seen = new Dictionary<ulong, string>();
                Walk(g, depth, 0, _ =>
                {
                    string snapshot = SearchGolden.NormalizedState(g);
                    ulong h = g.ZobristHash;
                    if (seen.TryGetValue(h, out string? known))
                    {
                        repeated++;
                        Assert.True(known == snapshot, $"{label}: equal hash 0x{h:X16} for different states\n  first: {known}\n  other: {snapshot}");
                    }
                    else
                    {
                        seen[h] = snapshot;
                        distinct++;
                    }
                    nodes++;
                });
                if ((g.Rules & (RuleSet.Ascension | RuleSet.Descension)) != 0) typed++;
                positions++;
            }
        }

        _output.WriteLine($"positions={positions} (with 同类 rules {typed}), nodes={nodes:N0}, distinct hashes={distinct:N0}, repeated visits (transpositions)={repeated:N0}");
        Assert.True(positions >= 500, $"only {positions} positions");
        Assert.True(repeated >= 5_000, $"only {repeated} transpositions: the check would prove little");
        Assert.True(typed >= 100);
    }

    [Fact]
    public void Equal_hashes_imply_equal_exact_values_in_both_evaluation_modes()
    {
        string path = GoldenFiles.EnsureSearchGolden("search");
        int positions = 0, nodes = 0, repeated = 0, typed = 0;
        long leaves = 0;

        foreach (var doc in SearchGolden.Records(path, "search"))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                var g = SearchGolden.BuildState(rec, out string label);
                int depth = Math.Min(4, g.AvailableCellCount);
                if (Oracle.EstimateLeaves(g, depth) > 4_000) continue;
                int ai = g.CurrentPlayer;
                var eval = new Evaluator(g, SearchGolden.Db);

                foreach (bool aggressive in new[] { false, true })
                {
                    var seen = new Dictionary<ulong, long>();
                    Walk(g, depth, 0, ply =>
                    {
                        int remaining = depth - ply;
                        bool maximizing = g.CurrentPlayer == ai;
                        double value = Oracle.Brute(g, eval, remaining, maximizing, ai, aggressive, ref leaves);
                        long bits = BitConverter.DoubleToInt64Bits(value);
                        ulong h = g.ZobristHash;
                        if (seen.TryGetValue(h, out long known))
                        {
                            repeated++;
                            Assert.True(known == bits,
                                $"{label} aggressive={aggressive}: equal hash 0x{h:X16} but exact values differ " +
                                $"({SearchGolden.Describe(BitConverter.Int64BitsToDouble(known))} vs {SearchGolden.Describe(value)}) at remaining depth {remaining}");
                        }
                        else
                        {
                            seen[h] = bits;
                        }
                        nodes++;
                    });
                }
                if ((g.Rules & (RuleSet.Ascension | RuleSet.Descension)) != 0) typed++;
                positions++;
            }
        }

        _output.WriteLine($"positions={positions} (with 同类 rules {typed}), nodes={nodes:N0}, transposed visits whose value was compared={repeated:N0}, brute-force leaves={leaves:N0}");
        Assert.True(positions >= 300, $"only {positions} positions");
        Assert.True(repeated >= 2_000, $"only {repeated} transposed visits");
        Assert.True(typed >= 50);
    }
}
