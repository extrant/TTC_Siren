using System.IO.Compression;
using System.Text;
using System.Text.Json;
using TTC.Siren.Core;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>Phase 4: the incremental Zobrist hash of <see cref="GameState"/>.</summary>
public sealed class ZobristTests
{
    private readonly ITestOutputHelper _output;

    public ZobristTests(ITestOutputHelper output) => _output = output;

    /// <summary>The information the hash is meant to cover, as text: side to move, board cell -> instance, owner and modifier of every instance.</summary>
    private static string HashRelevantKey(GameState g)
    {
        var sb = new StringBuilder();
        sb.Append(g.CurrentPlayer).Append('|');
        for (int cell = 0; cell < 9; cell++) sb.Append(g.BoardInstance(cell)).Append(',');
        sb.Append('|');
        for (int inst = 0; inst < g.InstanceCount; inst++)
            sb.Append(g.InstanceOwner(inst)).Append(':').Append(g.InstanceModifier(inst)).Append(',');
        return sb.ToString();
    }

    [Fact]
    public void Incremental_hash_equals_from_scratch_over_the_python_golden_sequences()
    {
        string path = GoldenFiles.EnsureRulesTrace();
        int sequences = 0, checks = 0, makes = 0, undos = 0, distinctStates = 0;

        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        reader.ReadLine(); // header (validated by the rules differential test)

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            int index = root.GetProperty("i").GetInt32();
            var rules = RuleParser.FromPythonList(root.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
            var g = GameState.FromSnapshot(StateSnapshot.FromJson(root.GetProperty("init"), rules));

            var keyToHash = new Dictionary<string, ulong>();
            var hashToKey = new Dictionary<ulong, string>();
            var before = new Stack<ulong>();

            void Check(string where)
            {
                ulong inc = g.ZobristHash;
                Assert.True(inc == g.ComputeZobristHash(), $"sequence {index} {where}: incremental hash 0x{inc:X16} != from scratch 0x{g.ComputeZobristHash():X16}");
                string key = HashRelevantKey(g);
                if (keyToHash.TryGetValue(key, out ulong known))
                    Assert.True(known == inc, $"sequence {index} {where}: equal positions have different hashes");
                else
                    keyToHash[key] = inc;
                if (hashToKey.TryGetValue(inc, out string? knownKey))
                    Assert.True(knownKey == key, $"sequence {index} {where}: hash collision between different positions");
                else
                    hashToKey[inc] = key;
                checks++;
            }

            Check("init");
            foreach (var step in root.GetProperty("steps").EnumerateArray())
            {
                if (step.GetProperty("op").GetString() == "make")
                {
                    before.Push(g.ZobristHash);
                    g.MakeMove(g.MoveFromSlot(step.GetProperty("slot").GetInt32(), step.GetProperty("cell").GetInt32()));
                    makes++;
                    Check("after make");
                }
                else
                {
                    g.UndoMove();
                    undos++;
                    Check("after undo");
                    Assert.True(before.Pop() == g.ZobristHash, $"sequence {index}: undo did not restore the pre-move hash");
                }
            }
            distinctStates += keyToHash.Count;
            sequences++;
        }

        _output.WriteLine($"sequences={sequences} hash checks={checks} (makes={makes}, undos={undos}) distinct positions={distinctStates}");
        Assert.True(sequences >= 20000);
        Assert.True(checks > 150_000);
    }

    [Fact]
    public void Hash_ignores_hand_order_and_display_edges_and_identifies_transpositions()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_none", db);
        ulong start = g.ZobristHash;

        // make + undo reorders the hand (D-001) but restores the position.
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        g.MakeMove(moves[n / 2]);
        g.UndoMove();
        Assert.Equal(start, g.ZobristHash);
        Assert.Equal(start, g.ComputeZobristHash());

        // Same four placements in two orders (corners are not adjacent, so nothing is flipped): same position, same hash.
        int[] red = { g.HandInstance(0, 0), g.HandInstance(0, 1) };
        int[] blue = { g.HandInstance(1, 0), g.HandInstance(1, 1) };
        void Play(bool redFirst)
        {
            if (redFirst)
            {
                g.MakeMove(new Move(red[0], 0));
                g.MakeMove(new Move(blue[0], 8));
                g.MakeMove(new Move(red[1], 2));
                g.MakeMove(new Move(blue[1], 6));
            }
            else
            {
                g.MakeMove(new Move(red[1], 2));
                g.MakeMove(new Move(blue[1], 6));
                g.MakeMove(new Move(red[0], 0));
                g.MakeMove(new Move(blue[0], 8));
            }
        }

        Play(true);
        ulong a = g.ZobristHash;
        string stateA = SearchGolden.NormalizedState(g);
        for (int i = 0; i < 4; i++) g.UndoMove();
        Assert.Equal(start, g.ZobristHash);
        Play(false);
        Assert.Equal(stateA, SearchGolden.NormalizedState(g));
        Assert.Equal(a, g.ZobristHash);
        for (int i = 0; i < 4; i++) g.UndoMove();

        // The side to move is part of the key: the same board after an odd number of plies differs by the side key only.
        g.MakeMove(new Move(red[0], 0));
        Assert.NotEqual(start, g.ZobristHash);
        Assert.Equal(1, g.CurrentPlayer);
    }

    [Fact]
    public void Same_type_modifiers_are_part_of_the_hash()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("mid3_ascension", db);
        ulong start = g.ZobristHash;
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = g.GenerateMoves(moves);
        int modifierChanges = 0;
        for (int i = 0; i < n; i++)
        {
            var before = new int[g.InstanceCount];
            for (int inst = 0; inst < before.Length; inst++) before[inst] = g.InstanceModifier(inst);
            g.MakeMove(moves[i]);
            Assert.Equal(g.ComputeZobristHash(), g.ZobristHash);
            for (int inst = 0; inst < before.Length; inst++)
                if (before[inst] != g.InstanceModifier(inst)) { modifierChanges++; break; }
            g.UndoMove();
            Assert.Equal(start, g.ZobristHash);
        }
        Assert.True(modifierChanges > 0, "the scenario never changed a modifier, the test would prove nothing");
    }

    [Fact]
    public void Make_and_undo_with_the_hash_still_allocate_nothing()
    {
        var db = SearchGolden.Db;
        var g = ReferenceScenarios.Create("open_all_rules", db);
        ReferenceScenarios.RunPlayouts(g, 1, 100_000);
        long before = GC.GetAllocatedBytesForCurrentThread();
        long checksum = ReferenceScenarios.RunPlayouts(g, 7, 500_000);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.NotEqual(long.MinValue, checksum);
        Assert.Equal(0, allocated);
        Assert.Equal(g.ComputeZobristHash(), g.ZobristHash);
    }
}
