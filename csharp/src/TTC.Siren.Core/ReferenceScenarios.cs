namespace TTC.Siren.Core;

/// <summary>
/// The fixed benchmark positions from CSHARP_MIGRATION.md §3.3 (plus one all-rules stress position) and a
/// deterministic random-playout driver used for the allocation test and the make/undo throughput benchmark.
/// </summary>
public static class ReferenceScenarios
{
    public static readonly string[] Names =
    {
        "open_none", "open_same_plus", "mid3_ascension", "mid4_rev_ace", "end5_same_plus", "open_all_rules",
    };

    private static CardSnapshot Card(CardDb db, int id, byte owner)
    {
        var def = db.Get(id);
        // Python Card.copy(): display edges are the clamped modified values.
        return new CardSnapshot(def.Up, def.Right, def.Down, def.Left, owner, id, CardTypes.NameOf(def.TypeId))
            .WithAppliedModifier(0);
    }

    private static GameState Build(CardDb db, (int Cell, int Id, byte Owner)[] board, int[] red, int[] blue, int current, RuleSet rules)
    {
        var s = new StateSnapshot { Current = current, Rules = rules };
        foreach (var (cell, id, owner) in board) s.Board[cell] = Card(db, id, owner);
        foreach (int id in red) s.Hands[0].Add(Card(db, id, OwnerId.Red));
        foreach (int id in blue) s.Hands[1].Add(Card(db, id, OwnerId.Blue));
        var g = GameState.FromSnapshot(s);
        g.RecalculateTypeModifiers(); // no-op unless a 同类 rule is active (Python: done at request time)
        return g;
    }

    public static GameState Create(string name, CardDb db)
    {
        int[] red5 = { 56, 64, 27, 46, 7 };
        int[] blue5 = { 59, 51, 35, 48, 8 };
        const byte R = OwnerId.Red, B = OwnerId.Blue;
        return name switch
        {
            "open_none" => Build(db, Array.Empty<(int, int, byte)>(), red5, blue5, 0, RuleSet.None),
            "open_same_plus" => Build(db, Array.Empty<(int, int, byte)>(), red5, blue5, 0, RuleSet.Same | RuleSet.Plus),
            "mid3_ascension" => Build(db, new[] { (4, 40, R), (0, 41, B), (8, 52, R) },
                new[] { 42, 43, 31, 15 }, new[] { 53, 54, 47, 14 }, 1, RuleSet.Ascension),
            "mid4_rev_ace" => Build(db, new[] { (1, 1, R), (3, 2, B), (5, 3, R), (7, 4, B) },
                new[] { 56, 64, 27 }, new[] { 59, 51, 35 }, 0, RuleSet.Reverse | RuleSet.FallenAce),
            "end5_same_plus" => Build(db, new[] { (0, 56, R), (1, 59, B), (4, 64, R), (6, 51, B), (8, 27, R) },
                new[] { 46, 7 }, new[] { 35, 48 }, 1, RuleSet.Same | RuleSet.Plus),
            "open_all_rules" => Build(db, Array.Empty<(int, int, byte)>(),
                new[] { 40, 41, 42, 43, 31 }, new[] { 53, 54, 47, 14, 15 }, 0,
                RuleSet.Same | RuleSet.Plus | RuleSet.Reverse | RuleSet.FallenAce | RuleSet.Ascension),
            _ => throw new ArgumentException($"Unknown scenario '{name}'", nameof(name)),
        };
    }

    /// <summary>
    /// Plays random games from the current position (xorshift64*, fixed seed), undoing every move after the game ends,
    /// until <paramref name="pairs"/> make/undo pairs were executed. Returns a checksum so the work cannot be elided.
    /// Allocation free.
    /// </summary>
    public static long RunPlayouts(GameState g, ulong seed, long pairs)
    {
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        ulong x = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
        long checksum = 0;
        long remaining = pairs;
        while (remaining > 0)
        {
            int depth = 0;
            while (remaining > 0)
            {
                int n = g.GenerateMoves(moves);
                if (n == 0) break;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                ulong r = x * 0x2545F4914F6CDD1DUL;
                g.MakeMove(moves[(int)((r >> 33) % (ulong)n)]);
                checksum += g.LastFlipMask;
                depth++;
                remaining--;
            }
            while (depth-- > 0) g.UndoMove();
            if (depth == -1 && remaining == pairs) break; // no legal move at all from this position
        }
        return checksum;
    }
}
