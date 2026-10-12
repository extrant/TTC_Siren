namespace TTC.Siren.Core;

/// <summary>
/// Fixed random keys for the incremental Zobrist hash of <see cref="GameState"/> (Phase 4). The keys come from a
/// SplitMix64 stream with a constant seed, so hashes are identical in every process and run (deterministic search).
///
/// What the hash covers (the information that decides the value of a position within one search of a fixed root):
/// the card instance on each board cell, the owner of every card instance (board and hands), the type modifier of
/// every card instance, and the side to move. Hand membership is implied by the board (a card is in its hand until
/// it is placed). It deliberately does NOT cover: hand order (D-001), display edges (D-003) and the undo history; they
/// only influence move ordering, never a score (<c>EvaluateState</c> reads owners and effective edges only).
///
/// Hash domain: ONE <see cref="GameState"/> object. Instance indices are reused between states, and card ids, edges,
/// types, rules and <c>can_use</c> (invariants of a state) are not hashed, so equal hashes of two different states
/// say nothing. A table keyed by this hash must therefore also record which state (and which AI side and evaluation
/// mode) it was filled for; <c>FastTranspositionTable</c> does this with a <c>SearchContext</c> (review H4-1).
/// </summary>
public static class ZobristKeys
{
    public const int ModStates = 256; // sbyte modifier, indexed by its byte pattern

    private static readonly ulong[] CellKeys = new ulong[9 * GameState.MaxInstances];
    private static readonly ulong[] OwnerKeys = new ulong[GameState.MaxInstances * 3];
    private static readonly ulong[] ModKeys = new ulong[GameState.MaxInstances * ModStates];

    public static readonly ulong Side;

    static ZobristKeys()
    {
        ulong state = 0x5EED_0F_7A11_D1CEUL;
        for (int i = 0; i < CellKeys.Length; i++) CellKeys[i] = Next(ref state);
        for (int i = 0; i < OwnerKeys.Length; i++) OwnerKeys[i] = Next(ref state);
        for (int i = 0; i < ModKeys.Length; i++) ModKeys[i] = Next(ref state);
        Side = Next(ref state);
    }

    private static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static ulong Cell(int cell, int inst) => CellKeys[cell * GameState.MaxInstances + inst];

    public static ulong Owner(int inst, int owner) => OwnerKeys[inst * 3 + owner];

    public static ulong Mod(int inst, int modifier) => ModKeys[inst * ModStates + (byte)modifier];
}
