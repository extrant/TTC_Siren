using System.Numerics;
using System.Runtime.CompilerServices;
using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// Thrown when the table would have to grow beyond its configured hard cap. The searcher unwinds the game state
/// before the exception leaves the public entry point (review H-1).
/// </summary>
public sealed class TranspositionTableFullException : Exception
{
    public int MaxCapacity { get; }
    public long Entries { get; }

    public TranspositionTableFullException(int maxCapacity, long entries)
        : base($"ReferenceCompat transposition table hard cap reached: cap = {maxCapacity} slots " +
               $"(load limit 75%), {entries} entries in use. Python's table is unbounded; " +
               "construct CompatTranspositionTable with a larger maxCapacity if this memory is acceptable.")
    {
        MaxCapacity = maxCapacity;
        Entries = entries;
    }
}

/// <summary>
/// Emulation of Python's global <c>TRANSPOSITION_TABLE</c> dict for the ReferenceCompat searcher.
/// Equivalence of keys is exactly that of <c>get_state_hash</c> (see <see cref="PythonStateKey"/>, including the
/// D-005 hand-id concatenation collisions). Entries carry no bound type (D-006): a hit is accepted iff
/// <c>entry.depth &gt;= depth</c>, and every node overwrites its entry.
/// Python's dict never refuses an insert, so this table grows on demand: when an insert would exceed the 75% load
/// limit, the slot array doubles and the live entries of the current generation are reinserted (lookups compare the
/// full 128-bit key, so reinsertion order cannot change any result). Growth happens only inside <see cref="Store"/>
/// and is rare and amortised, so steady-state searching on a pre-sized table still allocates nothing. A configurable
/// hard cap bounds memory; reaching it throws <see cref="TranspositionTableFullException"/>.
/// <see cref="NewGeneration"/> empties the table in O(1) and keeps the grown capacity.
/// </summary>
public sealed class CompatTranspositionTable
{
    public const int DefaultInitialCapacity = 1 << 21;

    /// <summary>1 GiB of 32-byte entries (see <see cref="EntryBytes"/>).</summary>
    public const int DefaultMaxCapacity = 1 << 25;

    public struct Entry
    {
        public ulong H1;
        public ulong H2;
        public double Score;
        public int Generation;
        public short Depth;
        public sbyte BestInstance; // -1 = Python None
        public byte BestCell;
    }

    public static readonly int EntryBytes = Unsafe.SizeOf<Entry>();

    private Entry[] _entries;
    private int _mask;
    private readonly int _maxCapacity;
    private int _generation = 1;
    private long _used;

    /// <summary>Current number of slots.</summary>
    public int Capacity => _entries.Length;

    /// <summary>Entries stored in the current generation.</summary>
    public long Used => _used;

    /// <summary>Largest number of entries any generation has held.</summary>
    public long PeakUsed { get; private set; }

    /// <summary>Number of times the slot array was doubled (diagnostics).</summary>
    public int Grows { get; private set; }

    public int MaxCapacity => _maxCapacity;

    /// <summary>Approximate memory held by the slot array.</summary>
    public long MemoryBytes => (long)_entries.Length * EntryBytes;

    public CompatTranspositionTable(int initialCapacity = DefaultInitialCapacity, int maxCapacity = DefaultMaxCapacity)
    {
        if (!BitOperations.IsPow2(initialCapacity) || initialCapacity < 4)
            throw new ArgumentException("initial capacity must be a power of two >= 4", nameof(initialCapacity));
        if (!BitOperations.IsPow2(maxCapacity) || maxCapacity < initialCapacity)
            throw new ArgumentException("max capacity must be a power of two >= the initial capacity", nameof(maxCapacity));
        _entries = new Entry[initialCapacity];
        _mask = initialCapacity - 1;
        _maxCapacity = maxCapacity;
    }

    /// <summary>Python: <c>TRANSPOSITION_TABLE = {}</c>.</summary>
    public void NewGeneration()
    {
        _generation++;
        _used = 0;
    }

    public bool TryGet(ulong h1, ulong h2, out Entry entry)
    {
        int i = (int)(h1 & (ulong)_mask);
        while (true)
        {
            ref Entry e = ref _entries[i];
            if (e.Generation != _generation)
            {
                entry = default;
                return false;
            }
            if (e.H1 == h1 && e.H2 == h2)
            {
                entry = e;
                return true;
            }
            i = (i + 1) & _mask;
        }
    }

    public void Store(ulong h1, ulong h2, int depth, double score, int bestInstance, int bestCell)
    {
        int i = (int)(h1 & (ulong)_mask);
        while (true)
        {
            ref Entry e = ref _entries[i];
            if (e.Generation != _generation)
            {
                // New key: make room first if the load limit would be exceeded, then insert into the (possibly new) array.
                if ((_used + 1) * 4 > (long)_entries.Length * 3)
                {
                    Grow();
                    i = (int)(h1 & (ulong)_mask);
                    while (_entries[i].Generation == _generation) i = (i + 1) & _mask;
                    e = ref _entries[i];
                }
                _used++;
                if (_used > PeakUsed) PeakUsed = _used;
                e.Generation = _generation;
                e.H1 = h1;
                e.H2 = h2;
                e.Depth = (short)depth;
                e.Score = score;
                e.BestInstance = (sbyte)bestInstance;
                e.BestCell = (byte)bestCell;
                return;
            }
            if (e.H1 == h1 && e.H2 == h2)
            {
                e.Depth = (short)depth;
                e.Score = score;
                e.BestInstance = (sbyte)bestInstance;
                e.BestCell = (byte)bestCell;
                return;
            }
            i = (i + 1) & _mask;
        }
    }

    private void Grow()
    {
        if (_entries.Length >= _maxCapacity) throw new TranspositionTableFullException(_maxCapacity, _used);

        var old = _entries;
        var grown = new Entry[old.Length * 2];
        int mask = grown.Length - 1;
        foreach (ref readonly Entry e in old.AsSpan())
        {
            if (e.Generation != _generation) continue; // stale generations are logically empty
            int i = (int)(e.H1 & (ulong)mask);
            while (grown[i].Generation == _generation) i = (i + 1) & mask;
            grown[i] = e;
        }
        _entries = grown;
        _mask = mask;
        Grows++;
    }
}
/// <summary>
/// 128-bit digest of the information in Python's <c>get_state_hash</c> string:
/// per board cell (owner, card id or None, the four BASE edge numbers, type_modifier), the concatenated hand card
/// ids of both players (red hand first, <c>None</c> printed as "None", no separators: D-005), and the player to move.
/// Not part of the key (so also not here): can_use, card type, hand cards' modifiers/owners, stale display edges.
/// Equality of the digests is equality of the Python strings up to a 2^-128 collision probability; the test
/// <c>PythonStateKeyTests</c> checks the induced partition against Python's SHA-1 of the real key strings.
/// </summary>
public static class PythonStateKey
{
    private const ulong K1 = 0xA0761D6478BD642FUL;
    private const ulong K2 = 0xE7037ED1A0B428DBUL;
    private const ulong K3 = 0x8EBC6AF09C88C6E3UL;

    private static ulong Mix(ulong a, ulong b)
    {
        ulong hi = Math.BigMul(a, b, out ulong lo);
        return hi ^ lo;
    }

    public static void Compute(GameState g, out ulong h1, out ulong h2)
    {
        // 9 cells * 12 bytes + at most 10 hand ids * 11 chars + 1 = 219 bytes.
        Span<byte> buf = stackalloc byte[256];
        int n = 0;
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = g.BoardInstance(cell);
            if (inst < 0)
            {
                buf[n++] = 0;
                continue;
            }
            buf[n++] = (byte)(1 + g.InstanceOwner(inst));
            int id = g.InstanceId(inst);
            buf[n++] = (byte)(id == -1 ? 0 : 1);
            buf[n++] = (byte)id;
            buf[n++] = (byte)(id >> 8);
            buf[n++] = (byte)(id >> 16);
            buf[n++] = (byte)(id >> 24);
            for (int side = 0; side < 4; side++) buf[n++] = (byte)g.InstanceBase(inst, side);
            buf[n++] = (byte)g.InstanceModifier(inst);
        }

        Span<byte> digits = stackalloc byte[10];
        for (int p = 0; p < 2; p++)
        {
            int hc = g.HandCount(p);
            for (int i = 0; i < hc; i++)
            {
                int id = g.InstanceId(g.HandInstance(p, i));
                if (id == -1)
                {
                    buf[n++] = (byte)'N'; buf[n++] = (byte)'o'; buf[n++] = (byte)'n'; buf[n++] = (byte)'e';
                    continue;
                }
                if (id < 0) buf[n++] = (byte)'-';
                uint v = id < 0 ? (uint)(-(long)id) : (uint)id;
                int d = 0;
                do { digits[d++] = (byte)('0' + v % 10); v /= 10; } while (v != 0);
                while (d > 0) buf[n++] = digits[--d];
            }
        }
        buf[n++] = (byte)(0xF0 | g.CurrentPlayer); // outside the hand-char alphabet: terminates the id string

        ulong a = 0x243F6A8885A308D3UL ^ (ulong)n;
        ulong b = 0x13198A2E03707344UL + (ulong)n * K3;
        int pos = 0;
        while (pos < n)
        {
            ulong w = 0;
            int take = Math.Min(8, n - pos);
            for (int j = 0; j < take; j++) w |= (ulong)buf[pos + j] << (8 * j);
            pos += take;
            a = Mix(a ^ w, K1);
            b = Mix(b + w, K2);
        }
        h1 = Mix(a ^ K3, b ^ K1);
        h2 = Mix(b ^ K2, a + K3);
    }
}
