using System.Numerics;
using System.Runtime.CompilerServices;

namespace TTC.Siren.Search;

/// <summary>What a stored score says about the true value of the position at the stored depth.</summary>
public enum Bound : byte
{
    /// <summary>The score is the exact value (it lay strictly inside the search window).</summary>
    Exact = 0,

    /// <summary>The true value is at least the score (the search failed high: <c>score &gt;= beta</c>).</summary>
    Lower = 1,

    /// <summary>The true value is at most the score (the search failed low: <c>score &lt;= alpha</c>).</summary>
    Upper = 2,
}

/// <summary>
/// Transposition table of the Fast search (Phase 4). Unlike the ReferenceCompat table (an emulation of Python's
/// unbounded dict, D-005/D-006) it is fixed-size (a power of two, allocated once, never grows), keyed by the 64-bit
/// Zobrist hash, stores a bound type with every score, and replaces entries by depth.
///
/// Layout: two-entry buckets (slots <c>2k</c> and <c>2k+1</c>). A probe looks at both slots of the bucket. A store
/// overwrites the entry with the same key if present, otherwise an empty or stale-generation slot, otherwise the
/// shallower of the two (so deep entries survive and the newest entry always gets a slot). The table never refuses a
/// store. <see cref="NewGeneration"/> empties it in O(1) (a generation byte; the array is cleared only when the byte
/// wraps). Only the full 64-bit key is compared; with the bucket index that is 64 + log2(capacity/2) bits, so a false
/// match needs a collision that is astronomically unlikely at the table sizes in use (see CSHARP_MIGRATION.md).
///
/// Validity context (review H4-1). The key identifies a position only inside one <see cref="Core.GameState"/>, and a
/// stored score is from a fixed AI perspective under one evaluation mode. The table therefore records the
/// <see cref="SearchContext"/> (state object, AI side, evaluation mode) of its current generation, and
/// <see cref="RequireContext"/> refuses reuse under any other context. A new table is unbound: it can only be used
/// after a generation was started with a context. The check happens once per search, never per node.
/// </summary>
public sealed class FastTranspositionTable
{
    public const int DefaultCapacity = 1 << 22; // 4 Mi entries, 96 MiB

    public struct Entry
    {
        public ulong Key;
        public double Score;
        public sbyte Depth;
        public Bound Bound;
        public sbyte BestInstance; // -1 = no move recorded
        public byte BestCell;
        public byte Generation;    // 0 = empty
    }

    public static readonly int EntryBytes = Unsafe.SizeOf<Entry>();

    private readonly Entry[] _entries;
    private readonly int _bucketMask; // index of the first slot of the bucket = (key & _bucketMask) with the low bit cleared
    private byte _generation = 1;
    private SearchContext _context;
    private bool _bound;

    /// <summary>True once a generation was started with a context (and not reset by the context-free overload).</summary>
    public bool IsBound => _bound;

    /// <summary>The context of the current generation. Only meaningful if <see cref="IsBound"/>.</summary>
    public SearchContext Context => _context;

    public int Capacity => _entries.Length;
    public long MemoryBytes => (long)_entries.Length * EntryBytes;

    /// <summary>Number of stores since construction (diagnostics).</summary>
    public long Stores { get; private set; }

    public FastTranspositionTable(int capacity = DefaultCapacity)
    {
        if (!BitOperations.IsPow2(capacity) || capacity < 4)
            throw new ArgumentException("capacity must be a power of two >= 4", nameof(capacity));
        _entries = new Entry[capacity];
        _bucketMask = capacity - 2;
    }

    /// <summary>
    /// Empties the table in O(1) (previous entries become invisible) and binds it to <paramref name="context"/>: only
    /// searches with an equal context may continue on its entries.
    /// </summary>
    public void NewGeneration(in SearchContext context)
    {
        NewGeneration();
        _context = context;
        _bound = true;
    }

    /// <summary>
    /// Throws <see cref="InvalidOperationException"/> unless the table is bound to a context equal to
    /// <paramref name="context"/>. Changes nothing.
    /// </summary>
    public void RequireContext(in SearchContext context)
    {
        if (!_bound)
            throw new InvalidOperationException(
                "Cannot reuse the transposition table (clearTable:false): it is not bound to any search context " +
                "(never used, or emptied without a context). Start the search with clearTable:true.");
        string? mismatch = _context.DescribeMismatch(context);
        if (mismatch is not null)
            throw new InvalidOperationException(
                "Cannot reuse the transposition table (clearTable:false): " + mismatch + ". Start the search with clearTable:true.");
    }

    /// <summary>
    /// Empties the table in O(1) and marks it unbound (no search may reuse it with <c>clearTable:false</c> until a
    /// generation is started with a context). Prefer <see cref="NewGeneration(in SearchContext)"/>.
    /// </summary>
    public void NewGeneration()
    {
        _bound = false;
        _context = default;
        if (_generation == byte.MaxValue)
        {
            Array.Clear(_entries);
            _generation = 1;
        }
        else
        {
            _generation++;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGet(ulong key, out Entry entry)
    {
        int i = (int)(key & (ulong)_bucketMask);
        ref Entry a = ref _entries[i];
        if (a.Generation == _generation && a.Key == key)
        {
            entry = a;
            return true;
        }
        ref Entry b = ref _entries[i + 1];
        if (b.Generation == _generation && b.Key == key)
        {
            entry = b;
            return true;
        }
        entry = default;
        return false;
    }

    public void Store(ulong key, int depth, double score, Bound bound, int bestInstance, int bestCell)
    {
        Stores++;
        int i = (int)(key & (ulong)_bucketMask);
        ref Entry a = ref _entries[i];
        ref Entry b = ref _entries[i + 1];

        ref Entry slot = ref a;
        if (a.Generation == _generation && a.Key == key)
        {
            slot = ref a;
        }
        else if (b.Generation == _generation && b.Key == key)
        {
            slot = ref b;
        }
        else if (a.Generation != _generation)
        {
            slot = ref a;
        }
        else if (b.Generation != _generation)
        {
            slot = ref b;
        }
        else
        {
            slot = ref (a.Depth <= b.Depth ? ref a : ref b);
        }

        slot.Key = key;
        slot.Score = score;
        slot.Depth = (sbyte)depth;
        slot.Bound = bound;
        slot.BestInstance = (sbyte)bestInstance;
        slot.BestCell = (byte)bestCell;
        slot.Generation = _generation;
    }
}
