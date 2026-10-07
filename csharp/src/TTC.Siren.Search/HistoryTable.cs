using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// Python's history dict <c>{(card_id, row, col): value}</c> as a fixed array. Keys are by card id (instances that
/// share an id share entries; <c>None</c> is its own id). Updates follow
/// <c>min(old + 2**depth, 1000000)</c>.
/// </summary>
public sealed class HistoryTable
{
    public const int Cap = 1_000_000;

    private readonly int[] _idIndexOfInstance;
    private readonly int[] _ids;
    private readonly int[] _values;

    public HistoryTable(GameState state)
    {
        int n = state.InstanceCount;
        _idIndexOfInstance = new int[n];
        var ids = new List<int>();
        for (int inst = 0; inst < n; inst++)
        {
            int id = state.InstanceId(inst);
            int idx = ids.IndexOf(id);
            if (idx < 0)
            {
                idx = ids.Count;
                ids.Add(id);
            }
            _idIndexOfInstance[inst] = idx;
        }
        _ids = ids.ToArray();
        _values = new int[_ids.Length * 9];
    }

    public int Get(int instance, int cell) => _values[_idIndexOfInstance[instance] * 9 + cell];

    /// <summary>Cutoff bonus <c>2**depth</c> with the 1,000,000 cap (Python ints do not overflow).</summary>
    public void Add(int instance, int cell, int depth)
    {
        int i = _idIndexOfInstance[instance] * 9 + cell;
        long add = depth >= 40 ? 1L << 40 : 1L << depth;
        long v = _values[i] + add;
        _values[i] = (int)(v > Cap ? Cap : v);
    }

    public void Clear() => Array.Clear(_values);

    /// <summary>Sets an entry by card id; ids not present in the state are ignored (they can never be read).</summary>
    public void SetById(int id, int row, int col, int value)
    {
        int idx = Array.IndexOf(_ids, id);
        if (idx >= 0) _values[idx * 9 + row * 3 + col] = value;
    }

    /// <summary>All non-zero entries sorted by (id, row, col) with <c>None</c> as -1. Allocates; tests only.</summary>
    public List<(int Id, int Row, int Col, int Value)> Entries()
    {
        var list = new List<(int, int, int, int)>();
        for (int idx = 0; idx < _ids.Length; idx++)
            for (int cell = 0; cell < 9; cell++)
                if (_values[idx * 9 + cell] != 0) list.Add((_ids[idx], cell / 3, cell % 3, _values[idx * 9 + cell]));
        list.Sort();
        return list;
    }
}
