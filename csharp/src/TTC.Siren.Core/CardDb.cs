using System.Text;

namespace TTC.Siren.Core;

/// <summary>Static card definition from the card database CSV.</summary>
public readonly record struct CardDef(ushort Id, byte Up, byte Right, byte Down, byte Left, byte Star, byte TypeId);

/// <summary>
/// Card database loaded from <c>data/幻卡数据库.csv</c> (UTF-8 BOM, columns read by name).
/// Includes the all-zero placeholder rows (ids 474/475) exactly like Python's <c>get_all_cards</c>.
/// </summary>
public sealed class CardDb
{
    public const string DataFileName = "幻卡数据库.csv";

    private readonly CardDef[] _byId;
    private readonly bool[] _present;
    private readonly Dictionary<uint, ushort> _firstIdByEdges;

    public IReadOnlyList<CardDef> Cards { get; }

    public int Count => Cards.Count;

    private CardDb(List<CardDef> cards)
    {
        Cards = cards;
        int maxId = 0;
        foreach (var c in cards) maxId = Math.Max(maxId, c.Id);
        _byId = new CardDef[maxId + 1];
        _present = new bool[maxId + 1];
        _firstIdByEdges = new Dictionary<uint, ushort>(cards.Count);
        foreach (var c in cards)
        {
            _byId[c.Id] = c;
            _present[c.Id] = true;
            // Python get_card_lookup: first row wins for identical (Top, Right, Bottom, Left).
            _firstIdByEdges.TryAdd(EdgeKey(c.Up, c.Right, c.Down, c.Left), c.Id);
        }
    }

    private static uint EdgeKey(int up, int right, int down, int left) =>
        (uint)up | (uint)right << 8 | (uint)down << 16 | (uint)left << 24;

    public bool TryGet(int id, out CardDef def)
    {
        if ((uint)id < (uint)_byId.Length && _present[id])
        {
            def = _byId[id];
            return true;
        }
        def = default;
        return false;
    }

    public CardDef Get(int id) =>
        TryGet(id, out var def) ? def : throw new KeyNotFoundException($"Card id {id} not in database");

    /// <summary>First database id with exactly these edges (Python <c>get_card_lookup</c>), or 0 if none.</summary>
    public int FindFirstIdByEdges(int up, int right, int down, int left) =>
        _firstIdByEdges.TryGetValue(EdgeKey(up, right, down, left), out ushort id) ? id : 0;

    /// <summary>Walks up from <paramref name="startDir"/> looking for <c>data/幻卡数据库.csv</c>.</summary>
    public static string LocateDataFile(string startDir)
    {
        string? dir = Path.GetFullPath(startDir);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir, "data", DataFileName);
            if (File.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        throw new FileNotFoundException($"Could not find data/{DataFileName} above '{startDir}'");
    }

    public static CardDb LoadFromRepository(string startDir) => Load(LocateDataFile(startDir));

    public static CardDb Load(string path)
    {
        using var reader = new StreamReader(path, new UTF8Encoding(false), detectEncodingFromByteOrderMarks: true);
        var rows = ParseCsv(reader);
        if (rows.Count == 0) throw new InvalidDataException("Empty card database");

        var header = rows[0];
        int Col(string name)
        {
            int idx = Array.IndexOf(header, name);
            return idx >= 0 ? idx : throw new InvalidDataException($"Missing column '{name}'");
        }

        int cId = Col("序号"), cUp = Col("Top"), cDown = Col("Bottom"), cLeft = Col("Left"), cRight = Col("Right");
        int cStar = Col("星级"), cType = Col("TripleTriadCardType");

        var cards = new List<CardDef>(rows.Count - 1);
        for (int i = 1; i < rows.Count; i++)
        {
            var r = rows[i];
            if (r.Length == 1 && r[0].Length == 0) continue; // blank line
            cards.Add(new CardDef(
                ushort.Parse(r[cId]),
                byte.Parse(r[cUp]),
                byte.Parse(r[cRight]),
                byte.Parse(r[cDown]),
                byte.Parse(r[cLeft]),
                byte.Parse(r[cStar]),
                CardTypes.IdOf(r[cType])));
        }
        return new CardDb(cards);
    }

    /// <summary>Minimal RFC-4180 reader (quoted fields, doubled quotes, CRLF or LF).</summary>
    private static List<string[]> ParseCsv(TextReader reader)
    {
        var rows = new List<string[]>();
        var fields = new List<string>();
        var sb = new StringBuilder();
        bool inQuotes = false, any = false;
        int ch;
        while ((ch = reader.Read()) >= 0)
        {
            char c = (char)ch;
            any = true;
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (reader.Peek() == '"') { reader.Read(); sb.Append('"'); }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else if (c == '"') inQuotes = true;
            else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
            else if (c == '\r') { }
            else if (c == '\n')
            {
                fields.Add(sb.ToString());
                sb.Clear();
                rows.Add(fields.ToArray());
                fields.Clear();
                any = false;
            }
            else sb.Append(c);
        }
        if (any)
        {
            fields.Add(sb.ToString());
            rows.Add(fields.ToArray());
        }
        return rows;
    }
}
