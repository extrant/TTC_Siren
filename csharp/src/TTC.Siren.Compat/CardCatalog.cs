using TTC.Siren.Core;

namespace TTC.Siren.Compat;

/// <summary>
/// The read-only card tables the Python request path builds from the CSV (<c>get_all_cards</c>, <c>get_card_lookup</c>, <c>get_card_star_map</c>,
/// <c>get_card_type_map</c>): every database row in order (including the all-zero rows 474/475, D-012), the first id per edge tuple, the star
/// and type of each id. Shared by all requests and never mutated: every card handed out is a new <see cref="LCard"/>.
/// </summary>
public sealed class CardCatalog
{
    private readonly CardDb _db;
    private readonly Dictionary<int, int> _star = new();
    private readonly Dictionary<int, string?> _type = new();

    /// <summary>Python <c>get_all_cards()</c>: one prototype card per database row, owner none, display = base. Treat as read-only.</summary>
    public IReadOnlyList<LCard> AllCards { get; }

    public CardDb Db => _db;

    public CardCatalog(CardDb db)
    {
        _db = db;
        var all = new List<LCard>(db.Cards.Count);
        foreach (var def in db.Cards)
        {
            string? type = CardTypes.NameOf(def.TypeId);
            all.Add(new LCard(def.Up, def.Right, def.Down, def.Left, null, def.Id, type));
            _star[def.Id] = def.Star;
            _type[def.Id] = type;
        }
        AllCards = all;
    }

    /// <summary>Python <c>find_card_id_by_stats(up, right, down, left)</c>: the first database row with these edges, or null.</summary>
    public int? FindIdByStats(int up, int right, int down, int left)
    {
        if (up is < 0 or > 255 || right is < 0 or > 255 || down is < 0 or > 255 || left is < 0 or > 255) return null;
        int id = _db.FindFirstIdByEdges(up, right, down, left);
        return id == 0 ? null : id;
    }

    /// <summary><c>star_map.get(card_id, default)</c>: the star of a database id, or null when the id is not in the database.</summary>
    public int? StarOf(int? id) => id is { } i && _star.TryGetValue(i, out int s) ? s : null;

    public bool HasStar(int? id) => id is { } i && _star.ContainsKey(i);

    /// <summary><c>type_map.get(card_id)</c>: null for untyped cards and unknown ids.</summary>
    public string? TypeOf(int? id) => id is { } i && _type.TryGetValue(i, out var t) ? t : null;
}
