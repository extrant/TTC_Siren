using System.Globalization;
using System.Text;
using System.Text.Json;

namespace TTC.Siren.Core;

/// <summary>Plain, allocating description of one card. Used for import/export and parity only (never on the search path).</summary>
public sealed class CardSnapshot
{
    /// <summary>Card id; -1 stands for Python <c>None</c>.</summary>
    public int Id = -1;
    public string? Type;
    public byte Owner = OwnerId.None;
    public int Mod;
    public byte[] Base = new byte[4];
    public byte[] Display = new byte[4];
    public bool CanUse = true;

    public CardSnapshot() { }

    public CardSnapshot(int up, int right, int down, int left, byte owner = OwnerId.None, int id = -1,
        string? type = null, bool canUse = true, int mod = 0)
    {
        Base = new[] { (byte)up, (byte)right, (byte)down, (byte)left };
        Display = (byte[])Base.Clone(); // Python Card.__init__: display == base (no clamp)
        Owner = owner;
        Id = id;
        Type = type;
        CanUse = canUse;
        Mod = mod;
    }

    /// <summary>Python <c>Card.apply_type_modifier</c>: sets the modifier and refreshes the display values.</summary>
    public CardSnapshot WithAppliedModifier(int mod)
    {
        Mod = mod;
        for (int i = 0; i < 4; i++) Display[i] = (byte)Math.Max(1, Math.Min(10, Base[i] + mod));
        return this;
    }

    internal void AppendCanonical(StringBuilder sb)
    {
        sb.Append('[').Append(Id.ToString(CultureInfo.InvariantCulture)).Append(',')
          .Append(string.IsNullOrEmpty(Type) ? "-" : Type).Append(',')
          .Append(Owner).Append(',')
          .Append(Mod.ToString(CultureInfo.InvariantCulture)).Append(',');
        for (int i = 0; i < 4; i++) sb.Append(Base[i]).Append('.');
        sb.Append(',');
        for (int i = 0; i < 4; i++) sb.Append(Display[i]).Append('.');
        sb.Append(',').Append(CanUse ? '1' : '0').Append(']');
    }

    /// <summary>Review L-9: "r", "b" or null only. Any other value used to be read as blue, silently.</summary>
    private static byte ParseOwner(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.Null) return OwnerId.None;
        return e.ValueKind == JsonValueKind.String && e.GetString() is { } text
            ? text switch
            {
                "r" => OwnerId.Red,
                "b" => OwnerId.Blue,
                _ => throw new InvalidGameStateException($"Unknown owner '{text}' (expected \"r\", \"b\" or null)"),
            }
            : throw new InvalidGameStateException($"Owner must be \"r\", \"b\" or null, got {e.ValueKind}");
    }

    /// <summary>
    /// Parses the compact golden form
    /// <c>[id|null, type|null, "r"|"b"|null, mod, bu,br,bd,bl, du,dr,dd,dl, canUse(0/1)]</c>.
    /// </summary>
    public static CardSnapshot FromJson(JsonElement e)
    {
        var c = new CardSnapshot
        {
            Id = e[0].ValueKind == JsonValueKind.Null ? -1 : e[0].GetInt32(),
            Type = e[1].ValueKind == JsonValueKind.Null ? null : e[1].GetString(),
            Owner = ParseOwner(e[2]),
            Mod = e[3].GetInt32(),
            CanUse = e[12].GetInt32() != 0,
        };
        for (int i = 0; i < 4; i++)
        {
            c.Base[i] = checked((byte)e[4 + i].GetInt32());
            c.Display[i] = checked((byte)e[8 + i].GetInt32());
        }
        return c;
    }
}

/// <summary>Plain, allocating description of a whole game state (board, ordered hands, side to move, rules).</summary>
public sealed class StateSnapshot
{
    public int Current;
    public RuleSet Rules;
    public CardSnapshot?[] Board = new CardSnapshot?[9];
    public List<CardSnapshot>[] Hands = { new(), new() };

    /// <summary>Canonical text form. Two states are identical iff their canonical strings are equal.</summary>
    public string ToCanonicalString()
    {
        var sb = new StringBuilder(512);
        sb.Append("cur=").Append(Current).Append(";rules=").Append((int)Rules).Append(";board=");
        for (int i = 0; i < 9; i++)
        {
            if (Board[i] is null) sb.Append('_');
            else Board[i]!.AppendCanonical(sb);
        }
        for (int p = 0; p < 2; p++)
        {
            sb.Append(";hand").Append(p).Append('=');
            foreach (var c in Hands[p]) c.AppendCanonical(sb);
        }
        return sb.ToString();
    }

    /// <summary>Parses <c>{"cur":n,"board":[...9],"hands":[[...],[...]]}</c>; rules come from the caller.</summary>
    public static StateSnapshot FromJson(JsonElement e, RuleSet rules)
    {
        var s = new StateSnapshot { Current = e.GetProperty("cur").GetInt32(), Rules = rules };
        var board = e.GetProperty("board");
        for (int i = 0; i < 9; i++)
        {
            var cell = board[i];
            s.Board[i] = cell.ValueKind == JsonValueKind.Null ? null : CardSnapshot.FromJson(cell);
        }
        var hands = e.GetProperty("hands");
        for (int p = 0; p < 2; p++)
        {
            foreach (var card in hands[p].EnumerateArray()) s.Hands[p].Add(CardSnapshot.FromJson(card));
        }
        return s;
    }
}
