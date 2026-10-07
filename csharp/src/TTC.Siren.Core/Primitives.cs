namespace TTC.Siren.Core;

/// <summary>Owner of a card instance. Mirrors Python's three-valued <c>'red' | 'blue' | None</c>.</summary>
public static class OwnerId
{
    public const byte Red = 0;
    public const byte Blue = 1;
    public const byte None = 2;

    public static byte Parse(string? name) => name switch
    {
        "red" => Red,
        "blue" => Blue,
        null => None,
        _ => throw new ArgumentException($"Unknown owner '{name}'", nameof(name)),
    };

    public static string? ToName(byte owner) => owner switch
    {
        Red => "red",
        Blue => "blue",
        None => null,
        _ => throw new ArgumentOutOfRangeException(nameof(owner)),
    };
}

/// <summary>Card edge indices. Same order as the Python-side golden export: up, right, down, left.</summary>
public static class Side
{
    public const int Up = 0;
    public const int Right = 1;
    public const int Down = 2;
    public const int Left = 3;
}

/// <summary>Rule flags. Only membership matters, exactly like Python's <c>'x' in rules</c>.</summary>
[Flags]
public enum RuleSet : ushort
{
    None = 0,
    Same = 1 << 0,        // 同数
    Plus = 1 << 1,        // 加算
    Reverse = 1 << 2,     // 逆转
    FallenAce = 1 << 3,   // 王牌杀手
    Ascension = 1 << 4,   // 同类强化
    Descension = 1 << 5,  // 同类弱化
    Order = 1 << 6,       // 秩序
    Chaos = 1 << 7,       // 混乱
    Draft = 1 << 8,       // 选拔
}

public enum OpenMode : byte
{
    None = 0,
    All = 1,
    Three = 2,
}

public static class RuleParser
{
    private static readonly (string Name, RuleSet Flag)[] Names =
    {
        ("同数", RuleSet.Same),
        ("加算", RuleSet.Plus),
        ("逆转", RuleSet.Reverse),
        ("王牌杀手", RuleSet.FallenAce),
        ("同类强化", RuleSet.Ascension),
        ("同类弱化", RuleSet.Descension),
        ("秩序", RuleSet.Order),
        ("混乱", RuleSet.Chaos),
        ("选拔", RuleSet.Draft),
    };

    /// <summary>
    /// Equivalent of <c>ai_server.parse_rules_and_open_mode</c>: substring matching on the raw rule text.
    /// 全明牌 takes precedence over 三明牌.
    /// </summary>
    public static (RuleSet Rules, OpenMode OpenMode) ParsePythonRuleString(string rulesStr)
    {
        ArgumentNullException.ThrowIfNull(rulesStr);
        OpenMode open = OpenMode.None;
        if (rulesStr.Contains("全明牌", StringComparison.Ordinal)) open = OpenMode.All;
        else if (rulesStr.Contains("三明牌", StringComparison.Ordinal)) open = OpenMode.Three;

        RuleSet rules = RuleSet.None;
        foreach (var (name, flag) in Names)
        {
            if (rulesStr.Contains(name, StringComparison.Ordinal)) rules |= flag;
        }
        return (rules, open);
    }

    /// <summary>Builds a <see cref="RuleSet"/> from a Python rule list (exact element membership).</summary>
    public static RuleSet FromPythonList(IEnumerable<string> rules)
    {
        RuleSet result = RuleSet.None;
        foreach (string rule in rules)
        {
            foreach (var (name, flag) in Names)
            {
                if (string.Equals(rule, name, StringComparison.Ordinal)) result |= flag;
            }
        }
        return result;
    }

    /// <summary>Rule names in the canonical order used by <see cref="ParsePythonRuleString"/>.</summary>
    public static List<string> ToPythonList(RuleSet rules)
    {
        var list = new List<string>();
        foreach (var (name, flag) in Names)
        {
            if ((rules & flag) != 0) list.Add(name);
        }
        return list;
    }
}

/// <summary>
/// A move: the card instance to play plus a row-major board cell (0..8), exactly like Python's <c>(card, (row, col))</c>.
/// The identity is the card instance, which is stable for the lifetime of a <see cref="GameState"/>. It is deliberately
/// NOT a hand slot: make+undo reorders the hand (D-001), so a slot named earlier would name another card later
/// (review H-3). <see cref="GameState.MakeMove"/> resolves the current hand slot itself.
/// </summary>
public readonly struct Move : IEquatable<Move>
{
    public readonly byte Inst;
    public readonly byte Cell;

    public Move(int inst, int cell)
    {
        Inst = (byte)inst;
        Cell = (byte)cell;
    }

    public int Row => Cell / 3;
    public int Col => Cell % 3;

    public bool Equals(Move other) => Inst == other.Inst && Cell == other.Cell;
    public override bool Equals(object? obj) => obj is Move other && Equals(other);
    public override int GetHashCode() => Inst << 8 | Cell;
    public static bool operator ==(Move a, Move b) => a.Equals(b);
    public static bool operator !=(Move a, Move b) => !a.Equals(b);
    public override string ToString() => $"(card instance {Inst}, r{Row}c{Col})";
}

/// <summary>Fixed table of card types. Id 0 means "no type" (Python <c>None</c> or empty string).</summary>
public static class CardTypes
{
    private static readonly string[] Names = { "", "兽人", "拂晓", "帝国", "蛮神" };

    public const int Count = 5;

    public static byte IdOf(string? name)
    {
        if (string.IsNullOrEmpty(name)) return 0;
        for (int i = 1; i < Names.Length; i++)
        {
            if (string.Equals(Names[i], name, StringComparison.Ordinal)) return (byte)i;
        }
        throw new ArgumentException($"Unknown card type '{name}'", nameof(name));
    }

    public static string? NameOf(byte id)
    {
        if (id == 0) return null;
        if (id >= Names.Length) throw new ArgumentOutOfRangeException(nameof(id));
        return Names[id];
    }
}

/// <summary>Thrown when an input state cannot be represented faithfully (for example D-004 duplicate hand ids).</summary>
public sealed class InvalidGameStateException : Exception
{
    public InvalidGameStateException(string message) : base(message) { }
}
