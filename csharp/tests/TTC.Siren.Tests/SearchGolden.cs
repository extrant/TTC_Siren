using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Tests;

/// <summary>Helpers shared by the Phase 3 differential tests.</summary>
internal static class SearchGolden
{
    /// <summary>D-004 coverage: does any hand hold two cards with the same card id (including two None ids)?</summary>
    public static bool HasDuplicateHandIds(GameState g)
    {
        for (int p = 0; p < 2; p++)
        {
            for (int a = 0; a < g.HandCount(p); a++)
            {
                for (int b = a + 1; b < g.HandCount(p); b++)
                {
                    if (g.InstanceId(g.HandInstance(p, a)) == g.InstanceId(g.HandInstance(p, b))) return true;
                }
            }
        }
        return false;
    }

    /// <summary>How many cards in the hand of <paramref name="player"/> have the same card id as <paramref name="inst"/> (itself included).</summary>
    public static int SameIdInHand(GameState g, int player, int inst)
    {
        int count = 0;
        for (int k = 0; k < g.HandCount(player); k++)
        {
            if (g.InstanceId(g.HandInstance(player, k)) == g.InstanceId(inst)) count++;
        }
        return count;
    }

    private static readonly Lazy<CardDb> LazyDb = new(() => CardDb.LoadFromRepository(AppContext.BaseDirectory));

    public static CardDb Db => LazyDb.Value;

    /// <summary>Streams the records of a golden file after validating its header (generator version, kind).</summary>
    public static IEnumerable<JsonDocument> Records(string path, string kind)
    {
        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        using (var header = JsonDocument.Parse(reader.ReadLine()!))
        {
            var h = header.RootElement.GetProperty("header");
            Assert.Equal(GoldenFiles.SearchGeneratorVersion, h.GetProperty("generator").GetString());
            Assert.Equal(kind, h.GetProperty("kind").GetString());
        }
        string? line;
        while ((line = reader.ReadLine()) is not null) yield return JsonDocument.Parse(line);
    }

    public static GameState BuildState(JsonElement rec, out string label)
    {
        var rules = RuleParser.FromPythonList(rec.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
        var snapshot = StateSnapshot.FromJson(rec.GetProperty("state"), rules);
        label = $"i={rec.GetProperty("i").GetInt32()}" +
                (rec.TryGetProperty("name", out var n) ? $" name={n.GetString()}" : "");
        return GameState.FromSnapshot(snapshot);
    }

    /// <summary>Python float.hex() string to the IEEE-754 bit pattern (normal numbers, zeros, +-inf).</summary>
    public static long HexBits(string hex)
    {
        if (hex == "inf") return BitConverter.DoubleToInt64Bits(double.PositiveInfinity);
        if (hex == "-inf") return BitConverter.DoubleToInt64Bits(double.NegativeInfinity);
        bool negative = hex.StartsWith('-');
        string body = negative ? hex[1..] : hex;
        if (!body.StartsWith("0x", StringComparison.Ordinal)) throw new FormatException($"Unsupported float literal '{hex}'");
        int p = body.IndexOf('p');
        string mantissa = body[2..p];
        int exponent = int.Parse(body[(p + 1)..], CultureInfo.InvariantCulture);
        long bits;
        if (mantissa.StartsWith("0.", StringComparison.Ordinal) && mantissa[2..].All(c => c == '0'))
        {
            bits = 0;
        }
        else
        {
            if (!mantissa.StartsWith("1.", StringComparison.Ordinal) || mantissa.Length != 15)
                throw new FormatException($"Unsupported float literal '{hex}' (subnormal?)");
            long frac = long.Parse(mantissa[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            bits = ((long)(exponent + 1023) << 52) | frac;
        }
        return negative ? bits | long.MinValue : bits;
    }

    public static string Describe(double value) =>
        $"{value:R} (0x{BitConverter.DoubleToInt64Bits(value):X16})";

    public static void AssertSameDouble(string expectedHex, double actual, string context)
    {
        long expected = HexBits(expectedHex);
        long got = BitConverter.DoubleToInt64Bits(actual);
        Assert.True(expected == got,
            $"{context}: expected {expectedHex} (0x{expected:X16} = {BitConverter.Int64BitsToDouble(expected):R}) but C# produced {Describe(actual)}");
    }

    /// <summary>Move as the golden files print it: [card_id|null, row, col] or null.</summary>
    public static string MoveText(JsonElement move) =>
        move.ValueKind == JsonValueKind.Null
            ? "none"
            : $"{(move[0].ValueKind == JsonValueKind.Null ? "None" : move[0].GetInt32().ToString())},{move[1].GetInt32()},{move[2].GetInt32()}";

    public static string MoveText(GameState g, int instance, int cell) =>
        instance < 0
            ? "none"
            : $"{(g.InstanceId(instance) == -1 ? "None" : g.InstanceId(instance).ToString())},{cell / 3},{cell % 3}";

    public static string HistoryText(JsonElement hist) =>
        string.Join(";", hist.EnumerateArray().Select(e =>
            $"{(e[0].ValueKind == JsonValueKind.Null ? -1 : e[0].GetInt32())},{e[1].GetInt32()},{e[2].GetInt32()}={e[3].GetInt32()}"));

    /// <summary>
    /// State text that ignores exactly what the documented Python quirks legitimately change during make/undo:
    /// the order of each hand (D-001) and the stale display edges (D-003). Everything else must be restored exactly.
    /// </summary>
    public static string NormalizedState(GameState g)
    {
        var s = g.ToSnapshot();
        static string Card(CardSnapshot c) => $"{c.Id}|{c.Type}|{c.Owner}|{c.Mod}|{string.Join(",", c.Base)}|{(c.CanUse ? 1 : 0)}";
        var sb = new System.Text.StringBuilder();
        sb.Append(s.Current).Append(";rules=").Append((int)s.Rules).Append(";board=");
        foreach (var c in s.Board) sb.Append(c is null ? "_" : Card(c)).Append('/');
        for (int p = 0; p < 2; p++)
        {
            sb.Append(";hand").Append(p).Append('=');
            foreach (string card in s.Hands[p].Select(Card).OrderBy(x => x, StringComparer.Ordinal)) sb.Append(card).Append('/');
        }
        return sb.ToString();
    }

    public static string HistoryText(HistoryTable h) =>
        string.Join(";", h.Entries().Select(e => $"{e.Id},{e.Row},{e.Col}={e.Value}"));
}
