using System.Globalization;
using System.Text.Json;

namespace TTC.Siren.Tests;

/// <summary>
/// Deep comparison of two JSON documents for the parity tests: objects must have the same keys, arrays the same length, strings and
/// booleans must be equal, and numbers must be of the same kind (integer or float, as written) and EQUAL BIT FOR BIT as doubles
/// (CLAUDE.md section 2: no epsilon). Paths named in <c>ignore</c> are skipped (timing fields); keys named in <c>additive</c> may exist
/// in the second document only (new fields).
/// </summary>
internal sealed class JsonCompare
{
    private readonly Func<string, bool> _ignore;
    private readonly HashSet<string> _additive;
    public int Leaves { get; private set; }
    public int Ignored { get; private set; }
    public List<string> Differences { get; } = new();

    public JsonCompare(Func<string, bool> ignore, IEnumerable<string>? additive = null)
    {
        _ignore = ignore;
        _additive = new HashSet<string>(additive ?? Array.Empty<string>(), StringComparer.Ordinal);
    }

    /// <summary>Normalises array indices to <c>[]</c> so ignore rules read <c>search_progress[].time_elapsed</c>.</summary>
    private static string Child(string path, string name) => path.Length == 0 ? name : path + "." + name;

    public void Compare(JsonElement expected, JsonElement actual, string path = "")
    {
        if (_ignore(path))
        {
            Ignored++;
            return;
        }
        if (expected.ValueKind == JsonValueKind.Object && actual.ValueKind == JsonValueKind.Object)
        {
            var e = expected.EnumerateObject().ToDictionary(p => p.Name, p => p.Value, StringComparer.Ordinal);
            var a = actual.EnumerateObject().ToDictionary(p => p.Name, p => p.Value, StringComparer.Ordinal);
            foreach (var (key, value) in e)
            {
                if (!a.TryGetValue(key, out var other)) Differences.Add($"{Child(path, key)}: missing in the C# response");
                else Compare(value, other, Child(path, key));
            }
            foreach (var key in a.Keys)
                if (!e.ContainsKey(key) && !(path.Length == 0 && _additive.Contains(key)) && !_ignore(Child(path, key)))
                    Differences.Add($"{Child(path, key)}: only in the C# response");
            return;
        }
        if (expected.ValueKind == JsonValueKind.Array && actual.ValueKind == JsonValueKind.Array)
        {
            if (expected.GetArrayLength() != actual.GetArrayLength())
            {
                Differences.Add($"{path}: array length python {expected.GetArrayLength()} csharp {actual.GetArrayLength()}");
                return;
            }
            for (int i = 0; i < expected.GetArrayLength(); i++) Compare(expected[i], actual[i], path + "[]");
            return;
        }
        Leaves++;
        if (expected.ValueKind == JsonValueKind.Number && actual.ValueKind == JsonValueKind.Number)
        {
            bool eInt = IsInteger(expected), aInt = IsInteger(actual);
            if (eInt != aInt)
            {
                Differences.Add($"{path}: number kind python {expected.GetRawText()} csharp {actual.GetRawText()}");
                return;
            }
            double x = expected.GetDouble(), y = actual.GetDouble();
            if (BitConverter.DoubleToInt64Bits(x) != BitConverter.DoubleToInt64Bits(y))
                Differences.Add($"{path}: python {expected.GetRawText()} csharp {actual.GetRawText()} (bits {BitConverter.DoubleToInt64Bits(x):X16} vs {BitConverter.DoubleToInt64Bits(y):X16})");
            return;
        }
        if (expected.ValueKind != actual.ValueKind)
        {
            Differences.Add($"{path}: python {expected.ValueKind} {Short(expected)} csharp {actual.ValueKind} {Short(actual)}");
            return;
        }
        if (expected.ValueKind == JsonValueKind.String && expected.GetString() != actual.GetString())
            Differences.Add($"{path}: python \"{expected.GetString()}\" csharp \"{actual.GetString()}\"");
    }

    private static bool IsInteger(JsonElement e)
    {
        string raw = e.GetRawText();
        return !raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E');
    }

    private static string Short(JsonElement e)
    {
        string t = e.GetRawText();
        return t.Length > 60 ? t[..60] + "..." : t;
    }
}
