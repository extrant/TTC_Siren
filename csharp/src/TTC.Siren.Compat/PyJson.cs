using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace TTC.Siren.Compat;

/// <summary>
/// JSON text that <see cref="PyJson"/> writes as it is: a value the caller already holds in its final form (for example a line of a game log that must reach
/// the client with its numbers exactly as they were written, review L8-2).
/// </summary>
public sealed record RawJson(string Text)
{
    /// <summary>
    /// The same JSON without the whitespace between tokens: every token keeps the exact text it had in the input (a number written <c>4.0</c> stays <c>4.0</c>,
    /// <c>4</c> stays <c>4</c>, a string keeps its escapes) and the members keep their order. A request body that was already compact comes back byte for byte.
    /// </summary>
    public static RawJson Compact(string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        var reader = new Utf8JsonReader(bytes, new JsonReaderOptions { MaxDepth = 64 });
        var sb = new StringBuilder(bytes.Length);
        bool comma = false;
        while (reader.Read())
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.StartObject:
                case JsonTokenType.StartArray:
                    if (comma) sb.Append(',');
                    sb.Append(reader.TokenType == JsonTokenType.StartObject ? '{' : '[');
                    comma = false;
                    break;
                case JsonTokenType.EndObject:
                case JsonTokenType.EndArray:
                    sb.Append(reader.TokenType == JsonTokenType.EndObject ? '}' : ']');
                    comma = true;
                    break;
                case JsonTokenType.PropertyName:
                    if (comma) sb.Append(',');
                    sb.Append('"').Append(Encoding.UTF8.GetString(reader.ValueSpan)).Append("\":");
                    comma = false;
                    break;
                case JsonTokenType.String:
                    if (comma) sb.Append(',');
                    sb.Append('"').Append(Encoding.UTF8.GetString(reader.ValueSpan)).Append('"');
                    comma = true;
                    break;
                case JsonTokenType.Number:
                case JsonTokenType.True:
                case JsonTokenType.False:
                case JsonTokenType.Null:
                    if (comma) sb.Append(',');
                    sb.Append(Encoding.UTF8.GetString(reader.ValueSpan));
                    comma = true;
                    break;
            }
        }
        return new RawJson(sb.ToString());
    }
}

/// <summary>
/// Writes response objects the way Flask's <c>jsonify</c> does: compact separators, keys sorted, floats as Python's <c>repr</c> (a
/// float is always written with a point or an exponent: <c>1.0</c>, not <c>1</c>), non-ASCII escaped as <c>\uXXXX</c>. The old plugins
/// parse Python's output; staying byte-identical in form keeps strongly typed clients (which expect a double where Python wrote
/// <c>1.0</c>) working. Values: null, bool, int, long, double (a Python float), string, string-keyed dictionaries, lists.
/// </summary>
public static class PyJson
{
    public static string Serialize(object? value)
    {
        var sb = new StringBuilder(1024);
        Write(sb, value);
        return sb.ToString();
    }

    public static void Write(StringBuilder sb, object? value)
    {
        switch (value)
        {
            case null: sb.Append("null"); break;
            case RawJson raw: sb.Append(raw.Text); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case int i: sb.Append(i.ToString(CultureInfo.InvariantCulture)); break;
            case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); break;
            case double d: WriteDouble(sb, d); break;
            case string s: WriteString(sb, s); break;
            case IDictionary<string, object?> dict:
            {
                sb.Append('{');
                bool first = true;
                foreach (var key in dict.Keys.OrderBy(k => k, StringComparer.Ordinal))
                {
                    if (!first) sb.Append(',');
                    first = false;
                    WriteString(sb, key);
                    sb.Append(':');
                    Write(sb, dict[key]);
                }
                sb.Append('}');
                break;
            }
            case IEnumerable seq:
            {
                sb.Append('[');
                bool first = true;
                foreach (var item in seq)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    Write(sb, item);
                }
                sb.Append(']');
                break;
            }
            default: throw new NotSupportedException("PyJson cannot write " + value.GetType());
        }
    }

    private static void WriteDouble(StringBuilder sb, double d)
    {
        // Python's json module writes NaN / Infinity / -Infinity (not valid JSON, but that is what the old server sent)
        if (double.IsNaN(d)) sb.Append("NaN");
        else if (double.IsPositiveInfinity(d)) sb.Append("Infinity");
        else if (double.IsNegativeInfinity(d)) sb.Append("-Infinity");
        else sb.Append(Py.FloatRepr(d));
    }

    public static void WriteString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < ' ' || c > '~') sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }
}
