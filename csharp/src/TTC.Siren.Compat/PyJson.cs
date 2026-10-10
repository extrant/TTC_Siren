using System.Collections;
using System.Globalization;
using System.Text;

namespace TTC.Siren.Compat;

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
