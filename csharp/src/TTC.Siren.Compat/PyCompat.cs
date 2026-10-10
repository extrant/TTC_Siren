using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace TTC.Siren.Compat;

/// <summary>An error that Python's <c>ai_move</c> would turn into <c>{"error": str(e)}</c> with HTTP 500 (the message is Python's).</summary>
public sealed class PyException : Exception
{
    public PyException(string message) : base(message) { }
}

/// <summary>Small helpers that reproduce Python semantics the request path depends on (rounding, truthiness, <c>int()</c>, <c>repr</c>).</summary>
public static class Py
{
    /// <summary>
    /// <c>round(x, ndigits)</c> exactly as CPython does it: the value is rounded as an exact binary fraction to the nearest multiple of
    /// 10^-ndigits (ties to even), then converted back to the nearest double. <c>Math.Round(x, 3)</c> scales by 1000 first and
    /// accumulates an error, so it is not equivalent.
    /// </summary>
    public static double Round(double x, int ndigits)
    {
        if (double.IsNaN(x) || double.IsInfinity(x) || x == 0.0) return x;
        long bits = BitConverter.DoubleToInt64Bits(x);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0) exponent = 1;
        else mantissa |= 1L << 52;
        exponent -= 1075; // |x| = mantissa * 2^exponent
        BigInteger numerator = mantissa, denominator = BigInteger.One;
        if (exponent >= 0) numerator <<= exponent;
        else denominator <<= -exponent;
        if (ndigits >= 0) numerator *= BigInteger.Pow(10, ndigits);
        else denominator *= BigInteger.Pow(10, -ndigits);
        BigInteger quotient = BigInteger.DivRem(numerator, denominator, out BigInteger remainder);
        int cmp = (remainder * 2).CompareTo(denominator);
        if (cmp > 0 || (cmp == 0 && !quotient.IsEven)) quotient += 1;
        string text = (negative ? "-" : "") + quotient.ToString(CultureInfo.InvariantCulture) + "e" + (-ndigits).ToString(CultureInfo.InvariantCulture);
        return double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    /// <summary><c>repr(float)</c>: shortest round-trip digits, exponent form below 1e-4 and from 1e16 on, always with a point or exponent.</summary>
    public static string FloatRepr(double d)
    {
        if (double.IsNaN(d)) return "nan";
        if (double.IsPositiveInfinity(d)) return "inf";
        if (double.IsNegativeInfinity(d)) return "-inf";
        if (d == 0.0) return double.IsNegative(d) ? "-0.0" : "0.0";
        bool negative = d < 0;
        // shortest round-trip digits and exponent from the "R" text (.NET Core 3.0+ prints the shortest string that round-trips)
        string s = Math.Abs(d).ToString("R", CultureInfo.InvariantCulture);
        string mantissa = s;
        int exp10 = 0;
        int e = s.IndexOfAny(new[] { 'E', 'e' });
        if (e >= 0)
        {
            mantissa = s[..e];
            exp10 = int.Parse(s[(e + 1)..], CultureInfo.InvariantCulture);
        }
        int point = mantissa.IndexOf('.');
        string digits = point >= 0 ? mantissa.Remove(point, 1) : mantissa;
        int decimalPoint = (point >= 0 ? point : mantissa.Length) + exp10; // value = 0.DIGITS * 10^decimalPoint after stripping zeros
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
            decimalPoint--;
        }
        digits = digits[lead..].TrimEnd('0');
        if (digits.Length == 0) digits = "0";
        var sb = new StringBuilder();
        if (negative) sb.Append('-');
        int sci = decimalPoint - 1; // exponent in d.ddd form
        if (sci < -4 || sci >= 16)
        {
            sb.Append(digits[0]);
            if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
            sb.Append('e').Append(sci < 0 ? '-' : '+').Append(Math.Abs(sci).ToString("00", CultureInfo.InvariantCulture));
        }
        else if (decimalPoint <= 0)
        {
            sb.Append("0.").Append('0', -decimalPoint).Append(digits);
        }
        else if (decimalPoint >= digits.Length)
        {
            sb.Append(digits).Append('0', decimalPoint - digits.Length).Append(".0");
        }
        else
        {
            sb.Append(digits, 0, decimalPoint).Append('.').Append(digits, decimalPoint, digits.Length - decimalPoint);
        }
        return sb.ToString();
    }

    /// <summary>Python truth value of a JSON value (<c>if value:</c>).</summary>
    public static bool Truthy(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.True => true,
        JsonValueKind.False or JsonValueKind.Null or JsonValueKind.Undefined => false,
        JsonValueKind.Number => e.GetDouble() != 0.0,
        JsonValueKind.String => e.GetString()!.Length > 0,
        JsonValueKind.Array => e.GetArrayLength() > 0,
        JsonValueKind.Object => e.EnumerateObject().Any(),
        _ => false,
    };

    /// <summary>
    /// The integer a JSON value stands for when Python compares or hashes it as one: an integral number (<c>6</c>, <c>6.0</c>) or a bool.
    /// Null for everything else (a string, a fractional number, ...), which never equals an integer in Python.
    /// </summary>
    public static int? AsInt(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.True: return 1;
            case JsonValueKind.False: return 0;
            case JsonValueKind.Number:
                if (e.TryGetInt32(out int i)) return i;
                double d = e.GetDouble();
                if (d == Math.Floor(d) && Math.Abs(d) < int.MaxValue) return (int)d;
                return null;
            default: return null;
        }
    }

    /// <summary><c>str(value)</c> of a JSON value as an f-string shows it (strings without quotes, bools as True/False, null as None).</summary>
    public static string Str(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.True => "True",
        JsonValueKind.False => "False",
        JsonValueKind.Null or JsonValueKind.Undefined => "None",
        JsonValueKind.String => e.GetString()!,
        JsonValueKind.Number => NumberRepr(e),
        _ => Repr(e),
    };

    /// <summary><c>repr(value)</c> of a JSON value (the form <c>%r</c> shows): strings with quotes.</summary>
    public static string Repr(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.String: return "'" + e.GetString()!.Replace("\\", "\\\\").Replace("'", "\\'") + "'";
            case JsonValueKind.Array: return "[" + string.Join(", ", e.EnumerateArray().Select(Repr)) + "]";
            case JsonValueKind.Object:
                return "{" + string.Join(", ", e.EnumerateObject().Select(p => "'" + p.Name + "': " + Repr(p.Value))) + "}";
            default: return Str(e);
        }
    }

    private static string NumberRepr(JsonElement e)
    {
        string raw = e.GetRawText();
        if (!raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E') && e.TryGetInt64(out long l)) return l.ToString(CultureInfo.InvariantCulture);
        return FloatRepr(e.GetDouble());
    }

    /// <summary>
    /// <c>int(item)</c> as <c>_parse_id_list</c> applies it: ints, bools, floats (truncated), and strings the way <c>int(str)</c> reads
    /// them (surrounding whitespace, an optional sign, digits with single underscores between them). Null when Python would raise
    /// <c>ValueError</c> or <c>TypeError</c> (which <c>_parse_id_list</c> skips).
    /// </summary>
    public static long? IntOf(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.True: return 1;
            case JsonValueKind.False: return 0;
            case JsonValueKind.Number:
                if (e.TryGetInt64(out long l)) return l;
                double d = e.GetDouble();
                if (double.IsNaN(d) || double.IsInfinity(d)) return null;
                return Math.Abs(d) < 9.2e18 ? (long)Math.Truncate(d) : null;
            case JsonValueKind.String:
                return IntOfString(e.GetString()!);
            default:
                return null;
        }
    }

    public static long? IntOfString(string text)
    {
        string s = text.Trim(' ', '\t', '\n', '\r', '\f', '\v');
        if (s.Length == 0) return null;
        bool negative = false;
        int i = 0;
        if (s[0] is '+' or '-')
        {
            negative = s[0] == '-';
            i = 1;
        }
        if (i >= s.Length) return null;
        var digits = new StringBuilder();
        bool previousUnderscore = true; // a leading underscore is not allowed
        for (; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '_')
            {
                if (previousUnderscore) return null;
                previousUnderscore = true;
            }
            else if (c is >= '0' and <= '9')
            {
                digits.Append(c);
                previousUnderscore = false;
            }
            else return null;
        }
        if (previousUnderscore) return null; // trailing underscore
        if (!BigInteger.TryParse(digits.ToString(), NumberStyles.None, CultureInfo.InvariantCulture, out var value)) return null;
        if (negative) value = -value;
        if (value > long.MaxValue || value < long.MinValue) return null;
        return (long)value;
    }
}
