using System.Globalization;
using TTC.Siren.Compat;

namespace TTC.Siren.Tests;

/// <summary>Python's <c>round(x, n)</c> and <c>repr(float)</c> reproduced exactly (Phase 8A): the response values are rounded and written as Python does.</summary>
public class PyFormatTests
{
    private static double FromHex(string text)
    {
        // Python float.hex(): [-]0x1.<13 hex digits>p<exp> (or 0x0.0p+0 for zero)
        bool negative = text.StartsWith('-');
        string t = negative ? text[1..] : text;
        int p = t.IndexOf('p');
        string mantissa = t[2..p];
        int exp = int.Parse(t[(p + 1)..], CultureInfo.InvariantCulture);
        int dot = mantissa.IndexOf('.');
        long lead = long.Parse(mantissa[..dot], CultureInfo.InvariantCulture);
        string frac = mantissa[(dot + 1)..].PadRight(13, '0');
        long fraction = Convert.ToInt64(frac, 16);
        double value;
        if (lead == 0 && fraction == 0) value = 0.0;
        else if (lead == 1) value = BitConverter.Int64BitsToDouble(((long)(exp + 1023) << 52) | fraction);
        else value = (lead + fraction / 4503599627370496.0) * Math.Pow(2, exp); // subnormal forms: 0x0.xxxp-1022
        return negative ? -value : value;
    }

    [Fact]
    public void Round_and_repr_match_Python_on_6000_values()
    {
        string path = Golden8.Ensure(Path.Combine(Golden8.GoldenDir, "p8_format_g1_s20261011_n6000.jsonl.gz"), "fmt", "tools/phase8a/py_format_trace.py",
            "--out", "{0}", "--seed", "20261011", "--count", "6000");
        int n = 0;
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                double x = FromHex(rec.GetProperty("x").GetString()!);
                int digits = rec.GetProperty("n").GetInt32();
                double wantRound = FromHex(rec.GetProperty("round").GetString()!);
                string where = $"x={rec.GetProperty("x").GetString()} ({rec.GetProperty("repr").GetString()}) ndigits={digits}";
                double have = Py.Round(x, digits);
                Assert.True(BitConverter.DoubleToInt64Bits(wantRound) == BitConverter.DoubleToInt64Bits(have), $"{where}: round python {rec.GetProperty("round").GetString()} csharp {have:R}");
                Assert.True(rec.GetProperty("repr").GetString() == Py.FloatRepr(x), $"{where}: repr python {rec.GetProperty("repr").GetString()} csharp {Py.FloatRepr(x)}");
                Assert.True(rec.GetProperty("repr_round").GetString() == Py.FloatRepr(have), $"{where}: repr(round) python {rec.GetProperty("repr_round").GetString()} csharp {Py.FloatRepr(have)}");
                n++;
            }
        }
        Assert.Equal(6000, n);
    }

    [Fact]
    public void PyJson_writes_floats_with_a_point_sorts_keys_and_escapes_like_Flask()
    {
        var body = new Dictionary<string, object?>
        {
            ["b"] = 1.0, ["a"] = new List<object?> { 1, 2.5, "中\"\n", null, true, 1e-5, 100000000000000000.0 }, ["c"] = new Dictionary<string, object?> { ["z"] = 0, ["y"] = -0.0 },
        };
        Assert.Equal("{\"a\":[1,2.5,\"\\u4e2d\\\"\\n\",null,true,1e-05,1e+17],\"b\":1.0,\"c\":{\"y\":-0.0,\"z\":0}}", PyJson.Serialize(body));
    }

    [Theory]
    [InlineData("5", 5L)]
    [InlineData(" 12 ", 12L)]
    [InlineData("+7", 7L)]
    [InlineData("1_0", 10L)]
    [InlineData("-3", -3L)]
    public void Int_of_strings_follows_Python_int(string text, long expected) => Assert.Equal(expected, Py.IntOfString(text));

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    [InlineData("1__0")]
    [InlineData("_1")]
    [InlineData("1_")]
    [InlineData("1.5")]
    [InlineData("0x10")]
    public void Strings_Python_int_rejects(string text) => Assert.Null(Py.IntOfString(text));
}
