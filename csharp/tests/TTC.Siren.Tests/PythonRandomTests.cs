using System.Numerics;
using System.Text.Json;
using TTC.Siren.Compat;

namespace TTC.Siren.Tests;

/// <summary>Phase 8A, decision A-23: the C# generator reproduces CPython 3.11's <c>random</c> bit for bit.</summary>
public class PythonRandomTests
{
    private static string Hex(double d)
    {
        // Python float.hex(): 0x1.<13 hex digits>p<exp>, sign first; subnormals and zero are not produced by these tests
        if (d == 0) return d.Equals(0.0) && !double.IsNegative(d) ? "0x0.0p+0" : "-0x0.0p+0";
        long bits = BitConverter.DoubleToInt64Bits(d);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF) - 1023;
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        string digits = mantissa.ToString("x13");
        return $"{(negative ? "-" : "")}0x1.{digits}p{(exponent >= 0 ? "+" : "")}{exponent}";
    }

    [Fact]
    public void Golden_sequences_of_600_seeded_generators_match_CPython_exactly()
    {
        string path = Golden8.EnsureRng();
        int records = 0, operations = 0;
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                string seedText = rec.GetProperty("seed").GetString()!;
                var seed = BigInteger.Parse(seedText);
                var rng = new PythonRandom(seed);
                var ops = rec.GetProperty("ops");
                var outs = rec.GetProperty("out");
                for (int i = 0; i < ops.GetArrayLength(); i++)
                {
                    var op = ops[i];
                    var expected = outs[i];
                    string where = $"record {records} (seed {seedText}) op #{i} {op}";
                    switch (op[0].GetString())
                    {
                        case "random":
                            Assert.True(Hex(rng.NextDouble()) == expected.GetString(), where);
                            break;
                        case "bits":
                            Assert.True(rng.GetRandBits(op[1].GetInt32()).ToString() == expected.GetString(), where);
                            break;
                        case "below":
                            Assert.True(BelowOfAnySize(rng, op[1].GetInt64()) == expected.GetInt64(), where);
                            break;
                        case "range":
                            Assert.True(rng.RandRange(op[1].GetInt32(), op[2].GetInt32()) == expected.GetInt32(), where);
                            break;
                        case "randint":
                            Assert.True(rng.RandInt(op[1].GetInt32(), op[2].GetInt32()) == expected.GetInt32(), where);
                            break;
                        case "choice":
                            Assert.True(rng.Choice(Enumerable.Range(0, op[1].GetInt32()).ToArray()) == expected.GetInt32(), where);
                            break;
                        case "sample":
                        {
                            var got = rng.Sample(Enumerable.Range(0, op[1].GetInt32()).ToArray(), op[2].GetInt32());
                            var want = expected.EnumerateArray().Select(e => e.GetInt32()).ToList();
                            Assert.True(got.SequenceEqual(want), where);
                            break;
                        }
                        case "shuffle":
                        {
                            var x = Enumerable.Range(0, op[1].GetInt32()).ToList();
                            rng.Shuffle(x);
                            Assert.True(x.SequenceEqual(expected.EnumerateArray().Select(e => e.GetInt32())), where);
                            break;
                        }
                        case "uniform":
                            Assert.True(Hex(rng.Uniform(op[1].GetDouble(), op[2].GetDouble())) == expected.GetString(), where);
                            break;
                        default:
                            throw new InvalidOperationException("unknown op " + op);
                    }
                    operations++;
                }
                records++;
            }
        }
        Assert.Equal(Golden8.RngCount, records);
        Assert.True(operations > 10_000, $"only {operations} operations compared");
    }

    /// <summary>Python's <c>_randbelow(n)</c> also serves n up to 2^32 - 1 (needs more than 31 bits); RandBelow takes an int, so use bits.</summary>
    private static long BelowOfAnySize(PythonRandom rng, long n)
    {
        if (n <= int.MaxValue) return rng.RandBelow((int)n);
        int k = 64 - BitOperations.LeadingZeroCount((ulong)n);
        long r = rng.GetRandBits32(k);
        while (r >= n) r = rng.GetRandBits32(k);
        return r;
    }

    [Fact]
    public void Known_first_outputs_of_seed_42()
    {
        // random.seed(42); random.random() -> 0.6394267984578837 ; random.getrandbits(32) after that is checked by the golden file
        var rng = new PythonRandom(42UL);
        Assert.Equal(0.6394267984578837, rng.NextDouble());
        Assert.Equal(0.025010755222666936, rng.NextDouble());
    }

    [Fact]
    public void Negative_seeds_use_the_absolute_value_like_CPython()
    {
        var a = new PythonRandom(new BigInteger(-12345));
        var b = new PythonRandom(12345UL);
        for (int i = 0; i < 50; i++) Assert.Equal(b.NextUInt32(), a.NextUInt32());
    }

    [Fact]
    public void The_same_seed_gives_the_same_stream_and_a_different_seed_a_different_one()
    {
        var a = new PythonRandom(7UL);
        var b = new PythonRandom(7UL);
        var c = new PythonRandom(8UL);
        var sa = Enumerable.Range(0, 20).Select(_ => a.NextUInt32()).ToArray();
        var sb = Enumerable.Range(0, 20).Select(_ => b.NextUInt32()).ToArray();
        var sc = Enumerable.Range(0, 20).Select(_ => c.NextUInt32()).ToArray();
        Assert.Equal(sa, sb);
        Assert.NotEqual(sa, sc);
    }
}
