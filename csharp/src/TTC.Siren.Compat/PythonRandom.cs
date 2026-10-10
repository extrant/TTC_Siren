using System.Numerics;

namespace TTC.Siren.Compat;

/// <summary>
/// CPython 3.11 compatible <c>random.Random</c> (decision A-23): MT19937 seeded exactly like <c>random.seed(int)</c>
/// (<c>init_by_array</c> over the 32-bit words of <c>abs(seed)</c>), and the Python-level algorithms built on it
/// (<c>getrandbits</c>, <c>_randbelow</c>, <c>randrange</c>/<c>randint</c>, <c>choice</c>, <c>sample</c> with both of its
/// algorithms, <c>shuffle</c>, <c>uniform</c>). Given the same seed every call returns exactly what CPython returns
/// (checked against golden sequences produced by Python itself, see PythonRandomTests).
/// The unknown-card sampling of <c>/ai_move</c> only uses <c>sample</c> and <c>choice</c>; the other methods exist so the
/// generator can be verified as a whole. Not thread safe: one instance per request.
/// </summary>
public sealed class PythonRandom
{
    private const int N = 624;
    private const int M = 397;
    private const uint MatrixA = 0x9908b0dfU;
    private const uint UpperMask = 0x80000000U;
    private const uint LowerMask = 0x7fffffffU;

    private readonly uint[] _mt = new uint[N];
    private int _index = N + 1;

    /// <summary>Seeded like <c>random.seed(seed)</c>.</summary>
    public PythonRandom(ulong seed) => Seed(seed);

    /// <summary>Seeded like <c>random.seed(seed)</c> for an arbitrary (also negative) Python int.</summary>
    public PythonRandom(BigInteger seed) => Seed(seed);

    /// <summary><c>random.seed(a)</c> for a non-negative 64-bit integer.</summary>
    public void Seed(ulong seed) => Seed(new BigInteger(seed));

    /// <summary><c>random.seed(a)</c> for an int: the key is the little-endian 32-bit words of <c>abs(a)</c> (one word for 0).</summary>
    public void Seed(BigInteger seed)
    {
        var magnitude = BigInteger.Abs(seed);
        var key = new List<uint>();
        while (!magnitude.IsZero)
        {
            key.Add((uint)(magnitude & uint.MaxValue));
            magnitude >>= 32;
        }
        if (key.Count == 0) key.Add(0);
        InitByArray(key.ToArray());
    }

    private void InitGenrand(uint s)
    {
        _mt[0] = s;
        for (int i = 1; i < N; i++)
            _mt[i] = unchecked(1812433253U * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i);
        _index = N;
    }

    private void InitByArray(uint[] key)
    {
        InitGenrand(19650218U);
        int i = 1, j = 0;
        int k = Math.Max(N, key.Length);
        for (; k > 0; k--)
        {
            _mt[i] = unchecked((_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint)j);
            i++;
            j++;
            if (i >= N)
            {
                _mt[0] = _mt[N - 1];
                i = 1;
            }
            if (j >= key.Length) j = 0;
        }
        for (k = N - 1; k > 0; k--)
        {
            _mt[i] = unchecked((_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1566083941U)) - (uint)i);
            i++;
            if (i >= N)
            {
                _mt[0] = _mt[N - 1];
                i = 1;
            }
        }
        _mt[0] = 0x80000000U; // MSB is 1: assuring non-zero initial array
        _index = N;
    }

    /// <summary>One 32-bit output of the Mersenne Twister (<c>genrand_uint32</c>).</summary>
    public uint NextUInt32()
    {
        if (_index >= N)
        {
            int kk;
            uint y;
            for (kk = 0; kk < N - M; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk + M] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }
            for (; kk < N - 1; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }
            y = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
            _mt[N - 1] = _mt[M - 1] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            _index = 0;
        }
        uint r = _mt[_index++];
        r ^= r >> 11;
        r ^= (r << 7) & 0x9d2c5680U;
        r ^= (r << 15) & 0xefc60000U;
        r ^= r >> 18;
        return r;
    }

    /// <summary><c>random.random()</c>: a float in [0, 1) with 53 bits.</summary>
    public double NextDouble()
    {
        uint a = NextUInt32() >> 5, b = NextUInt32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    /// <summary><c>random.getrandbits(k)</c> for 0 &lt;= k &lt;= 32 (what <c>_randbelow</c> needs for sequences shorter than 2^32).</summary>
    public uint GetRandBits32(int k)
    {
        if (k is < 0 or > 32) throw new ArgumentOutOfRangeException(nameof(k));
        if (k == 0) return 0;
        return NextUInt32() >> (32 - k);
    }

    /// <summary><c>random.getrandbits(k)</c> for any k &gt;= 0 (words are drawn low to high, the last one is shifted, as CPython does).</summary>
    public BigInteger GetRandBits(int k)
    {
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (k == 0) return BigInteger.Zero;
        if (k <= 32) return GetRandBits32(k);
        int words = (k - 1) / 32 + 1;
        BigInteger result = BigInteger.Zero;
        for (int i = 0; i < words; i++, k -= 32)
        {
            uint r = NextUInt32();
            if (k < 32) r >>= 32 - k; // drop the least significant bits
            result |= (BigInteger)r << (32 * i);
        }
        return result;
    }

    /// <summary><c>Random._randbelow_with_getrandbits(n)</c> for 1 &lt;= n &lt; 2^32: uniform in [0, n).</summary>
    public int RandBelow(int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        int k = 32 - BitOperations.LeadingZeroCount((uint)n); // n.bit_length()
        uint r = GetRandBits32(k);
        while (r >= (uint)n) r = GetRandBits32(k);
        return (int)r;
    }

    /// <summary><c>random.randrange(start, stop)</c> (step 1) for ints.</summary>
    public int RandRange(int start, int stop)
    {
        long width = (long)stop - start;
        if (width <= 0) throw new ArgumentException("empty range for randrange()");
        return start + RandBelow((int)width);
    }

    /// <summary><c>random.randrange(stop)</c>.</summary>
    public int RandRange(int stop)
    {
        if (stop <= 0) throw new ArgumentException("empty range for randrange()");
        return RandBelow(stop);
    }

    /// <summary><c>random.randint(a, b)</c> (inclusive).</summary>
    public int RandInt(int a, int b) => RandRange(a, b + 1);

    /// <summary><c>random.uniform(a, b)</c>.</summary>
    public double Uniform(double a, double b) => a + (b - a) * NextDouble();

    /// <summary><c>random.choice(seq)</c>.</summary>
    public T Choice<T>(IReadOnlyList<T> seq)
    {
        if (seq.Count == 0) throw new IndexOutOfRangeException("Cannot choose from an empty sequence");
        return seq[RandBelow(seq.Count)];
    }

    /// <summary><c>random.shuffle(x)</c>, in place.</summary>
    public void Shuffle<T>(IList<T> x)
    {
        for (int i = x.Count - 1; i >= 1; i--)
        {
            int j = RandBelow(i + 1);
            (x[i], x[j]) = (x[j], x[i]);
        }
    }

    /// <summary>
    /// <c>random.sample(population, k)</c> of CPython 3.11: a pool-copy algorithm for small populations and a "selected set"
    /// algorithm for large ones. The threshold <c>setsize</c> follows CPython exactly (<c>21</c>, plus <c>4 ** ceil(log(3k, 4))</c> for
    /// k &gt; 5), so both algorithms are used in the same situations and consume the same numbers.
    /// </summary>
    public List<T> Sample<T>(IReadOnlyList<T> population, int k)
    {
        int n = population.Count;
        if (k < 0 || k > n) throw new ArgumentException("Sample larger than population or is negative");
        var result = new List<T>(k);
        long setsize = 21; // size of a small set minus size of an empty list
        if (k > 5) setsize += (long)Math.Pow(4, Math.Ceiling(Math.Log(k * 3) / Math.Log(4)));
        if (n <= setsize)
        {
            var pool = new List<T>(population);
            for (int i = 0; i < k; i++)
            {
                int j = RandBelow(n - i);
                result.Add(pool[j]);
                pool[j] = pool[n - i - 1]; // move non-selected item into vacancy
            }
        }
        else
        {
            var selected = new HashSet<int>();
            for (int i = 0; i < k; i++)
            {
                int j = RandBelow(n);
                while (selected.Contains(j)) j = RandBelow(n);
                selected.Add(j);
                result.Add(population[j]);
            }
        }
        return result;
    }
}
