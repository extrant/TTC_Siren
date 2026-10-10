using System.Globalization;
using System.Text;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 8A, goal section B2/B3: the C# request parser (<c>RequestParser</c> with <c>UnknownCardHandler</c> and <c>PythonRandom</c>) produces,
/// field for field, the state the REAL <c>ai_server.ai_move</c> builds before any search — including the unknown cards it samples after
/// <c>random.seed(seed)</c> and the position of the random generator afterwards. The golden file holds the 360 requests of the Phase 7
/// corpus and 2,400 new seeded requests (every rule, open modes, NPC pools, inHand/canUse, duplicate ids, currentPlayer variants, boards
/// with the all-zero database rows, and 40-odd requests Python refuses).
/// </summary>
public class ParseParityTests
{
    public const int Seed = 20261011;
    public const int Count = 2400;

    internal static string EnsureGolden() =>
        Golden8.Ensure(Path.Combine(Golden8.GoldenDir, $"p8_parse_g1_s{Seed}_n{Count}.jsonl.gz"), "parse", "tools/phase8a/py_parse_trace.py",
            "--out", "{0}", "--seed", Seed.ToString(), "--count", Count.ToString());

    internal static CardCatalog Catalog { get; } = new(CardDb.LoadFromRepository(AppContext.BaseDirectory));

    internal static string CardJson(LCard c)
    {
        var sb = new StringBuilder();
        sb.Append('[').Append(c.CardId is { } id ? id.ToString(CultureInfo.InvariantCulture) : "null").Append(',');
        sb.Append(string.IsNullOrEmpty(c.CardType) ? "null" : "\"" + c.CardType + "\"").Append(',');
        sb.Append(c.Owner is null ? "null" : c.Owner == "red" ? "\"r\"" : "\"b\"").Append(',');
        sb.Append(c.TypeModifier.ToString(CultureInfo.InvariantCulture));
        for (int i = 0; i < 4; i++) sb.Append(',').Append(c.Base[i]);
        for (int i = 0; i < 4; i++) sb.Append(',').Append(c.Disp[i]);
        sb.Append(',').Append(c.CanUse ? 1 : 0);
        sb.Append(',').Append(c.IsGenerated ? 1 : 0).Append(',').Append(c.IsPrediction ? 1 : 0).Append(']');
        return sb.ToString();
    }

    private static string Describe(ParsedRequest r)
    {
        var sb = new StringBuilder();
        sb.Append("cur=").Append(r.CurrentPlayerIdx).Append(" rules=").Append(string.Join(',', r.Rules));
        sb.Append(" board=");
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                sb.Append(r.Board.Get(row, col) is { } c ? CardJson(c) : "null").Append(';');
        foreach (var p in r.State.Players)
        {
            sb.Append(" hand=");
            foreach (var c in p.Hand) sb.Append(CardJson(c)).Append(';');
        }
        return sb.ToString();
    }

    [Fact]
    public void Parsed_state_matches_Python_field_for_field_on_the_Phase7_corpus_and_2400_new_requests()
    {
        string path = EnsureGolden();
        int parsed = 0, refused = 0, withUnknown = 0, withSampling = 0, fields = 0;
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                int index = rec.GetProperty("i").GetInt32();
                string src = rec.GetProperty("src").GetString()!;
                long seed = rec.GetProperty("seed").GetInt64();
                string where = $"{src} request #{index} (seed {seed}) tags {rec.GetProperty("tags")}";

                var rng = new PythonRandom((ulong)seed);
                var parser = new RequestParser(Catalog, rng);
                var outcome = parser.TryParse(rec.GetProperty("request"));

                if (!rec.GetProperty("ok").GetBoolean())
                {
                    Assert.True(outcome.Request is null, $"{where}: Python refused the request, C# parsed it");
                    Assert.True(rec.GetProperty("status").GetInt32() == outcome.Status, $"{where}: status Python {rec.GetProperty("status")} C# {outcome.Status}");
                    string? error = rec.GetProperty("error").GetString();
                    Assert.True(error == outcome.Error, $"{where}: error text Python '{error}' C# '{outcome.Error}'");
                    refused++;
                    continue;
                }

                Assert.True(outcome.Request is not null, $"{where}: C# refused ({outcome.Status} {outcome.Error}), Python parsed it");
                var r = outcome.Request!;

                // the state in the engine payload form
                var state = rec.GetProperty("state");
                Assert.True(state.GetProperty("cur").GetInt32() == r.CurrentPlayerIdx, $"{where}: current player");
                var flags = rec.GetProperty("generated_flags");
                for (int cell = 0; cell < 9; cell++)
                {
                    var want = state.GetProperty("board")[cell];
                    var have = r.Board.Get(cell / 3, cell % 3);
                    string wantText = want.ValueKind == JsonValueKind.Null ? "null" : want.GetRawText();
                    string haveText = have is null ? "null" : CardJson(have);
                    Assert.True(wantText == haveText, $"{where}: board cell {cell}\n  python {wantText}\n  csharp {haveText}");
                    fields++;
                }
                for (int p = 0; p < 2; p++)
                {
                    var wantHand = state.GetProperty("hands")[p];
                    var haveHand = r.State.Players[p].Hand;
                    Assert.True(wantHand.GetArrayLength() == haveHand.Count, $"{where}: hand {p} size python {wantHand.GetArrayLength()} csharp {haveHand.Count}");
                    for (int k = 0; k < haveHand.Count; k++)
                    {
                        Assert.True(wantHand[k].GetRawText() == CardJson(haveHand[k]),
                            $"{where}: hand {p} slot {k}\n  python {wantHand[k].GetRawText()}\n  csharp {CardJson(haveHand[k])}");
                        Assert.True((flags[p][k].GetInt32() == 1) == RequestParser.IsGeneratedUnknownCard(haveHand[k]), $"{where}: generated flag hand {p} slot {k}");
                        fields++;
                    }
                }

                Assert.Equal(rec.GetProperty("rules").EnumerateArray().Select(e => e.GetString()), r.Rules);
                Assert.True(rec.GetProperty("open_mode").GetString() == r.OpenMode, $"{where}: open mode");
                Assert.True(rec.GetProperty("my_owner").GetString() == r.MyOwner && rec.GetProperty("opp_owner").GetString() == r.OppOwner, $"{where}: owners");
                Assert.True(rec.GetProperty("aggressive").GetBoolean() == r.Aggressive, $"{where}: aggressive");
                Assert.True(rec.GetProperty("npc_ids").EnumerateArray().Select(e => e.GetInt64()).SequenceEqual(r.NpcIds), $"{where}: npc ids");
                Assert.True(rec.GetProperty("used_cards").EnumerateArray().Select(e => e.GetInt32()).SequenceEqual(r.UsedCards.OrderBy(x => x)), $"{where}: used cards");
                Assert.True(rec.GetProperty("opp_unknown_count").GetInt32() == r.OppUnknownCount, $"{where}: opponent unknown count");
                Assert.True(rec.GetProperty("use_endgame_robust").GetBoolean() == r.UseEndgameRobust, $"{where}: robust endgame decision");
                Assert.True(rec.GetProperty("show_progress").GetBoolean() == r.ShowSearchProgress, $"{where}: show_search_progress");

                // the random generator is in the same position afterwards: exactly the same number of draws
                string wantNext = rec.GetProperty("rng_next").GetString()!;
                string haveNext = HexFloat.Of(rng.NextDouble());
                Assert.True(wantNext == haveNext, $"{where}: random generator position after parsing, python {wantNext} csharp {haveNext}\n{Describe(r)}");

                parsed++;
                int unknown = r.State.Players.Sum(pl => pl.Hand.Count(RequestParser.IsGeneratedUnknownCard));
                if (unknown > 0) withUnknown++;
                if (r.State.Players.Any(pl => pl.Hand.Any(c => c.CardId >= 1000))) withSampling++;
            }
        }
        Assert.True(parsed + refused >= 2760, $"only {parsed + refused} records");
        Assert.True(parsed >= 2600 && refused >= 50, $"parsed {parsed}, refused {refused}");
        Assert.True(withUnknown >= 1500, $"only {withUnknown} requests had sampled unknown cards");
        Console.WriteLine($"parse parity: {parsed} requests parsed and {refused} refused identically; {fields} card fields; {withUnknown} with unknown cards ({withSampling} predictions with id >= 1000)");
    }
}

/// <summary>Python's <c>float.hex()</c> for the finite, non-zero doubles of a random generator.</summary>
internal static class HexFloat
{
    public static string Of(double d)
    {
        if (d == 0) return double.IsNegative(d) ? "-0x0.0p+0" : "0x0.0p+0";
        long bits = BitConverter.DoubleToInt64Bits(d);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF) - 1023;
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        return $"{(negative ? "-" : "")}0x1.{mantissa:x13}p{(exponent >= 0 ? "+" : "")}{exponent}";
    }
}
