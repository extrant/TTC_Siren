using System.Globalization;
using System.Text;
using System.Text.Json;
using TTC.Siren.Compat;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 8A, goal section B3: the C# port of <c>ai/unknown_card_handler.py</c> draws, from <see cref="PythonRandom"/>, exactly the cards
/// the Python handler draws after <c>random.seed(seed)</c> — 5,400 seeded situations covering every rule family and combination (none,
/// each of the nine rules alone, pairs, triples, all), 1 to 5 unknown slots, the opponent's and the player's own estimate, empty to
/// nearly full boards, known hands with 4/5-star cards, and the slot selection (random sample, or the highest-threat selection late in
/// the game). The next value of the random generator is compared too, so the number of draws is the same.
/// </summary>
public class SamplingParityTests
{
    public const int Seed = 20261011;
    public const int Count = 5400;

    private static string EnsureGolden() =>
        Golden8.Ensure(Path.Combine(Golden8.GoldenDir, $"p8_sampling_g1_s{Seed}_n{Count}.jsonl.gz"), "sampling", "tools/phase8a/py_sampling_trace.py",
            "--out", "{0}", "--seed", Seed.ToString(), "--count", Count.ToString());

    private static string Rec(LCard c)
    {
        var sb = new StringBuilder("[");
        sb.Append(c.CardId is { } id ? id.ToString(CultureInfo.InvariantCulture) : "null").Append(',');
        sb.Append(string.IsNullOrEmpty(c.CardType) ? "null" : "\"" + c.CardType + "\"");
        for (int i = 0; i < 4; i++) sb.Append(',').Append(c.Disp[i]);
        sb.Append(',').Append(c.CanUse ? 1 : 0).Append(',').Append(c.IsGenerated ? 1 : 0).Append(',').Append(c.IsPrediction ? 1 : 0).Append(']');
        return sb.ToString();
    }

    [Fact]
    public void Sampled_unknown_cards_match_Python_exactly_in_5400_seeded_situations()
    {
        string path = EnsureGolden();
        var catalog = ParseParityTests.Catalog;
        int cases = 0, opponentCases = 0, cardsCompared = 0, selectedRisk = 0;
        var perRuleSet = new Dictionary<string, int>();
        var perCount = new int[6];
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                int index = rec.GetProperty("i").GetInt32();
                long seed = rec.GetProperty("seed").GetInt64();
                bool opponent = rec.GetProperty("opponent").GetBoolean();
                string owner = rec.GetProperty("owner").GetString()!;
                var rules = rec.GetProperty("rules").EnumerateArray().Select(e => e.GetString()!).ToList();
                int count = rec.GetProperty("count").GetInt32();
                string where = $"sampling case #{index} (seed {seed}) opponent={opponent} owner={owner} rules=[{string.Join(',', rules)}] count={count} occupied={rec.GetProperty("occupied")}";

                var board = new LBoard();
                foreach (var cell in rec.GetProperty("board").EnumerateArray())
                {
                    int id = cell[1].GetInt32();
                    var proto = catalog.AllCards[id - 1];
                    Assert.Equal(id, proto.CardId);
                    board.Place(cell[0].GetInt32() / 3, cell[0].GetInt32() % 3,
                        new LCard(proto.Base[0], proto.Base[1], proto.Base[2], proto.Base[3], cell[2].GetString(), id, proto.CardType));
                }
                var known = rec.GetProperty("known").EnumerateArray().Select(e =>
                {
                    var proto = catalog.AllCards[e.GetInt32() - 1];
                    return new LCard(proto.Base[0], proto.Base[1], proto.Base[2], proto.Base[3], owner, proto.CardId, proto.CardType);
                }).ToList();
                var used = new HashSet<int>(rec.GetProperty("used").EnumerateArray().Select(e => e.GetInt32()));

                var rng = new PythonRandom((ulong)seed);
                var parser = new RequestParser(catalog, rng);
                var cards = opponent
                    ? parser.Handler.GenerateOpponentCards(count, rules, new HashSet<int>(used), board, known, owner, true)
                    : parser.Handler.GenerateUnknownCards(count, rules, new HashSet<int>(used), board, known, owner, true);
                var selected = parser.SelectUnknownCardsForSlots(cards, rec.GetProperty("select_slots").GetInt32(), board, rules, owner, opponent);

                var wantCards = rec.GetProperty("cards").EnumerateArray().Select(e => e.GetRawText()).ToList();
                var haveCards = cards.Select(Rec).ToList();
                Assert.True(wantCards.SequenceEqual(haveCards), $"{where}: generated cards\n  python {string.Join(' ', wantCards)}\n  csharp {string.Join(' ', haveCards)}");
                var wantSelected = rec.GetProperty("selected").EnumerateArray().Select(e => e.GetRawText()).ToList();
                var haveSelected = selected.Select(Rec).ToList();
                Assert.True(wantSelected.SequenceEqual(haveSelected), $"{where}: selected cards\n  python {string.Join(' ', wantSelected)}\n  csharp {string.Join(' ', haveSelected)}");
                string wantNext = rec.GetProperty("rng_next").GetString()!;
                Assert.True(wantNext == HexFloat.Of(rng.NextDouble()), $"{where}: random generator position afterwards");

                cases++;
                if (opponent) opponentCases++;
                cardsCompared += cards.Count + selected.Count;
                perCount[count]++;
                string key = rules.Count == 0 ? "(none)" : string.Join('+', rules);
                perRuleSet[key] = perRuleSet.GetValueOrDefault(key) + 1;
                if (opponent && board.Occupancy() >= 4 && cards.Count > selected.Count) selectedRisk++;
            }
        }
        Assert.Equal(Count, cases);
        Assert.True(perCount[1] > 300 && perCount[2] > 300 && perCount[3] > 300 && perCount[4] > 100 && perCount[5] > 100,
            $"unknown counts covered: {string.Join(',', perCount)}");
        foreach (var single in new[] { "同数", "加算", "逆转", "王牌杀手", "同类强化", "同类弱化", "秩序", "混乱", "选拔", "(none)" })
            Assert.True(perRuleSet.GetValueOrDefault(single) >= 40, $"rule family {single}: {perRuleSet.GetValueOrDefault(single)} cases");
        Assert.True(selectedRisk > 100, $"only {selectedRisk} high-threat selections");
        Console.WriteLine($"sampling parity: {cases} cases ({opponentCases} opponent), {cardsCompared} cards compared, per unknown count 1..5: {string.Join('/', perCount.Skip(1))}, high-threat selections {selectedRisk}, {perRuleSet.Count} rule combinations");
    }
}
