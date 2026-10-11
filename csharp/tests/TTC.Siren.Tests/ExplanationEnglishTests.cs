using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Compat;
using TTC.Siren.Explain;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// The English explanation templates (decision A-29) have the same guarantees as the Chinese ones: they come from the same facts, every number in a text is a number of the facts,
/// a win is claimed only when the exact value says win, the same facts always give the same text (committed snapshots), and the two files have the same keys.
/// </summary>
public sealed class ExplanationEnglishTests
{
    private readonly ITestOutputHelper _output;

    public ExplanationEnglishTests(ITestOutputHelper output) => _output = output;

    private static readonly TemplateSet En = TemplateSet.Embedded("en");
    private static readonly TemplateSet Zh = TemplateSet.Embedded("zh-CN");

    [Fact]
    public void The_two_template_files_have_exactly_the_same_keys_and_no_empty_variant()
    {
        Assert.Equal("en", En.Language);
        Assert.Equal("zh-CN", Zh.Language);
        Assert.Equal(Zh.Templates.Keys.OrderBy(k => k, StringComparer.Ordinal), En.Templates.Keys.OrderBy(k => k, StringComparer.Ordinal));
        foreach (var (key, variants) in En.Templates)
        {
            Assert.NotEmpty(variants);
            // only the words/markers that are meant to be empty may be empty strings
            if (key != "word.known") Assert.All(variants, v => Assert.False(string.IsNullOrWhiteSpace(v) && key != "word.sep.sentence", $"'{key}' has an empty variant"));
        }
        // a template names only slots that exist in the other language's template of the same key, or none at all (the renderer fills a fixed set of slots per key)
        var slot = new Regex(@"\{([a-z_0-9]+)\}");
        foreach (var (key, variants) in En.Templates)
        {
            var enSlots = variants.SelectMany(v => slot.Matches(v).Select(m => m.Groups[1].Value == "card_cap" ? "card" : m.Groups[1].Value)).ToHashSet(); // card_cap is the card with a capital letter
            var zhSlots = Zh.Templates[key].SelectMany(v => slot.Matches(v).Select(m => m.Groups[1].Value)).ToHashSet();
            Assert.True(enSlots.IsSubsetOf(zhSlots) || zhSlots.Count == 0 && enSlots.Count == 0, $"'{key}': English uses {{{string.Join(",", enSlots.Except(zhSlots))}}} which the Chinese template does not");
        }
    }

    [Fact]
    public void Languages_are_normalised_from_the_tags_a_client_sends()
    {
        Assert.Equal("en", TemplateSet.Normalise("en"));
        Assert.Equal("en", TemplateSet.Normalise("EN-us"));
        Assert.Equal("zh-CN", TemplateSet.Normalise("zh"));
        Assert.Equal("zh-CN", TemplateSet.Normalise("zh-CN"));
        Assert.Equal("zh-CN", TemplateSet.Normalise("zh-Hans-CN"));
        Assert.Null(TemplateSet.Normalise("fr"));
        Assert.Null(TemplateSet.Normalise(""));
        Assert.Null(TemplateSet.Normalise(null));
        Assert.Null(TemplateSet.Normalise("english"));
    }

    private static HashSet<double> FactNumbers(JsonElement e)
    {
        var set = new HashSet<double>();
        void Walk(JsonElement x)
        {
            switch (x.ValueKind)
            {
                case JsonValueKind.Number: set.Add(x.GetDouble()); break;
                case JsonValueKind.Array: foreach (var i in x.EnumerateArray()) Walk(i); break;
                case JsonValueKind.Object: foreach (var p in x.EnumerateObject()) Walk(p.Value); break;
            }
        }
        Walk(e);
        return set;
    }

    private static int Sentences(string text) => Regex.Matches(text, @"[.!?](?=\s|$)").Count;

    [Fact]
    public void The_English_text_says_only_what_the_facts_say_every_number_comes_from_a_fact_and_a_win_only_from_an_exact_win()
    {
        var number = new Regex(@"-?\d+(\.\d+)?", RegexOptions.Compiled);
        var winWord = new Regex(@"\b(win|wins|won|winning|victory)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        int checkedNumbers = 0, exactWins = 0, withheld = 0;
        foreach (var d in ExplanationFactsTests.All.Value)
        {
            var text = ExplanationRenderer.Render(d.Facts, En);
            var numbers = FactNumbers(JsonDocument.Parse(PyJson.Serialize(d.Facts.ToJson())).RootElement);
            foreach (var (kind, content) in new[] { ("brief", text.Brief), ("full", text.Full) })
                foreach (Match m in number.Matches(Regex.Replace(content, "\"[^\"]*\"", ""))) // an NPC name in quotes is a label, not a number
                {
                    double value = double.Parse(m.Value, System.Globalization.CultureInfo.InvariantCulture);
                    int decimals = m.Groups[1].Success ? m.Groups[1].Value.Length - 1 : 0;
                    double magnitude = Math.Abs(value);
                    bool found = numbers.Any(f => Math.Round(Math.Abs(f), decimals, MidpointRounding.AwayFromZero) == magnitude ||
                                                  Math.Round(Math.Abs(f), decimals, MidpointRounding.ToEven) == magnitude ||
                                                  Math.Round(f * 100, 0, MidpointRounding.AwayFromZero) == magnitude);
                    Assert.True(found, $"{ExplanationFactsTests.Where(d)}: the {kind} text mentions {m.Value}, which is in none of the facts:\n{content}");
                    checkedNumbers++;
                }

            // no claim against the facts
            var o = d.Facts.Outcome;
            bool claimsWin = winWord.IsMatch(text.Brief);
            if (o.Kind == "exact" && o.Result == "win") { Assert.True(claimsWin, $"{ExplanationFactsTests.Where(d)}: an exact win is not said:\n{text.Brief}"); exactWins++; }
            else
            {
                Assert.False(claimsWin, $"{ExplanationFactsTests.Where(d)}: the brief text claims a win without an exact win ({o.Kind} {o.Result}):\n{text.Brief}");
                withheld++;
            }
            if (o.Kind == "exact" && o.Result != "draw") Assert.DoesNotContain("draw", text.Brief);
            if (o.Kind == "exact" && o.Result == "win") Assert.DoesNotContain("unfavourable", text.Brief);
            if (o.Kind != "exact")
            {
                Assert.DoesNotContain("solved", text.Brief);
                Assert.DoesNotContain("calculated to the end", text.Brief);
            }
            // a flip is mentioned exactly when the move flips something
            Assert.Equal(d.Facts.Immediate.Flips.Count > 0, Regex.IsMatch(text.Brief, @"\b(flips|takes|captures) \d+"));
            Assert.True(Sentences(text.Brief) is >= 1 and <= 2, $"{ExplanationFactsTests.Where(d)}: brief has more than two sentences: {text.Brief}");
            Assert.True(char.IsUpper(text.Brief[0]) || text.Brief[0] == '"', "the text starts like a sentence");
            Assert.DoesNotContain("  ", text.Brief);
            Assert.DoesNotContain("{", text.Full);
        }
        Assert.True(checkedNumbers > 3000, $"only {checkedNumbers} numbers checked");
        Assert.True(exactWins > 0 && withheld > 0, "the corpus has to contain both exact wins and decisions without one");
        _output.WriteLine($"{checkedNumbers} numbers in {ExplanationFactsTests.All.Value.Count} English brief and full texts, each found in the facts; {exactWins} exact wins said, {withheld} decisions without a win claim");
    }

    [Fact]
    public void The_same_facts_always_give_the_same_English_text_and_the_wording_varies_with_the_position()
    {
        var first = ExplanationFactsTests.All.Value.Select(d => ExplanationRenderer.Render(d.Facts, En)).ToList();
        var second = ExplanationFactsTests.All.Value.Select(d => ExplanationRenderer.Render(d.Facts, TemplateSet.Embedded("en"))).ToList();
        Assert.True(first.SequenceEqual(second));
        Assert.True(first.Select(t => t.Brief).Distinct().Count() > 50);
        // the Chinese text of the same facts is not touched by the English file
        var zh = ExplanationFactsTests.All.Value.Select(d => ExplanationRenderer.Render(d.Facts, Zh)).ToList();
        Assert.True(zh.SequenceEqual(ExplanationFactsTests.All.Value.Select(d => d.Text)));
        Assert.All(first.Zip(zh), p => Assert.NotEqual(p.First.Brief, p.Second.Brief));
        // and no Chinese character slips into the English text
        Assert.All(first, t => Assert.DoesNotMatch(@"[㐀-鿿]", t.Brief + t.Full.Replace("(no specific assumption)", "")));
    }

    private static string SnapshotPath => Path.Combine(GoldenFiles.RepoRoot, "tests", "data", "explanation_snapshots_v2_en.json");

    /// <summary>The same 30 decisions as the Chinese snapshots, in English. To accept an intended change: <c>TTC_UPDATE_SNAPSHOTS=1</c> once and review the diff.</summary>
    [Fact]
    public void The_English_text_matches_the_committed_snapshots()
    {
        var all = ExplanationFactsTests.All.Value;
        var chosen = all.Where((d, i) => i % (all.Count / 30) == 0).Take(30).ToList();
        var actual = chosen.ToDictionary(d => $"{d.Source}#{d.Index}", d =>
        {
            var t = ExplanationRenderer.Render(d.Facts, En);
            return new Dictionary<string, string> { ["brief"] = t.Brief, ["full"] = t.Full };
        });
        string path = SnapshotPath;
        if (Environment.GetEnvironmentVariable("TTC_UPDATE_SNAPSHOTS") == "1")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(actual, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        }
        Assert.True(File.Exists(path), "snapshot file missing: run once with TTC_UPDATE_SNAPSHOTS=1");
        var expected = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(File.ReadAllText(path))!;
        Assert.Equal(expected.Keys.OrderBy(k => k), actual.Keys.OrderBy(k => k));
        foreach (var (key, texts) in actual)
        {
            Assert.True(expected[key]["brief"] == texts["brief"], $"{key}: brief text changed" + Environment.NewLine + $"  expected {expected[key]["brief"]}" + Environment.NewLine + $"  actual   {texts["brief"]}");
            Assert.True(expected[key]["full"] == texts["full"], $"{key}: full text changed" + Environment.NewLine + $"  expected {expected[key]["full"]}" + Environment.NewLine + $"  actual   {texts["full"]}");
        }
    }
}
