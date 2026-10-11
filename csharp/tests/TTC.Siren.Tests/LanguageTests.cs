using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Explain;
using TTC.Siren.Server;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Decision A-29: the explanation can be read in Chinese or English (<c>?lang=</c>) in the answer of <c>/ai_move</c>, in a replay of the game log and in the live events,
/// from the same facts.
/// </summary>
public sealed class LanguageTests : ServerTestBase
{
    public LanguageTests(ITestOutputHelper output) : base(output)
    {
    }

    private static readonly Regex Chinese = new(@"[㐀-鿿]", RegexOptions.Compiled);

    private const string Game = """
        {"board":[],"myHand":[{"numU":4,"numL":2,"numD":3,"numR":4},{"numU":5,"numL":4,"numD":6,"numR":5},{"numU":2,"numL":2,"numD":7,"numR":2}],
        "oppHand":[{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0},{"numU":0,"numL":0,"numD":0,"numR":0}],
        "myOwner":1,"currentPlayer":1,"rules":"加算,同数","max_depth":3,"max_time":1E1,"gameId":"lang-1"}
        """;

    /// <summary>Two values are the same JSON: same members, same strings, same numbers (the facts of two answers differ only in the measured search time).</summary>
    private static void AssertSameJson(JsonElement expected, JsonElement actual)
    {
        var compare = new JsonCompare(p => p.EndsWith("search_ms", StringComparison.Ordinal), Array.Empty<string>());
        compare.Compare(expected, actual);
        Assert.True(compare.Differences.Count == 0, string.Join("; ", compare.Differences.Take(5)));
        Assert.True(compare.Leaves > 5);
    }

    private static async Task<JsonDocument> AskAsync(HttpClient http, ServerHost host, string body, string query = "", ulong seed = 7)
    {
        var message = new HttpRequestMessage(HttpMethod.Post, host.BaseUrl + "/ai_move" + query) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        message.Headers.Add("X-Ttc-Test-Seed", seed.ToString());
        using var response = await http.SendAsync(message);
        string text = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, text);
        return JsonDocument.Parse(text);
    }

    [Fact]
    public async Task The_answer_explains_in_the_language_asked_for_and_the_facts_are_the_same()
    {
        var host = await StartAsync(o => o.TestSeedHeader = true);
        using var http = new HttpClient();
        using var zh = await AskAsync(http, host, Game.Replace("lang-1", "l-a"));
        using var en = await AskAsync(http, host, Game.Replace("lang-1", "l-b"), "?lang=en");
        using var inBody = await AskAsync(http, host, Game.Replace("lang-1", "l-c").Replace("\"max_depth\"", "\"lang\":\"en\",\"max_depth\""));
        using var queryWins = await AskAsync(http, host, Game.Replace("lang-1", "l-d").Replace("\"max_depth\"", "\"lang\":\"en\",\"max_depth\""), "?lang=zh-CN");
        using var unknown = await AskAsync(http, host, Game.Replace("lang-1", "l-e"), "?lang=fr");

        string Lang(JsonDocument d) => d.RootElement.GetProperty("explanation").GetProperty("language").GetString()!;
        string Brief(JsonDocument d) => d.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString()!;
        Assert.Equal("zh-CN", Lang(zh));
        Assert.Equal("en", Lang(en));
        Assert.Equal("en", Lang(inBody));
        Assert.Equal("zh-CN", Lang(queryWins));          // the query parameter beats the body
        Assert.Equal("zh-CN", Lang(unknown));            // a language we have no templates for is ignored, not an error
        Assert.Matches(Chinese, Brief(zh));
        Assert.DoesNotMatch(Chinese, Brief(en));
        Assert.DoesNotMatch(Chinese, en.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("full").GetString()!);
        Assert.Equal(Brief(en), Brief(inBody));
        // the facts do not depend on the language: the same decision, the same numbers
        AssertSameJson(zh.RootElement.GetProperty("explanation").GetProperty("facts"), en.RootElement.GetProperty("explanation").GetProperty("facts"));
        Assert.Equal(zh.RootElement.GetProperty("card_id").GetInt32(), en.RootElement.GetProperty("card_id").GetInt32());
        Assert.False(en.RootElement.GetProperty("explanation").TryGetProperty("texts", out _)); // the answer to the plugin carries one language only
    }

    [Fact]
    public async Task The_log_keeps_the_text_in_every_language_and_a_replay_can_be_read_in_either_without_touching_a_number()
    {
        var host = await StartAsync(o => o.TestSeedHeader = true);
        using var http = new HttpClient();
        using var answer = await AskAsync(http, host, Game, "?lang=en");
        string briefAsked = answer.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString()!;

        string stored = await http.GetStringAsync(host.BaseUrl + "/v1/games/lang-1/download");
        string requestLine = stored.Split('\n', StringSplitOptions.RemoveEmptyEntries).Single(l => l.Contains("\"type\":\"request\""));
        using (var rec = JsonDocument.Parse(requestLine))
        {
            var ex = rec.RootElement.GetProperty("explanation");
            Assert.Equal("en", ex.GetProperty("language").GetString());                       // what was asked for is what the record shows
            Assert.Equal(briefAsked, ex.GetProperty("text").GetProperty("brief").GetString());
            Assert.Equal(briefAsked, ex.GetProperty("texts").GetProperty("en").GetProperty("brief").GetString());
            Assert.Matches(Chinese, ex.GetProperty("texts").GetProperty("zh-CN").GetProperty("brief").GetString()!);
        }

        // read as Chinese: the text and the language change, nothing else does -- not even the way a number was written in the request
        async Task<string> Request(string query)
        {
            using var doc = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/v1/games/lang-1" + query));
            return doc.RootElement.GetProperty("records").EnumerateArray().Single(r => r.GetProperty("type").GetString() == "request").GetRawText();
        }
        string asStored = await Request("");
        string asZh = await Request("?lang=zh-CN");
        string asEn = await Request("?lang=en");
        Assert.Equal(requestLine, asStored);
        AssertSameJson(JsonDocument.Parse(asStored).RootElement, JsonDocument.Parse(asEn).RootElement);   // already English: nothing changes
        using (var zhRec = JsonDocument.Parse(asZh))
        {
            Assert.Equal("zh-CN", zhRec.RootElement.GetProperty("explanation").GetProperty("language").GetString());
            Assert.Matches(Chinese, zhRec.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString()!);
            Assert.Contains("\"max_time\":1E1", zhRec.RootElement.GetProperty("request_raw").GetRawText());   // the request is the one that was sent
            using var original = JsonDocument.Parse(requestLine);
            AssertSameJson(original.RootElement.GetProperty("explanation").GetProperty("facts"), zhRec.RootElement.GetProperty("explanation").GetProperty("facts"));
            AssertSameJson(original.RootElement.GetProperty("result"), zhRec.RootElement.GetProperty("result"));
        }
        // the download is the file as it was written unless a language is asked for
        string downloadZh = await http.GetStringAsync(host.BaseUrl + "/v1/games/lang-1/download?lang=zh-CN");
        Assert.NotEqual(stored, downloadZh);
        Assert.Equal(stored.Split('\n').Length, downloadZh.Split('\n').Length);
    }

    [Fact]
    public async Task The_live_events_carry_the_brief_in_every_language_and_the_stream_can_choose_one()
    {
        var host = await StartAsync(o => o.TestSeedHeader = true);
        using var http = new HttpClient();
        // the test server has the key check off: every caller is the key "local"
        await using var zhStream = await SseClient.ConnectAsync(host.BaseUrl, "local");
        await using var enStream = await SseClient.ConnectAsync(host.BaseUrl, "local", "?lang=en");
        // no authenticator header is needed here: ServerTestBase starts without keys, the SSE helper sends X-Test-Key which is ignored
        using var answer = await AskAsync(http, host, Game);

        var zhEvents = await zhStream.ReadUntilAsync((type, _) => type == "decision");
        var enEvents = await enStream.ReadUntilAsync((type, _) => type == "decision");
        var zhDecision = zhEvents[^1].Data;
        var enDecision = enEvents[^1].Data;
        Assert.Matches(Chinese, zhDecision.GetProperty("brief").GetString()!);
        Assert.DoesNotMatch(Chinese, enDecision.GetProperty("brief").GetString()!);
        Assert.Equal("en", enDecision.GetProperty("language").GetString());
        Assert.Equal(zhDecision.GetProperty("briefs").GetProperty("en").GetString(), enDecision.GetProperty("brief").GetString()); // both streams hold every language
        // the record that the log announces follows the same choice
        string TextOf(List<(string? Id, string Event, JsonElement Data)> events) =>
            events.Where(e => e.Event == "game.updated" && e.Data.GetProperty("record").TryGetProperty("explanation", out _)).Select(e => e.Data.GetProperty("record").GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString()!).Single();
        Assert.Matches(Chinese, TextOf(zhEvents));
        Assert.DoesNotMatch(Chinese, TextOf(enEvents));
        Assert.Equal(answer.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString(), TextOf(zhEvents));
    }
}
