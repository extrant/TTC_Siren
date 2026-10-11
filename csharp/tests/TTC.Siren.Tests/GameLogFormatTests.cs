using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Reviews L8-2 and L8-4: integers stay integers and the request is stored as it was received (log schema version 2, version 1 files still read), and an inferred card
/// that cannot be told apart from others by its edges is marked ambiguous with its candidates.
/// </summary>
public sealed class GameLogFormatTests : ServerTestBase
{
    public GameLogFormatTests(ITestOutputHelper output) : base(output)
    {
    }

    private static readonly Regex IntegralFloat = new(@"(?<![\d.])-?\d+\.0(?![\d])", RegexOptions.Compiled);

    [Fact]
    public void JsonConv_keeps_integers_as_integers()
    {
        using var doc = JsonDocument.Parse("""{"a":357,"b":4.0,"c":-2,"d":1e2,"e":0.25,"f":[1,2.5],"g":9007199254740993}""");
        var obj = (Dictionary<string, object?>)JsonConv.ToObject(doc.RootElement)!;
        Assert.IsType<long>(obj["a"]);
        Assert.IsType<double>(obj["b"]);
        Assert.IsType<long>(obj["c"]);
        Assert.IsType<double>(obj["d"]);
        Assert.IsType<double>(obj["e"]);
        Assert.IsType<long>(((List<object?>)obj["f"]!)[0]);
        Assert.IsType<long>(obj["g"]);
        Assert.Equal("""{"a":357,"b":4.0,"c":-2,"d":100.0,"e":0.25,"f":[1,2.5],"g":9007199254740993}""", PyJson.Serialize(obj));
    }

    [Theory]
    [InlineData("""{"a":1,"b":[true,null,"xé\n"],"c":{"d":4.0,"e":1E2,"f":-0}}""", """{"a":1,"b":[true,null,"xé\n"],"c":{"d":4.0,"e":1E2,"f":-0}}""")]
    [InlineData("  { \"a\" : 1 ,\n \"b\" : [ 1 , 2.50 , \"s p\" ] , \"c\" : { } , \"d\" : [ ] }  ", """{"a":1,"b":[1,2.50,"s p"],"c":{},"d":[]}""")]
    [InlineData("""{"z":1,"a":2}""", """{"z":1,"a":2}""")] // the order of the members is the sender's, not sorted
    public void RawJson_compact_keeps_every_token_as_it_was_written(string input, string expected)
    {
        Assert.Equal(expected, RawJson.Compact(input).Text);
        using var doc = JsonDocument.Parse(RawJson.Compact(input).Text); // and it is valid JSON
    }

    [Fact]
    public async Task The_log_stores_the_request_as_received_and_every_integer_as_an_integer()
    {
        var host = await StartAsync();
        using var http = new HttpClient();
        // a compact body, as a plugin sends it: floats and integers written in their own forms
        string body = "{\"board\":[],\"myHand\":[{\"numU\":4,\"numL\":2,\"numD\":3,\"numR\":4},{\"numU\":5,\"numL\":4,\"numD\":6,\"numR\":5}],\"oppHand\":[{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0},{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0}]," +
               "\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"加算\",\"max_depth\":3,\"max_time\":5.0,\"gameId\":\"raw-1\"}";
        using (var r = await http.SendAsync(Post(host.BaseUrl, body))) Assert.Equal(200, (int)r.StatusCode);

        string download = await http.GetStringAsync(host.BaseUrl + "/v1/games/raw-1/download");
        string api = await http.GetStringAsync(host.BaseUrl + "/v1/games/raw-1");
        var lines = download.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(2, lines.Length); // game_start and the request
        string requestLine = lines.Single(l => l.Contains("\"type\":\"request\""));

        using var rec = JsonDocument.Parse(requestLine);
        Assert.Equal(body, rec.RootElement.GetProperty("request_raw").GetRawText()); // byte for byte what was posted
        Assert.Equal(2, rec.RootElement.GetProperty("v").GetInt32());

        // the games API returns the same lines, nothing parsed and written again
        using var apiDoc = JsonDocument.Parse(api);
        Assert.Equal(2, apiDoc.RootElement.GetProperty("schema").GetInt32());
        var apiRecords = apiDoc.RootElement.GetProperty("records").EnumerateArray().Select(e => e.GetRawText()).ToList();
        Assert.Equal(lines.ToList(), apiRecords);

        // fields that are integers are written without a point -- in the files and in the API alike
        foreach (var line in lines)
        {
            using var d = JsonDocument.Parse(line);
            foreach (string name in new[] { "v", "seq", "elapsed_ms", "request_no", "http_status" })
                if (d.RootElement.TryGetProperty(name, out var n)) Assert.DoesNotContain('.', n.GetRawText());
        }
        using var req = JsonDocument.Parse(requestLine);
        var result = req.RootElement.GetProperty("result");
        foreach (string name in new[] { "depth_reached", "nodes", "tt_hits", "cutoffs", "root_moves" }) Assert.DoesNotContain('.', result.GetProperty(name).GetRawText());
        Assert.DoesNotContain('.', result.GetProperty("move").GetProperty("card_id").GetRawText());
        Assert.Equal("3", req.RootElement.GetProperty("settings").GetProperty("max_depth").GetRawText());
        Assert.Equal("5.0", req.RootElement.GetProperty("settings").GetProperty("max_time").GetRawText()); // a float the sender wrote as a float stays one
        // no integral value is written as a float anywhere in the integer-typed parts of the record (the scores and timings are floats on purpose)
        var counted = new[] { "\"seq\":", "\"request_no\":", "\"http_status\":", "\"depth_reached\":", "\"card_id\":", "\"slot\":", "\"row\":", "\"col\":" };
        foreach (string key in counted)
            foreach (Match m in Regex.Matches(requestLine, Regex.Escape(key) + @"[^,}\]]+"))
                Assert.DoesNotContain(".0", m.Value);
        _output.WriteLine($"request_raw is the posted body byte for byte ({body.Length} bytes); the schema of the file is {rec.RootElement.GetProperty("v").GetInt32()}");
    }

    [Fact]
    public async Task A_pretty_printed_request_is_stored_without_its_white_space_and_with_every_number_as_written()
    {
        var host = await StartAsync();
        using var http = new HttpClient();
        string pretty = "{\n  \"board\": [],\n  \"myHand\": [ {\"numU\": 4, \"numL\": 2, \"numD\": 3, \"numR\": 4} ],\n  \"oppHand\": [ {\"numU\": 0, \"numL\": 0, \"numD\": 0, \"numR\": 0} ],\n" +
                        "  \"myOwner\": 1, \"currentPlayer\": 1, \"rules\": \"加算\", \"max_depth\": 2, \"max_time\": 1E1, \"gameId\": \"raw-2\"\n}";
        using (var r = await http.SendAsync(Post(host.BaseUrl, pretty))) Assert.Equal(200, (int)r.StatusCode);
        string line = (await http.GetStringAsync(host.BaseUrl + "/v1/games/raw-2/download")).Split('\n', StringSplitOptions.RemoveEmptyEntries).Single(l => l.Contains("\"type\":\"request\""));
        using var rec = JsonDocument.Parse(line);
        string stored = rec.RootElement.GetProperty("request_raw").GetRawText();
        Assert.Contains("\"max_time\":1E1", stored);
        Assert.DoesNotContain('\n', stored);
        Assert.Equal(RawJson.Compact(pretty).Text, stored);
    }

    [Fact]
    public async Task A_game_log_written_in_schema_1_is_still_read_and_a_game_may_continue_in_schema_2()
    {
        string logs = Path.Combine(_dir, "v1-logs");
        var host = await StartAsync(sub: "v1-logs");
        using var http = new HttpClient();
        using (var r = await http.SendAsync(Post(host.BaseUrl, Opening.Replace("fixes-1", "old-1")))) Assert.Equal(200, (int)r.StatusCode);
        string file = Path.Combine(logs, "local", "old-1.jsonl");
        // what Phase 8A wrote: version 1 and floats for some integers
        File.WriteAllText(file, File.ReadAllText(file).Replace("\"v\":2", "\"v\":1").Replace("\"schema\":2", "\"schema\":1").Replace("\"elapsed_ms\":0,", "\"elapsed_ms\":0.0,"));

        using (var api = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/v1/games/old-1")))
        {
            Assert.Equal(1, api.RootElement.GetProperty("schema").GetInt32());
            Assert.All(api.RootElement.GetProperty("records").EnumerateArray(), r => Assert.Equal(1, r.GetProperty("v").GetInt32()));
            Assert.Equal(0.0, api.RootElement.GetProperty("records")[0].GetProperty("elapsed_ms").GetDouble());
        }
        using (var list = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/v1/games"))) Assert.Equal("old-1", list.RootElement.GetProperty("games")[0].GetProperty("game_id").GetString());

        // the server continues the old game (a restart: a new service reads the file back) and appends records of version 2
        var second = await StartAsync(sub: "v1-logs");
        using (var r = await http.SendAsync(Post(second.BaseUrl, Opening.Replace("fixes-1", "old-1")))) Assert.Equal(200, (int)r.StatusCode);
        using var again = JsonDocument.Parse(await http.GetStringAsync(second.BaseUrl + "/v1/games/old-1"));
        var versions = again.RootElement.GetProperty("records").EnumerateArray().Select(r => r.GetProperty("v").GetInt32()).ToList();
        Assert.Equal(2, again.RootElement.GetProperty("schema").GetInt32());
        Assert.Contains(1, versions);
        Assert.Contains(2, versions);
        Assert.Equal(Enumerable.Range(0, versions.Count), again.RootElement.GetProperty("records").EnumerateArray().Select(r => r.GetProperty("seq").GetInt32())); // no gap in seq
    }

    /// <summary>A plugin request built from card definitions: <c>numU</c> up, <c>numL</c> the database's right edge, <c>numD</c> down, <c>numR</c> the database's left edge (D-011).</summary>
    private static string Card(CardDef c, string extra = "") => $"{{\"numU\":{c.Up},\"numL\":{c.Right},\"numD\":{c.Down},\"numR\":{c.Left}{extra}}}";

    private async Task<JsonElement> OpponentMoveWithAsync(ServerHost host, CardDb db, CardDef opponentCard, string gameId, List<JsonDocument> keep)
    {
        var mine = db.Cards.Where(c => c.Up + c.Right + c.Down + c.Left > 0 && db.FindFirstIdByEdges(c.Up, c.Right, c.Down, c.Left) == c.Id && c.Id > 100).Take(3).ToList();
        string hand = string.Join(",", mine.Select(c => Card(c, ",\"canUse\":true,\"inHand\":true")));
        string unknown = string.Join(",", Enumerable.Range(0, 3).Select(_ => "{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0,\"canUse\":true,\"inHand\":true}"));
        string first = $"{{\"board\":[],\"myHand\":[{hand}],\"oppHand\":[{unknown}],\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"\",\"max_depth\":3,\"max_time\":5,\"gameId\":\"{gameId}\"}}";
        using var http = new HttpClient();
        using var response = await http.SendAsync(Post(host.BaseUrl, first));
        Assert.Equal(200, (int)response.StatusCode);
        using var answer = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        int playedId = answer.RootElement.GetProperty("card_id").GetInt32();
        int row = answer.RootElement.GetProperty("pos")[0].GetInt32(), col = answer.RootElement.GetProperty("pos")[1].GetInt32();
        var played = db.Get(playedId);

        // the next request: my card where it was recommended (blue = 1), the opponent's card (red = 2) in the first other cell, my hand without the played card
        int oppCell = Enumerable.Range(0, 9).First(c => c != row * 3 + col);
        string board = $"{{\"pos\":[{row},{col}],\"numU\":{played.Up},\"numL\":{played.Right},\"numD\":{played.Down},\"numR\":{played.Left},\"owner\":1}}," +
                       $"{{\"pos\":[{oppCell / 3},{oppCell % 3}],\"numU\":{opponentCard.Up},\"numL\":{opponentCard.Right},\"numD\":{opponentCard.Down},\"numR\":{opponentCard.Left},\"owner\":2}}";
        string hand2 = string.Join(",", mine.Where(c => c.Id != playedId).Select(c => Card(c, ",\"canUse\":true,\"inHand\":true")));
        string unknown2 = string.Join(",", Enumerable.Range(0, 2).Select(_ => "{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0,\"canUse\":true,\"inHand\":true}"));
        string second = $"{{\"board\":[{board}],\"myHand\":[{hand2}],\"oppHand\":[{unknown2}],\"myOwner\":1,\"currentPlayer\":1,\"rules\":\"\",\"max_depth\":3,\"max_time\":5,\"gameId\":\"{gameId}\"}}";
        using (var r2 = await http.SendAsync(Post(host.BaseUrl, second))) Assert.Equal(200, (int)r2.StatusCode);

        var doc = JsonDocument.Parse(await http.GetStringAsync(host.BaseUrl + "/v1/games/" + gameId));
        keep.Add(doc);
        return doc.RootElement.GetProperty("records").EnumerateArray().Single(r => r.GetProperty("type").GetString() == "opponent_move");
    }

    [Fact]
    public async Task An_inferred_card_that_shares_its_edges_with_other_cards_is_marked_ambiguous_with_its_candidates()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        // a group of cards with identical edges (the review's example: ids 12 and 369)
        var group = db.Cards.Where(c => c.Up + c.Right + c.Down + c.Left > 0).GroupBy(c => (c.Up, c.Right, c.Down, c.Left)).Where(g => g.Count() > 1).OrderBy(g => g.First().Id).First().ToList();
        Assert.True(group.Count >= 2);
        var unique = db.Cards.First(c => c.Up + c.Right + c.Down + c.Left > 0 && db.Cards.Count(o => (o.Up, o.Right, o.Down, o.Left) == (c.Up, c.Right, c.Down, c.Left)) == 1 && c.Id > 200);
        var host = await StartAsync();
        var keep = new List<JsonDocument>();
        try
        {
            var amb = await OpponentMoveWithAsync(host, db, group[1], "amb-1", keep); // the SECOND card of the group is the one that was played
            Assert.True(amb.GetProperty("ambiguous").GetBoolean());
            var ids = amb.GetProperty("candidate_ids").EnumerateArray().Select(e => e.GetInt32()).ToList();
            Assert.Equal(group.Select(c => (int)c.Id), ids);                       // all of them, in database order
            Assert.Equal(group[0].Id, amb.GetProperty("card_id").GetInt32());      // the log names the first, as Python's lookup does
            Assert.Equal(new[] { (int)group[1].Up, group[1].Right, group[1].Down, group[1].Left }, amb.GetProperty("card_edges").EnumerateArray().Select(e => e.GetInt32()));

            var one = await OpponentMoveWithAsync(host, db, unique, "amb-2", keep);
            Assert.False(one.GetProperty("ambiguous").GetBoolean());
            Assert.Equal(new[] { (int)unique.Id }, one.GetProperty("candidate_ids").EnumerateArray().Select(e => e.GetInt32()));
            Assert.Equal(unique.Id, one.GetProperty("card_id").GetInt32());
            _output.WriteLine($"ambiguous example: edges of cards {string.Join(", ", group.Select(c => c.Id))}");
        }
        finally
        {
            foreach (var d in keep) d.Dispose();
        }
    }
}
