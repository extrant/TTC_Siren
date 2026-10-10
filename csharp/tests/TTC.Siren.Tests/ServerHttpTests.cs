using System.Net.Http;
using System.Text;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Server;

namespace TTC.Siren.Tests;

/// <summary>
/// The HTTP surface of the product server (a real socket, the production code path): the plugin contract of API.md — POST /ai_move with
/// the plugin's JSON, the answer in the form of Flask's jsonify — plus the production seeding rule and the test-only seed header.
/// </summary>
public class ServerHttpTests
{
    private static async Task<ServerHost> StartAsync(bool testSeedHeader)
    {
        var options = new ServerOptions { Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries, TestSeedHeader = testSeedHeader };
        var host = ServerHost.Create(options);
        await host.StartAsync();
        return host;
    }

    private static HttpRequestMessage Post(string url, string json, ulong? seed = null)
    {
        var message = new HttpRequestMessage(HttpMethod.Post, url + "/ai_move") { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        if (seed is not null) message.Headers.Add("X-Ttc-Test-Seed", seed.Value.ToString());
        return message;
    }

    [Fact]
    public async Task Http_ai_move_answers_like_the_Python_server_for_requests_of_the_corpus()
    {
        string path = AiMoveParityTests.EnsureGolden();
        await using var host = await StartAsync(testSeedHeader: true);
        using var client = new HttpClient();
        int compared = 0;
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                if (rec.GetProperty("i").GetInt32() % 12 != 0) continue; // a sample over the whole corpus, over a real socket
                using var response = await client.SendAsync(Post(host.BaseUrl, rec.GetProperty("request").GetRawText(), rec.GetProperty("seed").GetUInt64()));
                string text = await response.Content.ReadAsStringAsync();
                Assert.Equal(rec.GetProperty("status").GetInt32(), (int)response.StatusCode);
                Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
                Assert.EndsWith("\n", text); // like Flask's jsonify
                using var actual = JsonDocument.Parse(text);
                var compare = new JsonCompare(AiMoveParityTests.IsTiming, AiMoveParityTests.AdditiveKeys);
                compare.Compare(rec.GetProperty("response"), actual.RootElement);
                Assert.True(compare.Differences.Count == 0, $"request #{rec.GetProperty("i")}: " + string.Join("; ", compare.Differences.Take(5)));
                compared++;
            }
        }
        Assert.True(compared >= 70, $"only {compared} requests compared");
    }

    private const string OpeningRequest = """
        {"board":[],"myHand":[{"numU":4,"numL":2,"numD":3,"numR":4,"canUse":true,"inHand":true},{"numU":5,"numL":4,"numD":6,"numR":5,"canUse":true,"inHand":true},
        {"numU":0,"numR":0,"numD":0,"numL":0,"canUse":true,"inHand":true}],
        "oppHand":[{"numU":0,"numR":0,"numD":0,"numL":0},{"numU":0,"numR":0,"numD":0,"numL":0},{"numU":0,"numR":0,"numD":0,"numL":0}],
        "myOwner":1,"currentPlayer":1,"rules":"加算,同数","max_depth":3,"max_time":5}
        """;

    [Fact]
    public async Task Production_seeds_every_request_freshly_and_the_header_is_ignored_unless_the_test_switch_is_on()
    {
        await using var host = await StartAsync(testSeedHeader: false);
        var seeds = new List<ulong>();
        host.RequestCompleted += o => seeds.Add(o.Seed);
        using var client = new HttpClient();
        for (int i = 0; i < 4; i++)
        {
            using var response = await client.SendAsync(Post(host.BaseUrl, OpeningRequest, seed: 12345));
            Assert.Equal(200, (int)response.StatusCode);
        }
        Assert.Equal(4, seeds.Count);
        Assert.DoesNotContain(12345UL, seeds);                      // the header had no effect
        Assert.Equal(4, seeds.Distinct().Count());                  // fresh 64-bit seeds
        Assert.All(seeds, s => Assert.True(s > uint.MaxValue));     // 64-bit values (probability of a failure: 2^-32 per seed)

        await using var testHost = await StartAsync(testSeedHeader: true);
        var testSeeds = new List<ulong>();
        testHost.RequestCompleted += o => testSeeds.Add(o.Seed);
        using (var response = await client.SendAsync(Post(testHost.BaseUrl, OpeningRequest, seed: 12345))) Assert.Equal(200, (int)response.StatusCode);
        Assert.Equal(new ulong[] { 12345 }, testSeeds);
    }

    [Fact]
    public async Task Bad_bodies_are_answered_with_the_error_forms_of_API_md()
    {
        await using var host = await StartAsync(testSeedHeader: false);
        using var client = new HttpClient();
        foreach (var (body, status) in new[] { ("{", 400), ("null", 400), ("{}", 400), ("[1]", 400), ("{\"board\":[],\"myHand\":[],\"oppHand\":[]}", 400) })
        {
            using var response = await client.SendAsync(Post(host.BaseUrl, body));
            string text = await response.Content.ReadAsStringAsync();
            Assert.True(status == (int)response.StatusCode, $"{body}: {(int)response.StatusCode} {text}");
            Assert.Equal("{\"error\":\"Invalid input data\"}\n", text);
        }

        using (var response = await client.GetAsync(host.BaseUrl + "/health"))
        {
            Assert.Equal(200, (int)response.StatusCode);
            using var health = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal("ok", health.RootElement.GetProperty("status").GetString());
            Assert.Equal(475, health.RootElement.GetProperty("cardCount").GetInt32());
        }
    }
}
