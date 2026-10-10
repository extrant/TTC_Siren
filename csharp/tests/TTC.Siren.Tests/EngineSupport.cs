using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Core;
using TTC.Siren.Engine;

namespace TTC.Siren.Tests;

/// <summary>An engine on a real loopback socket (port 0), as in production but with small tables.</summary>
public class EngineFixture : IAsyncLifetime
{
    public EngineHost Host { get; private set; } = null!;
    public HttpClient Client { get; private set; } = null!;

    public virtual EngineOptions Options() => new()
    {
        Port = 0,
        Workers = 4,
        QueueLimit = 256,
        QueueWaitMs = 120_000,
        TtEntries = 1 << 18,
        MemoEntries = 1 << 15,
    };

    public async Task InitializeAsync()
    {
        Host = EngineHost.Create(Options());
        await Host.StartAsync();
        Client = new HttpClient { BaseAddress = new Uri(Host.BaseUrl), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task DisposeAsync()
    {
        Client.Dispose();
        await Host.StopAsync();
        await Host.DisposeAsync();
    }
}

/// <summary>Builds protocol requests and reads responses in tests.</summary>
internal static class EngineJson
{
    public static string Card(CardSnapshot? c)
    {
        if (c is null) return "null";
        string id = c.Id < 0 ? "null" : c.Id.ToString();
        string type = c.Type is null ? "null" : JsonSerializer.Serialize(c.Type);
        string owner = c.Owner == OwnerId.Red ? "\"r\"" : c.Owner == OwnerId.Blue ? "\"b\"" : "null";
        return $"[{id},{type},{owner},{c.Mod},{c.Base[0]},{c.Base[1]},{c.Base[2]},{c.Base[3]},{c.Display[0]},{c.Display[1]},{c.Display[2]},{c.Display[3]},{(c.CanUse ? 1 : 0)}]";
    }

    public static string State(StateSnapshot s)
    {
        var sb = new StringBuilder();
        sb.Append("{\"cur\":").Append(s.Current).Append(",\"board\":[");
        sb.Append(string.Join(",", s.Board.Select(Card)));
        sb.Append("],\"hands\":[");
        sb.Append(string.Join(",", s.Hands.Select(h => "[" + string.Join(",", h.Select(Card)) + "]")));
        sb.Append("]}");
        return sb.ToString();
    }

    public static string Rules(RuleSet rules) => "[" + string.Join(",", RuleParser.ToPythonList(rules).Select(r => JsonSerializer.Serialize(r))) + "]";

    public static string Request(StateSnapshot s, bool aggressive = false, string maxDepth = "4", string maxTime = "null",
        string robust = "null", string requestId = "t")
    {
        return $"{{\"protocol\":1,\"requestId\":\"{requestId}\",\"rules\":{Rules(s.Rules)},\"state\":{State(s)},\"aggressive\":{(aggressive ? "true" : "false")}," +
               $"\"maxDepth\":{maxDepth},\"maxTime\":{maxTime},\"robust\":{robust}}}";
    }

    public static string Robust(int aiPlayer, int opponent, bool[][] generated, int sampleCount, IEnumerable<int>? npc = null,
        bool handler = true, string fallback = "null")
    {
        string gen = "[" + string.Join(",", generated.Select(h => "[" + string.Join(",", h.Select(x => x ? "1" : "0")) + "]")) + "]";
        return $"{{\"aiPlayer\":{aiPlayer},\"opponent\":{opponent},\"generated\":{gen},\"sampleCount\":{sampleCount}," +
               $"\"npcIds\":[{string.Join(",", npc ?? Array.Empty<int>())}],\"handler\":{(handler ? "true" : "false")},\"fallbackCandidates\":{fallback}}}";
    }

    public static async Task<(int Status, JsonNode Body)> Post(HttpClient client, string json)
    {
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await client.PostAsync("/v1/solve", content);
        string text = await response.Content.ReadAsStringAsync();
        return ((int)response.StatusCode, JsonNode.Parse(text)!);
    }

    /// <summary>Removes the fields that measure time or name the worker, so that two responses can be compared field by field.</summary>
    public static string Normalized(JsonNode body)
    {
        var copy = JsonNode.Parse(body.ToJsonString())!.AsObject();
        copy.Remove("timings");
        copy.Remove("worker");
        if (copy["depths"] is JsonArray depths)
            foreach (var d in depths) d!.AsObject().Remove("elapsed");
        copy["totals"]?.AsObject().Remove("elapsed");
        copy["robust"]?.AsObject().Remove("elapsed");
        return copy.ToJsonString();
    }

    public static double FromBits(string hex) => BitConverter.Int64BitsToDouble(Convert.ToInt64(hex, 16));
}
