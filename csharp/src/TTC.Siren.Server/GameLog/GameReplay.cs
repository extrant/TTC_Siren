using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TTC.Siren.Server.GameLog;

/// <summary>
/// Replays a logged decision (goal section C2): the <c>request</c> record holds the plugin's request, the seed of the unknown-card sampling and the
/// settings, which is everything the decision depended on. The replay runs the same request with the same seed through the same code; the search
/// is run to the depth the original reached and without a time limit, so a decision that was cut short by the clock is reproduced by what it had
/// completed. The caller compares the outcome with the record (the move, the scores of every depth, the facts).
/// </summary>
public static class GameReplay
{
    /// <summary>True when the logged search stopped before the depth it was asked for and before the game was searched to its end (time or node limit).</summary>
    public static bool WasInterrupted(JsonElement requestRecord, out int depthReached)
    {
        depthReached = 0;
        if (!requestRecord.TryGetProperty("result", out var result) || result.ValueKind != JsonValueKind.Object) return false;
        depthReached = result.GetProperty("depth_reached").GetInt32();
        int empties = requestRecord.GetProperty("state").GetProperty("board").EnumerateArray().Count(c => c.ValueKind == JsonValueKind.Null);
        var requested = requestRecord.GetProperty("settings").GetProperty("max_depth");
        double wanted = requested.ValueKind == JsonValueKind.Number ? Math.Min(Math.Floor(requested.GetDouble()), 100) : 10;
        return depthReached < Math.Min(wanted, empties);
    }

    public static async Task<AiMoveOutcome> ReplayAsync(AiMoveService service, JsonElement requestRecord, CancellationToken ct = default)
    {
        var raw = JsonNode.Parse(requestRecord.GetProperty("request_raw").GetRawText())!.AsObject();
        ulong seed = ulong.Parse(requestRecord.GetProperty("seed").GetString()!, CultureInfo.InvariantCulture);
        // a search that ran out of clock is reproduced by what it completed; one that ended by itself (the whole game searched, or the requested depth)
        // keeps its requested depth, so that even the facts about the settings are the same
        if (WasInterrupted(requestRecord, out int reached) && reached > 0) raw["max_depth"] = reached;
        raw["max_time"] = 1_000_000;
        if (requestRecord.GetProperty("settings").TryGetProperty("explain", out var explain) && explain.ValueKind == JsonValueKind.String) raw["explain"] = explain.GetString();
        raw.Remove("gameId");
        using var doc = JsonDocument.Parse(raw.ToJsonString());
        return await service.HandleAsync(doc.RootElement, seed, ct).ConfigureAwait(false);
    }
}
