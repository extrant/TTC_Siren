using System.Buffers;
using System.Diagnostics;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Engine;

namespace TTC.Siren.Server;

/// <summary>What the warm-up did.</summary>
public sealed record WarmUpReport(int Workers, int Searches, double Milliseconds, long NodesSearched);

/// <summary>
/// Review O8-3. The first requests after a start are slow: the JIT has not compiled the search, the parser, the analysis and the explainer yet, and the
/// first search of each worker touches the pages of its 96 MiB transposition table for the first time. The warm-up runs the real request path once on every
/// worker (an opening, and an endgame with unknown cards, which takes the robust path) before the server accepts a connection, so a user's first request does
/// not pay for it. It writes no game log and publishes no event, and it does not change any later answer (each request starts from a cleared table, review H4-1).
/// </summary>
public static class WarmUp
{
    public static async Task<WarmUpReport> RunAsync(AiMoveService ai, WorkerPool pool, CardDb db, double searchSeconds = 0.3, CancellationToken ct = default)
    {
        long t0 = Stopwatch.GetTimestamp();
        string opening = BuildRequest(db, occupied: 0, searchSeconds, depth: 7);
        string endgame = BuildRequest(db, occupied: 5, 5.0, depth: 10);
        int searches = 0;
        long nodes = 0;

        // every worker gets one request of each kind: they are all in flight at once, so each takes a different worker (a request waits for a free one)
        foreach (string text in new[] { opening, endgame })
        {
            var tasks = new List<Task<AiMoveOutcome>>();
            for (int i = 0; i < pool.Workers; i++)
            {
                ulong seed = (ulong)(0x57A5_0000 + i);
                tasks.Add(Task.Run(async () =>
                {
                    using var doc = JsonDocument.Parse(text);
                    return await ai.HandleAsync(doc.RootElement, seed, ct).ConfigureAwait(false);
                }, ct));
            }
            foreach (var outcome in await Task.WhenAll(tasks).ConfigureAwait(false))
            {
                if (outcome.Status != 200) throw new InvalidOperationException($"the warm-up request was refused: {outcome.Status} {TTC.Siren.Compat.PyJson.Serialize(outcome.Body)}");
                searches++;
                nodes += outcome.Solve?.Nodes ?? 0;
            }
        }
        return new WarmUpReport(pool.Workers, searches, Stopwatch.GetElapsedTime(t0).TotalMilliseconds, nodes);
    }

    /// <summary>
    /// A plugin request built from the card database: <paramref name="occupied"/> cards on the board (alternating owners), my hand and the opponent's hand filled up
    /// to nine cards in all; the opponent's cards are unknown (all zero). With five cards on the board two cards are unknown, so that the robust path runs.
    /// </summary>
    public static string BuildRequest(CardDb db, int occupied, double maxTimeSeconds, int depth)
    {
        // plain cards of the database with distinct edges, so that nothing depends on which duplicate a lookup returns
        var cards = db.Cards.Where(c => c.Id is >= 1 and <= 400 && c.Up > 0 && c.Star <= 3).GroupBy(c => (c.Up, c.Right, c.Down, c.Left)).Where(g => g.Count() == 1).Select(g => g.First()).Take(24).ToList();
        if (occupied is < 0 or > 7 || (occupied > 0 && occupied % 2 == 0)) throw new ArgumentOutOfRangeException(nameof(occupied), "0, or an odd number of cards on the board up to 7");
        int mine = occupied == 0 ? 5 : (9 - occupied) / 2, theirs = mine;
        int onBoard = occupied;
        if (occupied > 0 && onBoard + mine + theirs != 9) throw new InvalidOperationException("the position does not add up to nine cards");
        var buffer = new ArrayBufferWriter<byte>(2048);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteStartArray("board");
            for (int i = 0; i < onBoard; i++)
            {
                var c = cards[i];
                w.WriteStartObject();
                w.WriteStartArray("pos");
                w.WriteNumberValue(i / 3);
                w.WriteNumberValue(i % 3);
                w.WriteEndArray();
                // the protocol names the edges as the plugin does (D-011): numL is the database's RIGHT edge, numR its LEFT edge
                w.WriteNumber("numU", c.Up);
                w.WriteNumber("numL", c.Right);
                w.WriteNumber("numD", c.Down);
                w.WriteNumber("numR", c.Left);
                w.WriteNumber("owner", i % 2 == 0 ? 1 : 2);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("myHand");
            for (int i = 0; i < mine; i++)
            {
                var c = cards[10 + i];
                w.WriteStartObject();
                w.WriteNumber("numU", c.Up);
                w.WriteNumber("numL", c.Right);
                w.WriteNumber("numD", c.Down);
                w.WriteNumber("numR", c.Left);
                w.WriteBoolean("canUse", true);
                w.WriteBoolean("inHand", true);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("oppHand");
            for (int i = 0; i < theirs; i++)
            {
                w.WriteStartObject();
                w.WriteNumber("numU", 0);
                w.WriteNumber("numL", 0);
                w.WriteNumber("numD", 0);
                w.WriteNumber("numR", 0);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteNumber("myOwner", 2);
            w.WriteNumber("currentPlayer", 2);
            w.WriteString("rules", "加算,同数");
            w.WriteNumber("max_depth", depth);
            w.WriteNumber("max_time", maxTimeSeconds);
            w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
