using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Live;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>A client of the Server-Sent Events stream: reads the events of <c>GET /v1/live</c> into a channel.</summary>
internal sealed class SseClient : IAsyncDisposable
{
    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly CancellationTokenSource _cts = new();
    private readonly Channel<(string? Id, string Event, string Data)> _events = Channel.CreateUnbounded<(string?, string, string)>();
    private Task? _pump;
    private HttpResponseMessage? _response;

    public static async Task<SseClient> ConnectAsync(string baseUrl, string key, string query = "", string? lastEventId = null)
    {
        var client = new SseClient();
        var request = new HttpRequestMessage(HttpMethod.Get, baseUrl + "/v1/live" + query);
        request.Headers.Add("X-Test-Key", key);
        if (lastEventId is not null) request.Headers.Add("Last-Event-ID", lastEventId);
        client._response = await client._http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, client._cts.Token);
        client.Status = (int)client._response.StatusCode;
        if (client.Status == 200) client._pump = Task.Run(client.PumpAsync);
        return client;
    }

    public int Status { get; private set; }

    private async Task PumpAsync()
    {
        try
        {
            await using var stream = await _response!.Content.ReadAsStreamAsync(_cts.Token);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            string? id = null, type = "message";
            var data = new StringBuilder();
            string? line;
            while ((line = await reader.ReadLineAsync(_cts.Token)) is not null)
            {
                if (line.Length == 0)
                {
                    if (data.Length > 0) _events.Writer.TryWrite((id, type, data.ToString()));
                    id = null;
                    type = "message";
                    data.Clear();
                }
                else if (line.StartsWith(':')) continue; // comment (connected, heartbeat)
                else if (line.StartsWith("id: ", StringComparison.Ordinal)) id = line[4..];
                else if (line.StartsWith("event: ", StringComparison.Ordinal)) type = line[7..];
                else if (line.StartsWith("data: ", StringComparison.Ordinal)) data.Append(line[6..]);
            }
        }
        catch (Exception e) when (e is OperationCanceledException or IOException or HttpRequestException)
        {
        }
        finally
        {
            _events.Writer.TryComplete();
        }
    }

    /// <summary>Reads events until <paramref name="stop"/> returns true for one (that event is included). Fails the test after the timeout.</summary>
    public async Task<List<(string? Id, string Event, JsonElement Data)>> ReadUntilAsync(Func<string, JsonElement, bool> stop, TimeSpan? timeout = null)
    {
        var list = new List<(string?, string, JsonElement)>();
        using var cts = new CancellationTokenSource(timeout ?? TimeSpan.FromSeconds(30));
        while (await _events.Reader.WaitToReadAsync(cts.Token))
            while (_events.Reader.TryRead(out var e))
            {
                var data = JsonDocument.Parse(e.Data).RootElement.Clone();
                list.Add((e.Id, e.Event, data));
                if (stop(e.Event, data)) return list;
            }
        throw new InvalidOperationException("the stream ended before the expected event");
    }

    /// <summary>Events that arrive within the given time (used to show that something does NOT arrive).</summary>
    public async Task<List<(string? Id, string Event, JsonElement Data)>> ReadForAsync(TimeSpan time)
    {
        var list = new List<(string?, string, JsonElement)>();
        using var cts = new CancellationTokenSource(time);
        try
        {
            while (await _events.Reader.WaitToReadAsync(cts.Token))
                while (_events.Reader.TryRead(out var e)) list.Add((e.Id, e.Event, JsonDocument.Parse(e.Data).RootElement.Clone()));
        }
        catch (OperationCanceledException)
        {
        }
        return list;
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
        if (_pump is not null) await Task.WhenAny(_pump, Task.Delay(2000));
        _response?.Dispose();
        _http.Dispose();
    }
}

/// <summary>
/// The live event stream (goal section C3): the events of one request in order, isolation between keys, and back-pressure — a consumer that does
/// not keep up loses its oldest events and is told so, and never slows the producer.
/// </summary>
public sealed class LiveEventTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ttc-live-" + Guid.NewGuid().ToString("N"));

    public LiveEventTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private async Task<ServerHost> StartAsync(int liveBuffer = 256)
    {
        var options = new ServerOptions
        {
            Host = "127.0.0.1", Port = 0, Workers = 2, TtEntries = AiMoveParityTests.TtEntries, LiveBuffer = liveBuffer,
            GameLog = new GameLogOptions { Directory = _dir },
            Security = new TTC.Siren.Server.Security.SecurityOptions { RequireKey = false, DefaultBurst = 1000, DefaultRatePerMinute = 100000, DefaultMaxInFlight = 64 },
        };
        var host = ServerHost.Create(options, authenticator: new HeaderAuthenticator());
        await host.StartAsync();
        return host;
    }

    [Fact]
    public async Task The_events_of_a_request_arrive_in_order_and_describe_the_same_decision_as_the_answer()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        await using var host = await StartAsync();
        await using var alice = await SseClient.ConnectAsync(host.BaseUrl, "alice");
        Assert.Equal(200, alice.Status);
        using var http = new HttpClient();

        // a game of the user (red, first): the second request carries the opponent's move; the last one ends the game
        var sim = new GameSimulator(db, opponentAi, new Random(3), "加算,同数,逆转", user: 0, first: 0);
        int deepest = 0, robustEvents = 0, requests = 0;
        var seenTypes = new SortedSet<string>();
        while (!sim.Finished)
        {
            if (!sim.UserToMove)
            {
                await sim.PlayOpponentAsync();
                continue;
            }
            var (status, text) = await SimHttp.PostAsync(http, host.BaseUrl, sim.UserRequest(7), key: "alice");
            Assert.Equal(200, status);
            using var answer = JsonDocument.Parse(text);
            string gameId = answer.RootElement.GetProperty("game_id").GetString()!;
            var events = await alice.ReadUntilAsync((type, data) => type == "decision");
            requests++;

            var types = events.Select(e => e.Event).ToList();
            foreach (var t in types) seenTypes.Add(t);
            Assert.Equal("request.received", types[0]);
            Assert.Equal(1, types.Count(t => t == "decision"));
            Assert.Equal("decision", types[^1]);
            Assert.All(events, e => Assert.Equal(gameId, e.Data.GetProperty("game_id").GetString()));
            // ids strictly increase
            var ids = events.Select(e => long.Parse(e.Id!)).ToList();
            Assert.True(ids.Zip(ids.Skip(1), (a, b) => a < b).All(x => x));
            // search.depth: 1, 2, 3, ... in order, between the request and the decision, with growing node counts
            var depths = events.Where(e => e.Event == "search.depth").ToList();
            Assert.True(depths.Count >= 1);
            Assert.Equal(Enumerable.Range(1, depths.Count), depths.Select(e => e.Data.GetProperty("depth").GetInt32()));
            var nodes = depths.Select(e => e.Data.GetProperty("nodes").GetInt64()).ToList();
            Assert.True(nodes.Zip(nodes.Skip(1), (a, b) => a <= b).All(x => x));
            deepest = Math.Max(deepest, depths.Count);
            int firstDepth = types.IndexOf("search.depth"), requestRecord = types.LastIndexOf("game.updated");
            Assert.True(types.IndexOf("request.received") < firstDepth && firstDepth < requestRecord && requestRecord < types.IndexOf("decision"));
            // the last depth is the depth of the answer, the decision is the move of the answer
            Assert.Equal(answer.RootElement.GetProperty("performance_stats").GetProperty("search_depth").GetInt32(), depths[^1].Data.GetProperty("depth").GetInt32());
            var decision = events[^1].Data;
            Assert.Equal(answer.RootElement.GetProperty("card_id").GetInt32(), decision.GetProperty("move").GetProperty("card_id").GetInt32());
            Assert.Equal(answer.RootElement.GetProperty("pos")[0].GetInt32(), decision.GetProperty("move").GetProperty("row").GetInt32());
            Assert.Equal(answer.RootElement.GetProperty("explanation").GetProperty("text").GetProperty("brief").GetString(), decision.GetProperty("brief").GetString());
            // robust progress comes as (0, total) then (total, total)
            var robust = events.Where(e => e.Event == "robust.progress").ToList();
            if (robust.Count > 0)
            {
                Assert.Equal(2, robust.Count);
                Assert.Equal(0, robust[0].Data.GetProperty("done").GetInt32());
                Assert.Equal(robust[0].Data.GetProperty("total").GetInt32(), robust[1].Data.GetProperty("done").GetInt32());
                robustEvents++;
            }
            sim.PlayUser(answer.RootElement);
        }
        // the end of the game arrives as game.updated(game_end) and game.ended
        var tail = await alice.ReadUntilAsync((type, data) => type == "game.ended");
        Assert.Contains(tail, e => e.Event == "game.updated" && e.Data.GetProperty("record").GetProperty("type").GetString() == "game_end");
        Assert.True(deepest >= 3 && requests == 5, $"deepest {deepest}, requests {requests}");
        _output.WriteLine($"{requests} requests streamed in order ({deepest} depths at most, {robustEvents} with robust progress); event types: {string.Join(", ", seenTypes)}");
    }

    [Fact]
    public async Task A_key_receives_only_its_own_events_and_a_game_filter_narrows_the_stream()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var opponentAi = AiMoveParityTests.CreateService();
        await using var host = await StartAsync();
        await using var alice = await SseClient.ConnectAsync(host.BaseUrl, "alice");
        await using var bob = await SseClient.ConnectAsync(host.BaseUrl, "bob");
        await using var nobody = await SseClient.ConnectAsync(host.BaseUrl, "alice", "?game_id=some-other-game");
        using var http = new HttpClient();

        var simA = new GameSimulator(db, opponentAi, new Random(21), "加算", user: 0, first: 0);
        var simB = new GameSimulator(db, opponentAi, new Random(22), "同数", user: 1, first: 0);
        var (_, ta) = await SimHttp.PostAsync(http, host.BaseUrl, simA.UserRequest(5), key: "alice");
        var gameA = JsonDocument.Parse(ta).RootElement.GetProperty("game_id").GetString()!;
        var aliceEvents = await alice.ReadUntilAsync((type, _) => type == "decision");
        Assert.All(aliceEvents, e => Assert.Equal(gameA, e.Data.GetProperty("game_id").GetString()));
        Assert.Empty(await bob.ReadForAsync(TimeSpan.FromMilliseconds(400)));        // alice's request is none of bob's business
        Assert.Empty(await nobody.ReadForAsync(TimeSpan.FromMilliseconds(100)));      // the filter names another game

        var (_, tb) = await SimHttp.PostAsync(http, host.BaseUrl, simB.UserRequest(5), key: "bob");
        var gameB = JsonDocument.Parse(tb).RootElement.GetProperty("game_id").GetString()!;
        var bobEvents = await bob.ReadUntilAsync((type, _) => type == "decision");
        Assert.All(bobEvents, e => Assert.Equal(gameB, e.Data.GetProperty("game_id").GetString()));
        Assert.Empty(await alice.ReadForAsync(TimeSpan.FromMilliseconds(400)));
        Assert.NotEqual(gameA, gameB);

        // a filter that names the game receives its events
        await using var filtered = await SseClient.ConnectAsync(host.BaseUrl, "alice", "?game_id=" + gameA);
        host.Hub.Publish("alice", "test.ping", gameA, () => new Dictionary<string, object?> { ["n"] = 1 });
        host.Hub.Publish("alice", "test.ping", "elsewhere", () => new Dictionary<string, object?> { ["n"] = 2 });
        var got = await filtered.ReadUntilAsync((type, _) => type == "test.ping");
        Assert.Equal(1, got[^1].Data.GetProperty("n").GetInt32());
        Assert.Empty(await filtered.ReadForAsync(TimeSpan.FromMilliseconds(200)));
    }

    [Fact]
    public async Task A_slow_consumer_loses_its_oldest_events_never_slows_the_producer_and_is_told_how_many()
    {
        var hub = new LiveHub(capacity: 8);
        using var slow = hub.Subscribe("k");      // never reads
        using var other = hub.Subscribe("other"); // another key: sees nothing
        // "the producer is never slowed" is a property of the code, not of the machine: if Publish waited for the consumer, this loop would never end.
        // The watchdog only turns a hang into a failure; no speed is asserted (review M8-2).
        var producer = Task.Run(() =>
        {
            for (int i = 1; i <= 20_000; i++) hub.Publish("k", "tick", "g", () => new Dictionary<string, object?> { ["i"] = i });
        });
        await producer.WaitAsync(TimeSpan.FromSeconds(120));
        _output.WriteLine("20,000 events published to a consumer that never reads: the producer never blocked");

        Assert.Equal(20_000 - 8, slow.TakeDropped());
        Assert.Equal(0, slow.TakeDropped()); // taken once
        var kept = new List<LiveEvent>();
        while (slow.Reader.TryRead(out var e)) kept.Add(e);
        Assert.Equal(8, kept.Count);
        Assert.Equal(Enumerable.Range(19_993, 8).Select(i => (long)i), kept.Select(e => e.Id)); // the newest 8, in order
        Assert.False(other.Reader.TryRead(out _));
        Assert.Equal(0, other.TakeDropped());
    }

    [Fact]
    public async Task The_stream_tells_a_lagging_client_how_many_events_it_missed_and_ends_with_the_newest()
    {
        await using var host = await StartAsync(liveBuffer: 4);
        await using var client = await SseClient.ConnectAsync(host.BaseUrl, "alice");
        // wait until the subscription exists
        for (int i = 0; i < 100 && host.Hub.SubscriberCount("alice") == 0; i++) await Task.Delay(20);
        Assert.Equal(1, host.Hub.SubscriberCount("alice"));

        for (int i = 1; i <= 5000; i++) host.Hub.Publish("alice", "tick", null, () => new Dictionary<string, object?> { ["i"] = i });
        var events = await client.ReadUntilAsync((type, data) => type == "tick" && data.GetProperty("i").GetInt32() == 5000);
        var lagged = events.Where(e => e.Event == "lagged").ToList();
        Assert.NotEmpty(lagged);
        long missed = lagged.Sum(e => e.Data.GetProperty("dropped").GetInt64());
        var ticks = events.Where(e => e.Event == "tick").Select(e => e.Data.GetProperty("i").GetInt32()).ToList();
        Assert.True(ticks.Zip(ticks.Skip(1), (a, b) => a < b).All(x => x)); // never out of order
        Assert.Equal(5000, ticks.Count + missed);                           // every event is either delivered or counted as missed
        _output.WriteLine($"5,000 events through a buffer of 4: {ticks.Count} delivered, {missed} reported missed in {lagged.Count} lagged notices");
    }
}
