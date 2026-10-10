using System.Collections.Concurrent;
using System.Threading.Channels;
using TTC.Siren.Compat;

namespace TTC.Siren.Server.Live;

/// <summary>One event of the live stream. <see cref="Json"/> is the data line (already serialised).</summary>
public sealed record LiveEvent(long Id, string Type, string? GameId, string Json);

/// <summary>
/// A subscriber of one key's events. It reads from a bounded buffer: when the consumer is slower than the producer the OLDEST events are dropped
/// (never the producer made to wait, so a slow consumer cannot slow a search), and <see cref="TakeDropped"/> tells how many were lost so the
/// stream can send a <c>lagged</c> notice.
/// </summary>
public sealed class LiveSubscription : IDisposable
{
    private readonly Channel<LiveEvent> _channel;
    private readonly LiveHub _hub;
    private long _dropped;

    internal LiveSubscription(LiveHub hub, string keyId, string? gameFilter, int capacity)
    {
        _hub = hub;
        KeyId = keyId;
        GameFilter = gameFilter;
        _channel = Channel.CreateBounded<LiveEvent>(
            new BoundedChannelOptions(capacity) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = false },
            _ => Interlocked.Increment(ref _dropped));
    }

    public string KeyId { get; }
    public string? GameFilter { get; }
    public ChannelReader<LiveEvent> Reader => _channel.Reader;

    internal void Offer(LiveEvent e)
    {
        if (GameFilter is not null && e.GameId != GameFilter) return;
        _channel.Writer.TryWrite(e);
    }

    /// <summary>The number of events dropped since the last call.</summary>
    public long TakeDropped() => Interlocked.Exchange(ref _dropped, 0);

    public void Dispose()
    {
        _hub.Remove(this);
        _channel.Writer.TryComplete();
    }
}

/// <summary>
/// The live event stream (goal section C3): events are published per API key and reach only that key's subscribers. Publishing never blocks and
/// never throws; with no subscriber it costs one dictionary lookup.
/// </summary>
public sealed class LiveHub
{
    public const int DefaultCapacity = 256;

    private readonly ConcurrentDictionary<string, List<LiveSubscription>> _subscribers = new();
    private long _nextId;

    /// <summary>Events a subscriber may fall behind before the oldest are dropped.</summary>
    public int Capacity { get; }

    public LiveHub(int capacity = DefaultCapacity)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
    }

    public LiveSubscription Subscribe(string keyId, string? gameFilter = null, int? capacity = null)
    {
        var sub = new LiveSubscription(this, keyId, gameFilter, capacity ?? Capacity);
        var list = _subscribers.GetOrAdd(keyId, _ => new List<LiveSubscription>());
        lock (list) list.Add(sub);
        return sub;
    }

    internal void Remove(LiveSubscription sub)
    {
        if (!_subscribers.TryGetValue(sub.KeyId, out var list)) return;
        lock (list) list.Remove(sub);
    }

    public int SubscriberCount(string keyId) => _subscribers.TryGetValue(keyId, out var list) ? Locked(list) : 0;

    private static int Locked(List<LiveSubscription> list)
    {
        lock (list) return list.Count;
    }

    /// <summary>Publishes an event to the subscribers of <paramref name="keyId"/>; <paramref name="payload"/> is serialised only if somebody listens.</summary>
    public void Publish(string keyId, string type, string? gameId, Func<Dictionary<string, object?>> payload)
    {
        if (!_subscribers.TryGetValue(keyId, out var list)) return;
        LiveSubscription[] targets;
        lock (list)
        {
            if (list.Count == 0) return;
            targets = list.ToArray();
        }
        var body = payload();
        body["type"] = type;
        if (gameId is not null) body["game_id"] = gameId;
        body["t"] = DateTimeOffset.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", System.Globalization.CultureInfo.InvariantCulture);
        var e = new LiveEvent(Interlocked.Increment(ref _nextId), type, gameId, PyJson.Serialize(body));
        foreach (var t in targets) t.Offer(e);
    }
}
