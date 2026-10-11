using System.Collections.Concurrent;
using System.Threading.Channels;
using TTC.Siren.Compat;

namespace TTC.Siren.Server.Live;

/// <summary>Settings of the live event stream (<c>Server:Live</c>).</summary>
public sealed class LiveOptions
{
    /// <summary>
    /// Seconds between the comment lines (<c>: heartbeat</c>) that keep an idle stream alive. Cloudflare closes a connection that sends nothing for 100 s, so the default is well below that.
    /// </summary>
    public double HeartbeatSeconds { get; set; } = 20;

    /// <summary>How many of the most recent events of each key are kept so that a client that reconnects with <c>Last-Event-ID</c> can be given what it missed. 0 switches resuming off.</summary>
    public int ReplayEvents { get; set; } = 256;

    /// <summary>The bytes the kept events of one key may take (an upper bound, by length of text): the oldest are dropped first.</summary>
    public int ReplayBytes { get; set; } = 2 << 20;

    /// <summary>Events of a key nobody listens to are forgotten after this many minutes.</summary>
    public double ReplayMinutes { get; set; } = 15;

    public TimeSpan Heartbeat => TimeSpan.FromSeconds(HeartbeatSeconds);

    public void Validate()
    {
        if (HeartbeatSeconds is < 0.05 or > 60) throw new ArgumentException("Live:HeartbeatSeconds must be between 0.05 and 60 (Cloudflare drops a connection that is silent for 100 s)");
        if (ReplayEvents < 0) throw new ArgumentException("Live:ReplayEvents must not be negative");
        if (ReplayBytes < 1024) throw new ArgumentException("Live:ReplayBytes is too small");
        if (ReplayMinutes <= 0) throw new ArgumentException("Live:ReplayMinutes must be positive");
    }
}

/// <summary>
/// One event of the live stream. <see cref="Id"/> numbers the events of one API key (1, 2, 3 ... from the start of the server), so a client can say where it stopped
/// (<c>Last-Event-ID</c>) and the hub can say how many events it missed. <see cref="Json"/> is the data line (already serialised).
/// </summary>
public sealed record LiveEvent(long Id, string Type, string? GameId, string Json);

/// <summary>What a subscriber that asked to resume (<c>Last-Event-ID</c>) is told before the events.</summary>
/// <param name="Kind"><c>none</c> (a fresh subscription), <c>resumed</c> (every missed event is replayed), <c>gap</c> (some of the missed events are gone: <paramref name="Missed"/> of them),
/// <c>reset</c> (the client's id is beyond the newest event of this server: the server restarted, or the id is not ours).</param>
public readonly record struct ResumeInfo(string Kind, long Missed, long Replayed)
{
    public static readonly ResumeInfo None = new("none", 0, 0);
}

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

    /// <summary>What the subscription was told about the events it missed while it was away.</summary>
    public ResumeInfo Resume { get; internal set; } = ResumeInfo.None;

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
/// never throws. Each key also keeps its most recent events (bounded by count and size, forgotten when idle) so that a client that lost its connection
/// can ask for what it missed (<c>Last-Event-ID</c>): the answer says whether everything was replayed or how many events are gone.
/// </summary>
public sealed class LiveHub
{
    public const int DefaultCapacity = 256;

    private sealed class KeyState
    {
        public readonly object Gate = new();
        public readonly List<LiveSubscription> Subscribers = new();
        public readonly LinkedList<LiveEvent> Recent = new();
        public int RecentChars;
        public long Seq;
        public long LastActivityTicks;
    }

    private readonly ConcurrentDictionary<string, KeyState> _keys = new();
    private readonly int _replayEvents, _replayChars;
    private readonly long _replayTicks;
    private readonly TimeProvider _time;
    private int _publishes;

    /// <summary>Events a subscriber may fall behind before the oldest are dropped.</summary>
    public int Capacity { get; }

    public LiveHub(int capacity = DefaultCapacity, int replayEvents = 0, TimeSpan? replayIdle = null, TimeProvider? time = null, int replayBytes = 2 << 20)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
        _replayEvents = replayEvents;
        _replayChars = Math.Max(1, replayBytes / 2);
        _replayTicks = (replayIdle ?? TimeSpan.FromMinutes(15)).Ticks;
        _time = time ?? TimeProvider.System;
    }

    private bool Replays => _replayEvents > 0;

    /// <summary>
    /// A new subscription. With <paramref name="lastEventId"/> (the id the client saw last) the events it missed are put into the subscription first, in order, and
    /// <see cref="LiveSubscription.Resume"/> says what could be replayed. The replay and the registration happen under one lock, so no event is delivered twice or lost between them.
    /// </summary>
    public LiveSubscription Subscribe(string keyId, string? gameFilter = null, int? capacity = null, long? lastEventId = null)
    {
        var sub = new LiveSubscription(this, keyId, gameFilter, capacity ?? Capacity);
        var state = _keys.GetOrAdd(keyId, _ => new KeyState { LastActivityTicks = _time.GetUtcNow().Ticks });
        lock (state.Gate)
        {
            state.LastActivityTicks = _time.GetUtcNow().Ticks;
            if (lastEventId is { } last)
            {
                if (last > state.Seq) sub.Resume = new ResumeInfo("reset", 0, 0);
                else
                {
                    long oldest = state.Recent.First?.Value.Id ?? state.Seq + 1;
                    long missed = Math.Max(0, oldest - 1 - last);
                    long replayed = 0;
                    foreach (var e in state.Recent)
                    {
                        if (e.Id <= last) continue;
                        sub.Offer(e);
                        replayed++;
                    }
                    sub.Resume = new ResumeInfo(missed > 0 ? "gap" : "resumed", missed, replayed);
                }
            }
            state.Subscribers.Add(sub);
        }
        return sub;
    }

    internal void Remove(LiveSubscription sub)
    {
        if (!_keys.TryGetValue(sub.KeyId, out var state)) return;
        lock (state.Gate)
        {
            state.Subscribers.Remove(sub);
            state.LastActivityTicks = _time.GetUtcNow().Ticks;
        }
    }

    public int SubscriberCount(string keyId)
    {
        if (!_keys.TryGetValue(keyId, out var state)) return 0;
        lock (state.Gate) return state.Subscribers.Count;
    }

    /// <summary>The id of the newest event of the key (0: none yet).</summary>
    public long LastEventId(string keyId)
    {
        if (!_keys.TryGetValue(keyId, out var state)) return 0;
        lock (state.Gate) return state.Seq;
    }

    /// <summary>
    /// Publishes an event to the subscribers of <paramref name="keyId"/>; <paramref name="payload"/> is serialised when somebody listens or when events are kept for resuming.
    /// </summary>
    public void Publish(string keyId, string type, string? gameId, Func<Dictionary<string, object?>> payload)
    {
        KeyState? state;
        if (!_keys.TryGetValue(keyId, out state))
        {
            if (!Replays) return;
            state = _keys.GetOrAdd(keyId, _ => new KeyState { LastActivityTicks = _time.GetUtcNow().Ticks });
        }
        else if (!Replays)
        {
            lock (state.Gate)
                if (state.Subscribers.Count == 0) return;
        }

        var body = payload();
        body["type"] = type;
        if (gameId is not null) body["game_id"] = gameId;
        body["t"] = _time.GetUtcNow().ToString("yyyy-MM-ddTHH:mm:ss.fffZ", System.Globalization.CultureInfo.InvariantCulture);
        string json = PyJson.Serialize(body);

        lock (state.Gate)
        {
            var e = new LiveEvent(++state.Seq, type, gameId, json);
            state.LastActivityTicks = _time.GetUtcNow().Ticks;
            if (Replays)
            {
                state.Recent.AddLast(e);
                state.RecentChars += json.Length;
                while (state.Recent.Count > _replayEvents || (state.RecentChars > _replayChars && state.Recent.Count > 1))
                {
                    state.RecentChars -= state.Recent.First!.Value.Json.Length;
                    state.Recent.RemoveFirst();
                }
            }
            foreach (var t in state.Subscribers) t.Offer(e);
        }

        if ((Interlocked.Increment(ref _publishes) & 0xFF) == 0) Sweep();
    }

    /// <summary>Forgets the kept events of keys that nobody listens to and that have been quiet for the idle time.</summary>
    public void Sweep()
    {
        long now = _time.GetUtcNow().Ticks;
        foreach (var (key, state) in _keys)
        {
            bool idle;
            lock (state.Gate) idle = state.Subscribers.Count == 0 && now - state.LastActivityTicks > _replayTicks;
            if (idle) _keys.TryRemove(key, out _);
        }
    }
}
