using System.Collections.Concurrent;
using System.Net;
using Microsoft.AspNetCore.Http;

namespace TTC.Siren.Server.Security;

/// <summary>
/// Authenticates a request by its API key: <c>Authorization: Bearer &lt;key&gt;</c> or <c>X-Api-Key: &lt;key&gt;</c> (Authorization wins if both are sent).
/// The key is compared as a hash; the secret never leaves this method, and a log line shows at most its first eight characters.
/// 401 = no key or an unknown one, 403 = a revoked key.
/// </summary>
public sealed class KeyAuthenticator : IApiKeyAuthenticator
{
    private readonly ApiKeyStore _store;
    private readonly SecurityOptions _options;
    private readonly Action<string>? _log;

    public KeyAuthenticator(ApiKeyStore store, SecurityOptions options, Action<string>? log = null)
    {
        _store = store;
        _options = options;
        _log = log;
    }

    public static string? ExtractKey(HttpRequest request)
    {
        if (request.Headers.TryGetValue("Authorization", out var auth))
        {
            string value = auth.ToString();
            if (value.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)) return value[7..].Trim();
        }
        if (request.Headers.TryGetValue("X-Api-Key", out var key) && key.ToString().Trim() is { Length: > 0 } k) return k;
        return null;
    }

    public AuthResult Authenticate(HttpRequest request)
    {
        string? secret = ExtractKey(request);
        if (secret is null)
        {
            if (_options.AllowLoopbackWithoutKey && request.HttpContext.Connection.RemoteIpAddress is { } ip && IPAddress.IsLoopback(ip)) return AuthResult.Allowed("local-dev");
            return AuthResult.Denied(401, "API key required (Authorization: Bearer <key> or X-Api-Key)");
        }
        var record = _store.Find(secret);
        if (record is null)
        {
            _log?.Invoke($"rejected unknown API key (no part of it is logged) from {request.HttpContext.Connection.RemoteIpAddress}");
            return AuthResult.Denied(401, "invalid API key");
        }
        if (!record.Enabled)
        {
            _log?.Invoke($"rejected revoked API key {record.Id}");
            return AuthResult.Denied(403, "API key revoked");
        }
        return AuthResult.Allowed(record.Id);
    }
}

/// <summary>What a rate-limit check decided: admitted (with a lease that must be disposed when the request ends) or refused (with the seconds to wait).</summary>
public sealed class LimitDecision
{
    public bool Allowed { get; init; }
    public int RetryAfterSeconds { get; init; }
    public string Reason { get; init; } = "";
    public IDisposable? Lease { get; init; }
}

/// <summary>
/// Per-key limits: a token bucket (<c>Burst</c> requests at once, refilled at <c>RatePerMinute</c>) and a maximum number of requests in flight.
/// Answered with 429 and <c>Retry-After</c>. The limits of a key are read from its record every time, so <c>keys set-limit</c> takes effect without a restart.
/// </summary>
public sealed class KeyLimiter
{
    private sealed class State
    {
        public double Tokens;
        public long LastTicks;
        public int InFlight;
        public int Streams;
        public readonly object Gate = new();
    }

    private sealed class Lease : IDisposable
    {
        private readonly State _state;
        private readonly bool _stream;
        private int _disposed;

        public Lease(State state, bool stream)
        {
            _state = state;
            _stream = stream;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 1) return;
            lock (_state.Gate)
            {
                if (_stream) _state.Streams--;
                else _state.InFlight--;
            }
        }
    }

    private readonly ApiKeyStore? _store;
    private readonly SecurityOptions _options;
    private readonly TimeProvider _time;
    private readonly ConcurrentDictionary<string, State> _states = new();

    public KeyLimiter(ApiKeyStore? store, SecurityOptions options, TimeProvider? time = null)
    {
        _store = store;
        _options = options;
        _time = time ?? TimeProvider.System;
    }

    private (int Rate, int Burst, int InFlight) LimitsOf(string keyId)
    {
        var record = _store?.GetById(keyId);
        return record is null
            ? (_options.DefaultRatePerMinute, _options.DefaultBurst, _options.DefaultMaxInFlight)
            : (record.RatePerMinute, record.Burst, record.MaxInFlight);
    }

    /// <summary>Admits one request of the key, or says how long to wait.</summary>
    public LimitDecision TryEnter(string keyId)
    {
        var (rate, burst, maxInFlight) = LimitsOf(keyId);
        var state = _states.GetOrAdd(keyId, _ => new State { Tokens = burst, LastTicks = _time.GetUtcNow().UtcTicks });
        lock (state.Gate)
        {
            long now = _time.GetUtcNow().UtcTicks;
            double elapsed = Math.Max(0, now - state.LastTicks) / (double)TimeSpan.TicksPerSecond;
            state.LastTicks = now;
            double perSecond = rate / 60.0;
            state.Tokens = Math.Min(burst, state.Tokens + elapsed * perSecond);
            if (state.InFlight >= maxInFlight)
                return new LimitDecision { Allowed = false, RetryAfterSeconds = 1, Reason = $"too many requests in flight (limit {maxInFlight})" };
            if (state.Tokens < 1.0)
                return new LimitDecision { Allowed = false, RetryAfterSeconds = Math.Max(1, (int)Math.Ceiling((1.0 - state.Tokens) / perSecond)), Reason = $"rate limit exceeded ({rate} requests per minute)" };
            state.Tokens -= 1.0;
            state.InFlight++;
            return new LimitDecision { Allowed = true, Lease = new Lease(state, stream: false) };
        }
    }

    /// <summary>Admits one more open event stream of the key (limited by <see cref="SecurityOptions.MaxLiveStreamsPerKey"/>).</summary>
    public LimitDecision TryOpenStream(string keyId)
    {
        var (_, burst, _) = LimitsOf(keyId);
        var state = _states.GetOrAdd(keyId, _ => new State { Tokens = burst, LastTicks = _time.GetUtcNow().UtcTicks });
        lock (state.Gate)
        {
            if (state.Streams >= _options.MaxLiveStreamsPerKey)
                return new LimitDecision { Allowed = false, RetryAfterSeconds = 5, Reason = $"too many open event streams (limit {_options.MaxLiveStreamsPerKey})" };
            state.Streams++;
            return new LimitDecision { Allowed = true, Lease = new Lease(state, stream: true) };
        }
    }
}
