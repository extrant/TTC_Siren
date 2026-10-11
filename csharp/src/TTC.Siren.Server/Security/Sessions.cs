using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http;

namespace TTC.Siren.Server.Security;

/// <summary>Settings of the web login (<c>Server:Security:Sessions</c>). The web login uses the API key: there are no accounts and no passwords.</summary>
public sealed class SessionOptions
{
    /// <summary>How long a session lasts, in hours. It ends sooner on logout, when the key is revoked or disabled, or when the server restarts (sessions live in memory).</summary>
    public double LifetimeHours { get; set; } = 12;

    /// <summary>The most sessions one key may have at once; the oldest is ended to make room.</summary>
    public int MaxPerKey { get; set; } = 20;

    /// <summary>Login attempts per minute and real client address that are refilled (a failed attempt and a successful one count the same).</summary>
    public int LoginAttemptsPerMinute { get; set; } = 6;

    /// <summary>Login attempts one address may make at once before it has to wait for the refill.</summary>
    public int LoginBurst { get; set; } = 5;

    /// <summary>The audit log of logins (one JSON object per line: time, address, success, key id on success). Relative paths are relative to the application folder; empty switches the file off.</summary>
    public string AuditLogFile { get; set; } = "audit.jsonl";

    public void Validate()
    {
        if (LifetimeHours is <= 0 or > 24 * 31) throw new ArgumentException("Sessions:LifetimeHours must be between 0 and 744 (31 days)");
        if (MaxPerKey < 1) throw new ArgumentException("Sessions:MaxPerKey must be at least 1");
        if (LoginAttemptsPerMinute < 1 || LoginBurst < 1) throw new ArgumentException("Sessions:LoginAttemptsPerMinute and LoginBurst must be at least 1");
    }

    public TimeSpan Lifetime => TimeSpan.FromHours(LifetimeHours);
}

/// <summary>A web session: whose key it belongs to and when it ends.</summary>
public sealed record Session(string KeyId, DateTimeOffset Created, DateTimeOffset Expires);

/// <summary>
/// The server-side sessions of the web login (decision A-27). The browser holds a random 256-bit session id in an HttpOnly cookie; the server holds only its SHA-256, so
/// neither a memory listing nor a log can reveal an id that works. A session belongs to the API key it was created with and ends when the key is revoked or disabled
/// (checked at every use, so no restart is needed), when it expires, on logout, or when the key has more sessions than allowed. The id is never logged.
/// </summary>
public sealed class SessionStore
{
    private sealed class Entry
    {
        public required Session Session;
        public required long Seq;
    }

    private readonly ApiKeyStore? _keys;
    private readonly SessionOptions _options;
    private readonly TimeProvider _time;
    private readonly ConcurrentDictionary<string, Entry> _byHash = new(StringComparer.Ordinal);
    private long _seq;

    public SessionStore(ApiKeyStore? keys, SessionOptions options, TimeProvider? time = null)
    {
        _keys = keys;
        _options = options;
        _time = time ?? TimeProvider.System;
    }

    public int Count => _byHash.Count;

    private static string HashOf(string id) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(id)));

    /// <summary>Starts a session for the key and returns its id (43 characters of base64url: the 256 random bits). The id is shown to the browser once, in the cookie.</summary>
    public (string Id, Session Session) Create(string keyId)
    {
        string id = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32)).TrimEnd('=').Replace('+', '-').Replace('/', '_');
        var now = _time.GetUtcNow();
        var session = new Session(keyId, now, now + _options.Lifetime);
        _byHash[HashOf(id)] = new Entry { Session = session, Seq = Interlocked.Increment(ref _seq) };
        EnforceLimit(keyId);
        if ((_seq & 0x3F) == 0) PurgeExpired();
        return (id, session);
    }

    private void EnforceLimit(string keyId)
    {
        var mine = _byHash.Where(kv => kv.Value.Session.KeyId == keyId).OrderBy(kv => kv.Value.Seq).ToList();
        for (int i = 0; i < mine.Count - _options.MaxPerKey; i++) _byHash.TryRemove(mine[i].Key, out _);
    }

    /// <summary>
    /// The session of an id, or null: unknown, expired, or its key is no longer enabled (then the session is ended). The key store is consulted at every use.
    /// </summary>
    public Session? Find(string? id)
    {
        if (string.IsNullOrEmpty(id) || id.Length > 128) return null;
        string hash = HashOf(id);
        if (!_byHash.TryGetValue(hash, out var entry)) return null;
        if (entry.Session.Expires <= _time.GetUtcNow())
        {
            _byHash.TryRemove(hash, out _);
            return null;
        }
        if (_keys is not null && _keys.GetById(entry.Session.KeyId) is not { Enabled: true })
        {
            _byHash.TryRemove(hash, out _);
            return null;
        }
        return entry.Session;
    }

    public bool Remove(string? id) => !string.IsNullOrEmpty(id) && id.Length <= 128 && _byHash.TryRemove(HashOf(id), out _);

    /// <summary>Ends every session of the key (used when a key is revoked while the server runs; lookups also check the key themselves).</summary>
    public int RemoveAllOf(string keyId)
    {
        int n = 0;
        foreach (var kv in _byHash)
            if (kv.Value.Session.KeyId == keyId && _byHash.TryRemove(kv.Key, out _)) n++;
        return n;
    }

    public void PurgeExpired()
    {
        var now = _time.GetUtcNow();
        foreach (var kv in _byHash)
            if (kv.Value.Session.Expires <= now) _byHash.TryRemove(kv.Key, out _);
    }
}

/// <summary>Reads and writes the session cookie.</summary>
public sealed class SessionCookie
{
    private readonly SecurityOptions _security;

    public SessionCookie(SecurityOptions security) => _security = security;

    /// <summary>
    /// With a Secure cookie the name starts with <c>__Host-</c>, which the browser accepts only for a Secure cookie without a Domain and with Path=/: a sibling
    /// subdomain cannot set or overwrite it. Without TLS (a loopback development server) the prefix is impossible and the name is plain.
    /// </summary>
    public const string SecureName = "__Host-ttc_session";
    public const string PlainName = "ttc_session";

    /// <summary>Secure whenever the browser talks TLS to us or to the proxy in front of us.</summary>
    public bool IsSecure(HttpRequest request) => request.IsHttps || _security.BehindTlsProxy || _security.Https.Enabled;

    public string NameFor(HttpRequest request) => IsSecure(request) ? SecureName : PlainName;

    /// <summary>The session id the request carries (either name is read: the one that does not apply simply is not there), or null.</summary>
    public string? Read(HttpRequest request)
    {
        if (request.Cookies.TryGetValue(IsSecure(request) ? SecureName : PlainName, out var value) && value.Length is > 0 and <= 128) return value;
        return null;
    }

    public bool Has(HttpRequest request) => Read(request) is not null;

    /// <summary>HttpOnly (no script can read it), Secure (as above), SameSite=Strict (never sent with a request from another site), Path=/.</summary>
    public void Write(HttpResponse response, HttpRequest request, string id, DateTimeOffset expires, TimeSpan lifetime)
    {
        response.Cookies.Append(NameFor(request), id, new CookieOptions
        {
            HttpOnly = true, Secure = IsSecure(request), SameSite = SameSiteMode.Strict, Path = "/", MaxAge = lifetime, Expires = expires, IsEssential = true,
        });
    }

    /// <summary>Tells the browser to forget the cookie.</summary>
    public void Clear(HttpResponse response, HttpRequest request)
    {
        response.Cookies.Append(NameFor(request), "", new CookieOptions
        {
            HttpOnly = true, Secure = IsSecure(request), SameSite = SameSiteMode.Strict, Path = "/", MaxAge = TimeSpan.Zero, Expires = DateTimeOffset.UnixEpoch, IsEssential = true,
        });
    }
}

/// <summary>
/// CSRF defence for requests that are authenticated by the cookie (decision A-27). The cookie is SameSite=Strict, and on top of that a state-changing request must
/// come from the page's own origin: its <c>Origin</c> header (or, without one, its <c>Referer</c>) must name this host; <c>Sec-Fetch-Site</c>, when the browser sends it,
/// must say same-origin (or none). A request without any of them is refused. Requests authenticated by an API key do not need this: nothing is sent by itself.
/// </summary>
public static class OriginPolicy
{
    public static bool IsStateChanging(string method) => !(HttpMethods.IsGet(method) || HttpMethods.IsHead(method) || HttpMethods.IsOptions(method) || HttpMethods.IsTrace(method));

    /// <summary>True when the request provably comes from this server's own pages.</summary>
    public static bool IsSameOrigin(HttpRequest request)
    {
        if (request.Headers.TryGetValue("Sec-Fetch-Site", out var site) && site.ToString() is { Length: > 0 } s && s != "same-origin" && s != "none") return false;
        string? origin = request.Headers["Origin"].ToString();
        if (!string.IsNullOrEmpty(origin)) return SameAuthority(origin, request);
        string? referer = request.Headers["Referer"].ToString();
        if (!string.IsNullOrEmpty(referer)) return SameAuthority(referer, request);
        return false;
    }

    /// <summary>Whether the header holds an origin or URL whose host and port are the request's own (the scheme is not compared: TLS may end at the proxy in front).</summary>
    private static bool SameAuthority(string value, HttpRequest request)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)) return false; // "null" and the like
        string host = request.Host.Host;
        int? port = request.Host.Port;
        if (!string.Equals(uri.Host, host, StringComparison.OrdinalIgnoreCase)) return false;
        // a missing port in the Host header means the default port of the scheme it was reached with
        int hostPort = port ?? (request.IsHttps ? 443 : 80);
        int originPort = uri.Port;
        return originPort == hostPort || (port is null && (originPort is 80 or 443));
    }
}

/// <summary>
/// The audit log of the web login: time, real client address, success, and the key id on success (never a key, never a session id). Written as JSON lines to a file
/// and to the application log.
/// </summary>
public sealed class AuditLog
{
    private readonly string? _path;
    private readonly Action<string> _log;
    private readonly TimeProvider _time;
    private readonly object _gate = new();

    public AuditLog(string? path, Action<string> log, TimeProvider? time = null)
    {
        _path = string.IsNullOrWhiteSpace(path) ? null : Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(AppContext.BaseDirectory, path));
        _log = log;
        _time = time ?? TimeProvider.System;
    }

    public string? FilePath => _path;

    /// <param name="outcome"><c>success</c>, <c>failed</c> or <c>throttled</c>.</param>
    public void Login(string ip, string outcome, string? keyId)
    {
        Write("login", ip, outcome, keyId);
    }

    public void Logout(string ip, string? keyId) => Write("logout", ip, "success", keyId);

    private void Write(string eventName, string ip, string outcome, string? keyId)
    {
        string line = TTC.Siren.Compat.PyJson.Serialize(new Dictionary<string, object?>
        {
            ["t"] = _time.GetUtcNow().ToString("yyyy-MM-ddTHH:mm:ss.fffZ", System.Globalization.CultureInfo.InvariantCulture), ["event"] = eventName, ["ip"] = ip,
            ["success"] = outcome == "success", ["outcome"] = outcome, ["key_id"] = keyId,
        });
        _log("audit " + line);
        if (_path is null) return;
        try
        {
            lock (_gate)
            {
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(_path)!);
                File.AppendAllText(_path, line + "\n", new UTF8Encoding(false));
            }
        }
        catch (IOException e)
        {
            _log("audit log could not be written: " + e.GetType().Name);
        }
    }
}

/// <summary>Per-address limits for visitors who are not logged in: a token bucket (rate per minute, burst) and a number of requests in flight.</summary>
public sealed class IpLimiter
{
    private sealed class State
    {
        public double Tokens;
        public long LastTicks;
        public int InFlight;
        public long LastUsed;
        public readonly object Gate = new();
    }

    private sealed class Lease : IDisposable
    {
        private readonly State _state;
        private int _done;

        public Lease(State state) => _state = state;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _done, 1) == 1) return;
            lock (_state.Gate) _state.InFlight--;
        }
    }

    private readonly TimeProvider _time;
    private readonly ConcurrentDictionary<string, State> _states = new();
    private readonly int _maxEntries;
    private long _calls;

    public IpLimiter(TimeProvider? time = null, int maxEntries = 200_000)
    {
        _time = time ?? TimeProvider.System;
        _maxEntries = maxEntries;
    }

    public int Entries => _states.Count;

    /// <summary>Admits one request of the address, or says how long to wait. Dispose the lease of an admitted request when it ends.</summary>
    public LimitDecision TryEnter(string address, int ratePerMinute, int burst, int maxInFlight)
    {
        long now = _time.GetUtcNow().UtcTicks;
        var state = _states.GetOrAdd(address, _ => new State { Tokens = burst, LastTicks = now });
        LimitDecision decision;
        lock (state.Gate)
        {
            double elapsed = Math.Max(0, now - state.LastTicks) / (double)TimeSpan.TicksPerSecond;
            state.LastTicks = now;
            state.LastUsed = now;
            double perSecond = ratePerMinute / 60.0;
            state.Tokens = Math.Min(burst, state.Tokens + elapsed * perSecond);
            if (state.InFlight >= maxInFlight) decision = new LimitDecision { Allowed = false, RetryAfterSeconds = 1, Reason = "too many requests in flight from this address" };
            else if (state.Tokens < 1.0)
                decision = new LimitDecision { Allowed = false, RetryAfterSeconds = Math.Max(1, (int)Math.Ceiling((1.0 - state.Tokens) / perSecond)), Reason = "too many requests from this address" };
            else
            {
                state.Tokens -= 1.0;
                state.InFlight++;
                decision = new LimitDecision { Allowed = true, Lease = new Lease(state) };
            }
        }
        if ((Interlocked.Increment(ref _calls) & 0x3FF) == 0 && _states.Count > _maxEntries) Prune(now);
        return decision;
    }

    /// <summary>Forgets the addresses that have been quiet for a while and hold nothing (their bucket would be full again anyway).</summary>
    private void Prune(long now)
    {
        long idle = TimeSpan.FromMinutes(10).Ticks;
        foreach (var kv in _states)
            lock (kv.Value.Gate)
                if (kv.Value.InFlight == 0 && now - kv.Value.LastUsed > idle) _states.TryRemove(kv.Key, out _);
    }
}
