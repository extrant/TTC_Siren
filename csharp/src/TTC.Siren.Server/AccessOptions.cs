using TTC.Siren.Engine;

namespace TTC.Siren.Server;

/// <summary>
/// Access classes and the capacity they get (<c>Server:Access</c>, decision A-27). The game plugin (an API key) has the highest priority and a share of the workers
/// that nothing else can take; a browser logged in with a key comes next; a visitor who is not logged in comes last, may hold only a small share of the workers at a time,
/// has its own short queue, and gets a smaller search (depth and time) and a rate limit per real client address.
/// </summary>
public sealed class AccessOptions
{
    /// <summary>The share of the workers that only the plugin may use (rounded down; none when there is a single worker).</summary>
    public double PluginReservedShare { get; set; } = 0.25;

    /// <summary>The most workers anonymous requests may hold at once, as a share of all workers (rounded down, at least one).</summary>
    public double AnonymousMaxShare { get; set; } = 0.25;

    /// <summary>The most anonymous requests that may wait for a worker in all; the next one is answered 503 busy with Retry-After.</summary>
    public int AnonymousQueueLimit { get; set; } = 16;

    /// <summary>How long an anonymous request waits for a worker before it is answered busy.</summary>
    public double AnonymousQueueWaitSeconds { get; set; } = 10;

    /// <summary>The deepest search an anonymous request may ask for (its <c>max_depth</c> is lowered to this).</summary>
    public int AnonymousMaxDepth { get; set; } = 8;

    /// <summary>The longest search an anonymous request may ask for, in seconds.</summary>
    public double AnonymousMaxSeconds { get; set; } = 3;

    /// <summary>Requests per minute, per real client address, that an anonymous visitor may make (the bucket is refilled at this rate).</summary>
    public int AnonymousRatePerMinute { get; set; } = 30;

    /// <summary>Requests an anonymous address may make at once before it has to wait for the refill.</summary>
    public int AnonymousBurst { get; set; } = 10;

    /// <summary>Requests of one anonymous address that may be in flight at the same time.</summary>
    public int AnonymousMaxInFlight { get; set; } = 2;

    public void Validate()
    {
        if (PluginReservedShare is < 0 or >= 1) throw new ArgumentException("Access:PluginReservedShare must be at least 0 and below 1");
        if (AnonymousMaxShare is <= 0 or > 1) throw new ArgumentException("Access:AnonymousMaxShare must be above 0 and at most 1");
        if (AnonymousQueueLimit < 0 || AnonymousQueueWaitSeconds < 0) throw new ArgumentException("Access:AnonymousQueueLimit and AnonymousQueueWaitSeconds must not be negative");
        if (AnonymousMaxDepth < 1) throw new ArgumentException("Access:AnonymousMaxDepth must be at least 1");
        if (AnonymousMaxSeconds <= 0) throw new ArgumentException("Access:AnonymousMaxSeconds must be above 0");
        if (AnonymousRatePerMinute < 1 || AnonymousBurst < 1 || AnonymousMaxInFlight < 1) throw new ArgumentException("Access:AnonymousRatePerMinute, AnonymousBurst and AnonymousMaxInFlight must be at least 1");
    }

    /// <summary>The numbers for the worker pool.</summary>
    public CapacityOptions CapacityFor(int workers) =>
        CapacityOptions.FromShares(workers, PluginReservedShare, AnonymousMaxShare, AnonymousQueueLimit, (int)Math.Round(AnonymousQueueWaitSeconds * 1000));

    /// <summary>The caps on the search of an anonymous request.</summary>
    public SearchCaps AnonymousCaps => new(AnonymousMaxSeconds, AnonymousMaxDepth);
}

/// <summary>
/// The limits for a visitor who is not logged in (decision A-27): a token bucket and a number of requests in flight per REAL client address (after the trusted-proxy rule, so
/// a header cannot choose the address). 8B has no anonymous compute endpoint -- the pages are static -- but the gate is complete and tested: game coaching (8D) admits its
/// anonymous requests through it and then through the anonymous class of the worker pool and the search caps of <see cref="AccessOptions"/>.
/// </summary>
public sealed class AnonymousGate
{
    private readonly AccessOptions _options;
    private readonly Security.IpLimiter _limiter;

    public AnonymousGate(AccessOptions options, TimeProvider? time = null)
    {
        _options = options;
        _limiter = new Security.IpLimiter(time);
    }

    public Security.LimitDecision TryEnter(System.Net.IPAddress client) =>
        _limiter.TryEnter(client.ToString(), _options.AnonymousRatePerMinute, _options.AnonymousBurst, _options.AnonymousMaxInFlight);
}
