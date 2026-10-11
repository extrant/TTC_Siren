namespace TTC.Siren.Server.Security;

/// <summary>HTTPS settings (<c>Server:Security:Https</c>): Kestrel terminates TLS with a PFX certificate.</summary>
public sealed class HttpsOptions
{
    public bool Enabled { get; set; }

    /// <summary>The port of the HTTPS listener (443 in production).</summary>
    public int Port { get; set; } = 443;

    public string? PfxPath { get; set; }

    /// <summary>The password of the PFX file. Prefer the environment variable named by <see cref="PfxPasswordEnvironmentVariable"/>: this value would sit in a config file.</summary>
    public string? PfxPassword { get; set; }

    public string PfxPasswordEnvironmentVariable { get; set; } = "TTC_PFX_PASSWORD";

    /// <summary>Send <c>Strict-Transport-Security</c> on HTTPS responses.</summary>
    public bool Hsts { get; set; } = true;

    public int HstsMaxAgeSeconds { get; set; } = 31_536_000;
}

/// <summary>Security settings of the server (<c>Server:Security</c>).</summary>
public sealed class SecurityOptions
{
    /// <summary>Every request needs an API key (the default). Off only for development on a loopback address.</summary>
    public bool RequireKey { get; set; } = true;

    /// <summary>With <see cref="RequireKey"/>: requests from the loopback address are admitted without a key (as the key "local-dev"). Off by default; for development only.</summary>
    public bool AllowLoopbackWithoutKey { get; set; }

    /// <summary>The JSON file with the key hashes (managed by the <c>keys</c> commands of the server binary).</summary>
    public string KeyStoreFile { get; set; } = Path.Combine(AppContext.BaseDirectory, "keys.json");

    public int DefaultRatePerMinute { get; set; } = 120;
    public int DefaultBurst { get; set; } = 20;
    public int DefaultMaxInFlight { get; set; } = 4;

    /// <summary>Open <c>/v1/live</c> streams per key.</summary>
    public int MaxLiveStreamsPerKey { get; set; } = 4;

    /// <summary>Origins that may call the API from a browser (the 8B web UI). Exact matches only; no wildcard.</summary>
    public List<string> CorsOrigins { get; set; } = new();

    public HttpsOptions Https { get; set; } = new();

    /// <summary>Which peers may name the real client in a header (Cloudflare, a local tunnel). Nothing is trusted by default.</summary>
    public TrustedProxyOptions TrustedProxies { get; set; } = new();

    /// <summary>
    /// The server listens on a non-loopback address without HTTPS because a reverse proxy in front of it terminates TLS. Without this flag (or
    /// HTTPS) the server refuses to listen on a non-loopback address: keys would cross the network in clear text.
    /// </summary>
    public bool BehindTlsProxy { get; set; }

    public void Validate(string host, bool hostIsLoopback)
    {
        if (DefaultRatePerMinute < 1 || DefaultBurst < 1 || DefaultMaxInFlight < 1) throw new ArgumentException("Security default limits must be at least 1");
        if (MaxLiveStreamsPerKey < 1) throw new ArgumentException("MaxLiveStreamsPerKey must be at least 1");
        if (CorsOrigins.Any(o => o == "*" || !Uri.TryCreate(o, UriKind.Absolute, out _))) throw new ArgumentException("CorsOrigins must be exact origins such as https://app.example.org (no wildcard)");
        TrustedProxies.Validate();
        if (Https.Enabled)
        {
            if (string.IsNullOrWhiteSpace(Https.PfxPath)) throw new ArgumentException("Security:Https:PfxPath is required when HTTPS is enabled");
            if (Https.Port is < 0 or > 65535) throw new ArgumentException("Security:Https:Port out of range");
        }
        if (!hostIsLoopback && !Https.Enabled && !BehindTlsProxy)
            throw new ArgumentException($"Refusing to listen on {host} without HTTPS: enable Security:Https, or set Security:BehindTlsProxy=true if a reverse proxy terminates TLS in front of this server.");
        if (AllowLoopbackWithoutKey && !hostIsLoopback) throw new ArgumentException("AllowLoopbackWithoutKey is for development and needs a loopback Host");
    }
}
