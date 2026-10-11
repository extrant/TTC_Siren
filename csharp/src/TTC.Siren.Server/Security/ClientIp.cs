using System.Net;
using System.Net.Sockets;
using Microsoft.AspNetCore.Http;

namespace TTC.Siren.Server.Security;

/// <summary>Which peers may tell the server who the real client is (<c>Server:Security:TrustedProxies</c>). Everything is off by default.</summary>
public sealed class TrustedProxyOptions
{
    /// <summary>Trust Cloudflare's published ranges (the file named by <see cref="CidrFile"/>).</summary>
    public bool UseCloudflareRanges { get; set; }

    /// <summary>A file with one CIDR per line (<c>#</c> starts a comment). Relative paths are relative to the application folder. Refresh it with <c>tools/refresh_cloudflare_ips.py</c>.</summary>
    public string CidrFile { get; set; } = "cloudflare-ips.txt";

    /// <summary>Further trusted proxies, for example a reverse proxy on a private address: CIDRs or plain addresses.</summary>
    public List<string> Cidrs { get; set; } = new();

    /// <summary>
    /// Trust the loopback addresses, for a <c>cloudflared</c> tunnel or a reverse proxy that runs on this machine. Requests of the operator on this machine are
    /// not affected: they carry no proxy header, and the operator endpoints refuse every request that does.
    /// </summary>
    public bool TrustLoopback { get; set; }

    /// <summary>The header that carries the client address. Only a trusted peer is believed.</summary>
    public string ClientIpHeader { get; set; } = "CF-Connecting-IP";

    public bool AnyTrusted => UseCloudflareRanges || TrustLoopback || Cidrs.Count > 0;

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(ClientIpHeader)) throw new ArgumentException("TrustedProxies:ClientIpHeader must name a header");
        foreach (string c in Cidrs)
            if (!Cidr.TryParse(c, out _)) throw new ArgumentException($"TrustedProxies:Cidrs: '{c}' is not an address or a CIDR");
    }
}

/// <summary>An IPv4 or IPv6 network in CIDR notation.</summary>
public readonly struct Cidr
{
    private readonly byte[] _network;
    private readonly int _prefix;

    private Cidr(byte[] network, int prefix)
    {
        _network = network;
        _prefix = prefix;
    }

    public static bool TryParse(string text, out Cidr cidr)
    {
        cidr = default;
        text = text.Trim();
        int slash = text.IndexOf('/');
        string address = slash < 0 ? text : text[..slash];
        if (!IPAddress.TryParse(address, out var ip)) return false;
        if (ip.IsIPv4MappedToIPv6) ip = ip.MapToIPv4();
        byte[] bytes = ip.GetAddressBytes();
        int prefix = bytes.Length * 8;
        if (slash >= 0 && (!int.TryParse(text[(slash + 1)..], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out prefix) || prefix < 0 || prefix > bytes.Length * 8))
            return false;
        // keep only the network bits: a CIDR with host bits set still matches the network it names
        for (int i = 0; i < bytes.Length; i++)
        {
            int bitsInByte = Math.Clamp(prefix - i * 8, 0, 8);
            bytes[i] &= (byte)(bitsInByte == 0 ? 0 : 0xFF << (8 - bitsInByte));
        }
        cidr = new Cidr(bytes, prefix);
        return true;
    }

    public bool Contains(IPAddress ip)
    {
        if (ip.IsIPv4MappedToIPv6) ip = ip.MapToIPv4();
        byte[] bytes = ip.GetAddressBytes();
        if (bytes.Length != _network.Length) return false;
        int full = _prefix / 8;
        for (int i = 0; i < full; i++)
            if (bytes[i] != _network[i]) return false;
        int rest = _prefix % 8;
        if (rest == 0) return true;
        int mask = 0xFF << (8 - rest) & 0xFF;
        return (bytes[full] & mask) == _network[full];
    }
}

/// <summary>
/// Who is the client? Behind Cloudflare (or another proxy) the TCP peer is the proxy, and the client address comes in a header. A header can be written
/// by anybody, so it is believed ONLY when the peer is a configured trusted proxy; from every other peer the header is ignored and the peer itself is the
/// client. The result is what rate limits, login throttling and the audit log use, so a client cannot choose its own identity by sending a header.
/// </summary>
public sealed class ClientIpResolver
{
    /// <summary>Headers that mean "a proxy handled this request". The operator endpoints refuse a request that carries any of them.</summary>
    public static readonly string[] ProxyHeaders =
    {
        "CF-Connecting-IP", "CF-Connecting-IPv6", "CF-Ray", "CF-IPCountry", "CF-Visitor", "CF-Worker", "True-Client-IP",
        "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Forwarded-Port", "Forwarded", "X-Real-IP", "Via",
    };

    private readonly TrustedProxyOptions _options;
    private readonly List<Cidr> _trusted = new();

    public ClientIpResolver(TrustedProxyOptions options, string applicationDirectory)
    {
        _options = options;
        foreach (string c in options.Cidrs) _trusted.Add(Cidr.TryParse(c, out var cidr) ? cidr : throw new ArgumentException($"'{c}' is not an address or a CIDR"));
        if (options.UseCloudflareRanges)
        {
            string path = Path.IsPathRooted(options.CidrFile) ? options.CidrFile : Path.Combine(applicationDirectory, options.CidrFile);
            if (!File.Exists(path)) throw new FileNotFoundException($"TrustedProxies:UseCloudflareRanges is on, but the range file '{path}' does not exist (tools/refresh_cloudflare_ips.py creates it)");
            int count = 0;
            foreach (string raw in File.ReadAllLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '#') continue;
                if (!Cidr.TryParse(line, out var cidr)) throw new InvalidDataException($"'{path}': '{line}' is not a CIDR");
                _trusted.Add(cidr);
                count++;
            }
            if (count == 0) throw new InvalidDataException($"'{path}' holds no ranges");
        }
        RangeCount = _trusted.Count;
    }

    public int RangeCount { get; }

    public bool IsTrustedPeer(IPAddress? peer)
    {
        if (peer is null) return false;
        if (peer.IsIPv4MappedToIPv6) peer = peer.MapToIPv4();
        if (_options.TrustLoopback && IPAddress.IsLoopback(peer)) return true;
        foreach (var c in _trusted)
            if (c.Contains(peer)) return true;
        return false;
    }

    /// <summary>The client address of a request: the header's address if the peer is trusted and the header holds one, otherwise the peer.</summary>
    public IPAddress Resolve(HttpContext context)
    {
        var peer = context.Connection.RemoteIpAddress ?? IPAddress.Loopback;
        if (peer.IsIPv4MappedToIPv6) peer = peer.MapToIPv4();
        if (!IsTrustedPeer(peer)) return peer;
        if (context.Request.Headers.TryGetValue(_options.ClientIpHeader, out var header))
        {
            // a header sent more than once, or a list, is not a single address: not believed
            string text = header.Count == 1 ? header[0]!.Trim() : "";
            if (IPAddress.TryParse(text, out var client) && client.AddressFamily is AddressFamily.InterNetwork or AddressFamily.InterNetworkV6)
                return client.IsIPv4MappedToIPv6 ? client.MapToIPv4() : client;
        }
        return peer;
    }

    /// <summary>True when the request carries any header that only a proxy sets.</summary>
    public static bool CarriesProxyHeaders(HttpRequest request)
    {
        foreach (string name in ProxyHeaders)
            if (request.Headers.ContainsKey(name)) return true;
        return false;
    }
}
