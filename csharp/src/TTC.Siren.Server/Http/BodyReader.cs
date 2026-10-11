using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using TTC.Siren.Compat;

namespace TTC.Siren.Server;

/// <summary>Keys of <c>HttpContext.Items</c> that the middleware sets.</summary>
public static class HttpItems
{
    /// <summary>The client address after the trusted-proxy rule (an <c>IPAddress</c>).</summary>
    public const string ClientIp = "ttc.clientIp";
}

/// <summary>
/// Reads a request body up to a limit. A body over the limit is answered with 413 in the usual error form (review L8-3).
/// A body that is only a little too large is read to its end and thrown away first (up to <see cref="DrainBudget"/>): a server that answers while the client is
/// still sending, and then closes the connection, makes the client see a reset instead of the answer. A body of more than limit + budget is refused unread.
/// </summary>
internal static class BodyReader
{
    /// <summary>How much more than the limit is read and discarded before the 413 is sent.</summary>
    public const int DrainBudget = 8 << 20;

    /// <summary>What Kestrel is told to accept per request: every endpoint enforces its own, smaller limit here.</summary>
    public const int HardLimit = 16 << 20;

    public static IResult TooLarge(int limit) => new JsonTextResult(413, PyJson.Serialize(new Dictionary<string, object?> { ["error"] = $"Request body too large (limit {limit} bytes)" }));

    /// <summary>The body, or the 413 answer.</summary>
    public static async Task<(byte[]? Bytes, IResult? Failure)> ReadAsync(HttpRequest request, int limit, CancellationToken ct)
    {
        if (request.ContentLength is { } declared && declared > limit + (long)DrainBudget) return (null, TooLarge(limit));
        if (request.HttpContext.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } feature) feature.MaxRequestBodySize = limit + (long)DrainBudget;
        try
        {
            using var ms = new MemoryStream();
            var buffer = new byte[64 * 1024];
            long total = 0;
            int read;
            while ((read = await request.Body.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
            {
                total += read;
                if (total <= limit) ms.Write(buffer, 0, read);
            }
            return total > limit ? (null, TooLarge(limit)) : (ms.ToArray(), null);
        }
        catch (BadHttpRequestException e) when (e.StatusCode == 413)
        {
            return (null, TooLarge(limit));
        }
    }
}
