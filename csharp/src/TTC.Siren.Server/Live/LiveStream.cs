using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using TTC.Siren.Compat;

namespace TTC.Siren.Server.Live;

/// <summary>The Server-Sent Events response of <c>GET /v1/live</c>.</summary>
public static class LiveStream
{
    public static readonly TimeSpan Heartbeat = TimeSpan.FromSeconds(15);

    public static async Task RunAsync(HttpContext ctx, LiveHub hub, string keyId, string? gameFilter)
    {
        var response = ctx.Response;
        response.StatusCode = 200;
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers["X-Accel-Buffering"] = "no"; // a reverse proxy must not buffer the stream
        ctx.Features.Get<IHttpResponseBodyFeature>()?.DisableBuffering();

        using var subscription = hub.Subscribe(keyId, gameFilter);
        var ct = ctx.RequestAborted;
        try
        {
            await WriteAsync(response, ": connected\n\n", ct).ConfigureAwait(false);
            while (!ct.IsCancellationRequested)
            {
                long dropped = subscription.TakeDropped();
                if (dropped > 0)
                    await WriteAsync(response, "event: lagged\ndata: " + PyJson.Serialize(new Dictionary<string, object?> { ["type"] = "lagged", ["dropped"] = dropped }) + "\n\n", ct).ConfigureAwait(false);

                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
                timeout.CancelAfter(Heartbeat);
                try
                {
                    if (!await subscription.Reader.WaitToReadAsync(timeout.Token).ConfigureAwait(false)) break;
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    await WriteAsync(response, ": heartbeat\n\n", ct).ConfigureAwait(false);
                    continue;
                }
                while (subscription.Reader.TryRead(out var e))
                {
                    long lost = subscription.TakeDropped();
                    if (lost > 0)
                        await WriteAsync(response, "event: lagged\ndata: " + PyJson.Serialize(new Dictionary<string, object?> { ["type"] = "lagged", ["dropped"] = lost }) + "\n\n", ct).ConfigureAwait(false);
                    await WriteAsync(response, $"id: {e.Id}\nevent: {e.Type}\ndata: {e.Json}\n\n", ct).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // the client went away
        }
    }

    private static async Task WriteAsync(HttpResponse response, string text, CancellationToken ct)
    {
        await response.Body.WriteAsync(Encoding.UTF8.GetBytes(text), ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }
}
