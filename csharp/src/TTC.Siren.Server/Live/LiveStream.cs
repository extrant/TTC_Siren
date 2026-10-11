using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using TTC.Siren.Compat;

namespace TTC.Siren.Server.Live;

/// <summary>The Server-Sent Events response of <c>GET /v1/live</c>.</summary>
public static class LiveStream
{
    /// <summary>The default time between heartbeat comments (see <see cref="LiveOptions.HeartbeatSeconds"/>).</summary>
    public static readonly TimeSpan Heartbeat = TimeSpan.FromSeconds(20);

    /// <summary>The id the client saw last: the <c>Last-Event-ID</c> header that a browser's EventSource sends when it reconnects, or <c>?last_event_id=</c> for clients that cannot set headers.</summary>
    public static long? LastEventId(HttpRequest request)
    {
        string? text = request.Headers["Last-Event-ID"].ToString();
        if (string.IsNullOrEmpty(text)) text = request.Query["last_event_id"].ToString();
        return long.TryParse(text, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out long id) ? id : null;
    }

    public static async Task RunAsync(HttpContext ctx, LiveHub hub, string keyId, string? gameFilter, TimeSpan? heartbeat = null, long? lastEventId = null, string? language = null)
    {
        var beat = heartbeat ?? Heartbeat;
        var response = ctx.Response;
        response.StatusCode = 200;
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-store, no-cache";
        response.Headers["X-Accel-Buffering"] = "no"; // a reverse proxy must not buffer the stream
        ctx.Features.Get<IHttpResponseBodyFeature>()?.DisableBuffering();

        using var subscription = hub.Subscribe(keyId, gameFilter, lastEventId: lastEventId);
        var ct = ctx.RequestAborted;
        try
        {
            // retry: how long a browser waits before it reconnects (and then sends Last-Event-ID)
            await WriteAsync(response, "retry: 3000\n: connected\n\n", ct).ConfigureAwait(false);
            if (subscription.Resume.Kind is "gap" or "reset")
            {
                // some events are gone (the replay buffer is bounded), or the id is not from this run of the server: tell the client to read the state again
                await WriteAsync(response, "event: resync\ndata: " + PyJson.Serialize(new Dictionary<string, object?>
                {
                    ["type"] = "resync", ["reason"] = subscription.Resume.Kind == "gap" ? "events_lost" : "stream_restarted",
                    ["missed"] = subscription.Resume.Kind == "gap" ? subscription.Resume.Missed : null, ["replayed"] = subscription.Resume.Replayed,
                }) + "\n\n", ct).ConfigureAwait(false);
            }
            while (!ct.IsCancellationRequested)
            {
                long dropped = subscription.TakeDropped();
                if (dropped > 0)
                    await WriteAsync(response, "event: lagged\ndata: " + PyJson.Serialize(new Dictionary<string, object?> { ["type"] = "lagged", ["dropped"] = dropped }) + "\n\n", ct).ConfigureAwait(false);

                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
                timeout.CancelAfter(beat);
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
                    // ?lang=: the text of the explanation in the language the client reads (the events carry every language)
                    string data = language is null ? e.Json : ExplanationLanguage.RewriteEvent(e.Json, e.Type, language);
                    await WriteAsync(response, $"id: {e.Id}\nevent: {e.Type}\ndata: {data}\n\n", ct).ConfigureAwait(false);
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
