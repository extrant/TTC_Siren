using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;

namespace TTC.Siren.Server.Proxy;

/// <summary>Settings of the proxy to the Python app (<c>Server:Proxy</c>).</summary>
public sealed class ProxyOptions
{
    public bool Enabled { get; set; } = true;

    /// <summary>The local Python app (waitress on loopback) that still serves the three <c>/api/*</c> endpoints until Phase 8C ports them.</summary>
    public string PythonUrl { get; set; } = "http://127.0.0.1:5001";

    /// <summary>Seconds to wait for the Python app (the draft analysis spends up to 20 s on its budget).</summary>
    public int TimeoutSeconds { get; set; } = 120;

    public int MaxBodyBytes { get; set; } = 4 << 20;

    public void Validate()
    {
        if (!Uri.TryCreate(PythonUrl, UriKind.Absolute, out var uri) || (uri.Scheme != "http" && uri.Scheme != "https")) throw new ArgumentException("Proxy:PythonUrl must be an http(s) URL");
        if (TimeoutSeconds < 1) throw new ArgumentException("Proxy:TimeoutSeconds must be at least 1");
        if (MaxBodyBytes < 1024) throw new ArgumentException("Proxy:MaxBodyBytes is too small");
    }
}

/// <summary>
/// Forwards <c>POST /api/recommend</c>, <c>/api/draft/analyze</c> and <c>/api/winrate</c> to the local Python app, after the caller's key was checked
/// and its limits applied. Only these three paths and only POST are forwarded; the body and its content type go through unchanged; nothing of the
/// caller's headers (above all not the API key) is passed on. A Python app that is down answers 502, one that is too slow 504.
/// </summary>
public sealed class PythonProxy
{
    public static readonly string[] Paths = { "/api/recommend", "/api/draft/analyze", "/api/winrate" };

    private readonly HttpClient _client;
    private readonly ProxyOptions _options;

    public PythonProxy(ProxyOptions options, HttpMessageHandler? handler = null)
    {
        options.Validate();
        _options = options;
        _client = handler is null ? new HttpClient() : new HttpClient(handler);
        _client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
    }

    public async Task<IResult> ForwardAsync(HttpRequest request, string path, CancellationToken ct)
    {
        request.HttpContext.Features.Get<IHttpMaxRequestBodySizeFeature>()?.MaxRequestBodySize = _options.MaxBodyBytes;
        byte[] body;
        try
        {
            using var ms = new MemoryStream();
            await request.Body.CopyToAsync(ms, ct).ConfigureAwait(false);
            body = ms.ToArray();
        }
        catch (Microsoft.AspNetCore.Http.BadHttpRequestException)
        {
            return Error(413, "request body too large");
        }

        using var upstream = new HttpRequestMessage(HttpMethod.Post, new Uri(new Uri(_options.PythonUrl.TrimEnd('/') + "/"), path.TrimStart('/'))) { Content = new ByteArrayContent(body) };
        string contentType = string.IsNullOrEmpty(request.ContentType) ? "application/json" : request.ContentType;
        upstream.Content.Headers.TryAddWithoutValidation("Content-Type", contentType);
        try
        {
            using var response = await _client.SendAsync(upstream, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false);
            var bytes = await response.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
            return new RawResult((int)response.StatusCode, response.Content.Headers.ContentType?.ToString() ?? "application/json", bytes);
        }
        catch (TaskCanceledException) when (!ct.IsCancellationRequested)
        {
            return Error(504, "the Python app did not answer in time");
        }
        catch (HttpRequestException)
        {
            return Error(502, "the Python app is not reachable");
        }
    }

    private static IResult Error(int status, string message) =>
        new RawResult(status, "application/json", System.Text.Encoding.UTF8.GetBytes("{\"error\":\"" + message + "\"}\n"));

    private sealed class RawResult : IResult
    {
        private readonly int _status;
        private readonly string _contentType;
        private readonly byte[] _body;

        public RawResult(int status, string contentType, byte[] body)
        {
            _status = status;
            _contentType = contentType;
            _body = body;
        }

        public Task ExecuteAsync(HttpContext httpContext)
        {
            httpContext.Response.StatusCode = _status;
            httpContext.Response.ContentType = _contentType;
            httpContext.Response.ContentLength = _body.Length;
            return httpContext.Response.Body.WriteAsync(_body).AsTask();
        }
    }
}
