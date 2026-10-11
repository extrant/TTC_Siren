using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Server;
using TTC.Siren.Server.GameLog;
using TTC.Siren.Server.Security;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Review L8-3: a request body over the limit is answered with 413 in the usual error form (it used to be a 400).
/// </summary>
public sealed class BodyLimitTests : ServerTestBase
{
    public BodyLimitTests(ITestOutputHelper output) : base(output)
    {
    }

    [Fact]
    public async Task A_body_over_the_limit_is_answered_with_413_in_the_usual_error_form()
    {
        var host = await StartAsync();
        using var http = new HttpClient();
        // a VALID json body of 2 MiB (the limit is 1 MiB)
        string padding = new('x', 2 << 20);
        string big = "{\"board\":[],\"myHand\":[],\"oppHand\":[],\"pad\":\"" + padding + "\"}";
        using (var r = await http.SendAsync(Post(host.BaseUrl, big)))
        {
            string text = await r.Content.ReadAsStringAsync();
            Assert.Equal(413, (int)r.StatusCode);
            Assert.Equal("{\"error\":\"Request body too large (limit 1048576 bytes)\"}\n", text);
            Assert.Equal("application/json", r.Content.Headers.ContentType?.MediaType);
        }

        // the same without a Content-Length (chunked)
        var chunked = new HttpRequestMessage(HttpMethod.Post, host.BaseUrl + "/ai_move") { Content = new StreamContent(new MemoryStream(Encoding.UTF8.GetBytes(big))) };
        chunked.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        chunked.Headers.TransferEncodingChunked = true;
        using (var r = await http.SendAsync(chunked))
        {
            Assert.Equal(413, (int)r.StatusCode);
            Assert.Contains("too large", await r.Content.ReadAsStringAsync());
        }

        // a body just under the limit is read (and refused for what it is: not a position)
        string under = "{\"pad\":\"" + new string('x', (1 << 20) - 64) + "\"}";
        using (var r = await http.SendAsync(Post(host.BaseUrl, under))) Assert.Equal(400, (int)r.StatusCode);
        // a malformed body is still 400
        using (var r = await http.SendAsync(Post(host.BaseUrl, "{"))) Assert.Equal(400, (int)r.StatusCode);
    }
}
