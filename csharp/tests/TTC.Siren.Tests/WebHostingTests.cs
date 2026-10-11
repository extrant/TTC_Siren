using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TTC.Siren.Explain;
using TTC.Siren.Server;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>The server hosts the web UI (decision A-29) as a single-page application under a strict Content-Security-Policy; the API never answers with the page.</summary>
public sealed class WebHostingTests : ServerTestBase
{
    public WebHostingTests(ITestOutputHelper output) : base(output)
    {
    }

    // ============================================================================================================================ hosting the web UI

    private string MakeSite()
    {
        string site = Path.Combine(_dir, "site-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(site, "assets"));
        File.WriteAllText(Path.Combine(site, "index.html"), "<!doctype html><html><head><title>spa</title></head><body><div id=\"root\"></div></body></html>");
        File.WriteAllText(Path.Combine(site, "assets", "app.12345678.js"), "export {}");
        return site;
    }

    [Fact]
    public async Task The_page_is_served_for_every_path_of_the_UI_under_a_strict_policy_and_never_for_the_API()
    {
        var host = await StartAsync(o => o.StaticDirectory = MakeSite());
        using var http = new HttpClient();
        const string csp = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
        Assert.Equal(csp, ServerHost.WebCsp);
        Assert.DoesNotContain("unsafe-inline", ServerHost.WebCsp);
        Assert.DoesNotContain("unsafe-eval", ServerHost.WebCsp);

        foreach (string path in new[] { "/", "/index.html", "/live", "/games", "/games/g-20261011-ab12", "/login", "/settings", "/some/deep/route" })
        {
            using var r = await http.GetAsync(host.BaseUrl + path);
            string body = await r.Content.ReadAsStringAsync();
            Assert.True(r.StatusCode == System.Net.HttpStatusCode.OK, $"{path}: {(int)r.StatusCode}");
            Assert.Contains("id=\"root\"", body);
            Assert.Equal(csp, r.Headers.GetValues("Content-Security-Policy").Single());
            Assert.Equal("no-cache", r.Headers.CacheControl?.ToString());
            Assert.Equal("nosniff", r.Headers.GetValues("X-Content-Type-Options").Single());
            Assert.Equal("DENY", r.Headers.GetValues("X-Frame-Options").Single());
        }
        using (var asset = await http.GetAsync(host.BaseUrl + "/assets/app.12345678.js"))
        {
            Assert.Equal(200, (int)asset.StatusCode);
            Assert.Equal("public, max-age=31536000, immutable", asset.Headers.CacheControl?.ToString());
            Assert.Equal(csp, asset.Headers.GetValues("Content-Security-Policy").Single());
        }
        foreach (string path in new[] { "/missing.js", "/assets/nothing.css", "/v1/nope", "/v1", "/api/unknown", "/admin/nope", "/ai_move/extra" })
        {
            using var r = await http.GetAsync(host.BaseUrl + path);
            Assert.True(r.StatusCode == System.Net.HttpStatusCode.NotFound || r.StatusCode == System.Net.HttpStatusCode.MethodNotAllowed, $"{path}: {(int)r.StatusCode}");
            Assert.DoesNotContain("<html", await r.Content.ReadAsStringAsync());
        }
        using (var post = await http.PostAsync(host.BaseUrl + "/live", new StringContent("{}")))
        {
            Assert.Equal(404, (int)post.StatusCode);
            Assert.DoesNotContain("<html", await post.Content.ReadAsStringAsync());
        }
        // the API is untouched by the fallback
        using (var health = await http.GetAsync(host.BaseUrl + "/health")) Assert.Equal("application/json", health.Content.Headers.ContentType?.MediaType);
        using (var games = await http.GetAsync(host.BaseUrl + "/v1/games")) Assert.Equal("application/json", games.Content.Headers.ContentType?.MediaType);
    }

    [Fact]
    public async Task Without_a_web_folder_the_root_is_the_placeholder_page()
    {
        var host = await StartAsync();
        using var http = new HttpClient();
        string html = await http.GetStringAsync(host.BaseUrl + "/");
        Assert.Contains("TTC Siren", html);
        using var live = await http.GetAsync(host.BaseUrl + "/live");
        Assert.Equal(404, (int)live.StatusCode);
    }
}
