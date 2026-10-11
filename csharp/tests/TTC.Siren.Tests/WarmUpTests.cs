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
/// Review O8-3: the start-up warm-up runs the real request path on every worker, leaves no trace (no game log, no event) and changes no answer.
/// </summary>
public sealed class WarmUpTests : ServerTestBase
{
    public WarmUpTests(ITestOutputHelper output) : base(output)
    {
    }

    [Fact]
    public async Task The_warm_up_runs_the_request_path_on_every_worker_without_a_trace_and_changes_no_answer()
    {
        var cold = await StartAsync(o =>
        {
            o.Workers = 3;
            o.TestSeedHeader = true;
        }, sub: "warm-b");
        // a warmed server with the same switch for the comparison
        var warmedSeeded = await StartAsync(o =>
        {
            o.Workers = 3;
            o.TestSeedHeader = true;
        }, sub: "warm-c");
        var report = await warmedSeeded.WarmUpAsync();
        Assert.Equal(3, report.Workers);
        Assert.Equal(6, report.Searches); // an opening and an endgame on each worker
        Assert.True(report.NodesSearched > 0);
        Assert.Equal(3, warmedSeeded.Pool.Created);   // every worker has its tables
        Assert.Equal(0, cold.Pool.Created);

        // no game log, no live event
        Assert.Empty(warmedSeeded.GameLog.Store.List("local", 0, 50).Items);
        Assert.False(Directory.Exists(Path.Combine(_dir, "warm-c", "local")));

        // the answers are the same with and without the warm-up (fixed seed, fixed depth)
        string request = Opening.Replace("\"max_time\":5.5", "\"max_time\":1000000");
        using var http = new HttpClient();
        using var a = await http.SendAsync(Post(cold.BaseUrl, request.Replace("fixes-1", "w-1"), seed: 99));
        using var b = await http.SendAsync(Post(warmedSeeded.BaseUrl, request.Replace("fixes-1", "w-1"), seed: 99));
        using var da = JsonDocument.Parse(await a.Content.ReadAsStringAsync());
        using var db = JsonDocument.Parse(await b.Content.ReadAsStringAsync());
        // measured times differ between two runs; everything else -- the move, the scores, the facts and the texts -- must not
        var compare = new JsonCompare(p => AiMoveParityTests.IsTiming(p) || p.EndsWith("search_ms", StringComparison.Ordinal), AiMoveParityTests.AdditiveKeys);
        compare.Compare(da.RootElement, db.RootElement);
        Assert.True(compare.Differences.Count == 0, string.Join("; ", compare.Differences.Take(8)));
        Assert.True(compare.Leaves > 50);
    }
}
