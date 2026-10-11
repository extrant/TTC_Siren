using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Server;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 8A, goal section B4: the native C# <c>/ai_move</c> answers every request of two corpora (the 360-request Phase 7 corpus and 600
/// fresh seeded requests, which together include the error forms) exactly like the Python server on its production path (the C# engine
/// on), given the same seed for the unknown-card sampling: same HTTP status, and a response equal field for field, numbers bit for bit.
/// Excluded: the timing fields (they measure the wall clock) and the worker number; the new additive fields (<c>solver_used</c>,
/// <c>solver_note</c>, ...) may exist in the C# response only. The excluded paths are listed in <see cref="IsTiming"/>.
/// </summary>
public class AiMoveParityTests
{
    public const int Seed = 20261011;
    public const int Count = 600;
    public const int TtEntries = 262144; // equal to the table size of the Python reference's engine process: node counts depend on it

    internal static string EnsureGolden() =>
        Golden8.Ensure(Path.Combine(Golden8.GoldenDir, $"p8_e2e_g1_s{Seed}_n{Count}.jsonl.gz"), "e2e", "tools/phase8a/py_e2e_trace.py",
            "--out", "{0}", "--seed", Seed.ToString(), "--count", Count.ToString());

    /// <summary>Fields that measure time or name a worker. Everything else is compared.</summary>
    internal static bool IsTiming(string path) =>
        path is "engine_info.timings" or "engine_info.worker"
            or "search_progress[].time_elapsed" or "search_progress[].time_remaining" or "search_progress[].nodes_per_second"
            or "search_summary.total_time_seconds" or "search_summary.search_efficiency";

    internal static readonly string[] AdditiveKeys = { "solver_used", "solver_note", "explanation", "game_id" };

    internal static AiMoveService CreateService(int workers = 2, int ttEntries = TtEntries) => CreateService(out _, workers, ttEntries);

    /// <summary>A service whose worker pool is returned too; <paramref name="time"/> is the clock the pool measures waiting with (a logical clock freezes the aging).</summary>
    internal static AiMoveService CreateService(out WorkerPool pool, int workers = 2, int ttEntries = TtEntries, TimeProvider? time = null)
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var catalog = new CardCatalog(db);
        pool = new WorkerPool(workers, ttEntries, 1 << 15, 256, 60_000, time: time);
        return new AiMoveService(catalog, new SolveService(db), pool, null, ttEntries);
    }

    [Fact]
    public async Task Native_ai_move_equals_the_Python_server_on_960_requests()
    {
        string path = EnsureGolden();
        var service = CreateService();
        int records = 0, answered = 0, errors = 0, leaves = 0, ignored = 0, robust = 0, overridden = 0, withProgress = 0, mc = 0;
        var statuses = new SortedDictionary<int, int>();
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                string src = rec.GetProperty("src").GetString()!;
                int index = rec.GetProperty("i").GetInt32();
                ulong seed = rec.GetProperty("seed").GetUInt64();
                string where = $"{src} request #{index} (seed {seed}) tags {rec.GetProperty("tags")}";

                var outcome = await service.HandleAsync(rec.GetProperty("request"), seed, CancellationToken.None);
                int wantStatus = rec.GetProperty("status").GetInt32();
                Assert.True(wantStatus == outcome.Status, $"{where}: HTTP status python {wantStatus} csharp {outcome.Status}: {PyJson.Serialize(outcome.Body)}");

                using var actual = JsonDocument.Parse(PyJson.Serialize(outcome.Body));
                var compare = new JsonCompare(IsTiming, AdditiveKeys);
                compare.Compare(rec.GetProperty("response"), actual.RootElement);
                Assert.True(compare.Differences.Count == 0, $"{where}:\n  " + string.Join("\n  ", compare.Differences.Take(12)));

                records++;
                leaves += compare.Leaves;
                ignored += compare.Ignored;
                statuses[wantStatus] = statuses.GetValueOrDefault(wantStatus) + 1;
                if (wantStatus == 200 && rec.GetProperty("response").TryGetProperty("pos", out _))
                {
                    answered++;
                    var info = rec.GetProperty("response").GetProperty("engine_info");
                    if (info.GetProperty("robust").GetBoolean()) robust++;
                    if (info.GetProperty("overridden").GetBoolean()) overridden++;
                    if (rec.GetProperty("response").TryGetProperty("search_progress", out _)) withProgress++;
                }
                else if (wantStatus != 200) errors++;
                if (rec.GetProperty("request").TryGetProperty("solver", out var solver) && solver.ValueKind == JsonValueKind.String && solver.GetString() == "monte_carlo")
                {
                    if (outcome.Status == 200)
                    {
                        mc++;
                        Assert.Equal("minimax", outcome.Body["solver_used"]);
                        Assert.Contains("蒙特卡洛", (string)outcome.Body["solver_note"]!);
                    }
                }
            }
        }
        Assert.Equal(960, records);
        Assert.True(answered > 700 && robust > 150 && overridden > 20 && withProgress > 30 && errors > 25, $"coverage: answered {answered}, robust {robust}, overridden {overridden}, progress {withProgress}, errors {errors}");
        Console.WriteLine($"ai_move parity: {records} requests equal ({answered} moves, {errors} error responses, {robust} robust endgames of which {overridden} overridden, " +
                          $"{withProgress} with search progress, {mc} Monte Carlo requests answered by minimax); {leaves} values compared bit for bit, {ignored} timing values excluded; statuses {string.Join(", ", statuses.Select(p => $"{p.Key}:{p.Value}"))}");
    }
}
