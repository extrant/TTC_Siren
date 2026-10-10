using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Search;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 7 end-to-end comparison (goal section 4.4, Q-5 (b)). The Python side (tools/phase7/e2e_reference.py) recorded, for every
/// request of the 360-request corpus, the payload the Python server sends to the engine and what Python computes (its standard search
/// per depth, the same without a transposition table, the robust table on the request-order state). Here every payload is sent to
/// the engine and compared:
///  1. the engine's root score at every completed depth equals the C# oracle (ReferenceCompat, table off) bit for bit;
///  2. the engine's move at every depth lies in the exact tie set (computed here, in the harness);
///  3. Python's move at every depth is classified: identical to the engine's, a different move in the tie set, or outside the tie
///     set but explained by D-006 (Python's table has no bound types: its own score differs from the exact value and Python run without
///     the table finds a move in the tie set); anything else is unexplained and fails the test;
///  4. the robust layer: the candidate table equals Python's bit for bit (final, safety, robust, corner, move, order), the scenario
///     count equals, and the final move equals the override decision computed from Python's table and the engine's standard move.
/// </summary>
public sealed class EngineEndToEndTests : IClassFixture<EngineFixture>
{
    private readonly EngineFixture _fx;
    private readonly ITestOutputHelper _output;

    public EngineEndToEndTests(EngineFixture fx, ITestOutputHelper output)
    {
        _fx = fx;
        _output = output;
    }

    private static CardDb Db => SearchGolden.Db;

    private sealed class Counts
    {
        public int Records, RejectedHandEmpty, NoMove, FallbackPathRecords, RobustRecords, Levels;
        public int EngineScoreEqualsOracle, EngineMoveInTieSet;
        public int PyIdentical, PyOtherInTieSet, PyD006, PyUnexplained;
        public int PyScoreExact, PyScoreDiffersD006;
        public int RobustTablesIdentical, RobustMovesCompared, RobustScenariosEqual, Overridden, NotOverridden, SameObjectScenarios;

        public readonly SortedDictionary<int, int> TieSizes = new();
        public readonly List<string> Problems = new();
    }

    private static long Hex(string pythonHex) => SearchGolden.HexBits(pythonHex);

    private static (int Slot, int Row, int Col, int CardId)? ReadMove(JsonNode? n) =>
        n is null ? null : (n["slot"]!.GetValue<int>(), n["row"]!.GetValue<int>(), n["col"]!.GetValue<int>(), n["cardId"]!.GetValue<int>());

    private static (int, int, int, int)? ReadPyMove(JsonElement e) =>
        e.ValueKind == JsonValueKind.Null ? null : (e[0].GetInt32(), e[1].GetInt32(), e[2].GetInt32(), e[3].GetInt32());

    [Fact]
    public async Task Engine_against_the_oracle_and_against_python_over_the_whole_corpus()
    {
        string path = GoldenFiles.EnsureE2eReference();
        var counts = new Counts();

        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        using (var header = JsonDocument.Parse(reader.ReadLine()!))
            Assert.Equal(GoldenFiles.E2eGeneratorVersion, header.RootElement.GetProperty("header").GetProperty("generator").GetString());

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            using var doc = JsonDocument.Parse(line);
            await CompareRecord(doc.RootElement, counts);
        }

        string summary =
            $"records={counts.Records} (robust {counts.RobustRecords}, fallback-candidates path {counts.FallbackPathRecords}, no move {counts.NoMove}, rejected: side to move has no cards {counts.RejectedHandEmpty}); " +
            $"search levels compared={counts.Levels}: engine score == oracle {counts.EngineScoreEqualsOracle}, engine move in exact tie set {counts.EngineMoveInTieSet}; " +
            $"Python move per level: identical to the engine's {counts.PyIdentical}, different but in the tie set {counts.PyOtherInTieSet}, outside the tie set and explained by D-006 {counts.PyD006}, unexplained {counts.PyUnexplained}; " +
            $"Python score == exact value {counts.PyScoreExact}, differs (D-006) {counts.PyScoreDiffersD006}; " +
            $"robust: tables bit-identical {counts.RobustTablesIdentical}, candidates compared {counts.RobustMovesCompared}, scenario counts equal {counts.RobustScenariosEqual}, " +
            $"base state returned as the only scenario {counts.SameObjectScenarios}, final move = override decision: overridden {counts.Overridden}, standard kept {counts.NotOverridden}; " +
            $"exact tie-set size over all levels: " + string.Join(", ", counts.TieSizes.Select(kv => $"{kv.Key}:{kv.Value}"));
        _output.WriteLine(summary);
        File.WriteAllText(Path.Combine(GoldenFiles.GoldenDir, "e2e_counts.txt"), summary + Environment.NewLine);

        Assert.True(counts.Problems.Count == 0, string.Join("\n", counts.Problems.Take(25)) + $"\n({counts.Problems.Count} problems)");
        Assert.True(counts.Records >= 350, "the corpus produced too few records");
        Assert.True(counts.RobustRecords >= 100, $"only {counts.RobustRecords} robust records");
        Assert.True(counts.FallbackPathRecords >= 5, $"only {counts.FallbackPathRecords} fallback-candidates records");
        Assert.Equal(0, counts.PyUnexplained);
    }

    private async Task CompareRecord(JsonElement rec, Counts c)
    {
        int index = rec.GetProperty("i").GetInt32();
        int maxDepthSent = rec.GetProperty("max_depth").GetInt32();
        string label = $"request {index} (max_depth {maxDepthSent}, tags {string.Join(",", rec.GetProperty("tags").EnumerateArray().Select(t => t.GetString()))})";
        c.Records++;

        var payload = rec.GetProperty("payload");
        bool aggressive = payload.GetProperty("aggressive").GetBoolean();
        var rules = RuleParser.FromPythonList(payload.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
        var snapshot = StateSnapshot.FromJson(payload.GetProperty("state"), rules);
        bool hasRobust = payload.GetProperty("robust").ValueKind != JsonValueKind.Null;
        bool expectsFallback = rec.GetProperty("expects_fallback").GetBoolean();

        // 1. the payload exactly as the Python server sends it (and, when the engine asks for them, the fallback candidates)
        var (status, body) = await EngineJson.Post(_fx.Client, payload.GetRawText());
        if (expectsFallback)
        {
            c.FallbackPathRecords++;
            if (status != 200 || body["status"]!.GetValue<string>() != "needs_fallback_candidates")
            {
                c.Problems.Add($"{label}: Python found no legal candidates but the engine answered {status} {body}");
                return;
            }
            var second = rec.GetProperty("payload_with_candidates");
            if (second.ValueKind == JsonValueKind.Null)
            {
                c.Problems.Add($"{label}: no second payload recorded");
                return;
            }
            (status, body) = await EngineJson.Post(_fx.Client, second.GetRawText());
        }
        else if (status == 200 && body["status"]!.GetValue<string>() == "needs_fallback_candidates")
        {
            c.Problems.Add($"{label}: the engine asks for fallback candidates but Python's legal enumeration is not empty");
            return;
        }

        if (status == 400)
        {
            // legal only for a side to move without cards (Python then has no move either)
            if (snapshot.Hands[snapshot.Current].Count == 0 && rec.GetProperty("py").GetArrayLength() == 0)
            {
                c.RejectedHandEmpty++;
                return;
            }
            c.Problems.Add($"{label}: the engine rejected the request: {body}");
            return;
        }
        if (status != 200 || body["status"]!.GetValue<string>() != "ok")
        {
            c.Problems.Add($"{label}: HTTP {status}: {body}");
            return;
        }

        var depths = body["depths"]!.AsArray();
        var py = rec.GetProperty("py").EnumerateArray().ToList();
        var pyNoTable = rec.GetProperty("py_no_table").EnumerateArray().ToList();

        if (body["move"] is null)
        {
            // no move: Python must have none either
            if (py.Count > 0 && py[^1].GetProperty("move").ValueKind != JsonValueKind.Null)
                c.Problems.Add($"{label}: the engine found no move but Python did");
            c.NoMove++;
            return;
        }

        // 2. levels: oracle score, exact tie set, Python classification
        (int, int, int, int)? engineStandardLast = null;
        foreach (var level in depths)
        {
            int depth = level!["depth"]!.GetValue<int>();
            c.Levels++;
            var oracle = Oracle.Exact(GameState.FromSnapshot(snapshot), depth, aggressive);
            long oracleBits = BitConverter.DoubleToInt64Bits(oracle.Score);
            long engineBits = Convert.ToInt64(level["scoreBits"]!.GetValue<string>(), 16);
            if (engineBits == oracleBits) c.EngineScoreEqualsOracle++;
            else c.Problems.Add($"{label}: depth {depth}: engine score {level["scoreBits"]} != oracle {oracleBits:X16}");

            var g = GameState.FromSnapshot(snapshot);
            int cur = g.CurrentPlayer;
            var slotInst = Enumerable.Range(0, g.HandCount(cur)).Select(k => g.HandInstance(cur, k)).ToArray(); // before RootMoves (D-001)
            var tie = Oracle.RootMoves(g, depth, aggressive);
            int tieSize = Oracle.TieSize(tie, oracle.Score);
            c.TieSizes[tieSize] = c.TieSizes.GetValueOrDefault(tieSize) + 1;

            bool InTie((int Slot, int Row, int Col, int CardId)? m) =>
                m is not null && m.Value.Slot < slotInst.Length && Oracle.InTieSet(tie, oracle.Score, slotInst[m.Value.Slot], m.Value.Row * 3 + m.Value.Col);

            var engineMove = ReadMove(level["move"]);
            if (InTie(engineMove)) c.EngineMoveInTieSet++;
            else c.Problems.Add($"{label}: depth {depth}: engine move {engineMove} is not in the exact tie set");
            if (depth == depths[^1]!["depth"]!.GetValue<int>()) engineStandardLast = engineMove;

            if (depth <= py.Count)
            {
                var pyLevel = py[depth - 1];
                var pyMove = ReadPyMove(pyLevel.GetProperty("move"));
                long pyBits = Hex(pyLevel.GetProperty("score").GetString()!);
                bool pyExact = pyBits == oracleBits;
                if (pyExact) c.PyScoreExact++; else c.PyScoreDiffersD006++;

                if (pyMove is not null && engineMove is not null && pyMove.Value == engineMove.Value) c.PyIdentical++;
                else if (InTie(pyMove)) c.PyOtherInTieSet++;
                else
                {
                    var noTableMove = depth <= pyNoTable.Count ? ReadPyMove(pyNoTable[depth - 1].GetProperty("move")) : null;
                    long noTableBits = depth <= pyNoTable.Count ? Hex(pyNoTable[depth - 1].GetProperty("score").GetString()!) : 0;
                    if (!pyExact && InTie(noTableMove) && noTableBits == oracleBits) c.PyD006++;
                    else
                    {
                        c.PyUnexplained++;
                        c.Problems.Add($"{label}: depth {depth}: Python move {pyMove} (score {pyLevel.GetProperty("score")}) is outside the tie set and not explained by D-006 (without table: {noTableMove})");
                    }
                }
            }
        }

        // 3. the robust layer
        var robustRef = rec.GetProperty("robust");
        if (!hasRobust)
        {
            if (body["robust"] is not null) c.Problems.Add($"{label}: the engine ran the robust layer without being asked");
            return;
        }
        c.RobustRecords++;
        var robust = body["robust"]!;
        if (robustRef.ValueKind == JsonValueKind.Null)
        {
            c.Problems.Add($"{label}: Python has no robust reference");
            return;
        }

        if (robustRef.GetProperty("scenarios").GetInt32() == robust["scenarios"]!.GetValue<int>()) c.RobustScenariosEqual++;
        else c.Problems.Add($"{label}: scenarios Python {robustRef.GetProperty("scenarios")} engine {robust["scenarios"]}");
        if (robustRef.GetProperty("same_object").GetBoolean()) c.SameObjectScenarios++;

        var pyTable = robustRef.GetProperty("table").EnumerateArray().ToList();
        var engTable = robust["candidates"]!.AsArray();
        bool identical = pyTable.Count == engTable.Count;
        var items = new List<OverrideItem>();
        for (int i = 0; identical && i < pyTable.Count; i++)
        {
            var p = pyTable[i];
            var e = engTable[i]!;
            var pm = ReadPyMove(p.GetProperty("move"))!.Value;
            var em = ReadMove(e["move"])!.Value;
            c.RobustMovesCompared++;
            bool same = pm == em &&
                        Hex(p.GetProperty("final").GetString()!) == Convert.ToInt64(e["finalBits"]!.GetValue<string>(), 16) &&
                        Hex(p.GetProperty("safety").GetString()!) == Convert.ToInt64(e["safetyBits"]!.GetValue<string>(), 16) &&
                        Hex(p.GetProperty("robust").GetString()!) == Convert.ToInt64(e["robustBits"]!.GetValue<string>(), 16) &&
                        Hex(p.GetProperty("corner").GetString()!) == Convert.ToInt64(e["cornerBits"]!.GetValue<string>(), 16);
            if (!same)
            {
                identical = false;
                c.Problems.Add($"{label}: candidate {i}: Python {p} vs engine {e}");
            }
            items.Add(new OverrideItem(new MoveKey(pm.Item4, pm.Item2, pm.Item3),
                BitConverter.Int64BitsToDouble(Hex(p.GetProperty("final").GetString()!)),
                BitConverter.Int64BitsToDouble(Hex(p.GetProperty("safety").GetString()!))));
        }
        if (pyTable.Count != engTable.Count) c.Problems.Add($"{label}: table sizes Python {pyTable.Count} engine {engTable.Count}");
        if (identical) c.RobustTablesIdentical++;

        var pyRobustMove = ReadPyMove(robustRef.GetProperty("robust_move"));
        var engRobustMove = ReadMove(robust["robustMove"]);
        if (pyRobustMove != engRobustMove) c.Problems.Add($"{label}: robust move Python {pyRobustMove} engine {engRobustMove}");

        // the override decision from Python's table and the ENGINE's standard move
        var std = engineStandardLast!.Value;
        var stdKey = new MoveKey(std.Item4, std.Item2, std.Item3);
        MoveKey? robustKey = pyRobustMove is null ? null : new MoveKey(pyRobustMove.Value.Item4, pyRobustMove.Value.Item2, pyRobustMove.Value.Item3);
        var decided = EndgameRobust.DecideOverride(stdKey, robustKey, items, aggressive)!.Value;
        var expectedFinal = decided != stdKey ? pyRobustMove : std;
        var finalMove = ReadMove(body["move"]);
        if (expectedFinal != finalMove) c.Problems.Add($"{label}: final move engine {finalMove}, override decision from Python's table gives {expectedFinal}");
        if (robust["overridden"]!.GetValue<bool>() != (decided != stdKey)) c.Problems.Add($"{label}: overridden flag {robust["overridden"]} vs decision {decided != stdKey}");
        if (decided != stdKey) c.Overridden++; else c.NotOverridden++;
    }
}
