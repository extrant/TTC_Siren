using System.Buffers;
using System.Text.Json;

namespace TTC.Siren.Engine;

/// <summary>
/// JSON responses of the protocol. Doubles are written twice where precision matters: as a JSON number (null when the
/// value is not finite: JSON has no infinity) and as <c>...Bits</c>, the 16 hex digits of the IEEE-754 bit pattern, which is
/// the exact value (the end-to-end comparison is bit for bit, CLAUDE.md section 2).
/// </summary>
public static class ResponseWriter
{
    public static string Bits(double d) => BitConverter.DoubleToInt64Bits(d).ToString("X16");

    public static byte[] Error(string status, string message, string? reason = null)
    {
        var buffer = new ArrayBufferWriter<byte>(256);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteNumber("protocol", Protocol.Version);
            w.WriteString("status", status);
            w.WriteString("message", message);
            if (reason is not null) w.WriteString("reason", reason);
            w.WriteEndObject();
        }
        return buffer.WrittenSpan.ToArray();
    }

    public static byte[] Solve(SolveResult r)
    {
        var buffer = new ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteNumber("protocol", Protocol.Version);
            w.WriteString("status", r.Status);
            if (r.RequestId is not null) w.WriteString("requestId", r.RequestId);

            if (r.Status == SolveResult.NeedsFallbackCandidates)
            {
                w.WriteNumber("unknownCount", r.UnknownCount);
                w.WriteEndObject();
                w.Flush();
                return buffer.WrittenSpan.ToArray();
            }

            WriteMove(w, "move", r.Final);

            w.WriteStartObject("standard");
            WriteMove(w, "move", r.Standard);
            Number(w, "score", r.StandardScore);
            w.WriteString("scoreBits", Bits(r.StandardScore));
            w.WriteEndObject();

            w.WriteStartArray("depths");
            foreach (var d in r.Depths)
            {
                w.WriteStartObject();
                w.WriteNumber("depth", d.Depth);
                WriteMove(w, "move", d.Move);
                Number(w, "score", d.Score);
                w.WriteString("scoreBits", Bits(d.Score));
                w.WriteNumber("nodes", d.Nodes);
                w.WriteNumber("nodesTotal", d.NodesTotal);
                w.WriteNumber("ttHitsTotal", d.TtHitsTotal);
                w.WriteNumber("cutoffsTotal", d.CutoffsTotal);
                w.WriteNumber("elapsed", d.ElapsedSeconds);
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartObject("totals");
            w.WriteNumber("nodes", r.Nodes);
            w.WriteNumber("depthCompleted", r.DepthCompleted);
            w.WriteNumber("ttHits", r.TtHits);
            w.WriteNumber("cutoffs", r.Cutoffs);
            w.WriteNumber("rootMoves", r.RootMoves);
            w.WriteNumber("elapsed", r.SearchSeconds);
            w.WriteEndObject();

            if (r.Robust is null)
            {
                w.WriteNull("robust");
            }
            else
            {
                var b = r.Robust;
                w.WriteStartObject("robust");
                w.WriteBoolean("ran", true);
                w.WriteNumber("scenarios", b.Scenarios);
                WriteMove(w, "robustMove", b.RobustMove);
                w.WriteBoolean("overridden", b.Overridden);
                w.WriteNumber("exactNodes", b.ExactNodes);
                w.WriteNumber("exactMakeMoves", b.ExactMakeMoves);
                w.WriteNumber("elapsed", b.ElapsedSeconds);
                w.WriteStartArray("candidates");
                foreach (var c in b.Candidates)
                {
                    w.WriteStartObject();
                    WriteMove(w, "move", c.Move);
                    Number(w, "finalScore", c.FinalScore);
                    w.WriteString("finalBits", Bits(c.FinalScore));
                    Number(w, "safetyRatio", c.SafetyRatio);
                    w.WriteString("safetyBits", Bits(c.SafetyRatio));
                    w.WriteString("robustBits", Bits(c.RobustScore));
                    w.WriteString("cornerBits", Bits(c.CornerRisk));
                    w.WriteEndObject();
                }
                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteStartObject("timings");
            w.WriteNumber("buildMs", r.BuildMs);
            w.WriteNumber("searchMs", r.SearchMs);
            w.WriteNumber("robustMs", r.RobustMs);
            w.WriteNumber("totalMs", r.TotalMs);
            w.WriteEndObject();

            w.WriteNumber("worker", r.WorkerId);
            w.WriteEndObject();
        }
        return buffer.WrittenSpan.ToArray();
    }

    private static void Number(Utf8JsonWriter w, string name, double value)
    {
        if (double.IsFinite(value)) w.WriteNumber(name, value);
        else w.WriteNull(name);
    }

    private static void WriteMove(Utf8JsonWriter w, string name, MoveOut? move)
    {
        if (move is null)
        {
            w.WriteNull(name);
            return;
        }
        w.WriteStartObject(name);
        w.WriteNumber("slot", move.Value.Slot);
        w.WriteNumber("row", move.Value.Row);
        w.WriteNumber("col", move.Value.Col);
        w.WriteNumber("cardId", move.Value.CardId);
        w.WriteEndObject();
    }
}
