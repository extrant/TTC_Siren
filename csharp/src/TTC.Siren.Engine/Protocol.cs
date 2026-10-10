using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Engine;

/// <summary>Wire protocol of <c>POST /v1/solve</c> (see docs/reports/PHASE7_INTEGRATION.md for the full description).</summary>
public static class Protocol
{
    /// <summary>Bumped whenever a field is added, removed or changes meaning. The Python client refuses another version.</summary>
    public const int Version = 1;

    public const int MaxBodyBytes = 1 << 20;
    public const int MaxEngineDepth = CompatSearcher.PythonMaxDepth;
}

/// <summary>A request that cannot be served as sent. Mapped to HTTP 400 <c>invalid_request</c>.</summary>
public sealed class RequestException : Exception
{
    public RequestException(string message) : base(message) { }
}

/// <summary>The robust-endgame inputs of Python's <c>ai_move</c> (only present when <c>_should_use_endgame_robust_mode</c> is true).</summary>
public sealed class RobustRequest
{
    /// <summary>Python <c>game_state.current_player_idx</c>; must equal the state's side to move.</summary>
    public int AiPlayer;

    /// <summary>Python <c>opponent_player_idx</c> (0 if my owner is blue, else 1): the hand whose unknown cards are enumerated.</summary>
    public int Opponent;

    /// <summary>Python <c>_is_generated_unknown_card</c> for every hand slot, per player.</summary>
    public bool[][] Generated = Array.Empty<bool[]>();

    /// <summary>Python: <c>1 if opp_unknown_count == 0 else min(24, max(16, opp_unknown_count * 12))</c>.</summary>
    public int SampleCount;

    public List<int> NpcIds = new();

    /// <summary>Whether Python's unknown-card handler exists (<c>if not handler: return [base_state]</c>).</summary>
    public bool Handler = true;

    /// <summary>The cards <c>handler.generate_opponent_cards</c> produced, when Python's legal enumeration found none.</summary>
    public List<UnknownCandidate>? FallbackCandidates;
}

public sealed class SolveRequest
{
    public int Protocol;
    public string? RequestId;
    public RuleSet Rules;
    public StateSnapshot State = new();
    public bool Aggressive;

    /// <summary>Python <c>min(max_depth, 100)</c> after flooring: 0..100. 0 searches nothing (no move).</summary>
    public int MaxDepth;

    /// <summary>Seconds. <c>null</c> in the JSON means no limit (Python NaN or +inf); zero or less is "already over".</summary>
    public double MaxTime = double.PositiveInfinity;

    public RobustRequest? Robust;
}

/// <summary>Strict reader for <see cref="SolveRequest"/>: every violation is a <see cref="RequestException"/> with a message.</summary>
public static class RequestReader
{
    private const int MaxIds = 600;

    public static SolveRequest Parse(ReadOnlyMemory<byte> body)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(body, new JsonDocumentOptions { MaxDepth = 32 });
        }
        catch (JsonException e)
        {
            throw new RequestException("malformed JSON: " + e.Message);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new RequestException("the request must be a JSON object");

            var req = new SolveRequest { Protocol = Int(root, "protocol") };
            if (req.Protocol != Engine.Protocol.Version)
                throw new RequestException($"unsupported protocol {req.Protocol} (this engine speaks {Engine.Protocol.Version})");

            if (root.TryGetProperty("requestId", out var rid) && rid.ValueKind == JsonValueKind.String)
                req.RequestId = rid.GetString();

            req.Rules = ReadRules(Prop(root, "rules"));
            req.Aggressive = Bool(root, "aggressive");
            req.MaxDepth = ReadMaxDepth(Prop(root, "maxDepth"));
            req.MaxTime = ReadMaxTime(Prop(root, "maxTime"));
            req.State = ReadState(Prop(root, "state"), req.Rules);

            if (root.TryGetProperty("robust", out var robust) && robust.ValueKind != JsonValueKind.Null)
                req.Robust = ReadRobust(robust, req.State);
            return req;
        }
    }

    // ------------------------------------------------------------------ pieces

    private static RuleSet ReadRules(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Array) throw new RequestException("rules must be an array of rule names");
        var names = new List<string>();
        foreach (var item in e.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String) throw new RequestException("rules must contain strings only");
            string name = item.GetString()!;
            if (!RuleParser.ToPythonList((RuleSet)ushort.MaxValue).Contains(name))
                throw new RequestException($"unknown rule name '{name}'");
            names.Add(name);
        }
        return RuleParser.FromPythonList(names);
    }

    /// <summary>
    /// Python: <c>depth &lt;= min(max_depth, 100)</c> with <c>depth</c> counting up from 1. A fractional limit therefore
    /// means its floor; zero or less means no level at all. The value is a JSON number (integer or double).
    /// </summary>
    private static int ReadMaxDepth(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Number) throw new RequestException("maxDepth must be a number");
        double d = e.GetDouble();
        if (double.IsNaN(d) || d < 1.0) return 0;
        return d >= Engine.Protocol.MaxEngineDepth ? Engine.Protocol.MaxEngineDepth : (int)Math.Floor(d);
    }

    /// <summary>
    /// Python: time is up once <c>elapsed &gt; max_time</c>. <c>null</c> = never (Python NaN or +inf). Zero or less: already
    /// over before the first level, so no move is returned. A huge value behaves as no limit.
    /// </summary>
    private static double ReadMaxTime(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.Null) return double.PositiveInfinity;
        if (e.ValueKind != JsonValueKind.Number) throw new RequestException("maxTime must be a number or null");
        return e.GetDouble();
    }

    private static StateSnapshot ReadState(JsonElement e, RuleSet rules)
    {
        if (e.ValueKind != JsonValueKind.Object) throw new RequestException("state must be an object");
        var s = new StateSnapshot { Rules = rules, Current = Int(e, "cur") };
        if (s.Current is not (0 or 1)) throw new RequestException("state.cur must be 0 or 1");

        var board = Prop(e, "board");
        if (board.ValueKind != JsonValueKind.Array || board.GetArrayLength() != 9)
            throw new RequestException("state.board must be an array of 9 cells");
        int i = 0;
        foreach (var cell in board.EnumerateArray())
            s.Board[i++] = cell.ValueKind == JsonValueKind.Null ? null : ReadCard(cell, $"state.board[{i - 1}]");

        var hands = Prop(e, "hands");
        if (hands.ValueKind != JsonValueKind.Array || hands.GetArrayLength() != 2)
            throw new RequestException("state.hands must be an array of 2 hands");
        int p = 0;
        foreach (var hand in hands.EnumerateArray())
        {
            if (hand.ValueKind != JsonValueKind.Array) throw new RequestException($"state.hands[{p}] must be an array");
            if (hand.GetArrayLength() > GameState.HandCapacity)
                throw new RequestException($"state.hands[{p}] has {hand.GetArrayLength()} cards (capacity {GameState.HandCapacity})");
            int k = 0;
            foreach (var card in hand.EnumerateArray()) s.Hands[p].Add(ReadCard(card, $"state.hands[{p}][{k++}]"));
            p++;
        }

        int total = s.Board.Count(c => c is not null) + s.Hands[0].Count + s.Hands[1].Count;
        if (total > GameState.MaxInstances) throw new RequestException($"too many cards ({total})");
        if (s.Hands[s.Current].Count == 0) throw new RequestException("the side to move has no cards in hand");
        return s;
    }

    /// <summary><c>[id|null, type|null, "r"|"b"|null, mod, bu,br,bd,bl, du,dr,dd,dl, canUse(0/1)]</c>.</summary>
    internal static CardSnapshot ReadCard(JsonElement e, string where)
    {
        if (e.ValueKind != JsonValueKind.Array || e.GetArrayLength() != 13)
            throw new RequestException($"{where} must be an array of 13 values");

        var c = new CardSnapshot();
        var id = e[0];
        if (id.ValueKind == JsonValueKind.Null) c.Id = -1;
        else if (id.ValueKind == JsonValueKind.Number && id.TryGetInt32(out int idv) && idv is >= 0 and <= 1_000_000) c.Id = idv;
        else throw new RequestException($"{where}: id must be null or an integer 0..1000000");

        var type = e[1];
        if (type.ValueKind == JsonValueKind.Null) c.Type = null;
        else if (type.ValueKind == JsonValueKind.String)
        {
            c.Type = type.GetString();
            try
            {
                CardTypes.IdOf(c.Type);
            }
            catch (ArgumentException)
            {
                throw new RequestException($"{where}: unknown card type '{c.Type}'");
            }
        }
        else throw new RequestException($"{where}: type must be null or a string");

        var owner = e[2];
        if (owner.ValueKind == JsonValueKind.Null) c.Owner = OwnerId.None;
        else if (owner.ValueKind == JsonValueKind.String && owner.GetString() == "r") c.Owner = OwnerId.Red;
        else if (owner.ValueKind == JsonValueKind.String && owner.GetString() == "b") c.Owner = OwnerId.Blue;
        else throw new RequestException($"{where}: unknown owner (expected \"r\", \"b\" or null)");

        if (e[3].ValueKind != JsonValueKind.Number || !e[3].TryGetInt32(out int mod) || mod is < sbyte.MinValue or > sbyte.MaxValue)
            throw new RequestException($"{where}: modifier must be an integer in the sbyte range");
        c.Mod = mod;

        for (int k = 0; k < 4; k++)
        {
            c.Base[k] = Edge(e[4 + k], where, "base edge");
            c.Display[k] = Edge(e[8 + k], where, "display edge");
        }

        if (e[12].ValueKind != JsonValueKind.Number || !e[12].TryGetInt32(out int canUse) || canUse is not (0 or 1))
            throw new RequestException($"{where}: canUse must be 0 or 1");
        c.CanUse = canUse == 1;
        return c;
    }

    private static byte Edge(JsonElement e, string where, string what)
    {
        if (e.ValueKind != JsonValueKind.Number || !e.TryGetInt32(out int v) || v is < 0 or > 10)
            throw new RequestException($"{where}: {what} must be an integer 0..10");
        return (byte)v;
    }

    private static RobustRequest ReadRobust(JsonElement e, StateSnapshot state)
    {
        if (e.ValueKind != JsonValueKind.Object) throw new RequestException("robust must be an object or null");
        var r = new RobustRequest { AiPlayer = Int(e, "aiPlayer"), Opponent = Int(e, "opponent") };
        if (r.AiPlayer != state.Current) throw new RequestException("robust.aiPlayer must be the side to move");
        if (r.Opponent is not (0 or 1)) throw new RequestException("robust.opponent must be 0 or 1");

        var gen = Prop(e, "generated");
        if (gen.ValueKind != JsonValueKind.Array || gen.GetArrayLength() != 2)
            throw new RequestException("robust.generated must hold one array per player");
        r.Generated = new bool[2][];
        int p = 0;
        foreach (var hand in gen.EnumerateArray())
        {
            if (hand.ValueKind != JsonValueKind.Array || hand.GetArrayLength() != state.Hands[p].Count)
                throw new RequestException($"robust.generated[{p}] must have one flag per hand card");
            r.Generated[p] = new bool[hand.GetArrayLength()];
            int k = 0;
            foreach (var flag in hand.EnumerateArray())
            {
                if (flag.ValueKind != JsonValueKind.Number || !flag.TryGetInt32(out int f) || f is not (0 or 1))
                    throw new RequestException("robust.generated flags must be 0 or 1");
                r.Generated[p][k++] = f == 1;
            }
            p++;
        }

        r.SampleCount = Int(e, "sampleCount");
        if (r.SampleCount is < 1 or > 1000) throw new RequestException("robust.sampleCount must be 1..1000");

        if (e.TryGetProperty("npcIds", out var npc) && npc.ValueKind != JsonValueKind.Null)
        {
            if (npc.ValueKind != JsonValueKind.Array || npc.GetArrayLength() > MaxIds)
                throw new RequestException("robust.npcIds must be an array of at most 600 ids");
            foreach (var item in npc.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int id) || id <= 0)
                    throw new RequestException("robust.npcIds must be positive integers");
                r.NpcIds.Add(id);
            }
        }

        r.Handler = Bool(e, "handler");

        if (e.TryGetProperty("fallbackCandidates", out var fb) && fb.ValueKind != JsonValueKind.Null)
        {
            if (fb.ValueKind != JsonValueKind.Array || fb.GetArrayLength() > MaxIds)
                throw new RequestException("robust.fallbackCandidates must be an array of at most 600 cards");
            r.FallbackCandidates = new List<UnknownCandidate>();
            int k = 0;
            foreach (var item in fb.EnumerateArray())
            {
                var c = ReadCard(item, $"robust.fallbackCandidates[{k++}]");
                r.FallbackCandidates.Add(new UnknownCandidate(c.Id, c.Base[0], c.Base[1], c.Base[2], c.Base[3], CardTypes.IdOf(c.Type)));
            }
        }
        return r;
    }

    // ------------------------------------------------------------------ helpers

    private static JsonElement Prop(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var v) ? v : throw new RequestException($"missing field '{name}'");

    private static int Int(JsonElement obj, string name)
    {
        var v = Prop(obj, name);
        if (v.ValueKind != JsonValueKind.Number || !v.TryGetInt32(out int i)) throw new RequestException($"'{name}' must be an integer");
        return i;
    }

    private static bool Bool(JsonElement obj, string name)
    {
        var v = Prop(obj, name);
        return v.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new RequestException($"'{name}' must be true or false"),
        };
    }
}
