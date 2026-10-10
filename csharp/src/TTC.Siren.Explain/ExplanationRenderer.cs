using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TTC.Siren.Explain;

/// <summary>
/// The wording of the explanations (decision A-24): a data file of Chinese templates, several phrasings per key. The file is the only place
/// that contains sentences, so another voice (the mascot of Phase 8D) is a different file, not different code. A template may use only the
/// slots the renderer fills from the facts; a template that names an unknown slot fails loudly instead of printing a placeholder.
/// </summary>
public sealed class TemplateSet
{
    private static readonly Regex Slot = new(@"\{([a-z_0-9]+)\}", RegexOptions.Compiled);
    private readonly Dictionary<string, string[]> _templates;

    public string Language { get; }
    public string Voice { get; }
    public int Version { get; }

    private TemplateSet(string language, string voice, int version, Dictionary<string, string[]> templates)
    {
        Language = language;
        Voice = voice;
        Version = version;
        _templates = templates;
    }

    public IReadOnlyDictionary<string, string[]> Templates => _templates;

    public static TemplateSet Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var map = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (var p in root.GetProperty("templates").EnumerateObject())
        {
            var variants = p.Value.EnumerateArray().Select(v => v.GetString()!).ToArray();
            if (variants.Length == 0) throw new InvalidDataException($"template '{p.Name}' has no variants");
            map[p.Name] = variants;
        }
        return new TemplateSet(root.GetProperty("language").GetString()!, root.GetProperty("voice").GetString()!, root.GetProperty("version").GetInt32(), map);
    }

    public static TemplateSet Load(string path) => Parse(File.ReadAllText(path, Encoding.UTF8));

    /// <summary>The templates compiled into this assembly (the plain voice).</summary>
    public static TemplateSet Default()
    {
        using var stream = typeof(TemplateSet).Assembly.GetManifestResourceStream("explain.zh-CN.json")
                           ?? throw new InvalidOperationException("embedded template file missing");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return Parse(reader.ReadToEnd());
    }

    public bool Has(string key) => _templates.ContainsKey(key);

    /// <summary>
    /// The variant chosen for <paramref name="key"/> and <paramref name="signature"/> (a string made of the numbers of the fact being told:
    /// the same facts always give the same wording; no randomness), with the slots filled.
    /// </summary>
    public string Fill(string key, string signature, IReadOnlyDictionary<string, string> slots)
    {
        if (!_templates.TryGetValue(key, out var variants)) throw new KeyNotFoundException($"template '{key}' not found");
        string template = variants[(int)(Hash(key + "|" + signature) % (uint)variants.Length)];
        return Slot.Replace(template, m =>
            slots.TryGetValue(m.Groups[1].Value, out var v) ? v : throw new InvalidDataException($"template '{key}' uses the unknown slot '{{{m.Groups[1].Value}}}'"));
    }

    /// <summary>FNV-1a over the UTF-8 bytes: stable across runs and machines (string.GetHashCode is not).</summary>
    private static uint Hash(string text)
    {
        uint h = 2166136261;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            h ^= b;
            h *= 16777619;
        }
        return h;
    }
}

public sealed record ExplanationText(string Brief, string Full);

/// <summary>Turns <see cref="ExplanationFacts"/> into the brief and the full text. Only what the facts contain is said.</summary>
public static class ExplanationRenderer
{
    private static string Inv(int v) => v.ToString(CultureInfo.InvariantCulture);

    private static string F2(double v)
    {
        double r = Math.Round(v, 2, MidpointRounding.AwayFromZero);
        if (r == 0) r = 0; // no "-0.00"
        return r.ToString("0.00", CultureInfo.InvariantCulture);
    }

    private static string Signed(int v) => v >= 0 ? "+" + Inv(v) : Inv(v);

    private static string Avg(double v)
    {
        double r = Math.Round(v, 2, MidpointRounding.AwayFromZero);
        return r == Math.Floor(r) ? Inv((int)r) : r.ToString("0.##", CultureInfo.InvariantCulture);
    }

    private static string CardText(MoveFact m) => $"第{m.CardId}号牌（上{m.Edges[0]}右{m.Edges[1]}下{m.Edges[2]}左{m.Edges[3]}）";

    private static string Cause(TemplateSet t, string cause) => t.Fill("word.cause." + cause, cause, new Dictionary<string, string>());

    public static ExplanationText Render(ExplanationFacts f, TemplateSet t) => new(RenderBrief(f, t), RenderFull(f, t));

    // =============================================================================================================== brief

    public static string RenderBrief(ExplanationFacts f, TemplateSet t)
    {
        var im = f.Immediate;
        var m = im.Move;
        string sig1 = $"{m.CardId}.{m.Row}.{m.Col}.{im.Flips.Count}";
        string sentence1;
        if (im.Flips.Count > 0)
        {
            string causes = string.Join("、", im.CauseCounts().Keys.Select(c => Cause(t, c)));
            sentence1 = t.Fill("brief.move_flips", sig1, new Dictionary<string, string>
            {
                ["card"] = CardText(m), ["cell"] = Names.Cell[m.Cell], ["flip_count"] = Inv(im.Flips.Count), ["causes"] = causes,
            });
        }
        else
        {
            sentence1 = t.Fill("brief.move_quiet", sig1, new Dictionary<string, string>
            {
                ["card"] = CardText(m), ["cell"] = Names.Cell[m.Cell],
            });
        }
        return sentence1 + OutcomeSentence(f, t);
    }

    private static string OutcomeSentence(ExplanationFacts f, TemplateSet t)
    {
        var o = f.Outcome;
        string sig = $"{o.Kind}.{o.Result}.{o.Depth}.{o.Difference}";
        if (o.Kind == "exact")
        {
            string basis = t.Fill("basis." + o.Basis, "", new Dictionary<string, string>());
            var slots = new Dictionary<string, string> { ["basis"] = basis, ["diff"] = Inv(Math.Abs(o.Difference!.Value)) };
            return t.Fill("brief.outcome_exact_" + o.Result, sig, slots);
        }
        return t.Fill("brief.outcome_heuristic_" + o.Result, sig, new Dictionary<string, string> { ["depth"] = Inv(o.Depth), ["score"] = F2(o.Score!.Value) });
    }

    // =============================================================================================================== full

    public static string RenderFull(ExplanationFacts f, TemplateSet t)
    {
        var paragraphs = new List<string>();
        var im = f.Immediate;
        var m = im.Move;

        // 1. the move and its immediate effect
        var slots = new Dictionary<string, string>
        {
            ["card"] = CardText(m), ["cell"] = Names.Cell[m.Cell],
            ["before_mine"] = Inv(im.Before.Mine), ["before_theirs"] = Inv(im.Before.Theirs), ["after_mine"] = Inv(im.After.Mine), ["after_theirs"] = Inv(im.After.Theirs),
        };
        string sig = $"{m.CardId}.{m.Row}.{m.Col}.{im.Flips.Count}";
        var p1 = new StringBuilder();
        if (im.Flips.Count > 0)
        {
            slots["flip_list"] = string.Join("；", im.Flips.Select(fl => FlipText(t, fl)));
            p1.Append(t.Fill("full.immediate_flips", sig, slots));
        }
        else p1.Append(t.Fill("full.immediate_quiet", sig, slots));
        foreach (var mod in im.Modifiers)
            p1.Append(t.Fill("full.modifier", $"{mod.Type}.{mod.From}.{mod.To}", new Dictionary<string, string>
            {
                ["type"] = mod.Type, ["from"] = Signed(mod.From), ["to"] = Signed(mod.To), ["board"] = Inv(mod.BoardCards), ["hand"] = Inv(mod.HandCards),
            }));
        paragraphs.Add(p1.ToString());

        // 2. exposure
        paragraphs.Add(ExposureParagraph(f, t));

        // 3. the opponent's reply and the line
        var p3 = new StringBuilder();
        if (f.Reply is { } r)
        {
            string effect = r.Flips.Count > 0
                ? t.Fill("full.reply_effect_flips", $"{r.Flips.Count}", new Dictionary<string, string>
                {
                    ["flip_count"] = Inv(r.Flips.Count),
                    ["captures"] = r.CapturesPlacedCard ? t.Fill("full.reply_captures", "", new Dictionary<string, string>()) : "",
                })
                : t.Fill("full.reply_effect_quiet", "", new Dictionary<string, string>());
            p3.Append(t.Fill("full.reply", $"{r.CardId}.{r.Row}.{r.Col}", new Dictionary<string, string>
            {
                ["reply_id"] = Inv(r.CardId), ["assumed"] = r.Assumed ? t.Fill("word.assumed", "", new Dictionary<string, string>()) : "",
                ["reply_cell"] = Names.Cell[r.Row * 3 + r.Col], ["reply_effect"] = effect,
            }));
        }
        if (f.Outcome.Pv.Count > 1)
        {
            string line = string.Join("，", f.Outcome.Pv.Select(p => t.Fill("full.pv_ply", $"{p.Ply}", new Dictionary<string, string>
            {
                ["who"] = t.Fill(p.Side == "self" ? "full.who_self" : "full.who_opponent", "", new Dictionary<string, string>()),
                ["card_id"] = Inv(p.CardId), ["cell"] = Names.Cell[p.Row * 3 + p.Col], ["flips"] = Inv(p.Flips),
            })));
            p3.Append(t.Fill("full.pv", $"{f.Outcome.Pv.Count}", new Dictionary<string, string> { ["line"] = line }));
        }
        if (p3.Length > 0) paragraphs.Add(p3.ToString());

        // 4. the outcome
        var p4 = new StringBuilder(OutcomeSentence(f, t));
        if (f.Outcome.Kind == "heuristic")
            p4.Append(t.Fill("full.outcome_note_heuristic", $"{f.Outcome.Depth}", new Dictionary<string, string>
            {
                ["depth"] = Inv(f.Outcome.Depth), ["score"] = F2(f.Outcome.Score!.Value), ["empty"] = Inv(f.Outcome.EmptyCells),
            }));
        paragraphs.Add(p4.ToString());

        // 5. alternatives
        if (f.Alternatives is { Items.Count: > 0 } alt) paragraphs.Add(AlternativesParagraph(alt, t));

        // 6. the robust endgame
        if (f.Robust is { } rb) paragraphs.Add(RobustParagraph(rb, t));

        // 7. unknown cards
        paragraphs.Add(UnknownParagraph(f.Unknown, t));

        // 8. confidence
        var c = f.Confidence;
        var cs = new Dictionary<string, string>
        {
            ["depth"] = Inv(c.DepthReached), ["requested"] = Inv(c.DepthRequested), ["empty"] = Inv(c.EmptyCells), ["nodes"] = c.Nodes.ToString(CultureInfo.InvariantCulture),
            ["ms"] = Inv((int)Math.Round(c.SearchMs, MidpointRounding.AwayFromZero)),
        };
        string conf = c.SolvedToEnd
            ? t.Fill("full.confidence_solved", $"{c.DepthReached}.{c.Nodes}", cs)
            : t.Fill("full.confidence_partial", $"{c.DepthReached}.{c.Nodes}", cs);
        if (c.StoppedEarly) conf += t.Fill("full.confidence_stopped", "", new Dictionary<string, string>());
        paragraphs.Add(conf);

        return string.Join("\n\n", paragraphs);
    }

    private static string FlipText(TemplateSet t, FlipFact fl)
    {
        var slots = new Dictionary<string, string>
        {
            ["cell"] = Names.Cell[fl.Row * 3 + fl.Col], ["their_id"] = Inv(fl.CardId), ["my_side"] = fl.MySide, ["my_value"] = Inv(fl.MyValue),
            ["their_value"] = Inv(fl.TheirValue), ["sum"] = Inv(fl.MyValue + fl.TheirValue),
            ["via"] = fl.ViaRow >= 0 ? Names.Cell[fl.ViaRow * 3 + fl.ViaCol] : "",
        };
        return t.Fill("full.flip." + fl.Cause, $"{fl.Row}{fl.Col}{fl.CardId}", slots);
    }

    private static string ExposureParagraph(ExplanationFacts f, TemplateSet t)
    {
        var e = f.Exposure;
        var sb = new StringBuilder();
        string position = t.Fill("word.position." + e.PositionKind, "", new Dictionary<string, string>());
        sb.Append(t.Fill("full.exposure_position", e.PositionKind, new Dictionary<string, string> { ["position"] = position }));

        var open = e.Sides.Where(s => s.Neighbour == "empty").ToList();
        if (open.Count > 0)
        {
            string sides = string.Join("、", open.Select(s => $"{s.Side}边{s.Value}"));
            sb.Append(t.Fill("full.exposure_sides_open", sides, new Dictionary<string, string> { ["sides"] = sides }));
        }
        else sb.Append(t.Fill("full.exposure_sides_closed", "", new Dictionary<string, string>()));

        if (e.Threats.Count == 0 && e.UnknownCaptureProbability is null or 0)
            sb.Append(t.Fill("full.exposure_safe", Inv(open.Count), new Dictionary<string, string>()));
        foreach (var th in e.Threats.Take(3))
            sb.Append(t.Fill("full.exposure_threat", $"{th.CardId}.{th.Row}.{th.Col}", new Dictionary<string, string>
            {
                ["threat_id"] = Inv(th.CardId), ["assumed"] = th.Assumed ? t.Fill("word.assumed", "", new Dictionary<string, string>()) : "",
                ["threat_cell"] = Names.Cell[th.Row * 3 + th.Col], ["threat_cause"] = Cause(t, th.Cause), ["captured"] = Inv(th.CapturedCards),
            }));
        if (e.UnknownCaptureProbability is { } p && p > 0)
            sb.Append(t.Fill("full.exposure_unknown", $"{e.UnknownCandidates}", new Dictionary<string, string>
            {
                ["candidates"] = Inv(e.UnknownCandidates), ["percent"] = Inv((int)Math.Round(p * 100.0, MidpointRounding.AwayFromZero)),
            }));
        return sb.ToString();
    }

    private static string AlternativesParagraph(AlternativesFact alt, TemplateSet t)
    {
        var lines = new List<string>
        {
            t.Fill("full.alt_intro", $"{alt.EvalDepth}", new Dictionary<string, string> { ["eval_depth"] = Inv(alt.EvalDepth), ["chosen_score"] = F2(alt.ChosenScore) }),
        };
        foreach (var a in alt.Items)
        {
            string deltaKey = a.Tied ? "full.delta_equal" : a.ScoreDelta < 0 ? "full.delta_lower" : "full.delta_higher";
            string delta = t.Fill(deltaKey, $"{a.Rank}", new Dictionary<string, string> { ["delta"] = F2(Math.Abs(a.ScoreDelta)) });
            lines.Add(t.Fill("full.alt_item", $"{a.Rank}.{a.Move.CardId}", new Dictionary<string, string>
            {
                ["rank"] = Inv(a.Rank), ["alt_id"] = Inv(a.Move.CardId), ["alt_cell"] = Names.Cell[a.Move.Cell], ["alt_score"] = F2(a.Score),
                ["delta_text"] = delta, ["why"] = WhyText(a, t),
            }));
        }
        return string.Join("\n", lines);
    }

    private static string WhyText(AlternativeFact a, TemplateSet t)
    {
        var d = a.Why.Data;
        var slots = new Dictionary<string, string>();
        switch (a.Why.Kind)
        {
            case "exposed_to_reply":
                slots["reply_id"] = Inv((int)d["reply_card_id"]!);
                slots["assumed"] = (bool)d["reply_assumed"]! ? t.Fill("word.assumed", "", new Dictionary<string, string>()) : "";
                slots["reply_cell"] = Names.Cell[(int)d["reply_row"]! * 3 + (int)d["reply_col"]!];
                slots["cause"] = Cause(t, (string)d["cause"]!);
                slots["their_side"] = (string)d["their_side"]!;
                slots["their_value"] = Inv((int)d["their_value"]!);
                slots["my_value"] = Inv((int)d["my_value"]!);
                break;
            case "fewer_captures":
                slots["alt_flips"] = Inv((int)d["alt_flips"]!);
                slots["chosen_flips"] = Inv((int)d["chosen_flips"]!);
                break;
            case "bigger_counter":
                slots["alt_reply_flips"] = Inv((int)d["alt_reply_flips"]!);
                slots["chosen_reply_flips"] = Inv((int)d["chosen_reply_flips"]!);
                break;
        }
        return t.Fill("full.why." + a.Why.Kind, $"{a.Rank}", slots);
    }

    private static string RobustParagraph(RobustFact r, TemplateSet t)
    {
        var sb = new StringBuilder();
        if (r.Average is { } avg && r.Worst is { } worst)
        {
            if (r.Scenarios > 1)
                sb.Append(t.Fill("full.robust", $"{r.Scenarios}", new Dictionary<string, string>
                {
                    ["scenarios"] = Inv(r.Scenarios), ["safety_percent"] = Inv((int)Math.Round(r.SafetyRatio * 100.0, MidpointRounding.AwayFromZero)),
                    ["average"] = Avg(avg), ["worst"] = Avg(worst),
                }));
            else
                sb.Append(t.Fill("full.robust_single", $"{r.Scenarios}", new Dictionary<string, string> { ["average"] = Avg(avg) }));
        }
        if (r.Overridden && r.OverrideRule is not null)
            sb.Append(t.Fill("full.robust_override", r.OverrideRule, new Dictionary<string, string>
            {
                ["rule"] = t.Fill("full.override_rule." + r.OverrideRule, "", new Dictionary<string, string>()),
            }));
        else if (r.StandardMove is not null) sb.Append(t.Fill("full.robust_keep", "", new Dictionary<string, string>()));
        return sb.ToString();
    }

    private static string UnknownParagraph(UnknownFact u, TemplateSet t)
    {
        if (u.OpponentUnknown == 0) return t.Fill("full.unknown_none", "", new Dictionary<string, string>());
        string strategy;
        switch (u.Strategy)
        {
            case "npc_pool":
                string npcText = u.NpcName is { Length: > 0 } && u.NpcName != "unknown"
                    ? t.Fill("full.npc_text.named", u.NpcName, new Dictionary<string, string> { ["npc_name"] = u.NpcName, ["pool"] = Inv(u.NpcPoolSize) })
                    : t.Fill("full.npc_text.anon", "", new Dictionary<string, string> { ["pool"] = Inv(u.NpcPoolSize) });
                strategy = t.Fill("full.strategy.npc_pool", "", new Dictionary<string, string> { ["npc_text"] = npcText });
                break;
            case "rule_model":
                strategy = t.Fill("full.strategy.rule_model", u.PrimaryRule ?? "", new Dictionary<string, string> { ["rule"] = u.PrimaryRule ?? "平衡" });
                break;
            default:
                strategy = t.Fill("full.strategy." + u.Strategy, "", new Dictionary<string, string>());
                break;
        }
        string cards = u.Assumed.Count == 0
            ? "（无具体假设）"
            : string.Join("、", u.Assumed.Select(a => t.Fill("full.unknown_card", $"{a.CardId}", new Dictionary<string, string> { ["card_id"] = Inv(a.CardId), ["star"] = Inv(a.Star) })));
        return t.Fill("full.unknown_assumed", $"{u.OpponentUnknown}", new Dictionary<string, string>
        {
            ["count"] = Inv(u.OpponentUnknown), ["strategy"] = strategy, ["cards"] = cards,
        });
    }
}
