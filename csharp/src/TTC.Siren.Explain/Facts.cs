namespace TTC.Siren.Explain;

/// <summary>Names used in the facts and the texts: cells, sides, rules.</summary>
public static class Names
{
    public static readonly string[] Cell =
    {
        "左上角", "上边", "右上角",
        "左边", "中心", "右边",
        "左下角", "下边", "右下角",
    };

    public static string CellKind(int cell) => cell is 0 or 2 or 6 or 8 ? "corner" : cell == 4 ? "center" : "edge";

    public static readonly string[] Side = { "上", "右", "下", "左" };

    public static string Wire(FlipCause c) => FlipCauseNames.Wire(c);
}

/// <summary>JSON helpers: facts are exported as plain dictionaries and lists (the types <c>PyJson</c> writes).</summary>
internal static class J
{
    public static Dictionary<string, object?> Obj(params (string Key, object? Value)[] items)
    {
        var d = new Dictionary<string, object?>();
        foreach (var (k, v) in items) d[k] = v;
        return d;
    }

    /// <summary>A finite number, or null (JSON has no infinity or NaN; the engine uses -inf for "no value").</summary>
    public static object? Num(double? v) => v is { } d && double.IsFinite(d) ? d : null;

    public static List<object?> List<T>(IEnumerable<T> items, Func<T, object?> map) => items.Select(map).ToList();
}

/// <summary>A card placed on a cell: what is needed to name it in the facts.</summary>
/// <param name="CardId">The real database id of the card (for a card the server assumed, the id behind the assumption).</param>
/// <param name="Assumed">True when the card is one of the cards the server assumed for an unknown slot.</param>
public sealed record MoveFact(int CardId, int Row, int Col, int Slot, int[] Edges, string? CardType, bool Assumed)
{
    public int Cell => Row * 3 + Col;

    public Dictionary<string, object?> ToJson() => J.Obj(
        ("card_id", CardId), ("row", Row), ("col", Col), ("slot", Slot), ("cell_name", Names.Cell[Cell]), ("cell_kind", Names.CellKind(Cell)),
        ("edges", new List<object?> { Edges[0], Edges[1], Edges[2], Edges[3] }), ("type", CardType), ("assumed", Assumed));
}

public sealed record FlipFact(int Row, int Col, int CardId, string Cause, string MySide, int MyValue, int TheirValue, int ViaRow, int ViaCol, string? ComboKind)
{
    public Dictionary<string, object?> ToJson()
    {
        var d = J.Obj(("row", Row), ("col", Col), ("cell_name", Names.Cell[Row * 3 + Col]), ("card_id", CardId), ("cause", Cause), ("my_side", MySide),
            ("my_value", MyValue), ("their_value", TheirValue));
        if (Cause == "plus") d["sum"] = MyValue + TheirValue;
        if (Cause == "combo")
        {
            d["via_row"] = ViaRow;
            d["via_col"] = ViaCol;
            d["via_cell_name"] = Names.Cell[ViaRow * 3 + ViaCol];
            d["combo_kind"] = ComboKind;
        }
        return d;
    }

    public static FlipFact From(FlipRecord f) => new(f.Cell / 3, f.Cell % 3, f.CardId, Names.Wire(f.Cause), Names.Side[f.MySide], f.MyValue, f.TheirValue,
        f.Via >= 0 ? f.Via / 3 : -1, f.Via >= 0 ? f.Via % 3 : -1, f.Cause == FlipCause.Combo ? Names.Wire(f.ComboKind) : null);
}

public sealed record CardCounts(int Mine, int Theirs);

public sealed record ModifierChange(string Type, int From, int To, int BoardCards, int HandCards)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("type", Type), ("from", From), ("to", To), ("board_cards", BoardCards), ("hand_cards", HandCards));
}

/// <summary>Fact family 1: what the move does at once.</summary>
public sealed record ImmediateEffectFact(MoveFact Move, List<FlipFact> Flips, CardCounts Before, CardCounts After, List<ModifierChange> Modifiers,
    List<string> ActiveRules)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "immediate_effect"), ("move", Move.ToJson()), ("flips", J.List(Flips, f => f.ToJson())),
        ("flip_count", Flips.Count), ("cause_counts", CauseCounts()),
        ("cards_before", J.Obj(("mine", Before.Mine), ("theirs", Before.Theirs))),
        ("cards_after", J.Obj(("mine", After.Mine), ("theirs", After.Theirs))),
        ("modifier_changes", J.List(Modifiers, m => m.ToJson())),
        ("active_rules", ActiveRules));

    /// <summary>How many flips each cause produced, in the order the causes first occur.</summary>
    public Dictionary<string, object?> CauseCounts()
    {
        var d = new Dictionary<string, object?>();
        foreach (var f in Flips) d[f.Cause] = d.TryGetValue(f.Cause, out var n) ? (int)n! + 1 : 1;
        return d;
    }
}

public sealed record SideFact(string Side, string Neighbour, int Value)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("side", Side), ("neighbour", Neighbour), ("value", Value));
}

/// <param name="CardId">The opponent card (a database id; for an assumed card the id behind the assumption).</param>
/// <param name="Assumed">True when this is one of the cards the server assumed for the opponent's unknown slots.</param>
public sealed record ThreatFact(int CardId, bool Assumed, int Row, int Col, string Cause, int CapturedCards)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("card_id", CardId), ("assumed", Assumed), ("row", Row), ("col", Col),
        ("cell_name", Names.Cell[Row * 3 + Col]), ("cause", Cause), ("cards_captured", CapturedCards));
}

/// <summary>Fact family 2: how safe the placed card is on the next turn.</summary>
/// <param name="UnknownCaptureProbability">
/// Among the legal candidates for an unknown opponent card (database cards not seen, within the deck limits and the NPC pool), the share that
/// could capture the placed card with some move next turn; null when the opponent has no unknown card.
/// </param>
public sealed record ExposureFact(List<SideFact> Sides, List<ThreatFact> Threats, double? UnknownCaptureProbability, int UnknownCandidates,
    string PositionKind, double PositionWeight, double PlacementEdgeScore)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "exposure"), ("sides", J.List(Sides, s => s.ToJson())),
        ("exposed_sides", Sides.Count(s => s.Neighbour == "empty")),
        ("threats", J.List(Threats, t => t.ToJson())),
        ("capturable_now", Threats.Any(t => !t.Assumed)),
        ("unknown_capture_probability", UnknownCaptureProbability),
        ("unknown_capture_percent", UnknownCaptureProbability is { } p ? (int)Math.Round(p * 100.0, MidpointRounding.AwayFromZero) : null),
        ("unknown_candidates", UnknownCandidates),
        ("position_kind", PositionKind), ("position_weight", PositionWeight), ("placement_edge_score", PlacementEdgeScore));
}

public sealed record PlyFact(int Ply, string Side, int CardId, int Row, int Col, int Flips, bool Assumed)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("ply", Ply), ("side", Side), ("card_id", CardId), ("row", Row), ("col", Col),
        ("cell_name", Names.Cell[Row * 3 + Col]), ("flips", Flips), ("assumed", Assumed));
}

/// <summary>Fact family 3: the opponent's strongest reply and what it does.</summary>
public sealed record ReplyFact(int CardId, bool Assumed, int Row, int Col, List<FlipFact> Flips, bool CapturesPlacedCard, string Source)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "opponent_reply"), ("card_id", CardId), ("assumed", Assumed), ("row", Row), ("col", Col), ("cell_name", Names.Cell[Row * 3 + Col]),
        ("flips", J.List(Flips, f => f.ToJson())), ("flip_count", Flips.Count), ("captures_placed_card", CapturesPlacedCard), ("source", Source));
}

/// <summary>Fact family 4: the principal variation and the expected outcome.</summary>
/// <param name="Kind">"exact" (the card difference at the end of the game is known) or "heuristic" (the search score at <paramref name="Depth"/>).</param>
/// <param name="Result">For exact: "win", "draw" or "loss" for the AI side. For heuristic: "ahead", "level" or "behind".</param>
/// <param name="Difference">Exact: cards of the AI side minus cards of the opponent at the end (hands included). Heuristic: null.</param>
/// <param name="Basis">Exact: "known_hands" when all cards are known, "assumed_hands" when it holds under the assumed unknown cards.</param>
public sealed record OutcomeFact(List<PlyFact> Pv, int ReliablePlies, string Kind, string Result, int? Difference, double? Score, int Depth, int EmptyCells,
    string? Basis, string PvSource)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "outcome"), ("pv", J.List(Pv, p => p.ToJson())), ("pv_plies", Pv.Count), ("pv_source", PvSource), ("kind", Kind), ("result", Result),
        ("difference", Difference), ("score", Score), ("depth", Depth), ("empty_cells", EmptyCells), ("basis", Basis));
}

/// <summary>Why an alternative is worse than the chosen move, as a recomputable fact.</summary>
/// <param name="Kind">"exposed_to_reply" | "fewer_captures" | "bigger_counter" | "score_gap".</param>
public sealed record WhyWorse(string Kind, Dictionary<string, object?> Data)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("kind", Kind), ("data", Data));
}

public sealed record AlternativeFact(int Rank, MoveFact Move, double Score, double ScoreDelta, bool Tied, int Flips, ReplyFact? Reply, WhyWorse Why)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("rank", Rank), ("move", Move.ToJson()), ("score", Score), ("score_delta", ScoreDelta),
        ("tied", Tied), ("flips", Flips), ("reply", Reply?.ToJson()), ("why_worse", Why.ToJson()));
}

/// <summary>Fact family 5: the other moves, scored at the same depth as the chosen one.</summary>
public sealed record AlternativesFact(int EvalDepth, double ChosenScore, bool ChosenIsSearchBest, int MovesScored, List<AlternativeFact> Items)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "alternatives"), ("eval_depth", EvalDepth), ("chosen_score", ChosenScore), ("chosen_is_search_best", ChosenIsSearchBest),
        ("moves_scored", MovesScored), ("items", J.List(Items, a => a.ToJson())));
}

/// <summary>Fact family 6: the information-set endgame.</summary>
/// <param name="OverrideRule">When overridden: "safety_floor", "safer_and_close", "clearly_better" or "aggressive_margin" (the clause of the decision that fired).</param>
public sealed record RobustFact(int Scenarios, bool Aggressive, double SafetyRatio, double? Average, double? Worst, double FinalScore, bool Overridden,
    string? OverrideRule, MoveFact? StandardMove, double? StandardSafety, double? StandardFinal)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "robust_endgame"), ("scenarios", Scenarios), ("aggressive", Aggressive), ("safety_ratio", SafetyRatio),
        ("safety_percent", (int)Math.Round(SafetyRatio * 100.0, MidpointRounding.AwayFromZero)), ("average_difference", J.Num(Average)),
        ("worst_difference", J.Num(Worst)), ("final_score", J.Num(FinalScore)), ("overridden", Overridden), ("override_rule", OverrideRule),
        ("standard_move", StandardMove?.ToJson()), ("standard_safety_ratio", J.Num(StandardSafety)), ("standard_final_score", J.Num(StandardFinal)));
}

public sealed record AssumedCardFact(int Slot, int CardId, int Star, string? CardType)
{
    public Dictionary<string, object?> ToJson() => J.Obj(("slot", Slot), ("card_id", CardId), ("star", Star), ("type", CardType));
}

/// <summary>Fact family 7: the unknown cards and how they were assumed.</summary>
/// <param name="Strategy">"none" | "npc_pool" | "rule_model" | "random_sample" | "threat_ranking".</param>
public sealed record UnknownFact(int OpponentUnknown, int OwnUnknown, string Strategy, string? PrimaryRule, string? NpcName, int NpcPoolSize, List<AssumedCardFact> Assumed)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "unknown_cards"), ("opponent_unknown", OpponentUnknown), ("own_unknown", OwnUnknown), ("strategy", Strategy), ("primary_rule", PrimaryRule),
        ("npc_name", NpcName), ("npc_pool_size", NpcPoolSize), ("assumed", J.List(Assumed, a => a.ToJson())));
}

/// <summary>Fact family 8: how reliable the answer is.</summary>
public sealed record ConfidenceFact(bool SolvedToEnd, int DepthReached, int DepthRequested, int EmptyCells, long Nodes, double SearchMs, bool StoppedEarly, bool RobustUsed)
{
    public Dictionary<string, object?> ToJson() => J.Obj(
        ("type", "confidence"), ("solved_to_end", SolvedToEnd), ("depth_reached", DepthReached), ("depth_requested", DepthRequested), ("empty_cells", EmptyCells),
        ("nodes", Nodes), ("search_ms", SearchMs), ("stopped_early", StoppedEarly), ("robust_used", RobustUsed));
}

/// <summary>All facts of one decision (explanation facts v2). Optional families are null when they do not apply or were not requested.</summary>
public sealed class ExplanationFacts
{
    public const int Version = 2;

    public required ImmediateEffectFact Immediate { get; init; }
    public required ExposureFact Exposure { get; init; }
    public ReplyFact? Reply { get; init; }
    public required OutcomeFact Outcome { get; init; }
    public AlternativesFact? Alternatives { get; init; }
    public RobustFact? Robust { get; init; }
    public required UnknownFact Unknown { get; init; }
    public required ConfidenceFact Confidence { get; init; }

    public List<object?> ToJson()
    {
        var facts = new List<object?> { Immediate.ToJson(), Exposure.ToJson() };
        if (Reply is not null) facts.Add(Reply.ToJson());
        facts.Add(Outcome.ToJson());
        if (Alternatives is not null) facts.Add(Alternatives.ToJson());
        if (Robust is not null) facts.Add(Robust.ToJson());
        facts.Add(Unknown.ToJson());
        facts.Add(Confidence.ToJson());
        return facts;
    }
}
