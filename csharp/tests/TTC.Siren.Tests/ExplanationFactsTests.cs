using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Explain;
using TTC.Siren.Search;
using TTC.Siren.Server;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

/// <summary>One decision of the corpus: the request, the seed, and everything the server produced for it with <c>explain = full</c>.</summary>
internal sealed record Decision(int Index, string Source, ulong Seed, string Tags, AiMoveOutcome Outcome)
{
    public ExplanationFacts Facts => Outcome.Facts!;
    public ExplanationText Text => Outcome.Text!;
    public SolveResult Solve => Outcome.Solve!;
    public SolveRequest Request => Outcome.Request!;
}

/// <summary>
/// Explanation facts v2 (A-24): every family is RECOMPUTED here, by code that does not share the explainer's logic — the engine's own
/// <c>MakeMove</c> for flips and captures, a brute-force minimax over the engine's state for exact outcomes, fixed-depth searches for
/// scores, the engine's candidate table for the robust endgame — on the decisions of the end-to-end corpus (explain = full).
/// </summary>
public sealed class ExplanationFactsTests
{
    private readonly ITestOutputHelper _output;

    public ExplanationFactsTests(ITestOutputHelper output) => _output = output;

    internal static readonly Lazy<List<Decision>> All = new(LoadDecisions);

    private static List<Decision> LoadDecisions()
    {
        string path = AiMoveParityTests.EnsureGolden();
        var service = AiMoveParityTests.CreateService(workers: 2);
        var list = new List<Decision>();
        int n = 0;
        foreach (var doc in Golden8.ReadRecords(path))
        {
            using (doc)
            {
                var rec = doc.RootElement;
                if (rec.GetProperty("status").GetInt32() != 200 || !rec.GetProperty("response").TryGetProperty("pos", out _)) continue;
                var robust = rec.GetProperty("response").GetProperty("engine_info").GetProperty("robust").GetBoolean();
                // every robust decision, and about a third of the others (keeps the run short; the full corpus is covered by the parity test)
                if (!robust && n++ % 3 != 0) continue;
                var node = JsonNode.Parse(rec.GetProperty("request").GetRawText())!.AsObject();
                node["explain"] = "full";
                using var request = JsonDocument.Parse(node.ToJsonString());
                var outcome = service.HandleAsync(request.RootElement, rec.GetProperty("seed").GetUInt64(), CancellationToken.None).GetAwaiter().GetResult();
                Assert.Equal(200, outcome.Status);
                Assert.True(outcome.Facts is not null, $"{rec.GetProperty("src")} #{rec.GetProperty("i")}: {outcome.Body.GetValueOrDefault("explanation_error")}");
                list.Add(new Decision(rec.GetProperty("i").GetInt32(), rec.GetProperty("src").GetString()!, rec.GetProperty("seed").GetUInt64(), rec.GetProperty("tags").ToString(), outcome));
            }
        }
        return list;
    }

    internal static string Where(Decision d) => $"{d.Source} request #{d.Index} (seed {d.Seed}) {d.Tags}";

    // =============================================================================================================== family 1

    [Fact]
    public void Immediate_effect_equals_what_the_engine_does_when_the_move_is_made()
    {
        int flipsChecked = 0, modifierChanges = 0;
        foreach (var d in All.Value)
        {
            var im = d.Facts.Immediate;
            var g = GameState.FromSnapshot(d.Request.State);
            int me = g.CurrentPlayer;
            g.CountCards(out int r0, out int b0);
            g.MakeMove(new Move(g.HandInstance(me, im.Move.Slot), im.Move.Row * 3 + im.Move.Col));
            g.CountCards(out int r1, out int b1);

            int mask = 0;
            foreach (var f in im.Flips) mask |= 1 << (f.Row * 3 + f.Col);
            Assert.True(mask == g.LastFlipMask, $"{Where(d)}: flipped cells {mask:X3} engine {g.LastFlipMask:X3}");
            Assert.Equal(BitOperations.PopCount((uint)mask), im.Flips.Count);
            Assert.Equal(me == 0 ? r0 : b0, im.Before.Mine);
            Assert.Equal(me == 0 ? b0 : r0, im.Before.Theirs);
            Assert.Equal(me == 0 ? r1 : b1, im.After.Mine);
            Assert.Equal(me == 0 ? b1 : r1, im.After.Theirs);
            // the move gains exactly the flipped cards (the placed card was already counted in the hand)
            Assert.Equal(im.Before.Mine + im.Flips.Count, im.After.Mine);
            Assert.Equal(im.Before.Theirs - im.Flips.Count, im.After.Theirs);
            // every flip names a card that really was the opponent's
            foreach (var f in im.Flips)
            {
                int inst = g.BoardInstance(f.Row * 3 + f.Col);
                Assert.Equal(f.CardId, g.InstanceId(inst));
                Assert.Equal(me, (int)g.InstanceOwner(inst));
            }
            flipsChecked += im.Flips.Count;
            modifierChanges += im.Modifiers.Count;
        }
        Assert.True(flipsChecked > 300, $"only {flipsChecked} flips");
        _output.WriteLine($"{All.Value.Count} decisions, {flipsChecked} flips and {modifierChanges} modifier changes recomputed with the engine's MakeMove");
    }

    // =============================================================================================================== family 2

    private static bool CapturesWithEngine(GameState afterMyMove, int opp, int slot, int cell, int placedCell, out int flipCount)
    {
        var g = GameState.FromSnapshot(afterMyMove.ToSnapshot()); // a copy: instance numbers change, so the card is found again by its slot
        g.MakeMove(new Move(g.HandInstance(opp, slot), cell));
        flipCount = BitOperations.PopCount(g.LastFlipMask);
        return (g.LastFlipMask >> placedCell & 1) != 0;
    }

    [Fact]
    public void Exposure_threats_are_exactly_the_opponent_cards_that_take_the_placed_card_next_turn()
    {
        int threats = 0, checkedDecisions = 0, quiet = 0;
        foreach (var d in All.Value)
        {
            var e = d.Facts.Exposure;
            var im = d.Facts.Immediate;
            var g = GameState.FromSnapshot(d.Request.State);
            int me = g.CurrentPlayer, opp = 1 - me;
            int placed = im.Move.Row * 3 + im.Move.Col;
            g.MakeMove(new Move(g.HandInstance(me, im.Move.Slot), placed));

            // brute force with the engine: every opponent card, every empty cell
            var expected = new Dictionary<int, (int Cell, int Captured)>(); // by slot
            for (int slot = 0; slot < g.HandCount(opp); slot++)
            {
                int inst = g.HandInstance(opp, slot);
                if ((g.Rules & (RuleSet.Order | RuleSet.Chaos)) != 0 && !g.InstanceCanUse(inst)) continue;
                for (int c = 0; c < 9; c++)
                {
                    if (!g.IsEmpty(c)) continue;
                    if (CapturesWithEngine(g, opp, slot, c, placed, out int flips) && (!expected.TryGetValue(slot, out var best) || flips > best.Captured))
                        expected[slot] = (c, flips);
                }
            }
            Assert.True(expected.Count == e.Threats.Count, $"{Where(d)}: {e.Threats.Count} threats in the facts, {expected.Count} by brute force");
            var byCard = e.Threats.OrderBy(t => t.CardId).ThenBy(t => t.Row * 3 + t.Col).Select(t => (t.CardId, t.Row * 3 + t.Col, t.CapturedCards)).ToList();
            var brute = expected.Select(p => (EndgameScenarios.BaseCardId(g.InstanceId(g.HandInstance(opp, p.Key))), p.Value.Cell, p.Value.Captured))
                .OrderBy(t => t.Item1).ThenBy(t => t.Item2).ToList();
            Assert.True(byCard.SequenceEqual(brute), $"{Where(d)}: threats\n  facts {string.Join(' ', byCard)}\n  brute {string.Join(' ', brute)}");
            threats += e.Threats.Count;
            if (e.Threats.Count == 0) quiet++;

            // the sides: neighbour kinds and values of the placed card as the engine holds them
            int inst0 = g.BoardInstance(placed);
            for (int s = 0; s < 4; s++) Assert.Equal(g.InstanceEffective(inst0, s), e.Sides[s].Value);
            checkedDecisions++;
        }
        Assert.True(threats > 100 && quiet > 100, $"{threats} threats, {quiet} quiet decisions");
        _output.WriteLine($"{checkedDecisions} decisions: {threats} threats confirmed one by one with MakeMove, {quiet} decisions without any");
    }

    [Fact]
    public void The_capture_probability_of_unknown_cards_equals_a_brute_force_over_the_legal_candidates()
    {
        int compared = 0;
        foreach (var d in All.Value.Where(x => x.Facts.Exposure.UnknownCaptureProbability is not null).Take(60))
        {
            var e = d.Facts.Exposure;
            var im = d.Facts.Immediate;
            var snapshot = d.Request.State;
            var g0 = GameState.FromSnapshot(snapshot);
            int me = g0.CurrentPlayer, opp = 1 - me;
            int placed = im.Move.Row * 3 + im.Move.Col;
            var generated = d.Outcome.Generated!;
            var npc = d.Outcome.Parsed!.NpcIds.Where(i => i <= int.MaxValue).Select(i => (int)i).ToList();
            var candidates = EndgameScenarios.LegalCandidates(AiMoveParityTests.CreateService().Catalog.Db, g0, opp, generated, npc);
            Assert.Equal(candidates.Count, e.UnknownCandidates);

            int slotOfUnknown = Array.IndexOf(generated[opp], true);
            int capturing = 0;
            foreach (var cand in candidates)
            {
                // put the candidate into the first unknown slot of the opponent's hand, then let the engine play every empty cell
                var s = GameState.FromSnapshot(snapshot);
                s.MakeMove(new Move(s.HandInstance(me, im.Move.Slot), placed));
                var snap = s.ToSnapshot();
                var card = snap.Hands[opp][slotOfUnknown];
                int mod = 0;
                if (cand.TypeId != 0 && (s.Rules & (RuleSet.Ascension | RuleSet.Descension)) != 0)
                {
                    int onBoard = Enumerable.Range(0, 9).Count(c => snap.Board[c] is { } b && b.Type == CardTypes.NameOf(cand.TypeId));
                    mod = (s.Rules & RuleSet.Ascension) != 0 ? onBoard : -onBoard;
                }
                card.Id = cand.Id;
                card.Type = CardTypes.NameOf(cand.TypeId);
                card.Mod = mod;
                var edges = new[] { cand.Up, cand.Right, cand.Down, cand.Left };
                for (int k = 0; k < 4; k++)
                {
                    card.Base[k] = edges[k];
                    card.Display[k] = (byte)Math.Max(1, Math.Min(10, edges[k] + mod));
                }
                var withCandidate = GameState.FromSnapshot(snap); // the slot keeps its own can_use flag
                bool any = false;
                bool usable = (s.Rules & (RuleSet.Order | RuleSet.Chaos)) == 0 || card.CanUse;
                for (int c = 0; c < 9 && !any && usable; c++)
                    if (withCandidate.IsEmpty(c)) any = CapturesWithEngine(withCandidate, opp, slotOfUnknown, c, placed, out _);
                if (any) capturing++;
            }
            double expected = candidates.Count == 0 ? double.NaN : (double)capturing / candidates.Count;
            Assert.True(candidates.Count > 0 && e.UnknownCaptureProbability!.Value == expected, $"{Where(d)}: probability facts {e.UnknownCaptureProbability} brute force {expected}");
            compared++;
        }
        Assert.True(compared >= 20, $"only {compared} decisions with unknown cards");
    }

    // =============================================================================================================== families 3 and 4

    /// <summary>Exact value of the position for <paramref name="ai"/> by plain minimax over the engine's state: cards of the AI side minus the opponent's at the end.</summary>
    private static int BruteForce(GameState g, int ai)
    {
        if (g.AvailableCellCount == 0)
        {
            g.CountCards(out int red, out int blue);
            return ai == 0 ? red - blue : blue - red;
        }
        var buf = new Move[GameState.MaxMoves];
        int n = g.GenerateMoves(buf);
        if (n == 0) // Python has no pass rule: the position is scored as it stands
        {
            g.CountCards(out int r, out int b);
            return ai == 0 ? r - b : b - r;
        }
        bool maximize = g.CurrentPlayer == ai;
        int best = maximize ? int.MinValue : int.MaxValue;
        for (int i = 0; i < n; i++)
        {
            g.MakeMove(buf[i]);
            int v = BruteForce(g, ai);
            g.UndoMove();
            best = maximize ? Math.Max(best, v) : Math.Min(best, v);
        }
        return best;
    }

    [Fact]
    public void Exact_outcomes_equal_a_brute_force_minimax_and_say_win_draw_loss_consistently()
    {
        int exact = 0, win = 0, draw = 0, loss = 0;
        foreach (var d in All.Value)
        {
            var o = d.Facts.Outcome;
            var g = GameState.FromSnapshot(d.Request.State);
            int me = g.CurrentPlayer;
            bool shouldBeExact = g.AvailableCellCount <= 6;
            Assert.True(shouldBeExact == (o.Kind == "exact"), $"{Where(d)}: kind {o.Kind} with {g.AvailableCellCount} empty cells");
            if (o.Kind == "exact")
            {
                Assert.Equal(g.AvailableCellCount, o.EmptyCells);
                Assert.Equal(o.Difference > 0 ? "win" : o.Difference == 0 ? "draw" : "loss", o.Result);
                if (g.AvailableCellCount <= 5) // brute force on the 5-cell positions (the exact solver's own memoised search covers 6)
                {
                    var im = d.Facts.Immediate;
                    g.MakeMove(new Move(g.HandInstance(me, im.Move.Slot), im.Move.Row * 3 + im.Move.Col));
                    int value = BruteForce(g, me);
                    Assert.True(value == o.Difference, $"{Where(d)}: exact difference facts {o.Difference} brute force {value}");
                }
                Assert.Equal(d.Facts.Unknown.OpponentUnknown + d.Facts.Unknown.OwnUnknown > 0 ? "assumed_hands" : "known_hands", o.Basis);
                exact++;
                if (o.Result == "win") win++; else if (o.Result == "draw") draw++; else loss++;
            }
            else
            {
                Assert.Null(o.Difference);
                Assert.NotNull(o.Score);
                Assert.Equal(o.Score > 0.25 ? "ahead" : o.Score < -0.25 ? "behind" : "level", o.Result);
                Assert.Equal(d.Solve.DepthCompleted, o.Depth);
            }
            // the line starts with the chosen move and alternates sides
            Assert.Equal(d.Facts.Immediate.Move.CardId, o.Pv[0].CardId);
            for (int i = 0; i < o.Pv.Count; i++) Assert.Equal(i % 2 == 0 ? "self" : "opponent", o.Pv[i].Side);
        }
        Assert.True(exact > 100 && win > 10 && loss > 5, $"exact {exact}: win {win} draw {draw} loss {loss}");
        _output.WriteLine($"{exact} exact outcomes (win {win}, draw {draw}, loss {loss}), the rest heuristic");
    }

    [Fact]
    public void The_principal_variation_is_a_legal_line_and_the_reply_is_its_second_move()
    {
        int lines = 0, plies = 0, replies = 0, searched = 0;
        foreach (var d in All.Value)
        {
            var o = d.Facts.Outcome;
            var g = GameState.FromSnapshot(d.Request.State);
            for (int i = 0; i < o.Pv.Count; i++)
            {
                var p = o.Pv[i];
                int player = g.CurrentPlayer;
                int cell = p.Row * 3 + p.Col;
                Assert.True(g.IsEmpty(cell), $"{Where(d)}: ply {i} cell {cell} is not empty");
                int inst = -1;
                for (int slot = 0; slot < g.HandCount(player); slot++)
                {
                    int cand = g.HandInstance(player, slot);
                    if (EndgameScenarios.BaseCardId(g.InstanceId(cand)) == p.CardId && g.InstanceCanUse(cand) == g.InstanceCanUse(cand)) { inst = cand; break; }
                }
                Assert.True(inst >= 0, $"{Where(d)}: ply {i} card {p.CardId} is not in the hand of the side to move");
                g.MakeMove(new Move(inst, cell));
                Assert.Equal(BitOperations.PopCount(g.LastFlipMask), p.Flips);
                plies++;
            }
            if (d.Facts.Reply is { } reply)
            {
                Assert.Equal(o.Pv[1].CardId, reply.CardId);
                Assert.Equal(o.Pv[1].Row, reply.Row);
                Assert.Equal(o.Pv[1].Col, reply.Col);
                replies++;
                if (reply.Source == "line_search") searched++;
            }
            lines++;
        }
        Assert.True(plies > 400 && replies > 150, $"{plies} plies, {replies} replies");
        _output.WriteLine($"{lines} lines, {plies} plies replayed legally with MakeMove, {replies} replies ({searched} from a fresh line search, the rest from the search table)");
    }

    // =============================================================================================================== family 5

    [Fact]
    public void Alternative_scores_are_the_root_values_of_a_fixed_depth_search_ties_are_ties()
    {
        int decisions = 0, alternatives = 0, ties = 0;
        foreach (var d in All.Value)
        {
            var a = d.Facts.Alternatives;
            if (a is null) continue;
            // the best of all root scores at the evaluated depth is the root score of a fixed-depth search at that depth
            var g = GameState.FromSnapshot(d.Request.State);
            var searcher = new FastSearcher(g, AiMoveParityTests.CreateService().Catalog.Db, new FastTranspositionTable(1 << 16));
            var root = searcher.RunFixedDepth(a.EvalDepth, d.Outcome.Parsed!.Aggressive, SearchLimits.Unlimited);
            double best = Math.Max(a.ChosenScore, a.Items.Count == 0 ? double.NegativeInfinity : a.Items.Max(i => i.Score));
            if (a.MovesScored <= a.Items.Count + 1) // every move is listed: the maximum is the root value
                Assert.True(best == root.Score, $"{Where(d)}: best alternative score {best} root value {root.Score} at depth {a.EvalDepth}");
            else
                Assert.True(best <= root.Score, $"{Where(d)}: listed score {best} above the root value {root.Score}");
            if (a.ChosenIsSearchBest && a.EvalDepth == d.Solve.DepthCompleted)
                Assert.True(a.ChosenScore == d.Solve.StandardScore, $"{Where(d)}: chosen score {a.ChosenScore} engine {d.Solve.StandardScore}");

            double previous = double.PositiveInfinity;
            foreach (var item in a.Items)
            {
                Assert.True(item.Score <= previous, $"{Where(d)}: alternatives not in descending order of score");
                previous = item.Score;
                Assert.Equal(item.Score == a.ChosenScore, item.Tied);
                Assert.Equal(item.Score - a.ChosenScore == 0 ? 0 : item.Score - a.ChosenScore, item.ScoreDelta);
                if (item.Tied) ties++;
                alternatives++;
            }
            Assert.True(a.Items.Count <= 4);
            decisions++;
        }
        Assert.True(decisions > 100 && alternatives > 300, $"{decisions} decisions, {alternatives} alternatives");
        _output.WriteLine($"{decisions} decisions with alternatives: {alternatives} scores checked against fixed-depth root values, {ties} exact ties");
    }

    // =============================================================================================================== family 6

    [Fact]
    public void Robust_facts_reproduce_the_engines_candidate_table_and_the_override_clause()
    {
        int robust = 0, overridden = 0;
        foreach (var d in All.Value.Where(x => x.Facts.Robust is not null))
        {
            var r = d.Facts.Robust!;
            var table = d.Solve.Robust!;
            var chosen = d.Facts.Immediate.Move;
            var candidate = table.Candidates.Last(c => c.Move.Slot == chosen.Slot && c.Move.Row == chosen.Row && c.Move.Col == chosen.Col);
            Assert.Equal(table.Scenarios, r.Scenarios);
            Assert.Equal(table.Overridden, r.Overridden);
            Assert.True(candidate.SafetyRatio == r.SafetyRatio, $"{Where(d)}: safety ratio");
            Assert.True(candidate.FinalScore == r.FinalScore, $"{Where(d)}: final score");

            // the final score is a function of average, worst, safety and corner risk: recompute it from the recomputed average and worst values
            bool aggressive = d.Outcome.Parsed!.Aggressive;
            if (r.Average is null)
            {
                // the engine plays the side whose own hand holds the assumed cards: the scenarios replace that very card, no value exists, the
                // engine's table says so too (final score minus infinity)
                Assert.True(double.IsNegativeInfinity(candidate.FinalScore) && d.Facts.Immediate.Move.Assumed, $"{Where(d)}: no recomputed value but the table has {candidate.FinalScore}");
                continue;
            }
            double avg = r.Average!.Value, worst = r.Worst!.Value;
            double robustScore = aggressive ? avg * 0.75 + worst * 0.25 : avg * 0.45 + worst * 0.55;
            double safetyBonus = aggressive ? r.SafetyRatio * 0.75 : r.SafetyRatio * 2.0;
            double cornerPenalty = aggressive ? candidate.CornerRisk * 0.05 : candidate.CornerRisk * 0.15;
            Assert.True(BitConverter.DoubleToInt64Bits(robustScore + safetyBonus - cornerPenalty) == BitConverter.DoubleToInt64Bits(candidate.FinalScore),
                $"{Where(d)}: final score {candidate.FinalScore} recomputed {robustScore + safetyBonus - cornerPenalty}");
            Assert.True(BitConverter.DoubleToInt64Bits(robustScore) == BitConverter.DoubleToInt64Bits(candidate.RobustScore), $"{Where(d)}: robust score");

            if (r.Overridden)
            {
                var std = table.Candidates.Last(c => c.Move.Slot == r.StandardMove!.Slot && c.Move.Row == r.StandardMove.Row && c.Move.Col == r.StandardMove.Col);
                switch (r.OverrideRule)
                {
                    case "aggressive_margin": Assert.True(aggressive && candidate.FinalScore > std.FinalScore + 2.5 && candidate.SafetyRatio >= 0.75); break;
                    case "safety_floor": Assert.True(!aggressive && std.SafetyRatio < 0.5 && candidate.SafetyRatio >= 0.5); break;
                    case "safer_and_close":
                        Assert.True(!aggressive && !(std.SafetyRatio < 0.5 && candidate.SafetyRatio >= 0.5) && candidate.SafetyRatio > std.SafetyRatio && candidate.FinalScore >= std.FinalScore - 1.0);
                        break;
                    case "clearly_better":
                        Assert.True(!aggressive && !(std.SafetyRatio < 0.5 && candidate.SafetyRatio >= 0.5) &&
                                    !(candidate.SafetyRatio > std.SafetyRatio && candidate.FinalScore >= std.FinalScore - 1.0) && candidate.FinalScore > std.FinalScore + 0.05);
                        break;
                    default: Assert.Fail($"{Where(d)}: unknown override rule {r.OverrideRule}");
                        break;
                }
                overridden++;
            }
            robust++;
        }
        Assert.True(robust > 100 && overridden > 20, $"{robust} robust decisions, {overridden} overridden");
        _output.WriteLine($"{robust} robust decisions recomputed bit for bit against the engine's table; {overridden} overrides, each with the clause that fired");
    }

    // =============================================================================================================== families 7 and 8

    [Fact]
    public void Unknown_and_confidence_facts_follow_the_request_and_the_search()
    {
        int unknownDecisions = 0;
        var strategies = new SortedDictionary<string, int>();
        foreach (var d in All.Value)
        {
            var u = d.Facts.Unknown;
            var c = d.Facts.Confidence;
            var g = GameState.FromSnapshot(d.Request.State);
            int opp = 1 - g.CurrentPlayer;
            Assert.Equal(d.Outcome.Generated![opp].Count(x => x), u.OpponentUnknown);
            Assert.Equal(d.Outcome.Generated![1 - opp].Count(x => x), u.OwnUnknown);
            if (u.OpponentUnknown == 0) Assert.Equal("none", u.Strategy);
            else strategies[u.Strategy] = strategies.GetValueOrDefault(u.Strategy) + 1;
            foreach (var a in u.Assumed)
            {
                int inst = g.HandInstance(opp, a.Slot);
                Assert.Equal(EndgameScenarios.BaseCardId(g.InstanceId(inst)), a.CardId);
            }
            if (u.OpponentUnknown > 0) unknownDecisions++;

            Assert.Equal(d.Solve.DepthCompleted, c.DepthReached);
            Assert.Equal(d.Request.MaxDepth, c.DepthRequested);
            Assert.Equal(g.AvailableCellCount, c.EmptyCells);
            Assert.Equal(d.Solve.Nodes, c.Nodes);
            Assert.Equal(d.Solve.DepthCompleted >= g.AvailableCellCount, c.SolvedToEnd);
            Assert.Equal(d.Solve.Robust is not null, c.RobustUsed);
        }
        Assert.True(unknownDecisions > 50, $"only {unknownDecisions} decisions with unknown opponent cards");
        _output.WriteLine("assumption strategies: " + string.Join(", ", strategies.Select(p => $"{p.Key} {p.Value}")));
    }

    // =============================================================================================================== text

    [Fact]
    public void Text_says_only_what_the_facts_say_every_number_and_every_win_claim_comes_from_a_fact()
    {
        var number = new Regex(@"-?\d+(\.\d+)?", RegexOptions.Compiled);
        int checkedNumbers = 0;
        foreach (var d in All.Value)
        {
            var text = d.Text;
            var numbers = FactNumbers(JsonSerializer.SerializeToElement(d.Outcome.Body["explanation"], new JsonSerializerOptions()).GetProperty("facts"));
            numbers.UnionWith(FactNumbers(JsonDocument.Parse(PyJson.Serialize(d.Facts.ToJson())).RootElement));
            foreach (var (kind, content) in new[] { ("brief", text.Brief), ("full", text.Full) })
                foreach (Match m in number.Matches(Regex.Replace(content, "「[^」]*」", ""))) // an NPC name in quotes is a label, not a number
                {
                    double value = double.Parse(m.Value, System.Globalization.CultureInfo.InvariantCulture);
                    int decimals = m.Groups[1].Success ? m.Groups[1].Value.Length - 1 : 0;
                    double magnitude = Math.Abs(value);
                    bool found = numbers.Any(f => Math.Round(Math.Abs(f), decimals, MidpointRounding.AwayFromZero) == magnitude ||
                                                  Math.Round(Math.Abs(f), decimals, MidpointRounding.ToEven) == magnitude ||
                                                  Math.Round(f * 100, 0, MidpointRounding.AwayFromZero) == magnitude);
                    Assert.True(found, $"{Where(d)}: the {kind} text mentions {m.Value}, which is in none of the facts:\n{content}");
                    checkedNumbers++;
                }

            // no claim against the facts
            var o = d.Facts.Outcome;
            if (o.Kind == "exact")
            {
                if (o.Result != "win") Assert.DoesNotContain("获胜", text.Brief);
                if (o.Result == "win") Assert.DoesNotContain("局面偏不利", text.Brief);
                if (o.Result != "draw") Assert.DoesNotContain("平局", text.Brief);
            }
            else
            {
                Assert.DoesNotContain("获胜", text.Brief);
                Assert.DoesNotContain("已经算到终局", text.Brief);
            }
            Assert.Equal(d.Facts.Immediate.Flips.Count > 0, text.Brief.Contains("翻转") || text.Brief.Contains("翻回") || text.Brief.Contains("吃掉"));
            Assert.True(text.Brief.Count(ch => ch == '。') is >= 1 and <= 2, $"{Where(d)}: brief has more than two sentences: {text.Brief}");
        }
        Assert.True(checkedNumbers > 3000, $"only {checkedNumbers} numbers checked");
        _output.WriteLine($"{checkedNumbers} numbers in {All.Value.Count} brief and full texts, each found in the facts");
    }

    private static HashSet<double> FactNumbers(JsonElement e)
    {
        var set = new HashSet<double>();
        void Walk(JsonElement x)
        {
            switch (x.ValueKind)
            {
                case JsonValueKind.Number: set.Add(x.GetDouble()); break;
                case JsonValueKind.Array: foreach (var i in x.EnumerateArray()) Walk(i); break;
                case JsonValueKind.Object: foreach (var p in x.EnumerateObject()) Walk(p.Value); break;
            }
        }
        Walk(e);
        return set;
    }

    [Fact]
    public void The_same_facts_always_give_the_same_text_and_the_variants_differ_between_positions()
    {
        var templates = TemplateSet.Default();
        var first = All.Value.Select(d => ExplanationRenderer.Render(d.Facts, templates)).ToList();
        var second = All.Value.Select(d => ExplanationRenderer.Render(d.Facts, TemplateSet.Default())).ToList();
        Assert.True(first.SequenceEqual(second));
        Assert.True(first.Select(t => t.Brief).Distinct().Count() > 50);
        // every template key the renderer can use exists and has at least one variant (a missing key fails at render time with its name)
        Assert.True(templates.Templates.Count > 60);
        Assert.Equal("zh-CN", templates.Language);
    }

    // =============================================================================================================== snapshots

    private static string SnapshotPath => Path.Combine(GoldenFiles.RepoRoot, "tests", "data", "explanation_snapshots_v2.json");

    /// <summary>
    /// Golden snapshots of the generated text: 30 decisions of the corpus with the exact brief and full text. The text is a pure function of the
    /// facts and the template file, so any change of wording (or of a fact) shows up here. To accept an intended change, run the tests once with
    /// the environment variable <c>TTC_UPDATE_SNAPSHOTS=1</c> and review the diff of the file.
    /// </summary>
    [Fact]
    public void The_generated_text_matches_the_committed_snapshots()
    {
        var chosen = All.Value.Where((d, i) => i % (All.Value.Count / 30) == 0).Take(30).ToList();
        var actual = chosen.ToDictionary(d => $"{d.Source}#{d.Index}", d => new Dictionary<string, string> { ["brief"] = d.Text.Brief, ["full"] = d.Text.Full });
        string path = SnapshotPath;
        if (Environment.GetEnvironmentVariable("TTC_UPDATE_SNAPSHOTS") == "1")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(actual, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        }
        Assert.True(File.Exists(path), "snapshot file missing: run once with TTC_UPDATE_SNAPSHOTS=1");
        var expected = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(File.ReadAllText(path))!;
        Assert.Equal(expected.Keys.OrderBy(k => k), actual.Keys.OrderBy(k => k));
        foreach (var (key, texts) in actual)
        {
            Assert.True(expected[key]["brief"] == texts["brief"], $"{key}: brief text changed" + Environment.NewLine + $"  expected {expected[key]["brief"]}" + Environment.NewLine + $"  actual   {texts["brief"]}");
            Assert.True(expected[key]["full"] == texts["full"], $"{key}: full text changed" + Environment.NewLine + $"  expected {expected[key]["full"]}" + Environment.NewLine + $"  actual   {texts["full"]}");
        }
    }
}
