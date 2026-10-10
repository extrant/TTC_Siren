using System.Numerics;
using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Explain;

/// <summary>How the unknown cards of the side not to move were assumed (from the request parser).</summary>
/// <param name="Strategy">"none" | "npc_pool" | "rule_model" | "random_sample" | "threat_ranking".</param>
public sealed record UnknownInfo(string Strategy, string? PrimaryRule, string? NpcName, int NpcPoolSize);

/// <summary>What the robust (information-set) endgame layer ran with and returned.</summary>
public sealed class RobustInput
{
    public required int Opponent { get; init; }
    public required bool[][] Generated { get; init; }
    public required int SampleCount { get; init; }
    public required IReadOnlyList<int> NpcIds { get; init; }
    public IReadOnlyList<UnknownCandidate>? FallbackCandidates { get; init; }
    public required int Scenarios { get; init; }
    public required bool Overridden { get; init; }

    /// <summary>The candidate table: (hand slot, row, col, card id, final score, safety ratio, robust score, corner risk) in the order the layer returned it.</summary>
    public required IReadOnlyList<RobustCandidate> Candidates { get; init; }

    /// <summary>The move the layer itself preferred (it may be the standard one), by slot/row/col.</summary>
    public (int Slot, int Row, int Col)? RobustMove { get; init; }
}

public readonly record struct RobustCandidate(int Slot, int Row, int Col, int CardId, double FinalScore, double SafetyRatio, double RobustScore, double CornerRisk);

/// <summary>Everything the explainer needs; built by the server from the parsed request and the engine result.</summary>
public sealed class ExplainInput
{
    public required CardDb Db { get; init; }

    /// <summary>The request state before the move, hands in the order of the request.</summary>
    public required StateSnapshot State { get; init; }

    public required int ChosenSlot { get; init; }
    public required int Row { get; init; }
    public required int Col { get; init; }
    public required bool Aggressive { get; init; }

    /// <summary>Compute the alternatives (full explanation); false = brief.</summary>
    public required bool Full { get; init; }

    public required int DepthCompleted { get; init; }
    public required int DepthRequested { get; init; }
    public required long Nodes { get; init; }
    public required double SearchMs { get; init; }
    public required double StandardScore { get; init; }

    /// <summary>The move of the standard search (slot, row, col); null when the search found none.</summary>
    public (int Slot, int Row, int Col)? StandardMove { get; init; }

    /// <summary>The worker's table right after the standard search (valid only while it is not reused). Null: the line is searched anew.</summary>
    public FastTranspositionTable? Tt { get; init; }

    public RobustInput? Robust { get; init; }
    public required UnknownInfo Unknown { get; init; }

    /// <summary>The NPC card pool of the request (database ids), empty when there is none.</summary>
    public IReadOnlyList<int> NpcIds { get; init; } = Array.Empty<int>();

    /// <summary>The "generated" flag of every hand card, per player, in request order (an unknown slot filled with an assumed card).</summary>
    public required bool[][] Generated { get; init; }
}

/// <summary>
/// Computes the explanation facts v2 of one decision (decision A-24). Every fact is derived from the position by the engine's own rules and
/// searches; none comes from text. The facts are verified by recomputation in ExplainTests (flip causes against <c>MakeMove</c>, scores
/// against fixed-depth searches, the exact outcome against the exact solver, the robust table against the engine's table).
/// All searches here use node budgets, never the wall clock, so the facts are reproducible from the same input.
/// </summary>
public static class Explainer
{
    private const int AlternativeNodeBudget = 400_000;
    private const int LineNodeBudget = 80_000;
    private const int MaxAlternatives = 4;
    private const int MaxPvPlies = 6;
    private const int ExactMaxEmptyCells = 6;
    private const double LevelThreshold = 0.25;

    public static ExplanationFacts Build(ExplainInput inp)
    {
        var db = inp.Db;
        var g0 = GameState.FromSnapshot(inp.State);
        int me = g0.CurrentPlayer, opp = 1 - me;
        int cell = inp.Row * 3 + inp.Col;
        int chosenInst = g0.HandInstance(me, inp.ChosenSlot);
        var chosen = new Move(chosenInst, cell);
        int empties0 = g0.AvailableCellCount;
        string[] activeRules = RuleParser.ToPythonList(g0.Rules).ToArray();

        // ---- family 1: immediate effect
        var chosenFact = MoveOf(g0, inp.Generated, me, inp.ChosenSlot, chosenInst, cell);
        var view0 = FlipAnalyzer.ViewOf(g0);
        var flips = FlipAnalyzer.Analyze(view0, cell, FlipAnalyzer.ViewOfInstance(g0, chosenInst), (byte)me, g0.Rules);
        var before = Counts(g0, me);
        var modsBefore = new int[g0.InstanceCount];
        for (int i = 0; i < modsBefore.Length; i++) modsBefore[i] = g0.InstanceModifier(i);
        double placementEdgeScore = new Evaluator(g0, db).PlacementEdgeScore(chosenInst, inp.Row, inp.Col);

        var g1 = GameState.FromSnapshot(inp.State);
        g1.MakeMove(new Move(g1.HandInstance(me, inp.ChosenSlot), cell));
        ushort engineMask = g1.LastFlipMask;
        int analyzerMask = 0;
        foreach (var f in flips) analyzerMask |= 1 << f.Cell;
        if (analyzerMask != engineMask)
            throw new InvalidOperationException($"flip analysis disagrees with the engine: analyzer {analyzerMask:X3}, engine {engineMask:X3}");
        var after = Counts(g1, me);

        var modifiers = new List<ModifierChange>();
        var changedByType = new Dictionary<byte, List<int>>();
        for (int i = 0; i < modsBefore.Length; i++)
            if (g1.InstanceModifier(i) != modsBefore[i] && g1.InstanceType(i) != 0)
            {
                if (!changedByType.TryGetValue(g1.InstanceType(i), out var list)) changedByType[g1.InstanceType(i)] = list = new List<int>();
                list.Add(i);
            }
        foreach (var (typeId, insts) in changedByType.OrderBy(p => p.Key))
        {
            int onBoard = insts.Count(i => Enumerable.Range(0, 9).Any(c => g1.BoardInstance(c) == i));
            modifiers.Add(new ModifierChange(CardTypes.NameOf(typeId)!, modsBefore[insts[0]], g1.InstanceModifier(insts[0]), onBoard, insts.Count - onBoard));
        }
        var immediate = new ImmediateEffectFact(chosenFact, flips.Select(FlipFact.From).ToList(), before, after, modifiers, activeRules.ToList());

        // ---- family 2: exposure of the placed card (position after the move: g1)
        var exposure = BuildExposure(inp, g0, g1, me, opp, cell, chosenInst, placementEdgeScore);

        // ---- families 3 and 4: the line, the opponent's reply, the outcome
        bool chosenIsStandard = inp.StandardMove is { } sm && sm.Slot == inp.ChosenSlot && sm.Row == inp.Row && sm.Col == inp.Col;
        var (pv, pvSource) = ExtractLine(inp, chosenIsStandard, me, cell, chosenInst);
        ReplyFact? reply = null;
        if (pv.Count > 1) reply = ReplyOf(inp, g1, pv[1], opp, cell, pvSource);

        double? chosenScore = chosenIsStandard ? inp.StandardScore : null;
        var outcome = BuildOutcome(inp, g1, me, empties0, pv, pvSource, chosenIsStandard, ref chosenScore);

        // ---- family 5: alternatives (full only)
        AlternativesFact? alternatives = null;
        if (inp.Full) alternatives = BuildAlternatives(inp, me, chosen, immediate, reply, chosenIsStandard, chosenScore);

        // ---- family 6: robust endgame
        RobustFact? robust = inp.Robust is not null ? BuildRobust(inp, me) : null;

        // ---- families 7 and 8
        var unknown = BuildUnknown(inp, g0, opp);
        var confidence = new ConfidenceFact(inp.DepthCompleted >= empties0, inp.DepthCompleted, inp.DepthRequested, empties0, inp.Nodes, inp.SearchMs,
            inp.DepthCompleted < Math.Min(inp.DepthRequested, empties0), inp.Robust is not null);

        return new ExplanationFacts
        {
            Immediate = immediate, Exposure = exposure, Reply = reply, Outcome = outcome, Alternatives = alternatives, Robust = robust,
            Unknown = unknown, Confidence = confidence,
        };
    }

    // =============================================================================================================== helpers

    private static MoveFact MoveOf(GameState g, bool[][] generated, int player, int slot, int inst, int cell) =>
        new(EndgameScenarios.BaseCardId(g.InstanceId(inst)), cell / 3, cell % 3, slot,
            new[] { g.InstanceEffective(inst, 0), g.InstanceEffective(inst, 1), g.InstanceEffective(inst, 2), g.InstanceEffective(inst, 3) },
            CardTypes.NameOf(g.InstanceType(inst)), slot < generated[player].Length && generated[player][slot]);

    private static CardCounts Counts(GameState g, int me)
    {
        g.CountCards(out int red, out int blue);
        return me == 0 ? new CardCounts(red, blue) : new CardCounts(blue, red);
    }

    /// <summary>The weight the evaluation gives a card on this cell: corners 1.5, centre 1.2, edges 1.0.</summary>
    private static double PositionWeight(int cell) => cell is 0 or 2 or 6 or 8 ? 1.5 : cell == 4 ? 1.2 : 1.0;

    // =============================================================================================================== exposure

    private static ExposureFact BuildExposure(ExplainInput inp, GameState g0, GameState g1, int me, int opp, int cell, int chosenInst, double placementEdgeScore)
    {
        var sides = new List<SideFact>();
        int row = cell / 3, col = cell % 3;
        int[] nr = { row - 1, row, row + 1, row };
        int[] nc = { col, col + 1, col, col - 1 };
        int placedInst = g1.BoardInstance(cell);
        for (int s = 0; s < 4; s++)
        {
            string neighbour;
            if (nr[s] is < 0 or > 2 || nc[s] is < 0 or > 2) neighbour = "wall";
            else
            {
                int other = g1.BoardInstance(nr[s] * 3 + nc[s]);
                neighbour = other < 0 ? "empty" : g1.InstanceOwner(other) == me ? "own" : "enemy";
            }
            sides.Add(new SideFact(Names.Side[s], neighbour, g1.InstanceEffective(placedInst, s)));
        }

        var view = FlipAnalyzer.ViewOf(g1);
        var threats = new List<ThreatFact>();
        for (int slot = 0; slot < g1.HandCount(opp); slot++)
        {
            int inst = g1.HandInstance(opp, slot);
            if (!PlayableUnderRules(g1, inst)) continue;
            bool assumed = slot < inp.Generated[opp].Length && inp.Generated[opp][slot];
            ThreatFact? best = null;
            for (int c = 0; c < 9; c++)
            {
                if (!g1.IsEmpty(c)) continue;
                var f = FlipAnalyzer.Analyze(view, c, FlipAnalyzer.ViewOfInstance(g1, inst), (byte)opp, g1.Rules);
                int hitIndex = f.FindIndex(x => x.Cell == cell);
                if (hitIndex < 0) continue; // the placed card is not captured by this move
                var hit = f[hitIndex];
                if (best is null || f.Count > best.CapturedCards)
                    best = new ThreatFact(EndgameScenarios.BaseCardId(g1.InstanceId(inst)), assumed, c / 3, c % 3, Names.Wire(hit.Cause), f.Count);
            }
            if (best is not null) threats.Add(best);
        }

        double? probability = null;
        int candidateCount = 0;
        int unknownSlots = inp.Generated[opp].Count(x => x);
        if (unknownSlots > 0)
        {
            var candidates = EndgameScenarios.LegalCandidates(inp.Db, g0, opp, inp.Generated, inp.NpcIds);
            candidateCount = candidates.Count;
            bool anyUsable = Enumerable.Range(0, g1.HandCount(opp)).Any(slot => inp.Generated[opp][slot] && g1.InstanceCanUse(g1.HandInstance(opp, slot)));
            int capturing = 0;
            if (anyUsable)
            {
                foreach (var cand in candidates)
                {
                    var candidate = CandidateView(g1, cand);
                    for (int c = 0; c < 9; c++)
                    {
                        if (!g1.IsEmpty(c)) continue;
                        if (FlipAnalyzer.Analyze(view, c, candidate, (byte)opp, g1.Rules).Any(x => x.Cell == cell))
                        {
                            capturing++;
                            break;
                        }
                    }
                }
            }
            probability = candidates.Count == 0 ? null : (double)capturing / candidates.Count;
        }

        return new ExposureFact(sides, threats, probability, candidateCount, Names.CellKind(cell), PositionWeight(cell), placementEdgeScore);
    }

    private static bool PlayableUnderRules(GameState g, int inst) =>
        (g.Rules & (RuleSet.Order | RuleSet.Chaos)) == 0 || g.InstanceCanUse(inst);

    /// <summary>A possible unknown card as it would stand in the opponent's hand: its type modifier follows the types on the board.</summary>
    private static BoardCard CandidateView(GameState g, UnknownCandidate cand)
    {
        int mod = 0;
        if (cand.TypeId != 0 && (g.Rules & (RuleSet.Ascension | RuleSet.Descension)) != 0)
        {
            int onBoard = 0;
            for (int c = 0; c < 9; c++)
            {
                int inst = g.BoardInstance(c);
                if (inst >= 0 && g.InstanceType(inst) == cand.TypeId) onBoard++;
            }
            mod = (g.Rules & RuleSet.Ascension) != 0 ? onBoard : -onBoard;
        }
        int E(int v) => Math.Max(1, Math.Min(10, v + mod));
        return new BoardCard(cand.Id, 0, cand.TypeId, E(cand.Up), E(cand.Right), E(cand.Down), E(cand.Left));
    }

    // =============================================================================================================== the line

    private static (List<PlyFact> Pv, string Source) ExtractLine(ExplainInput inp, bool chosenIsStandard, int me, int cell, int chosenInst)
    {
        var pv = new List<PlyFact>();
        var gp = GameState.FromSnapshot(inp.State);
        int empties = gp.AvailableCellCount;

        // ply 0 is the chosen move
        int slot0 = inp.ChosenSlot;
        int inst0 = gp.HandInstance(me, slot0);
        var f0 = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(gp), cell, FlipAnalyzer.ViewOfInstance(gp, inst0), (byte)me, gp.Rules);
        pv.Add(new PlyFact(0, "self", EndgameScenarios.BaseCardId(gp.InstanceId(inst0)), cell / 3, cell % 3, f0.Count, false));

        FastTranspositionTable? table = null;
        string source;
        int remaining;
        if (chosenIsStandard && inp.Tt is not null)
        {
            // the principal variation from the worker's table: the entry at the root must name the chosen move
            if (inp.Tt.TryGet(gp.ZobristHash, out var root) && root.BestInstance == inst0 && root.BestCell == cell)
            {
                table = inp.Tt;
                source = "search_table";
                remaining = inp.DepthCompleted;
                gp.MakeMove(new Move(inst0, cell));
                remaining--;
            }
            else
            {
                (table, source, remaining) = SearchAfter(inp, gp, inst0, cell, empties);
            }
        }
        else
        {
            (table, source, remaining) = SearchAfter(inp, gp, inst0, cell, empties);
        }

        for (int ply = 1; ply < MaxPvPlies && table is not null && remaining > 0 && gp.AvailableCellCount > 0; ply++)
        {
            if (!table.TryGet(gp.ZobristHash, out var e) || e.BestInstance < 0 || e.Depth != remaining) break;
            int player = gp.CurrentPlayer;
            int slot = gp.HandSlotOf(player, e.BestInstance);
            if (slot < 0 || !gp.IsEmpty(e.BestCell) || !PlayableUnderRules(gp, e.BestInstance)) break;
            var flips = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(gp), e.BestCell, FlipAnalyzer.ViewOfInstance(gp, e.BestInstance), (byte)player, gp.Rules);
            bool assumed = slot < inp.Generated[player].Length && player != me && inp.Generated[player][slot];
            pv.Add(new PlyFact(ply, player == me ? "self" : "opponent", EndgameScenarios.BaseCardId(gp.InstanceId(e.BestInstance)), e.BestCell / 3, e.BestCell % 3,
                flips.Count, assumed));
            gp.MakeMove(new Move(e.BestInstance, e.BestCell));
            remaining--;
        }
        return (pv, pv.Count > 1 ? source : "none");
    }

    /// <summary>Plays the chosen move on <paramref name="gp"/> and searches the position that follows (the opponent to move) with a node budget.</summary>
    private static (FastTranspositionTable? Table, string Source, int Remaining) SearchAfter(ExplainInput inp, GameState gp, int inst0, int cell, int empties)
    {
        gp.MakeMove(new Move(inst0, cell));
        int depth = Math.Max(1, Math.Min(Math.Max(inp.DepthCompleted - 1, 1), Math.Min(empties - 1, 6)));
        if (gp.AvailableCellCount == 0) return (null, "none", 0);
        var table = new FastTranspositionTable(1 << 16);
        var searcher = new FastSearcher(gp, inp.Db, table);
        searcher.RunFixedDepth(depth, inp.Aggressive, SearchLimits.WithNodeBudget(LineNodeBudget));
        bool complete = searcher.Stats.NodesSearched < LineNodeBudget;
        return (complete ? table : null, "line_search", depth);
    }

    private static ReplyFact ReplyOf(ExplainInput inp, GameState g1, PlyFact ply, int opp, int placedCell, string source)
    {
        int c = ply.Row * 3 + ply.Col;
        // find the opponent's card by id in its hand (the first with that base id) to analyse its effects on the position after my move
        int inst = -1;
        for (int slot = 0; slot < g1.HandCount(opp); slot++)
        {
            int candidate = g1.HandInstance(opp, slot);
            if (EndgameScenarios.BaseCardId(g1.InstanceId(candidate)) == ply.CardId) { inst = candidate; break; }
        }
        var flips = inst < 0
            ? new List<FlipRecord>()
            : FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(g1), c, FlipAnalyzer.ViewOfInstance(g1, inst), (byte)opp, g1.Rules);
        return new ReplyFact(ply.CardId, ply.Assumed, ply.Row, ply.Col, flips.Select(FlipFact.From).ToList(), flips.Any(f => f.Cell == placedCell), source);
    }

    // =============================================================================================================== outcome

    private static OutcomeFact BuildOutcome(ExplainInput inp, GameState g1, int me, int empties0, List<PlyFact> pv, string pvSource, bool chosenIsStandard,
        ref double? chosenScore)
    {
        bool assumedHands = inp.Generated.Any(h => h.Any(x => x));
        if (empties0 <= ExactMaxEmptyCells)
        {
            var gE = GameState.FromSnapshot(g1.ToSnapshot());
            int diff;
            if (gE.AvailableCellCount == 0)
            {
                gE.CountCards(out int red, out int blue);
                diff = me == 0 ? red - blue : blue - red;
            }
            else
            {
                var solver = new ExactEndgameSolver(gE, ExactEndgameSolver.CreateMemo(1 << 15));
                solver.Bind(me);
                diff = solver.Value(me);
            }
            return new OutcomeFact(pv, pv.Count, "exact", diff > 0 ? "win" : diff == 0 ? "draw" : "loss", diff, chosenScore, inp.DepthCompleted, empties0,
                assumedHands ? "assumed_hands" : "known_hands", pvSource);
        }

        if (chosenScore is null)
        {
            // an override of the search's move: the chosen move's heuristic score at the depth reached (node budget; falls back to shallower depths)
            chosenScore = ScoreMove(inp, me, inp.ChosenSlot, inp.Row * 3 + inp.Col, inp.DepthCompleted, out _);
        }
        double score = chosenScore.Value;
        string level = score > LevelThreshold ? "ahead" : score < -LevelThreshold ? "behind" : "level";
        return new OutcomeFact(pv, pv.Count, "heuristic", level, null, score, inp.DepthCompleted, empties0, null, pvSource);
    }

    /// <summary>
    /// The heuristic score of one root move at <paramref name="depth"/> plies (own perspective), from a search of the position after the
    /// move from the opponent's side: the evaluation is antisymmetric, so the value is exactly the negated opponent value.
    /// </summary>
    private static double ScoreMove(ExplainInput inp, int me, int slot, int cell, int depth, out bool complete)
    {
        for (int d = depth; d >= 1; d--)
        {
            var g = GameState.FromSnapshot(inp.State);
            g.MakeMove(new Move(g.HandInstance(me, slot), cell));
            var searcher = new FastSearcher(g, inp.Db, new FastTranspositionTable(1 << 16));
            var r = searcher.RunFixedDepth(d - 1, inp.Aggressive, SearchLimits.WithNodeBudget(AlternativeNodeBudget));
            if (searcher.Stats.NodesSearched < AlternativeNodeBudget)
            {
                complete = true;
                return Normalize(-r.Score);
            }
        }
        complete = false;
        return Normalize(-0.0);
    }

    private static double Normalize(double v) => v == 0.0 ? 0.0 : v;

    // =============================================================================================================== alternatives

    private static AlternativesFact? BuildAlternatives(ExplainInput inp, int me, Move chosen, ImmediateEffectFact immediate, ReplyFact? chosenReply,
        bool chosenIsStandard, double? chosenScoreHint)
    {
        var g0 = GameState.FromSnapshot(inp.State);
        Span<Move> buffer = stackalloc Move[GameState.MaxMoves];
        int n = g0.GenerateMoves(buffer);
        var moves = buffer[..n].ToArray();
        if (n <= 1) return null;
        int chosenIndex = Array.FindIndex(moves, m => m.Inst == chosen.Inst && m.Cell == chosen.Cell);
        if (chosenIndex < 0) return null;
        var slots = moves.Select(m => g0.HandSlotOf(me, m.Inst)).ToArray();

        var bestScores = new double[n];
        var bestReplies = new (int Inst, int Cell)[n];
        int evalDepth = 0;
        long used = 0;
        for (int depth = 1; depth <= Math.Max(inp.DepthCompleted, 1); depth++)
        {
            var scores = new double[n];
            var replies = new (int Inst, int Cell)[n];
            bool complete = true;
            long levelNodes = 0;
            for (int i = 0; i < n && complete; i++)
            {
                var g = GameState.FromSnapshot(inp.State);
                g.MakeMove(new Move(g.HandInstance(me, slots[i]), moves[i].Cell));
                var table = new FastTranspositionTable(1 << 14);
                var searcher = new FastSearcher(g, inp.Db, table);
                long budget = AlternativeNodeBudget - used - levelNodes;
                if (budget <= 0)
                {
                    complete = false;
                    break;
                }
                var r = searcher.RunFixedDepth(depth - 1, inp.Aggressive, SearchLimits.WithNodeBudget(budget));
                if (searcher.Stats.NodesSearched >= budget)
                {
                    complete = false;
                    break;
                }
                levelNodes += searcher.Stats.NodesSearched;
                scores[i] = Normalize(-r.Score);
                if (depth >= 2 && g.AvailableCellCount > 0 && table.TryGet(g.ZobristHash, out var e) && e.BestInstance >= 0) replies[i] = (e.BestInstance, e.BestCell);
                else replies[i] = (-1, 0);
            }
            if (!complete) break;
            used += levelNodes;
            evalDepth = depth;
            bestScores = scores;
            bestReplies = replies;
        }
        if (evalDepth == 0) return null;

        double chosenScore = bestScores[chosenIndex];
        var order = Enumerable.Range(0, n).Where(i => i != chosenIndex).OrderByDescending(i => bestScores[i]).ThenBy(i => i).Take(MaxAlternatives).ToList();
        var items = new List<AlternativeFact>();
        int rank = 0;
        foreach (int i in order)
        {
            rank++;
            var move = moves[i];
            var gA = GameState.FromSnapshot(inp.State);
            int inst = gA.HandInstance(me, slots[i]);
            var moveFact = MoveOf(gA, inp.Generated, me, slots[i], inst, move.Cell);
            var altFlips = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(gA), move.Cell, FlipAnalyzer.ViewOfInstance(gA, inst), (byte)me, gA.Rules);
            gA.MakeMove(new Move(inst, move.Cell));
            ReplyFact? reply = null;
            if (bestReplies[i].Inst >= 0)
            {
                int opp = 1 - me;
                int replyInst = bestReplies[i].Inst;
                var replyFlips = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(gA), bestReplies[i].Cell, FlipAnalyzer.ViewOfInstance(gA, replyInst), (byte)opp, gA.Rules);
                int replySlot = gA.HandSlotOf(opp, replyInst);
                bool assumed = replySlot >= 0 && replySlot < inp.Generated[opp].Length && inp.Generated[opp][replySlot];
                reply = new ReplyFact(EndgameScenarios.BaseCardId(gA.InstanceId(replyInst)), assumed, bestReplies[i].Cell / 3, bestReplies[i].Cell % 3,
                    replyFlips.Select(FlipFact.From).ToList(), replyFlips.Any(f => f.Cell == move.Cell), "search");
            }
            double delta = Normalize(bestScores[i] - chosenScore);
            bool tied = bestScores[i] == chosenScore;
            items.Add(new AlternativeFact(rank, moveFact, bestScores[i], delta, tied, altFlips.Count, reply,
                WhyOf(tied, delta, altFlips.Count, move.Cell, reply, immediate, chosenReply)));
        }
        return new AlternativesFact(evalDepth, chosenScore, chosenIsStandard, n, items);
    }

    /// <summary>The decisive reason an alternative is not better, as a fact (first applicable, in this order).</summary>
    private static WhyWorse WhyOf(bool tied, double delta, int altFlips, int altCell, ReplyFact? altReply, ImmediateEffectFact chosen, ReplyFact? chosenReply)
    {
        if (tied) return new WhyWorse("tied", J.Obj(("score_delta", delta)));
        if (delta > 0) return new WhyWorse("heuristic_better", J.Obj(("score_delta", delta)));
        if (altReply is { CapturesPlacedCard: true } && chosenReply is not { CapturesPlacedCard: true })
        {
            var hit = altReply.Flips.First(f => f.Row * 3 + f.Col == altCell); // how the reply takes the placed card
            return new WhyWorse("exposed_to_reply", J.Obj(
                ("reply_card_id", altReply.CardId), ("reply_assumed", altReply.Assumed), ("reply_row", altReply.Row), ("reply_col", altReply.Col),
                ("cause", hit.Cause), ("their_side", hit.MySide), ("their_value", hit.MyValue), ("my_value", hit.TheirValue), ("score_delta", delta)));
        }
        if (altFlips < chosen.Flips.Count) return new WhyWorse("fewer_captures", J.Obj(("alt_flips", altFlips), ("chosen_flips", chosen.Flips.Count), ("score_delta", delta)));
        if (altReply is not null && chosenReply is not null && altReply.Flips.Count > chosenReply.Flips.Count)
            return new WhyWorse("bigger_counter", J.Obj(("alt_reply_flips", altReply.Flips.Count), ("chosen_reply_flips", chosenReply.Flips.Count), ("score_delta", delta)));
        return new WhyWorse("score_gap", J.Obj(("score_delta", delta)));
    }

    // =============================================================================================================== robust endgame

    private static RobustFact BuildRobust(ExplainInput inp, int me)
    {
        var r = inp.Robust!;
        var cand = r.Candidates.FirstOrDefault(c => c.Slot == inp.ChosenSlot && c.Row == inp.Row && c.Col == inp.Col);
        bool found = cand.CardId != 0 || r.Candidates.Any(c => c.Slot == inp.ChosenSlot && c.Row == inp.Row && c.Col == inp.Col);

        // recompute the chosen move's values over the same scenarios (exact card differences)
        var baseState = GameState.FromSnapshot(inp.State);
        List<GameState> scenarios;
        int unknown = r.Generated[r.Opponent].Count(x => x);
        if (unknown == 0) scenarios = new List<GameState> { baseState };
        else
        {
            var build = EndgameScenarios.Build(inp.Db, baseState, r.Opponent, r.Generated, r.SampleCount, r.NpcIds.ToList(), r.FallbackCandidates?.ToList());
            scenarios = build.Scenarios ?? new List<GameState> { baseState };
        }
        var values = new List<int>();
        int cardId = baseState.InstanceId(baseState.HandInstance(me, inp.ChosenSlot));
        int cell = inp.Row * 3 + inp.Col;
        var memo = ExactEndgameSolver.CreateMemo(1 << 15);
        foreach (var scenario in scenarios)
        {
            var s = GameState.FromSnapshot(scenario.ToSnapshot());
            int inst = -1;
            for (int k = 0; k < s.HandCount(s.CurrentPlayer); k++)
                if (s.InstanceId(s.HandInstance(s.CurrentPlayer, k)) == cardId) { inst = s.HandInstance(s.CurrentPlayer, k); break; }
            if (inst < 0 || !s.IsEmpty(cell)) continue;
            s.MakeMove(new Move(inst, cell));
            if (s.AvailableCellCount == 0)
            {
                s.CountCards(out int red, out int blue);
                values.Add(me == 0 ? red - blue : blue - red);
            }
            else
            {
                var solver = new ExactEndgameSolver(s, memo);
                solver.Bind(me);
                values.Add(solver.Value(me));
            }
        }
        double? avg = values.Count > 0 ? (double)values.Sum() / values.Count : null;
        double? worst = values.Count > 0 ? values.Min() : null;
        double safety = values.Count > 0 ? (double)values.Count(v => v >= 0) / values.Count : 0.0;

        // which clause of the override decision fired
        string? rule = null;
        MoveFact? standard = null;
        double? stdSafety = null, stdFinal = null;
        if (inp.StandardMove is { } sm)
        {
            var gs = GameState.FromSnapshot(inp.State);
            standard = MoveOf(gs, inp.Generated, me, sm.Slot, gs.HandInstance(me, sm.Slot), sm.Row * 3 + sm.Col);
            var stdCand = r.Candidates.Where(c => c.Slot == sm.Slot && c.Row == sm.Row && c.Col == sm.Col).Cast<RobustCandidate?>().LastOrDefault();
            if (stdCand is { } sc)
            {
                stdSafety = sc.SafetyRatio;
                stdFinal = sc.FinalScore;
                if (r.Overridden && found)
                {
                    if (inp.Aggressive) rule = "aggressive_margin";
                    else if (sc.SafetyRatio < 0.5 && cand.SafetyRatio >= 0.5) rule = "safety_floor";
                    else if (cand.SafetyRatio > sc.SafetyRatio && cand.FinalScore >= sc.FinalScore - 1.0) rule = "safer_and_close";
                    else rule = "clearly_better";
                }
            }
        }
        return new RobustFact(scenarios.Count, inp.Aggressive, found ? cand.SafetyRatio : safety, avg, worst, found ? cand.FinalScore : double.NaN, r.Overridden, rule,
            standard, stdSafety, stdFinal);
    }

    // =============================================================================================================== unknown cards

    private static UnknownFact BuildUnknown(ExplainInput inp, GameState g0, int opp)
    {
        var assumed = new List<AssumedCardFact>();
        for (int slot = 0; slot < g0.HandCount(opp); slot++)
        {
            if (!inp.Generated[opp][slot]) continue;
            int inst = g0.HandInstance(opp, slot);
            int id = EndgameScenarios.BaseCardId(g0.InstanceId(inst));
            if (id <= 0 || !inp.Db.TryGet(id, out var def)) continue;
            assumed.Add(new AssumedCardFact(slot, id, def.Star, CardTypes.NameOf(def.TypeId)));
        }
        int oppUnknown = inp.Generated[opp].Count(x => x);
        string strategy = oppUnknown == 0 ? "none" : inp.Unknown.Strategy;
        int ownUnknown = inp.Generated[1 - opp].Count(x => x);
        return new UnknownFact(oppUnknown, ownUnknown, strategy, oppUnknown == 0 ? null : inp.Unknown.PrimaryRule, inp.Unknown.NpcName,
            inp.Unknown.NpcPoolSize, assumed);
    }
}
