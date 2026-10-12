using System.Numerics;
using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>Python's <c>_move_key(move)</c> = <c>(card_id, row, col)</c>; <see cref="CardId"/> is -1 for Python <c>None</c>.</summary>
public readonly record struct MoveKey(int CardId, int Row, int Col);

/// <summary>One entry of Python's <c>scored_moves</c> (final / safety / robust / corner values of one move).</summary>
public readonly record struct RobustMoveScore(Move Move, MoveKey Key, double FinalScore, double SafetyRatio, double RobustScore, double CornerRisk);

/// <summary>An entry of Python's <c>robust_lookup</c> as the override logic reads it.</summary>
public readonly record struct OverrideItem(MoveKey Key, double FinalScore, double SafetyRatio);

/// <summary>Result of <see cref="EndgameRobust.Select"/>: the selected move and <c>scored_moves</c> in Python's returned order.</summary>
public sealed class RobustSelection
{
    public bool HasBest { get; init; }

    /// <summary>The selected move (an instance of the BASE state's hand). Only valid if <see cref="HasBest"/>.</summary>
    public Move Best { get; init; }

    public MoveKey BestKey { get; init; }

    public List<RobustMoveScore> Scored { get; init; } = new();

    /// <summary>Counters of the exact solver over the whole selection (benchmark / diagnostics).</summary>
    public long ExactNodes { get; init; }
    public long ExactMakeMoves { get; init; }
}

/// <summary>
/// Port of the information-set robust endgame of <c>ai_server</c> (Phase 5): <c>_calculate_corner_safety_risk</c>,
/// <c>_best_immediate_reply_score</c>, <c>_evaluate_endgame_move_robustly</c>, <c>select_endgame_robust_move</c>,
/// <c>_should_use_endgame_robust_mode</c> and the override block of <c>ai_move</c>. All floating point follows Python's
/// expressions and accumulation order (bit-for-bit, CLAUDE.md section 2): the average is a left-to-right sum over the
/// scenarios in scenario order, divided by the count, exactly like <c>sum(list) / len(list)</c> on CPython 3.11
/// (CPython 3.12 changed <c>sum()</c> of floats to compensated summation; golden data is bound to the Python
/// version, see golden_sources.py).
/// </summary>
public static class EndgameRobust
{
    private static readonly int[] Dr = { -1, 1, 0, 0 };
    private static readonly int[] Dc = { 0, 0, -1, 1 };
    private static readonly int[] MySide = { Side.Up, Side.Down, Side.Left, Side.Right };
    private static readonly int[] OppSide = { Side.Down, Side.Up, Side.Right, Side.Left };

    private static bool IsCorner(int cell) => cell is 0 or 2 or 6 or 8;

    /// <summary>Python <c>_should_use_endgame_robust_mode</c>.</summary>
    public static bool ShouldUse(int occupiedCells, int opponentUnknownCount) => occupiedCells >= 5 && opponentUnknownCount <= 2;

    /// <summary>
    /// Python <c>_calculate_corner_safety_risk(card, row, col, board, rules)</c> for the card <paramref name="move"/>.Inst
    /// of <paramref name="g"/> placed on <paramref name="move"/>.Cell of <paramref name="g"/>'s board. The comparison
    /// "<c>adj_card.owner != card.owner</c>" uses the card instance's own owner attribute, not the side to move.
    /// </summary>
    public static double CornerSafetyRisk(GameState g, Move move)
    {
        int occupied = BitOperations.PopCount(g.OccupiedMask);
        if (occupied < 5) return 0.0;

        int row = move.Row, col = move.Col, inst = move.Inst;
        double stageWeight = 1.0 + Math.Min((occupied - 4) / 4.0, 1.0);
        double risk = 0.0;
        int attackable = 0, weak = 0;
        byte myOwner = g.InstanceOwner(inst);

        for (int k = 0; k < 4; k++)
        {
            int nr = row + Dr[k], nc = col + Dc[k];
            if (nr < 0 || nr > 2 || nc < 0 || nc > 2) continue;

            attackable++;
            int myValue = g.InstanceEffective(inst, MySide[k]);
            if (myValue <= 3)
            {
                weak++;
                risk += (4 - myValue) * 2.2;
            }
            else if (myValue == 4)
            {
                risk += 1.0;
            }

            int adj = g.BoardInstance(nr * 3 + nc);
            if (adj >= 0 && g.InstanceOwner(adj) != myOwner)
            {
                int oppValue = g.InstanceEffective(adj, OppSide[k]);
                if (oppValue > myValue) risk += (oppValue - myValue) * 1.5;
                else if (oppValue == myValue) risk += 1.0;
            }
        }

        if (IsCorner(move.Cell) && weak > 0) risk += weak * 1.5;
        else if (attackable >= 3 && weak > 0) risk += weak * 1.0;

        return risk * stageWeight;
    }

    /// <summary>
    /// Python <c>_best_immediate_reply_score(state_after_my_move, ai)</c>: the opponent's best reply by the heuristic
    /// <c>evaluate_state</c> (non-aggressive), one ply. Runs Make/Undo on <paramref name="state"/> (hand order, D-001,
    /// and display edges, D-003, may differ afterwards, which cannot change any value).
    /// </summary>
    public static double BestImmediateReplyScore(GameState state, Evaluator evaluator, int aiPlayer)
    {
        Span<Move> moves = stackalloc Move[GameState.MaxMoves];
        int n = state.GenerateMoves(moves);
        if (n == 0) return evaluator.EvaluateState(aiPlayer, false);

        // PY-QUIRK(D-019): Python copies the state for every reply and plays the FIRST card of the reply card's id in
        // the hand as it is on entry (`next(c for c in hand if c.card_id == reply_card.card_id)`). With duplicate ids in
        // the hand (D-004) every duplicate maps to the first card of that id. The mapping is resolved for all replies
        // BEFORE the first Make: Make/Undo reorders the hand (D-001), and Python's copies never see that.
        for (int i = 0; i < n; i++)
            moves[i] = new Move(FindInHand(state, state.CurrentPlayer, state.InstanceId(moves[i].Inst)), moves[i].Cell);

        double best = double.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            state.MakeMove(moves[i]);
            double v = evaluator.EvaluateState(aiPlayer, false);
            state.UndoMove();
            if (v < best) best = v;
        }
        return best != double.PositiveInfinity ? best : evaluator.EvaluateState(aiPlayer, false);
    }

    /// <summary>
    /// The first card in hand order with this card id (Python's <c>next(c for c in hand if c.card_id == card_id)</c>),
    /// or -1. PY-QUIRK(D-019): with duplicate ids in one hand every duplicate maps to the first card of that id; this is
    /// harmless for real data because equal ids are equal card definitions.
    /// </summary>
    private static int FindInHand(GameState g, int player, int cardId)
    {
        for (int slot = 0; slot < g.HandCount(player); slot++)
        {
            int inst = g.HandInstance(player, slot);
            if (g.InstanceId(inst) == cardId) return inst; // Python: first card in hand order with that card_id
        }
        return -1;
    }

    private const double SafetyMargin = 0.75;

    // One memo table per thread, reused by every Select call: allocating and zeroing a 32K-entry table (~768 KiB) per call was
    // visible in the robust pipeline (review L5-2). Every scenario starts its own generation (ExactEndgameSolver.Bind), which
    // empties the table in O(1) and binds it to that scenario's state, side and evaluation mode.
    [ThreadStatic] private static FastTranspositionTable? t_memo;

    private static FastTranspositionTable SharedMemo() => t_memo ??= ExactEndgameSolver.CreateMemo();

    /// <summary>
    /// Python <c>select_endgame_robust_move(base_state, scenario_states, ai_player_idx, aggressive)</c> including the
    /// per-move <c>_evaluate_endgame_move_robustly</c>. The scenarios are never modified (each is cloned first), nor is
    /// <paramref name="baseState"/>. With five or more occupied cells every (move, scenario) pair is solved exactly
    /// (<see cref="ExactEndgameSolver"/>, own memo generation per scenario state); otherwise the one-ply heuristic reply
    /// is used.
    /// </summary>
    public static RobustSelection Select(CardDb db, GameState baseState, IReadOnlyList<GameState> scenarios, int aiPlayer,
        bool aggressive, FastTranspositionTable? memo = null)
    {
        Span<Move> buffer = stackalloc Move[GameState.MaxMoves];
        int n = baseState.GenerateMoves(buffer);
        if (n == 0) return new RobustSelection { HasBest = false };
        var moves = buffer[..n].ToArray();

        int occupied = BitOperations.PopCount(baseState.OccupiedMask);
        bool useExact = occupied >= 5;
        var corner = new double[n];
        var cardIds = new int[n];
        for (int i = 0; i < n; i++)
        {
            corner[i] = CornerSafetyRisk(baseState, moves[i]);
            cardIds[i] = baseState.InstanceId(moves[i].Inst);
        }

        var exactScores = new List<int>[n];
        var heuristicScores = new List<double>[n];
        for (int i = 0; i < n; i++)
        {
            exactScores[i] = new List<int>();
            heuristicScores[i] = new List<double>();
        }
        var safetyVotes = new int[n];
        var mapped = new int[n];
        if (useExact) memo ??= SharedMemo(); // a generation per scenario state is started by Bind, so reuse across calls is safe
        long exactNodes = 0, exactMakes = 0;

        foreach (var scenario in scenarios)
        {
            var s = GameState.FromSnapshot(scenario.ToSnapshot());
            var evaluator = new Evaluator(s, db);
            double scenarioEval = useExact ? 0.0 : evaluator.EvaluateState(aiPlayer, false); // only the heuristic path reads it
            ExactEndgameSolver? solver = null;
            if (useExact)
            {
                solver = new ExactEndgameSolver(s, memo!);
                solver.Bind(aiPlayer);
            }

            // PY-QUIRK(D-019): Python copies the scenario for every move and plays the first card of that id in the
            // scenario hand as it is on entry; resolve it before the first Make reorders the hand (D-001).
            for (int i = 0; i < n; i++) mapped[i] = FindInHand(s, s.CurrentPlayer, cardIds[i]);

            for (int i = 0; i < n; i++)
            {
                int inst = mapped[i];
                if (inst < 0 || !s.IsEmpty(moves[i].Cell)) continue; // Python: `continue` (no score for this scenario)
                s.MakeMove(new Move(inst, moves[i].Cell));
                if (useExact)
                {
                    int reply = solver!.Value(aiPlayer);
                    exactScores[i].Add(reply);
                    if (reply >= 0) safetyVotes[i]++;
                }
                else
                {
                    double reply = BestImmediateReplyScore(s, evaluator, aiPlayer);
                    heuristicScores[i].Add(reply);
                    if (reply >= scenarioEval - SafetyMargin) safetyVotes[i]++;
                }
                s.UndoMove();
            }
            if (solver is not null)
            {
                exactNodes += solver.Nodes;
                exactMakes += solver.MakeMoves;
            }
        }

        var scored = new List<RobustMoveScore>(n);
        for (int i = 0; i < n; i++)
        {
            int count = useExact ? exactScores[i].Count : heuristicScores[i].Count;
            var key = KeyOf(baseState, moves[i]);
            if (count == 0)
            {
                scored.Add(new RobustMoveScore(moves[i], key, double.NegativeInfinity, 0.0, double.NegativeInfinity, corner[i]));
                continue;
            }

            double avg, worst;
            if (useExact)
            {
                long sum = 0;
                int min = int.MaxValue;
                foreach (int v in exactScores[i])
                {
                    sum += v;
                    if (v < min) min = v;
                }
                avg = (double)sum / count; // int / int true division
                worst = min;
            }
            else
            {
                double sum = 0.0; // Python: sum() starts at int 0 and adds left to right
                double min = double.PositiveInfinity;
                foreach (double v in heuristicScores[i])
                {
                    sum += v;
                    if (v < min) min = v;
                }
                avg = sum / count;
                worst = min;
            }

            double safetyRatio = (double)safetyVotes[i] / count;
            double robust, safetyBonus, cornerPenalty;
            if (aggressive)
            {
                robust = avg * 0.75 + worst * 0.25;
                safetyBonus = safetyRatio * 0.75;
                cornerPenalty = corner[i] * 0.05;
            }
            else
            {
                robust = avg * 0.45 + worst * 0.55;
                safetyBonus = safetyRatio * 2.0;
                cornerPenalty = corner[i] * 0.15;
            }

            double cornerSafetyThreshold = aggressive ? 0.5 : 0.8;
            if (!useExact && IsCorner(moves[i].Cell) && corner[i] >= 5.0 && safetyRatio < cornerSafetyThreshold)
            {
                scored.Add(new RobustMoveScore(moves[i], key, double.NegativeInfinity, safetyRatio, double.NegativeInfinity, corner[i]));
                continue;
            }

            double final = robust + safetyBonus - cornerPenalty;
            scored.Add(new RobustMoveScore(moves[i], key, final, safetyRatio, robust, corner[i]));
        }

        // Ranking (Python sorts with reverse=True, which is stable: equal keys keep their relative order).
        List<RobustMoveScore> ranked;
        if (aggressive)
        {
            ranked = scored; // sorted in place: the returned scored_moves list is reordered too
        }
        else
        {
            var fullySafe = scored.Where(it => it.SafetyRatio >= 1.0).ToList();
            var safe = fullySafe.Count > 0
                ? fullySafe
                : scored.Where(it => it.SafetyRatio >= 0.75 && (it.CornerRisk < 5.0 || it.SafetyRatio >= 0.8)).ToList();
            ranked = safe.Count > 0 ? safe : scored; // an empty `safe` falls back to scored_moves itself (aliased: sorted in place)
        }
        StableSortDescending(ranked, aggressive);

        var best = ranked[0];
        return new RobustSelection
        {
            HasBest = true,
            Best = best.Move,
            BestKey = best.Key,
            Scored = scored,
            ExactNodes = exactNodes,
            ExactMakeMoves = exactMakes,
        };
    }

    public static MoveKey KeyOf(GameState g, Move move) => new(g.InstanceId(move.Inst), move.Row, move.Col);

    /// <summary>Python tuple compare of the sort keys: aggressive (final, robust, safety, -corner), else (safety, final, robust, -corner).</summary>
    private static bool KeyLess(in RobustMoveScore a, in RobustMoveScore b, bool aggressive)
    {
        Span<double> ka = stackalloc double[4];
        Span<double> kb = stackalloc double[4];
        if (aggressive)
        {
            ka[0] = a.FinalScore; ka[1] = a.RobustScore; ka[2] = a.SafetyRatio; ka[3] = -a.CornerRisk;
            kb[0] = b.FinalScore; kb[1] = b.RobustScore; kb[2] = b.SafetyRatio; kb[3] = -b.CornerRisk;
        }
        else
        {
            ka[0] = a.SafetyRatio; ka[1] = a.FinalScore; ka[2] = a.RobustScore; ka[3] = -a.CornerRisk;
            kb[0] = b.SafetyRatio; kb[1] = b.FinalScore; kb[2] = b.RobustScore; kb[3] = -b.CornerRisk;
        }
        for (int i = 0; i < 4; i++)
        {
            if (ka[i] != kb[i]) return ka[i] < kb[i];
        }
        return false;
    }

    private static void StableSortDescending(List<RobustMoveScore> items, bool aggressive)
    {
        for (int i = 1; i < items.Count; i++)
        {
            var item = items[i];
            int j = i - 1;
            while (j >= 0 && KeyLess(items[j], item, aggressive))
            {
                items[j + 1] = items[j];
                j--;
            }
            items[j + 1] = item;
        }
    }

    /// <summary>
    /// The override block of <c>ai_move</c> (it is inline in the Flask handler; the golden data executes that very
    /// block, extracted from the source by AST). Returns the move to play: <paramref name="robust"/> if the rules say
    /// to override the standard minimax move <paramref name="standard"/>, otherwise <paramref name="standard"/>.
    /// A move that is missing from the table never overrides (Python: <c>robust_lookup.get</c> gives <c>None</c>).
    /// </summary>
    public static MoveKey? DecideOverride(MoveKey? standard, MoveKey? robust, IReadOnlyList<OverrideItem> items, bool aggressive)
    {
        if (standard is null || robust is null) return standard;

        OverrideItem? standardItem = null, robustItem = null;
        foreach (var item in items) // dict semantics: the last entry of a key wins
        {
            if (item.Key == standard.Value) standardItem = item;
            if (item.Key == robust.Value) robustItem = item;
        }
        if (standardItem is null || robustItem is null) return standard;

        var std = standardItem.Value;
        var rob = robustItem.Value;
        bool shouldOverride;
        if (aggressive)
        {
            shouldOverride = rob.FinalScore > std.FinalScore + 2.5 && rob.SafetyRatio >= 0.75;
        }
        else
        {
            shouldOverride = (std.SafetyRatio < 0.5 && rob.SafetyRatio >= 0.5) ||
                             (rob.SafetyRatio > std.SafetyRatio && rob.FinalScore >= std.FinalScore - 1.0) ||
                             (rob.FinalScore > std.FinalScore + 0.05);
        }
        return shouldOverride && robust.Value != standard.Value ? robust : standard;
    }
}
