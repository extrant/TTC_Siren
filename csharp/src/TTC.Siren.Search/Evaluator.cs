using System.Numerics;
using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// Port of <c>ai/ai.py</c>: <c>evaluate_state</c>, <c>evaluate_move</c>, <c>_calculate_placement_edge_score</c> and
/// <c>_calculate_endgame_exposure_penalty</c>. Every coefficient, accumulation order and clamp is kept so that the
/// results are bit-identical to Python's doubles (CLAUDE.md section 2/4). Bound to one <see cref="GameState"/>.
/// </summary>
public sealed class Evaluator
{
    /// <summary>Python <c>CORNER_STRATEGY_WEIGHT</c>.</summary>
    public const double CornerStrategyWeight = 5.0;

    // Directions in Python's order: up, down, left, right.
    private static readonly int[] Dr = { -1, 1, 0, 0 };
    private static readonly int[] Dc = { 0, 0, -1, 1 };
    private static readonly int[] MySide = { Side.Up, Side.Down, Side.Left, Side.Right };
    private static readonly int[] OppSide = { Side.Down, Side.Up, Side.Right, Side.Left };

    private readonly GameState _g;
    private readonly byte[] _star = new byte[GameState.MaxInstances];

    /// <summary>
    /// Star ratings come from the card database by id (<c>get_card_star_map().get(card_id, 0)</c>): predicted ids
    /// (1000+), <c>None</c> and unknown ids give 0.
    /// </summary>
    public Evaluator(GameState state, CardDb db)
    {
        _g = state;
        for (int inst = 0; inst < state.InstanceCount; inst++)
        {
            int id = state.InstanceId(inst);
            _star[inst] = id > 0 && db.TryGet(id, out var def) ? def.Star : (byte)0;
        }
    }

    public int Star(int inst) => _star[inst];

    // ---------------------------------------------------------------- aggressive-mode weights

    private static void StateWeights(int occupied, bool aggressive, out double baseWeight, out double positionWeight, out double edgeWeight)
    {
        if (!aggressive || occupied <= 2) // "balanced" equals the normal weights
        {
            baseWeight = 0.7; positionWeight = 0.3; edgeWeight = 0.25;
        }
        else if (occupied <= 5) // "hard"
        {
            baseWeight = 0.9; positionWeight = 0.25; edgeWeight = 0.15;
        }
        else // "ultra"
        {
            baseWeight = 1.15; positionWeight = 0.15; edgeWeight = 0.05;
        }
    }

    private static void MoveWeights(int occupied, out double capture, out double exposure, out double power)
    {
        if (occupied <= 2)
        {
            capture = 1.0; exposure = 1.0; power = 1.0;
        }
        else if (occupied <= 5)
        {
            capture = 1.25; exposure = 0.65; power = 1.1;
        }
        else
        {
            capture = 1.65; exposure = 0.35; power = 1.25;
        }
    }

    // ---------------------------------------------------------------- placement edge score

    // Per-edge terms of _calculate_placement_edge_score for the effective value 1..10, exposed (row > 0 for up, col < 2 for
    // right, row < 2 for down, col > 0 for left) or hidden. They are exactly the terms Python adds or subtracts; "score -= x"
    // is "score += -x" in IEEE arithmetic. Lookups replace the compare chains, whose branches are hard to predict.
    private static readonly double[] ExposedTerm = BuildTerms(true);
    private static readonly double[] HiddenTerm = BuildTerms(false);
    private static readonly byte[] ExposedMask = BuildExposedMasks(); // bit per Side index, per cell
    private static readonly byte[] ExposedCount = BuildExposedCounts();

    private static double[] BuildTerms(bool exposed)
    {
        var t = new double[11];
        for (int value = 1; value <= 10; value++)
        {
            double term = 0.0;
            if (exposed)
            {
                if (value >= 8) term = (value - 7) * 3.0;
                else if (value <= 3) term = -((4 - value) * 4.0);
                else if (value <= 5) term = -((6 - value) * 1.0);
            }
            else
            {
                if (value <= 3) term = (4 - value) * 2.5;
                else if (value >= 8) term = -((value - 7) * 2.0);
            }
            t[value] = term;
        }
        return t;
    }

    private static byte[] BuildExposedCounts()
    {
        var c = new byte[9];
        for (int cell = 0; cell < 9; cell++) c[cell] = (byte)System.Numerics.BitOperations.PopCount(ExposedMask[cell]);
        return c;
    }

    private static byte[] BuildExposedMasks()
    {
        var m = new byte[9];
        for (int cell = 0; cell < 9; cell++)
        {
            int row = cell / 3, col = cell % 3;
            int mask = 0;
            if (row > 0) mask |= 1 << Side.Up;
            if (col < 2) mask |= 1 << Side.Right;
            if (row < 2) mask |= 1 << Side.Down;
            if (col > 0) mask |= 1 << Side.Left;
            m[cell] = (byte)mask;
        }
        return m;
    }

    /// <summary>
    /// Python <c>_calculate_placement_edge_score</c>. All terms are multiples of 0.5 and small, so every partial sum is
    /// exact and the result does not depend on the order of the additions (bit-identical to Python).
    /// </summary>
    public double PlacementEdgeScore(int inst, int row, int col)
    {
        int mask = ExposedMask[row * 3 + col];
        double[] exposedTerm = ExposedTerm, hiddenTerm = HiddenTerm; // one static load each (cheap in unoptimised code too)
        int up = _g.InstanceEffective(inst, Side.Up), right = _g.InstanceEffective(inst, Side.Right);
        int down = _g.InstanceEffective(inst, Side.Down), left = _g.InstanceEffective(inst, Side.Left);
        double score = ((mask & 1 << Side.Up) != 0 ? exposedTerm[up] : hiddenTerm[up])
                     + ((mask & 1 << Side.Right) != 0 ? exposedTerm[right] : hiddenTerm[right])
                     + ((mask & 1 << Side.Down) != 0 ? exposedTerm[down] : hiddenTerm[down])
                     + ((mask & 1 << Side.Left) != 0 ? exposedTerm[left] : hiddenTerm[left]);

        // two exposed edges (corner cells): the pair bonus / penalty looks at those two values
        if (ExposedCount[row * 3 + col] == 2)
        {
            int ev0 = 0, ev1 = 0, n = 0;
            if ((mask & 1 << Side.Up) != 0) { ev0 = up; n++; }
            if ((mask & 1 << Side.Right) != 0) { if (n == 0) ev0 = right; else ev1 = right; n++; }
            if ((mask & 1 << Side.Down) != 0) { if (n == 0) ev0 = down; else ev1 = down; n++; }
            if ((mask & 1 << Side.Left) != 0) { if (n == 0) ev0 = left; else ev1 = left; }
            if (ev0 >= 8 && ev1 >= 8) score += 8.0;
            if (ev0 <= 3 || ev1 <= 3) score -= 4.0;
        }
        return score;
    }

    // ---------------------------------------------------------------- endgame exposure penalty

    /// <summary>Python <c>_calculate_endgame_exposure_penalty</c> (board occupancy is taken before the move).</summary>
    public double EndgameExposurePenalty(int inst, int row, int col, int occupied)
    {
        if (occupied < 5) return 0.0;

        double q = (occupied - 4) / 4.0;
        double stageWeight = 1.0 + (q > 1.0 ? 1.0 : q);
        double penalty = 0.0;
        int attackable = 0, weakAttackable = 0;
        byte myOwner = _g.InstanceOwner(inst);

        for (int k = 0; k < 4; k++)
        {
            int nr = row + Dr[k], nc = col + Dc[k];
            if (nr < 0 || nr > 2 || nc < 0 || nc > 2) continue;

            attackable++;
            int myValue = _g.InstanceEffective(inst, MySide[k]);
            int adj = _g.BoardInstance(nr * 3 + nc);

            if (myValue <= 3)
            {
                weakAttackable++;
                penalty += (4 - myValue) * 2.5;
            }
            else if (myValue == 4)
            {
                penalty += 1.0;
            }

            if (adj >= 0 && _g.InstanceOwner(adj) != myOwner)
            {
                int oppValue = _g.InstanceEffective(adj, OppSide[k]);
                if (oppValue > myValue) penalty += (oppValue - myValue) * 1.8;
                else if (oppValue == myValue) penalty += 1.0;
            }
        }

        bool corner = (row == 0 || row == 2) && (col == 0 || col == 2);
        if (corner && weakAttackable > 0) penalty += weakAttackable * 2.0;
        else if (attackable >= 3 && weakAttackable > 0) penalty += weakAttackable * 1.0;

        return penalty * stageWeight;
    }

    // ---------------------------------------------------------------- evaluate_state

    /// <summary>Python <c>evaluate_state(state, ai_player_idx, aggressive)</c>. The end of the game is not special-cased (D-007).</summary>
    public double EvaluateState(int aiPlayer, bool aggressive)
    {
        var g = _g;
        g.CountCards(out int red, out int blue);
        int baseScore = aiPlayer == 0 ? red - blue : blue - red;

        double positionScore = 0.0;
        double placementEdgeScore = 0.0;
        for (int cell = 0; cell < 9; cell++) // row-major, same accumulation order as Python
        {
            int inst = g.BoardInstance(cell);
            if (inst < 0) continue;
            int row = cell / 3, col = cell % 3;
            double weight = cell is 0 or 2 or 6 or 8 ? 1.5 : cell == 4 ? 1.2 : 1.0;
            byte owner = g.InstanceOwner(inst);
            if ((owner == OwnerId.Red && aiPlayer == 0) || (owner == OwnerId.Blue && aiPlayer == 1))
            {
                positionScore += weight;
                placementEdgeScore += PlacementEdgeScore(inst, row, col);
            }
            else
            {
                positionScore -= weight;
                placementEdgeScore -= PlacementEdgeScore(inst, row, col);
            }
        }

        StateWeights(BitOperations.PopCount(g.OccupiedMask), aggressive, out double bw, out double pw, out double ew);
        return baseScore * bw + positionScore * pw + placementEdgeScore * ew;
    }

    // ---------------------------------------------------------------- evaluate_move

    /// <summary>
    /// Python <c>evaluate_move(move, state, history, aggressive)</c> for a move of the side to move (the card is
    /// <see cref="Move.Inst"/>, so the result does not depend on the current hand order).
    /// PY-QUIRK(D-003): edge numbers used for card power, star bonuses and capture differences are the stale
    /// "display" values, not the effective ones; capture decisions use effective values.
    /// </summary>
    public double EvaluateMove(Move move, HistoryTable? history, bool aggressive)
    {
        var g = _g;
        int inst = move.Inst;
        int cell = move.Cell;
        int row = cell / 3, col = cell % 3;
        int occupied = BitOperations.PopCount(g.OccupiedMask);

        double captureWeight = 1.0, exposureWeight = 1.0, powerWeight = 1.0;
        if (aggressive) MoveWeights(occupied, out captureWeight, out exposureWeight, out powerWeight);

        double score = 0.0;

        // 1. history heuristic (capped at 1000.0)
        int h = history is null ? 0 : history.Get(inst, cell);
        double hs = h * 0.1;
        score += hs > 1000.0 ? 1000.0 : hs;

        // 2. star / edge power (display edges: D-003)
        int star = _star[inst];
        int up = g.InstanceDisplay(inst, Side.Up), down = g.InstanceDisplay(inst, Side.Down);
        int left = g.InstanceDisplay(inst, Side.Left), right = g.InstanceDisplay(inst, Side.Right);
        if (aggressive)
        {
            int cardPower = up + down + left + right + star * 3;
            score += cardPower * powerWeight;
        }

        bool corner = cell is 0 or 2 or 6 or 8;
        int eights = (up == 8 ? 1 : 0) + (down == 8 ? 1 : 0) + (left == 8 ? 1 : 0) + (right == 8 ? 1 : 0);
        int nines = (up == 9 ? 1 : 0) + (down == 9 ? 1 : 0) + (left == 9 ? 1 : 0) + (right == 9 ? 1 : 0);
        if (star == 3 && eights >= 2 && corner) score += 40.0;
        if (star == 4 && nines >= 2) score += 30.0;
        if (star == 5 && nines >= 3) score += 50.0;

        // 3. position
        if (corner) score += 15.0;
        else if (cell == 4) score += 10.0;

        // 4. captures (compare_values with the rule set; reversal and fallen ace included)
        byte myOwner = g.InstanceOwner(inst);
        RuleSet rules = g.Rules;
        for (int k = 0; k < 4; k++)
        {
            int nr = row + Dr[k], nc = col + Dc[k];
            if (nr < 0 || nr > 2 || nc < 0 || nc > 2) continue;
            int opp = g.BoardInstance(nr * 3 + nc);
            if (opp < 0 || g.InstanceOwner(opp) == myOwner) continue;
            int result = GameState.CompareValues(g.InstanceEffective(inst, MySide[k]), g.InstanceEffective(opp, OppSide[k]), rules);
            if (result != 1) continue;

            int myValue = g.InstanceDisplay(inst, MySide[k]);
            int oppValue = g.InstanceDisplay(opp, OppSide[k]);
            int diff = myValue > oppValue ? myValue - oppValue : oppValue - myValue;
            double captureScore = 30.0 + diff * 5.0;
            if (star != 0)
            {
                int oppStar = _star[opp];
                if (star > oppStar) captureScore += (star - oppStar) * 10.0;
            }
            score += captureScore * captureWeight;
        }

        // 5. adjacent own cards
        for (int k = 0; k < 4; k++)
        {
            int nr = row + Dr[k], nc = col + Dc[k];
            if (nr < 0 || nr > 2 || nc < 0 || nc > 2) continue;
            int adj = g.BoardInstance(nr * 3 + nc);
            if (adj >= 0 && g.InstanceOwner(adj) == myOwner) score += 10.0;
        }

        // 6. placement edge orientation, 7. endgame exposure penalty
        score += PlacementEdgeScore(inst, row, col) * CornerStrategyWeight;
        score -= EndgameExposurePenalty(inst, row, col, occupied) * exposureWeight;

        return score > 1000.0 ? 1000.0 : score;
    }
}
