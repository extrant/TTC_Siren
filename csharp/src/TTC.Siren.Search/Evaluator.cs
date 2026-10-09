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

    /// <summary>
    /// Python <c>_calculate_placement_edge_score</c>. All terms are multiples of 0.5, so the sum is exact in any order.
    /// </summary>
    public double PlacementEdgeScore(int inst, int row, int col)
    {
        double score = 0.0;
        int exposedCount = 0;
        int ev0 = 0, ev1 = 0;
        for (int side = 0; side < 4; side++)
        {
            bool exposed = side switch
            {
                Side.Up => row > 0,
                Side.Right => col < 2,
                Side.Down => row < 2,
                _ => col > 0,
            };
            int value = _g.InstanceEffective(inst, side);
            if (exposed)
            {
                if (value >= 8) score += (value - 7) * 3.0;
                else if (value <= 3) score -= (4 - value) * 4.0;
                else if (value <= 5) score -= (6 - value) * 1.0;
                if (exposedCount == 0) ev0 = value; else ev1 = value;
                exposedCount++;
            }
            else
            {
                if (value <= 3) score += (4 - value) * 2.5;
                else if (value >= 8) score -= (value - 7) * 2.0;
            }
        }

        if (exposedCount == 2)
        {
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
        // Neighbours come from the board's precomputed table in Python's order (up, down, left, right).
        byte myOwner = g.InstanceOwner(inst);
        RuleSet rules = g.Rules;
        int neighbours = GameState.NeighborCount(cell);
        for (int k = 0; k < neighbours; k++)
        {
            int opp = g.BoardInstance(GameState.NeighborCell(cell, k));
            if (opp < 0 || g.InstanceOwner(opp) == myOwner) continue;
            int mySide = GameState.NeighborMySide(cell, k), oppSide = GameState.NeighborOppSide(cell, k);
            int result = GameState.CompareValues(g.InstanceEffective(inst, mySide), g.InstanceEffective(opp, oppSide), rules);
            if (result != 1) continue;

            int myValue = g.InstanceDisplay(inst, mySide);
            int oppValue = g.InstanceDisplay(opp, oppSide);
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
        for (int k = 0; k < neighbours; k++)
        {
            int adj = g.BoardInstance(GameState.NeighborCell(cell, k));
            if (adj >= 0 && g.InstanceOwner(adj) == myOwner) score += 10.0;
        }

        // 6. placement edge orientation, 7. endgame exposure penalty
        score += PlacementEdgeScore(inst, row, col) * CornerStrategyWeight;
        score -= EndgameExposurePenalty(inst, row, col, occupied) * exposureWeight;

        return score > 1000.0 ? 1000.0 : score;
    }
}
