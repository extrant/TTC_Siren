using TTC.Siren.Core;

namespace TTC.Siren.Search;

/// <summary>
/// A possible identity of an unknown opponent card (Python: a candidate <c>Card</c> made by
/// <c>_clone_unknown_candidate</c>: base edges, base card id, no modifier, display edges equal to the base edges).
/// <see cref="Id"/> is -1 for Python <c>None</c>.
/// </summary>
public readonly record struct UnknownCandidate(int Id, byte Up, byte Right, byte Down, byte Left, byte TypeId);

/// <summary>Result of <see cref="EndgameScenarios.Build"/>.</summary>
public sealed class ScenarioBuild
{
    /// <summary>
    /// The possible worlds, in Python's order. <c>null</c> only if <see cref="RequiresFallback"/>. When there are no
    /// unknown cards, or no assignment survives, this is the base state itself (the same object), as in Python.
    /// </summary>
    public List<GameState>? Scenarios { get; init; }

    /// <summary>
    /// Python found no legal candidate and would ask <c>handler.generate_opponent_cards</c>, which samples at random
    /// (outside the deterministic scope, CSHARP_MIGRATION.md section 6 and A-7). The caller may supply those cards.
    /// </summary>
    public bool RequiresFallback { get; init; }

    public int LegalCandidateCount { get; init; }
}

/// <summary>
/// Port of the deterministic part of <c>ai_server._build_endgame_scenarios</c> and its helpers
/// (<c>_build_legal_unknown_candidates</c>, <c>_unique_unknown_candidates</c>, <c>_score_unknown_candidate</c>,
/// <c>_is_legal_unknown_assignment</c>, <c>_copy_state_with_unknown_assignment</c>): every possible-world state
/// the information-set robust endgame evaluates, in the order Python builds them. Not a search hot path (scenario
/// construction happens once per request, allocation is fine).
///
/// Python's hand cards carry attributes <c>_is_generated</c> / <c>_is_prediction</c> and ids &gt;= 1000 that mark
/// "unknown" slots; a C# <see cref="GameState"/> has no such attribute, so the caller passes the result of Python's
/// <c>_is_generated_unknown_card</c> for every hand slot (<c>generated[player][slot]</c>).
/// </summary>
public static class EndgameScenarios
{
    // Python's direction order: up, down, left, right.
    private static readonly int[] Dr = { -1, 1, 0, 0 };
    private static readonly int[] Dc = { 0, 0, -1, 1 };
    private static readonly int[] MySide = { Side.Up, Side.Down, Side.Left, Side.Right };
    private static readonly int[] OppSide = { Side.Down, Side.Up, Side.Right, Side.Left };

    /// <summary>Python <c>_base_card_id</c>: predicted ids (1000 + id) map back to the real id; None stays -1.</summary>
    public static int BaseCardId(int id) => id == -1 ? -1 : id >= 1000 ? id - 1000 : id;

    /// <summary>Python <c>star_map.get(base_id, 1)</c>: ids outside the database (and None) count as 1 star.</summary>
    private static int StarOrOne(CardDb db, int baseId) =>
        baseId != -1 && db.TryGet(baseId, out var def) ? def.Star : 1;

    /// <summary>Python <c>_parse_id_list</c> for an already numeric list: positive ids, duplicates dropped, order kept.</summary>
    public static List<int> ParseIdList(IReadOnlyList<int>? raw)
    {
        var result = new List<int>();
        if (raw is null) return result;
        foreach (int id in raw)
        {
            if (id > 0 && !result.Contains(id)) result.Add(id);
        }
        return result;
    }

    public static List<int> UnknownSlots(bool[] generatedFlags)
    {
        var slots = new List<int>();
        for (int i = 0; i < generatedFlags.Length; i++)
        {
            if (generatedFlags[i]) slots.Add(i);
        }
        return slots;
    }

    private static void StarUsage(CardDb db, IEnumerable<int> knownBaseIds, out int highStar, out int fiveStar)
    {
        highStar = 0;
        fiveStar = 0;
        foreach (int id in knownBaseIds)
        {
            int star = StarOrOne(db, id);
            if (star >= 4) highStar++;
            if (star == 5) fiveStar++;
        }
    }

    /// <summary>Python <c>_is_legal_unknown_assignment</c>: distinct ids and at most two 4+ star and one 5 star card in the deck.</summary>
    private static bool IsLegalAssignment(CardDb db, ReadOnlySpan<int> assignedIds, int knownHigh, int knownFive)
    {
        int high = knownHigh, five = knownFive;
        for (int i = 0; i < assignedIds.Length; i++)
        {
            int id = BaseCardId(assignedIds[i]);
            for (int j = 0; j < i; j++)
            {
                if (BaseCardId(assignedIds[j]) == id) return false;
            }
            int star = StarOrOne(db, id);
            if (star >= 4) high++;
            if (star == 5) five++;
        }
        return high <= 2 && five <= 1;
    }

    private static bool IsLegalSingle(CardDb db, int id, int knownHigh, int knownFive)
    {
        int star = StarOrOne(db, BaseCardId(id));
        return knownHigh + (star >= 4 ? 1 : 0) <= 2 && knownFive + (star == 5 ? 1 : 0) <= 1;
    }

    private static List<int> KnownOpponentBaseIds(GameState g, int oppPlayer, bool[] oppGenerated)
    {
        var ids = new List<int>();
        for (int slot = 0; slot < g.HandCount(oppPlayer); slot++)
        {
            if (oppGenerated[slot]) continue;
            ids.Add(BaseCardId(g.InstanceId(g.HandInstance(oppPlayer, slot))));
        }
        return ids;
    }

    /// <summary>
    /// Python <c>_build_legal_unknown_candidates</c>: every database card (database order) that is not already known
    /// (board, and the non-generated cards of both hands), optionally restricted to an NPC card pool, and that fits
    /// the opponent's deck limits together with the opponent's known cards.
    /// </summary>
    public static List<UnknownCandidate> LegalCandidates(CardDb db, GameState g, int oppPlayer, bool[][] generated, IReadOnlyList<int>? npcIds)
    {
        var knownGlobal = new HashSet<int>();
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = g.BoardInstance(cell);
            if (inst < 0) continue;
            int id = BaseCardId(g.InstanceId(inst));
            if (id != -1) knownGlobal.Add(id);
        }
        for (int p = 0; p < 2; p++)
        {
            for (int slot = 0; slot < g.HandCount(p); slot++)
            {
                if (generated[p][slot]) continue;
                int id = BaseCardId(g.InstanceId(g.HandInstance(p, slot)));
                if (id != -1) knownGlobal.Add(id);
            }
        }

        StarUsage(db, KnownOpponentBaseIds(g, oppPlayer, generated[oppPlayer]), out int knownHigh, out int knownFive);
        var npcSet = new HashSet<int>(ParseIdList(npcIds));

        var result = new List<UnknownCandidate>();
        foreach (var def in db.Cards)
        {
            if (npcSet.Count > 0 && !npcSet.Contains(def.Id)) continue;
            if (knownGlobal.Contains(def.Id)) continue;
            if (!IsLegalSingle(db, def.Id, knownHigh, knownFive)) continue;
            result.Add(new UnknownCandidate(def.Id, def.Up, def.Right, def.Down, def.Left, def.TypeId));
        }
        return result;
    }

    /// <summary>Python <c>_unique_unknown_candidates</c>: by (id, base edges, type), first occurrence wins.</summary>
    public static List<UnknownCandidate> Unique(IReadOnlyList<UnknownCandidate> candidates)
    {
        var seen = new HashSet<UnknownCandidate>();
        var unique = new List<UnknownCandidate>();
        foreach (var c in candidates)
        {
            if (seen.Add(c)) unique.Add(c);
        }
        return unique;
    }

    /// <summary>
    /// Python <c>_score_unknown_candidate(card, board_state, rules, owner)</c>: the largest immediate threat the
    /// candidate could pose on any empty cell, times ten, plus a small value-pressure term. The candidate's display
    /// edges are its base edges (a fresh clone), its effective edges are the clamped base edges.
    /// </summary>
    public static double ThreatScore(UnknownCandidate c, GameState g, RuleSet rules, byte opponentOwner)
    {
        Span<int> baseEdges = stackalloc int[4];
        baseEdges[Side.Up] = c.Up;
        baseEdges[Side.Right] = c.Right;
        baseEdges[Side.Down] = c.Down;
        baseEdges[Side.Left] = c.Left;

        Span<int> plusSum = stackalloc int[4];
        Span<int> plusSize = stackalloc int[4];
        Span<int> plusEnemies = stackalloc int[4];

        double bestScore = 0.0;
        for (int cell = 0; cell < 9; cell++)
        {
            if (!g.IsEmpty(cell)) continue;
            int row = cell / 3, col = cell % 3;

            double score = 0.0;
            int sameCount = 0, sameEnemies = 0;
            plusSum.Clear();
            plusSize.Clear();
            plusEnemies.Clear();
            int groups = 0;

            for (int k = 0; k < 4; k++)
            {
                int nr = row + Dr[k], nc = col + Dc[k];
                if (nr < 0 || nr > 2 || nc < 0 || nc > 2) continue;
                int target = g.BoardInstance(nr * 3 + nc);
                if (target < 0) continue;

                bool enemy = g.InstanceOwner(target) != opponentOwner;
                int myValue = Math.Max(1, Math.Min(10, baseEdges[MySide[k]]));
                int targetValue = g.InstanceEffective(target, OppSide[k]);
                if (enemy && GameState.CompareValues(myValue, targetValue, rules) == 1) score += 6.0;

                if (myValue == targetValue)
                {
                    sameCount++;
                    if (enemy) sameEnemies++;
                }

                int sum = myValue + targetValue;
                int group = -1;
                for (int j = 0; j < groups; j++)
                {
                    if (plusSum[j] == sum) { group = j; break; }
                }
                if (group < 0)
                {
                    group = groups++;
                    plusSum[group] = sum;
                }
                plusSize[group]++;
                if (enemy) plusEnemies[group]++;
            }

            if ((rules & RuleSet.Same) != 0 && sameCount >= 2 && sameEnemies > 0) score += 8.0 * sameEnemies;
            if ((rules & RuleSet.Plus) != 0)
            {
                for (int j = 0; j < groups; j++)
                {
                    if (plusSize[j] >= 2 && plusEnemies[j] > 0) score += 8.0 * plusEnemies[j];
                }
            }
            if (score > bestScore) bestScore = score;
        }

        int maxEdge = Math.Max(Math.Max(c.Up, c.Right), Math.Max(c.Down, c.Left));
        int sumEdges = c.Up + c.Right + c.Down + c.Left;
        double valuePressure = maxEdge + sumEdges / 40.0;
        return bestScore * 10.0 + valuePressure;
    }

    /// <summary>
    /// Python's ranking in <c>_build_endgame_scenarios</c>: <c>sorted(key=(threat score, sum of display edges, -card_id),
    /// reverse=True)</c>, stable (equal keys keep their relative order).
    /// </summary>
    public static List<UnknownCandidate> Rank(IReadOnlyList<UnknownCandidate> candidates, GameState g, RuleSet rules, byte opponentOwner)
    {
        int n = candidates.Count;
        var scores = new double[n];
        var items = new UnknownCandidate[n];
        for (int i = 0; i < n; i++)
        {
            items[i] = candidates[i];
            scores[i] = ThreatScore(candidates[i], g, rules, opponentOwner);
        }

        // stable insertion sort, descending by the key tuple
        for (int i = 1; i < n; i++)
        {
            var item = items[i];
            double score = scores[i];
            int j = i - 1;
            while (j >= 0 && KeyLess(scores[j], items[j], score, item))
            {
                items[j + 1] = items[j];
                scores[j + 1] = scores[j];
                j--;
            }
            items[j + 1] = item;
            scores[j + 1] = score;
        }
        return new List<UnknownCandidate>(items);
    }

    /// <summary>Python tuple comparison <c>(score, edge sum, -(id or 0)) &lt;</c>.</summary>
    private static bool KeyLess(double scoreA, UnknownCandidate a, double scoreB, UnknownCandidate b)
    {
        if (scoreA != scoreB) return scoreA < scoreB;
        int sumA = a.Up + a.Right + a.Down + a.Left, sumB = b.Up + b.Right + b.Down + b.Left;
        if (sumA != sumB) return sumA < sumB;
        int idA = a.Id == -1 ? 0 : a.Id, idB = b.Id == -1 ? 0 : b.Id;
        return -idA < -idB;
    }

    /// <summary>
    /// Python <c>_build_endgame_scenarios</c>. <paramref name="generated"/> holds, per player and hand slot, Python's
    /// <c>_is_generated_unknown_card</c>; <paramref name="oppPlayer"/> is the opponent (the side that is not to move).
    /// <paramref name="fallbackCandidates"/> replaces Python's random <c>handler.generate_opponent_cards</c> for the
    /// case that no legal candidate exists; without it that case reports <see cref="ScenarioBuild.RequiresFallback"/>.
    /// </summary>
    public static ScenarioBuild Build(CardDb db, GameState baseState, int oppPlayer, bool[][] generated, int sampleCount,
        IReadOnlyList<int>? npcIds = null, IReadOnlyList<UnknownCandidate>? fallbackCandidates = null)
    {
        var unknownSlots = UnknownSlots(generated[oppPlayer]);
        if (unknownSlots.Count == 0)
            return new ScenarioBuild { Scenarios = new List<GameState> { baseState } };

        var legal = LegalCandidates(db, baseState, oppPlayer, generated, npcIds);
        IReadOnlyList<UnknownCandidate> candidates;
        if (legal.Count > 0)
        {
            candidates = legal;
        }
        else if (fallbackCandidates is not null)
        {
            candidates = fallbackCandidates;
        }
        else
        {
            return new ScenarioBuild { RequiresFallback = true };
        }

        var unique = Unique(candidates);
        if (unique.Count == 0)
            return new ScenarioBuild { Scenarios = new List<GameState> { baseState }, LegalCandidateCount = legal.Count };

        byte oppOwner = (byte)oppPlayer; // player 0 = red, 1 = blue
        var ranked = Rank(unique, baseState, baseState.Rules, oppOwner);

        int unknownCount = unknownSlots.Count;
        int candidateLimit = Math.Min(ranked.Count, Math.Max(sampleCount * unknownCount, unknownCount));
        int scenarioLimit = Math.Max(1, sampleCount);

        StarUsage(db, KnownOpponentBaseIds(baseState, oppPlayer, generated[oppPlayer]), out int knownHigh, out int knownFive);
        var scenarios = new List<GameState>();
        var seenAssignments = new HashSet<string>();
        var chosen = new int[unknownCount];
        var ids = new int[unknownCount];

        // itertools.permutations(ranked[:candidate_limit], unknown_count): lexicographic by position.
        var used = new bool[candidateLimit];
        bool Recurse(int depth)
        {
            if (depth == unknownCount)
            {
                for (int k = 0; k < unknownCount; k++) ids[k] = ranked[chosen[k]].Id;
                string key = string.Join(",", ids);
                if (seenAssignments.Contains(key)) return false;
                if (!IsLegalAssignment(db, ids, knownHigh, knownFive)) return false;
                seenAssignments.Add(key);
                scenarios.Add(CopyWithAssignment(baseState, oppPlayer, unknownSlots, ranked, chosen));
                return scenarios.Count >= scenarioLimit;
            }
            for (int i = 0; i < candidateLimit; i++)
            {
                if (used[i]) continue;
                chosen[depth] = i;
                used[i] = true;
                bool done = Recurse(depth + 1);
                used[i] = false;
                if (done) return true;
            }
            return false;
        }

        if (candidateLimit >= unknownCount) Recurse(0);

        return new ScenarioBuild
        {
            Scenarios = scenarios.Count > 0 ? scenarios : new List<GameState> { baseState },
            LegalCandidateCount = legal.Count,
        };
    }

    /// <summary>
    /// Python <c>_copy_state_with_unknown_assignment</c>: a copy of the state with each unknown slot of the opponent's
    /// hand replaced by <c>candidate.copy()</c> (no type modifier, display edges = clamped base edges), keeping the slot's
    /// owner and <c>can_use</c>. Type modifiers are NOT recomputed (PY-QUIRK noted in CSHARP_MIGRATION.md, section 1.9).
    /// </summary>
    private static GameState CopyWithAssignment(GameState baseState, int oppPlayer, List<int> unknownSlots,
        List<UnknownCandidate> ranked, int[] chosen)
    {
        var snapshot = baseState.ToSnapshot();
        for (int k = 0; k < unknownSlots.Count; k++)
        {
            var c = ranked[chosen[k]];
            var old = snapshot.Hands[oppPlayer][unknownSlots[k]];
            var card = new CardSnapshot(c.Up, c.Right, c.Down, c.Left, old.Owner, c.Id, CardTypes.NameOf(c.TypeId)).WithAppliedModifier(0);
            card.CanUse = old.CanUse;
            snapshot.Hands[oppPlayer][unknownSlots[k]] = card;
        }
        return GameState.FromSnapshot(snapshot);
    }
}
