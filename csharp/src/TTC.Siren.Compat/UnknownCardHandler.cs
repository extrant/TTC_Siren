namespace TTC.Siren.Compat;

/// <summary>
/// Port of <c>ai/unknown_card_handler.py</c> (Phase 8A, decision A-23): the rule-driven random sampling of the unknown opponent/own cards.
/// The LOGIC IS UNCHANGED — same pools, same weights, same order of random draws, drawn from <see cref="PythonRandom"/> — so that with
/// the same seed it produces exactly the cards the Python handler produces. Improving the sampling is a later phase that uses this port
/// as its measurable baseline; do not "fix" anything here. Quirks are kept and marked <c>PY-QUIRK</c>.
/// The configuration (<c>config/unknown_card_config.py</c>) is baked into constants below; the golden data is bound to that file's hash,
/// so a change there makes the parity tests fail until this port follows.
/// Functions the Python module defines but never calls (<c>_intelligent_star_distribution</c>, <c>_select_best_cards_for_rules</c>,
/// <c>_simple_star_matching</c>, <c>_draft_fallback_sampling</c>) are not ported.
/// </summary>
public sealed class UnknownCardHandler
{
    // ---- config/unknown_card_config.py ----
    private const int MaxCardsPerUnknown = 150;          // max_unknown_cards_per_hand
    private const int FallbackSampleMultiplier = 5;      // fallback_sample_multiplier
    // performance_mode = False, aggressive_sampling = False and advanced.opponent_behavior_modeling = True select the branches below;
    // the other branches of the Python code are dead with this configuration and are not ported.
    private static readonly int[] PreferredSameValues = { 2, 3, 4, 5, 6 };
    private const double SameComboSetupRatio = 0.5, SameDefensiveRatio = 0.3;
    private const double PlusSumComboRatio = 0.5, PlusDefensiveRatio = 0.3;
    private const double ReverseLowPreference = 2.0, ReverseHighAvoidance = 0.3;
    private const int ReverseMaxPreferred = 5;
    private const double AceKillerPreference = 1.8, AceMidPreference = 0.7;
    private static readonly int[] StarLimits = { 0, 2, 2, 2, 2, 2 }; // draft_mode.total_star_limits, index = star

    private readonly IReadOnlyList<LCard> _all;
    private readonly CardCatalog _catalog;
    private readonly PythonRandom _rng;

    public UnknownCardHandler(CardCatalog catalog, PythonRandom rng)
    {
        _catalog = catalog;
        _all = catalog.AllCards;
        _rng = rng;
    }

    // =====================================================================================================================
    // entry points

    /// <summary><c>generate_unknown_cards</c>: the estimate for the player's OWN unknown cards.</summary>
    public List<LCard> GenerateUnknownCards(int count, IReadOnlyList<string> rules, HashSet<int> usedCards, LBoard? board,
        IReadOnlyList<LCard>? knownHand, string? owner, bool canUse)
    {
        int actualSamples = rules.Contains("选拔") ? count : Math.Min(count * FallbackSampleMultiplier, MaxCardsPerUnknown);
        var available = _all.Where(c => !usedCards.Contains(c.CardId!.Value)).ToList();
        if (available.Count == 0) available = new List<LCard>(_all);
        return SmartSamplingByRules(available, actualSamples, rules, board, knownHand, owner, canUse);
    }

    /// <summary><c>generate_opponent_cards</c>: the estimate for the opponent's unknown cards (behaviour modelling).</summary>
    public List<LCard> GenerateOpponentCards(int count, IReadOnlyList<string> rules, HashSet<int> usedCards, LBoard? board,
        IReadOnlyList<LCard>? knownHand, string? owner, bool canUse)
    {
        double occupancy = BoardOccupancy(board);
        int baseMultiplier = FallbackSampleMultiplier;
        int sampleMultiplier;
        if (occupancy >= 0.65) sampleMultiplier = Math.Max(baseMultiplier, count == 1 ? 16 : 12);
        else if (occupancy >= 0.45 || count <= 2) sampleMultiplier = Math.Max(baseMultiplier, count <= 2 ? 12 : 8);
        else sampleMultiplier = baseMultiplier;
        int actualSamples = Math.Min(count * sampleMultiplier, MaxCardsPerUnknown);

        var available = new List<LCard>(_all);

        switch (DeterminePrimaryRule(rules))
        {
            case "选拔": return SampleForDraftRule(available, actualSamples, board, knownHand, owner, canUse, rules);
            case "同数": return SampleStrategicSameNumberCards(available, actualSamples, board, knownHand, owner, canUse);
            case "加算": return SampleForAdditionRule(available, actualSamples, board, knownHand, owner, canUse, isOpponent: true);
            case "同类强化":
            case "同类弱化": return SampleForSameTypeRules(available, actualSamples, board, knownHand, owner, canUse, isOpponent: true);
            case "逆转": return EnhancedReverseSampling(available, actualSamples, owner, canUse);
            case "王牌杀手": return SampleStrategicAceKillerCards(available, actualSamples, owner, canUse);
            default: return SampleBalancedCards(available, actualSamples, owner, canUse, isOpponent: true);
        }
    }

    /// <summary>The rule family that drives the sampling (for the explanation facts): the opponent model's primary rule, or the first matching family for the own hand.</summary>
    public string PrimaryRuleOf(IReadOnlyList<string> rules, bool isOpponent)
    {
        if (isOpponent) return DeterminePrimaryRule(rules);
        foreach (var name in new[] { "选拔", "同数", "加算" })
            if (rules.Contains(name)) return name;
        if (rules.Contains("同类强化")) return "同类强化";
        if (rules.Contains("同类弱化")) return "同类弱化";
        foreach (var name in new[] { "逆转", "王牌杀手" })
            if (rules.Contains(name)) return name;
        return "balanced";
    }

    private static double BoardOccupancy(LBoard? board) => board is null ? 0.0 : board.Occupancy() / 9.0;

    private List<LCard> SmartSamplingByRules(List<LCard> available, int maxSamples, IReadOnlyList<string> rules, LBoard? board,
        IReadOnlyList<LCard>? knownHand, string? owner, bool canUse)
    {
        if (rules.Contains("选拔")) return SampleForDraftRule(available, maxSamples, board, knownHand, owner, canUse, rules);
        if (rules.Contains("同数")) return SampleForSameNumberRule(available, maxSamples, board, knownHand, owner, canUse, isOpponent: false);
        if (rules.Contains("加算")) return SampleForAdditionRule(available, maxSamples, board, knownHand, owner, canUse, isOpponent: false);
        if (rules.Contains("同类强化") || rules.Contains("同类弱化"))
            return SampleForSameTypeRules(available, maxSamples, board, knownHand, owner, canUse, isOpponent: false);
        if (rules.Contains("逆转")) return EnhancedReverseSampling(available, maxSamples, owner, canUse);
        if (rules.Contains("王牌杀手")) return SampleStrategicAceKillerCards(available, maxSamples, owner, canUse);
        return SampleBalancedCards(available, maxSamples, owner, canUse, isOpponent: false);
    }

    /// <summary><c>_determine_primary_rule</c>: the applicable rule of highest priority (the first one on ties).</summary>
    private static string DeterminePrimaryRule(IReadOnlyList<string> rules)
    {
        static int Priority(string rule) => rule switch
        {
            "选拔" => 15, "同数" => 10, "加算" => 9, "同类强化" => 8, "同类弱化" => 7, "逆转" => 6, "王牌杀手" => 5, "秩序" => 3, "混乱" => 2,
            _ => 0,
        };
        string best = "balanced";
        int bestPriority = int.MinValue;
        foreach (var rule in rules)
        {
            int p = Priority(rule);
            if (p == 0) continue;
            if (p > bestPriority)
            {
                bestPriority = p;
                best = rule;
            }
        }
        return best;
    }

    // =====================================================================================================================
    // strategies

    private List<LCard> SampleStrategicSameNumberCards(List<LCard> available, int count, LBoard? board, IReadOnlyList<LCard>? knownHand,
        string? owner, bool canUse)
    {
        var trap = new List<LCard>();
        var counter = new List<LCard>();
        foreach (var card in available)
        {
            if (IsExcellentTrapCard(card)) trap.Add(card);
            else if (IsCounterPlayCard(card)) counter.Add(card);
            // else: a defensive card, which the Python code classifies and then never uses
        }

        var result = new List<LCard>();
        int trapCount = Math.Max(1, (int)(count * 0.6));
        if (trap.Count > 0) result.AddRange(SampleFromPool(trap, Math.Min(trapCount, trap.Count), owner, canUse, isPrediction: true));

        int remaining = count - result.Count;
        int counterCount = remaining > 0 ? Math.Max(1, (int)(remaining * 0.4)) : 0;
        if (counter.Count > 0 && counterCount > 0)
            result.AddRange(SampleFromPool(counter, Math.Min(counterCount, counter.Count), owner, canUse, isPrediction: true));

        remaining = count - result.Count;
        if (remaining > 0)
        {
            // PY-QUIRK: the cards sampled above carry the prediction id 1000+id while `available` holds plain ids, so this filter never
            // removes anything: already chosen cards can be drawn again.
            var allRemaining = available.Where(c => !result.Any(r => c.CardId == r.CardId)).ToList();
            if (allRemaining.Count > 0) result.AddRange(SampleFromPool(allRemaining, remaining, owner, canUse, isPrediction: true));
        }
        return Take(result, count);
    }

    private List<LCard> SampleForSameNumberRule(List<LCard> available, int count, LBoard? board, IReadOnlyList<LCard>? knownHand,
        string? owner, bool canUse, bool isOpponent)
    {
        var combo = new List<LCard>();
        var defensive = new List<LCard>();
        foreach (var card in available)
        {
            if (IsGoodForSameNumberCombo(card)) combo.Add(card);
            else if (IsDefensiveCard(card)) defensive.Add(card);
        }

        var result = new List<LCard>();
        int comboCount = Math.Max(1, (int)(count * SameComboSetupRatio));
        int defensiveCount = Math.Max(1, (int)(count * SameDefensiveRatio));

        if (combo.Count > 0 && comboCount > 0) result.AddRange(SampleFromPool(combo, comboCount, owner, canUse, isPrediction: isOpponent));

        int remaining = count - result.Count;
        if (defensive.Count > 0 && defensiveCount > 0 && remaining > 0)
            result.AddRange(SampleFromPool(defensive, Math.Min(defensiveCount, remaining), owner, canUse, isPrediction: isOpponent));

        remaining = count - result.Count;
        if (remaining > 0)
        {
            var rest = available.Where(c => !result.Any(r => c.CardId == r.CardId)).ToList();
            if (rest.Count > 0) result.AddRange(SampleFromPool(rest, remaining, owner, canUse, isPrediction: isOpponent));
        }
        return Take(result, count);
    }

    private List<LCard> SampleForAdditionRule(List<LCard> available, int count, LBoard? board, IReadOnlyList<LCard>? knownHand,
        string? owner, bool canUse, bool isOpponent)
    {
        var combo = new List<LCard>();
        var defensive = new List<LCard>();
        foreach (var card in available)
        {
            if (IsGoodForAdditionCombo(card)) combo.Add(card);
            else if (IsDefensiveCard(card)) defensive.Add(card);
        }

        var result = new List<LCard>();
        int comboCount = Math.Max(1, (int)(count * PlusSumComboRatio));
        int defensiveCount = Math.Max(1, (int)(count * PlusDefensiveRatio));

        if (combo.Count > 0 && comboCount > 0) result.AddRange(SampleFromPool(combo, comboCount, owner, canUse, isPrediction: isOpponent));

        int remaining = count - result.Count;
        if (defensive.Count > 0 && defensiveCount > 0 && remaining > 0)
            result.AddRange(SampleFromPool(defensive, Math.Min(defensiveCount, remaining), owner, canUse, isPrediction: isOpponent));

        remaining = count - result.Count;
        if (remaining > 0)
        {
            var rest = available.Where(c => !result.Any(r => c.CardId == r.CardId)).ToList();
            if (rest.Count > 0) result.AddRange(SampleFromPool(rest, remaining, owner, canUse, isPrediction: isOpponent));
        }
        return Take(result, count);
    }

    private List<LCard> SampleForSameTypeRules(List<LCard> available, int count, LBoard? board, IReadOnlyList<LCard>? knownHand,
        string? owner, bool canUse, bool isOpponent)
    {
        var result = new List<LCard>();
        var typePriority = AnalyzeTypePriority(board, knownHand);

        int priorityCount = Math.Max(1, (int)(count * 0.3));
        int balancedCount = Math.Max(1, (int)(count * 0.4));

        if (typePriority.Count > 0 && priorityCount > 0)
        {
            string priorityType = typePriority[0];
            var typeCards = available.Where(c => _catalog.TypeOf(c.CardId) == priorityType).ToList();
            if (typeCards.Count > 0) result.AddRange(SampleFromPool(typeCards, priorityCount, owner, canUse, isPrediction: isOpponent));
        }

        if (balancedCount > 0)
        {
            var balanced = GetBalancedTypeSample(available, balancedCount);
            result.AddRange(SampleFromPool(balanced, balancedCount, owner, canUse, isPrediction: isOpponent));
        }

        int remaining = count - result.Count;
        if (remaining > 0)
        {
            var rest = available.Where(c => !result.Any(r => c.CardId == r.CardId)).ToList();
            if (rest.Count > 0) result.AddRange(SampleFromPool(rest, remaining, owner, canUse, isPrediction: isOpponent));
        }
        return Take(result, count);
    }

    private List<LCard> SampleStrategicAceKillerCards(List<LCard> available, int count, string? owner, bool canUse)
    {
        var aceKiller = new List<LCard>();
        var midValue = new List<LCard>();
        var other = new List<LCard>();
        int aceCopies = (int)(AceKillerPreference * 10);
        int midCopies = (int)(AceMidPreference * 10);
        foreach (var card in available)
        {
            int[] values = { card.Up, card.Right, card.Down, card.Left };
            bool hasAce = values.Contains(1) || values.Contains(10);
            if (hasAce) for (int i = 0; i < aceCopies; i++) aceKiller.Add(card);
            else if (values.Any(v => v is >= 4 and <= 7)) for (int i = 0; i < midCopies; i++) midValue.Add(card);
            else other.Add(card);
        }

        var allWeighted = new List<LCard>(aceKiller.Count + midValue.Count + other.Count);
        allWeighted.AddRange(aceKiller);
        allWeighted.AddRange(midValue);
        allWeighted.AddRange(other);
        if (allWeighted.Count >= count)
        {
            var sampled = _rng.Sample(allWeighted, count);
            var unique = DedupById(sampled, count);
            return SampleFromPool(Take(unique, count), count, owner, canUse, isPrediction: true);
        }
        return SampleFromPool(Take(available, count), count, owner, canUse, isPrediction: true);
    }

    private List<LCard> EnhancedReverseSampling(List<LCard> available, int count, string? owner, bool canUse)
    {
        var weighted = new List<LCard>();
        foreach (var card in available)
        {
            double avg = (card.Up + card.Right + card.Down + card.Left) / 4.0;
            double weight = avg <= ReverseMaxPreferred ? ReverseLowPreference : avg >= 8 ? ReverseHighAvoidance : 1.0;
            int copies = (int)(weight * 10);
            for (int i = 0; i < copies; i++) weighted.Add(card);
        }

        if (weighted.Count >= count)
        {
            var sampled = _rng.Sample(weighted, count);
            var unique = DedupById(sampled, count);
            return SampleFromPool(Take(unique, count), count, owner, canUse, isPrediction: true);
        }
        return SampleFromPool(Take(available, count), count, owner, canUse, isPrediction: true);
    }

    private List<LCard> SampleForDraftRule(List<LCard> available, int count, LBoard? board, IReadOnlyList<LCard>? knownHand,
        string? owner, bool canUse, IReadOnlyList<string> rules)
    {
        var starUsage = AnalyzeStarUsage(board, knownHand);
        var availableStars = new int[6];
        for (int star = 1; star <= 5; star++) availableStars[star] = Math.Max(0, StarLimits[star] - starUsage[star]);

        var weighted = new List<LCard>();
        foreach (var card in available)
        {
            int starLevel = _catalog.StarOf(card.CardId) ?? 1;
            // `available_stars.get(star_level, 0) > 0` then `== 0`: a star outside 1..5 (the all-zero rows have star 0) counts as 0 left
            int left = starLevel is >= 1 and <= 5 ? availableStars[starLevel] : 0;
            double baseWeight = left > 0 ? 2.0 : 0.3;
            double ruleScore = CalculateCardRuleScore(card, rules);
            double cornerScore = CalculateCornerStrategyScore(card, board);
            double finalWeight = baseWeight * (1.0 + ruleScore * 0.3 + cornerScore * 0.4);
            int copies = Math.Max(1, (int)(finalWeight * 10));
            for (int i = 0; i < copies; i++) weighted.Add(card);
        }

        if (weighted.Count < count)
            return SampleFromPool(available, Math.Min(count, available.Count), owner, canUse, isPrediction: true);

        var sampled = _rng.Sample(weighted, count);
        var unique = new List<LCard>();
        var seen = new HashSet<int?>();
        foreach (var card in sampled)
        {
            if (seen.Add(card.CardId)) unique.Add(card);
            if (unique.Count >= count) break;
        }
        if (unique.Count < count)
        {
            var remaining = available.Where(c => !seen.Contains(c.CardId)).ToList();
            int additional = Math.Min(count - unique.Count, remaining.Count);
            if (additional > 0) unique.AddRange(_rng.Sample(remaining, additional));
        }
        return SampleFromPool(Take(unique, count), count, owner, canUse, isPrediction: true);
    }

    private List<LCard> SampleBalancedCards(List<LCard> available, int count, string? owner, bool canUse, bool isOpponent)
    {
        // star_distribution of _sample_balanced_cards, in dict order
        (int Star, double Ratio)[] distribution = { (1, 0.4), (2, 0.3), (3, 0.2), (4, 0.08), (5, 0.02) };
        var result = new List<LCard>();
        foreach (var (star, ratio) in distribution)
        {
            int starCount = Math.Max(1, (int)(count * ratio));
            var starCards = available.Where(c => (_catalog.StarOf(c.CardId) ?? 1) == star).ToList();
            if (starCards.Count > 0 && starCount > 0)
            {
                var sampled = SampleFromPool(starCards, starCount, owner, canUse, isPrediction: isOpponent);
                result.AddRange(sampled);
                // the sampled cards are copies whose id is 1000+id for predictions: for the opponent the filter below removes nothing (PY-QUIRK)
                var sampledIds = new HashSet<int?>(sampled.Select(c => c.CardId));
                available = available.Where(c => !sampledIds.Contains(c.CardId)).ToList();
            }
        }
        int remaining = count - result.Count;
        if (remaining > 0 && available.Count > 0) result.AddRange(SampleFromPool(available, remaining, owner, canUse, isPrediction: isOpponent));
        return Take(result, count);
    }

    /// <summary><c>_sample_cards_from_pool</c>: samples without replacement and returns NEW cards (id 1000+id for predictions).</summary>
    private List<LCard> SampleFromPool(IReadOnlyList<LCard> pool, int count, string? owner, bool canUse, bool isPrediction)
    {
        var result = new List<LCard>();
        if (pool.Count == 0) return result;
        count = Math.Min(count, pool.Count);
        var sampled = _rng.Sample(pool, count);
        foreach (var card in sampled)
        {
            int? cardId = card.CardId;
            if (isPrediction) cardId = 1000 + card.CardId;
            var created = new LCard(card.Up, card.Right, card.Down, card.Left, owner, cardId, card.CardType, canUse);
            if (isPrediction)
            {
                created.IsGenerated = true;
                created.IsPrediction = true;
            }
            result.Add(created);
        }
        return result;
    }

    // =====================================================================================================================
    // helpers

    private static List<LCard> Take(List<LCard> list, int count) => list.Count <= count ? list : list.GetRange(0, count);

    private static List<LCard> DedupById(List<LCard> sampled, int count)
    {
        var unique = new List<LCard>();
        var seen = new HashSet<int?>();
        foreach (var card in sampled)
        {
            if (seen.Add(card.CardId)) unique.Add(card);
            if (unique.Count >= count) break;
        }
        return unique;
    }

    /// <summary><c>_analyze_type_priority</c>: types by weight (board cards 2, known hand cards 1), highest first, ties in first-seen order.</summary>
    private static List<string> AnalyzeTypePriority(LBoard? board, IReadOnlyList<LCard>? knownHand)
    {
        var counts = new List<(string Type, int Count)>();
        void Add(string type, int n)
        {
            int idx = counts.FindIndex(t => t.Type == type);
            if (idx < 0) counts.Add((type, n));
            else counts[idx] = (type, counts[idx].Count + n);
        }
        if (board is not null)
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var card = board.Get(r, c);
                    if (card is not null && !string.IsNullOrEmpty(card.CardType)) Add(card.CardType, 2);
                }
        if (knownHand is not null)
            foreach (var card in knownHand)
                if (!string.IsNullOrEmpty(card.CardType)) Add(card.CardType, 1);
        // sorted(..., reverse=True) is stable: equal counts keep the first-seen order. LINQ OrderByDescending is stable as well.
        return counts.Select((t, i) => (t, i)).OrderByDescending(x => x.t.Count).ThenBy(x => x.i).Select(x => x.t.Type).ToList();
    }

    /// <summary><c>_get_balanced_type_sample</c>: round-robin over the types present (None = untyped, in first-seen order), a random card each.</summary>
    private List<LCard> GetBalancedTypeSample(List<LCard> available, int count)
    {
        var order = new List<string?>();
        var groups = new Dictionary<string, List<LCard>>();
        List<LCard>? untyped = null;
        foreach (var card in available)
        {
            string? type = _catalog.TypeOf(card.CardId);
            if (type is null)
            {
                if (untyped is null)
                {
                    untyped = new List<LCard>();
                    order.Add(null);
                }
                untyped.Add(card);
            }
            else
            {
                if (!groups.TryGetValue(type, out var list))
                {
                    list = new List<LCard>();
                    groups[type] = list;
                    order.Add(type);
                }
                list.Add(card);
            }
        }
        var result = new List<LCard>();
        if (order.Count == 0) return result; // Python: `types[i % len(types)]` on an empty list cannot happen with a non-empty pool
        for (int i = 0; i < count; i++)
        {
            string? typeName = order[i % order.Count];
            var group = typeName is null ? untyped! : groups[typeName];
            if (group.Count > 0)
            {
                var card = _rng.Choice(group);
                result.Add(card);
                group.Remove(card); // removes the first equal entry; entries are unique objects, so this is the chosen one
            }
        }
        return result;
    }

    /// <summary><c>_analyze_star_usage</c>. PY-QUIRK: a board card whose star is 0 (the all-zero rows) raises <c>KeyError: 0</c> in Python.</summary>
    private int[] AnalyzeStarUsage(LBoard? board, IReadOnlyList<LCard>? knownHand)
    {
        var usage = new int[6];
        if (board is not null)
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var card = board.Get(r, c);
                    if (card is { CardId: > 0 } && _catalog.HasStar(card.CardId))
                    {
                        int star = _catalog.StarOf(card.CardId)!.Value;
                        if (star is < 1 or > 5) throw new PyException(star.ToString());
                        usage[star]++;
                    }
                }
        if (knownHand is not null)
            foreach (var card in knownHand)
            {
                if (card.CardId is > 0 && _catalog.HasStar(card.CardId) && !(card.Up == 0 && card.Right == 0 && card.Down == 0 && card.Left == 0))
                {
                    int star = _catalog.StarOf(card.CardId)!.Value;
                    if (star is < 1 or > 5) throw new PyException(star.ToString());
                    usage[star]++;
                }
            }
        return usage;
    }

    private double CalculateCardRuleScore(LCard card, IReadOnlyList<string> rules)
    {
        double score = 0.0;
        double avg = (card.Up + card.Right + card.Down + card.Left) / 4.0;
        score += avg * 0.1;
        int[] values = { card.Up, card.Right, card.Down, card.Left };

        if (rules.Contains("同数") && values.Distinct().Count() < 4) score += 2.0;

        if (rules.Contains("加算"))
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    if (i != j && values[i] + values[j] is 8 or 10 or 12) score += 1.5;

        if (rules.Contains("逆转"))
        {
            if (avg <= 5) score += 3.0;
            else if (avg >= 8) score -= 2.0;
        }

        if (rules.Contains("王牌杀手")) score += values.Count(v => v is 1 or 10) * 2.0;

        if (rules.Contains("同类强化") && !string.IsNullOrEmpty(card.CardType)) score += 1.0;
        if (rules.Contains("同类弱化") && !string.IsNullOrEmpty(card.CardType)) score += 0.5;
        return score;
    }

    private static double CalculateCornerStrategyScore(LCard card, LBoard? board)
    {
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        double score = 0.0;
        int high = values.Count(v => v >= 8);
        int low = values.Count(v => v <= 4);

        if (high >= 3)
        {
            score += 5.0;
            if (low >= 1) score += 2.0;
        }
        else if (high >= 2)
        {
            score += 3.0;
            var highPositions = Enumerable.Range(0, 4).Where(i => values[i] >= 8).ToList();
            if (AreAdjacentSides(highPositions)) score += 2.0;
        }
        else if (high == 1)
        {
            score += 1.0;
        }

        var acePositions = Enumerable.Range(0, 4).Where(i => values[i] == 10).ToList();
        if (acePositions.Count >= 2)
        {
            score += 4.0;
            if (AreAdjacentSides(acePositions))
            {
                score += 3.0;
                if (IsOptimalAaCombination(acePositions)) score += 5.0;
            }
        }

        if (board is not null) score += EvaluateCornerAvailability(values, board);

        score -= CalculateWeaknessExposurePenalty(values);
        return score;
    }

    private static bool AreAdjacentSides(List<int> positions)
    {
        if (positions.Count < 2) return false;
        // adjacency: 0(up)-1(right), 1(right)-2(down), 2(down)-3(left), 3(left)-0(up)
        static bool Adjacent(int a, int b) => (a, b) switch
        {
            (0, 1) or (0, 3) or (1, 0) or (1, 2) or (2, 1) or (2, 3) or (3, 0) or (3, 2) => true,
            _ => false,
        };
        for (int i = 0; i < positions.Count; i++)
            for (int j = i + 1; j < positions.Count; j++)
                if (Adjacent(positions[i], positions[j])) return true;
        return false;
    }

    private static bool IsOptimalAaCombination(List<int> acePositions)
    {
        if (acePositions.Count < 2) return false;
        if (acePositions.Contains(1) && acePositions.Contains(2)) return true;
        if (acePositions.Contains(3) && acePositions.Contains(2)) return true;
        if (acePositions.Contains(1) && acePositions.Contains(0)) return true;
        if (acePositions.Contains(3) && acePositions.Contains(0)) return true;
        return false;
    }

    private static double EvaluateCornerAvailability(int[] values, LBoard board)
    {
        (int Row, int Col)[] corners = { (0, 0), (0, 2), (2, 0), (2, 2) };
        var available = corners.Where(p => board.Get(p.Row, p.Col) is null).ToList();
        if (available.Count == 0) return 0.0;
        double bonus = 0.0;
        foreach (var corner in available) bonus += CalculateCornerFitScore(values, corner.Row, corner.Col) * 0.5;
        return Math.Min(bonus, 3.0);
    }

    private static double CalculateCornerFitScore(int[] values, int row, int col)
    {
        double fit = 0.0;
        if (row == 0 && col == 0)
        {
            if (values[1] >= 8) fit += 2.0;
            if (values[2] >= 8) fit += 2.0;
        }
        else if (row == 0 && col == 2)
        {
            if (values[3] >= 8) fit += 2.0;
            if (values[2] >= 8) fit += 2.0;
        }
        else if (row == 2 && col == 0)
        {
            if (values[0] >= 8) fit += 2.0;
            if (values[1] >= 8) fit += 2.0;
        }
        else if (row == 2 && col == 2)
        {
            if (values[0] >= 8) fit += 2.0;
            if (values[3] >= 8) fit += 2.0;
        }
        return fit;
    }

    private static double CalculateWeaknessExposurePenalty(int[] values)
    {
        double penalty = 0.0;
        int veryWeak = values.Count(v => v <= 3);
        int weak = values.Count(v => v is >= 4 and <= 5);
        penalty += veryWeak * 1.5;
        penalty += weak * 0.8;
        if (values.Max() - values.Min() >= 7) penalty += 2.0;
        return penalty;
    }

    private static bool IsExcellentTrapCard(LCard card)
    {
        if (!IsGoodForSameNumberCombo(card)) return false;
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        int maxCount = values.GroupBy(v => v).Max(g => g.Count());
        return maxCount >= 2;
    }

    private static bool IsCounterPlayCard(LCard card)
    {
        double avg = (card.Up + card.Right + card.Down + card.Left) / 4.0;
        return 5.5 <= avg && avg <= 7.5;
    }

    private static bool IsGoodForSameNumberCombo(LCard card)
    {
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        bool hasPreferred = values.Any(v => PreferredSameValues.Contains(v));
        bool hasDuplicates = values.Distinct().Count() < 4;
        if (values.Contains(1) || values.Contains(10)) return false; // avoid_extreme_values
        return hasPreferred || hasDuplicates;
    }

    private static bool IsGoodForAdditionCombo(LCard card)
    {
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
                if (i == j) continue;
                int sum = values[i] + values[j];
                if ((5 <= sum && sum <= 9) || (10 <= sum && sum <= 14)) return true;
            }
        foreach (int target in new[] { 8, 10, 12 })
            foreach (int value in values)
            {
                int complement = target - value;
                if (values.Contains(complement) && complement != value) return true;
            }
        return false;
    }

    private static bool IsDefensiveCard(LCard card) => (card.Up + card.Right + card.Down + card.Left) / 4.0 >= 6.0;
}
