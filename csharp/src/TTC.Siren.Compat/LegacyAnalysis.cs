using System.Globalization;

namespace TTC.Siren.Compat;

/// <summary>A move in the legacy model: the card OBJECT of the side to move's hand and the target cell (Python's <c>(card, (row, col))</c>).</summary>
public readonly record struct LMove(LCard Card, int Row, int Col);

/// <summary>
/// Port of the analysis and text functions of <c>ai_server.py</c> that fill the compatibility fields of the <c>/ai_move</c> response
/// (<c>opponent_hand_analysis</c>, <c>win_probability</c>, <c>recommendation</c>, the card description, the search summary). They are
/// deterministic and are ported FAITHFULLY, including their quirks (the make/undo calls that reorder the hand and leave stale display
/// values, D-001/D-003, the copied card that replaces a hand card in <c>count_captured_cards</c>, D-010, prediction ids that defeat
/// a duplicate filter). They stay exactly as plugins know them; the new, richer text goes into the additive <c>explanation</c> field.
/// The order in which they are called matters because they mutate the legacy state: see <see cref="AiMoveCompat"/>.
/// </summary>
public sealed class LegacyAnalysis
{
    private readonly CardCatalog _catalog;

    public LegacyAnalysis(CardCatalog catalog) => _catalog = catalog;

    // =============================================================================================================== formatting

    /// <summary><c>star_map.get(id, '?')</c> as a JSON value: the star as an int, or the string "?".</summary>
    public object StarOrQuestion(int? id) => _catalog.StarOf(id) is { } s ? s : "?";

    private static string StarText(object star) => star is int i ? i.ToString(CultureInfo.InvariantCulture) : (string)star;

    private static string Signed(int value) => value >= 0 ? "+" + value.ToString(CultureInfo.InvariantCulture) : value.ToString(CultureInfo.InvariantCulture);

    /// <summary><c>format_card_display</c>.</summary>
    public static string FormatCardDisplay(LCard card, object star)
    {
        if (card.TypeModifier != 0)
        {
            string modifier = Signed(card.TypeModifier);
            string type = string.IsNullOrEmpty(card.CardType) ? "无类型" : card.CardType;
            return $"U{card.Base[0]} R{card.Base[1]} D{card.Base[2]} L{card.Base[3]} → U{card.Modified(0)} R{card.Modified(1)} D{card.Modified(2)} L{card.Modified(3)} " +
                   $"({type}{modifier}) 星级:{StarText(star)}";
        }
        return $"U{card.Base[0]} R{card.Base[1]} D{card.Base[2]} L{card.Base[3]} 星级:{StarText(star)}";
    }

    /// <summary><c>format_move_for_display</c> (display values, as the search-progress entries show them).</summary>
    public string FormatMoveForDisplay(LMove? move)
    {
        if (move is null) return "无移动";
        var m = move.Value;
        return $"U{m.Card.Up}R{m.Card.Right}D{m.Card.Down}L{m.Card.Left}(★{StarText(StarOrQuestion(m.Card.CardId))}) → ({m.Row},{m.Col})";
    }

    // =============================================================================================================== opponent hand

    /// <summary><c>analyze_opponent_hand</c>.</summary>
    public Dictionary<string, object?> AnalyzeOpponentHand(List<LCard> oppHand, List<string> rules, LBoard board)
    {
        if (oppHand.Count == 0)
            return new Dictionary<string, object?> { ["predicted_cards"] = new List<object?>(), ["total_unknown"] = 0, ["strategy_analysis"] = "无对手手牌信息" };

        var knownCards = new List<Dictionary<string, object?>>();
        var predictedCards = new List<Dictionary<string, object?>>();
        int unknownCount = 0;

        foreach (var card in oppHand)
        {
            if (card.Up == 0 && card.Right == 0 && card.Down == 0 && card.Left == 0)
            {
                unknownCount++;
                continue;
            }
            object star = StarOrQuestion(card.CardId);
            double confidence = 1.0;
            string reasoning = "已知卡牌";
            bool isPredicted = false;
            if (card.CardId is { } id && id != 0)
            {
                if (id >= 1000 || card.IsGenerated)
                {
                    isPredicted = true;
                    unknownCount++;
                    confidence = GetPredictionConfidence(card, rules);
                    reasoning = GetPredictionReasoning(card, rules);
                }
            }
            var info = new Dictionary<string, object?>
            {
                ["card"] = FormatCardDisplay(card, star), ["star"] = star, ["confidence"] = confidence, ["reasoning"] = reasoning, ["is_predicted"] = isPredicted,
            };
            if (isPredicted || confidence < 1.0) predictedCards.Add(info);
            else knownCards.Add(info);
        }

        string strategy = AnalyzeOpponentStrategy(rules);
        string? draftAnalysis = null;
        if (rules.Contains("选拔"))
        {
            draftAnalysis = AnalyzeDraftModeConstraints(oppHand, board);
            if (draftAnalysis.Length > 0) strategy += $" | 选拔分析：{draftAnalysis}";
            else draftAnalysis = null;
        }

        int remainingUnknown = unknownCount - predictedCards.Count;
        if (remainingUnknown > 0) predictedCards.AddRange(PredictUnknownCards(rules, remainingUnknown));

        var all = new List<object?>();
        all.AddRange(knownCards);
        all.AddRange(predictedCards);
        var result = new Dictionary<string, object?>
        {
            ["predicted_cards"] = all.Count > 10 ? all.GetRange(0, 10) : all,
            ["total_unknown"] = unknownCount,
            ["strategy_analysis"] = strategy,
        };
        if (rules.Contains("选拔") && draftAnalysis is not null) result["draft_analysis"] = draftAnalysis;
        return result;
    }

    private static bool HasSameNumberPotential(LCard card) => card.Base.Distinct().Count() < 4;

    private static bool HasAdditionPotential(LCard card)
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (i != j && card.Base[i] + card.Base[j] is 8 or 10 or 12) return true;
        return false;
    }

    private static double Average(IEnumerable<int> values) => values.Sum() / 4.0;

    /// <summary><c>get_prediction_confidence</c>.</summary>
    public static double GetPredictionConfidence(LCard card, IReadOnlyList<string> rules)
    {
        double confidence = 0.6;
        var bonuses = new List<double>();
        if (rules.Contains("选拔")) bonuses.Add(0.85);

        if (rules.Contains("同数") && HasSameNumberPotential(card))
        {
            if (rules.Contains("加算") && HasAdditionPotential(card)) bonuses.Add(0.9);
            else bonuses.Add(0.8);
        }
        else if (rules.Contains("加算") && HasAdditionPotential(card)) bonuses.Add(0.7);

        if (rules.Contains("逆转"))
        {
            double avg = Average(new[] { card.Modified(0), card.Modified(1), card.Modified(2), card.Modified(3) });
            if (avg <= 5) bonuses.Add(0.8);
            else if (avg >= 8) bonuses.Add(0.4);
            else bonuses.Add(0.6);
        }

        if (rules.Contains("王牌杀手"))
        {
            int ace = Enumerable.Range(0, 4).Count(i => card.Modified(i) is 1 or 10);
            bonuses.Add(ace >= 2 ? 0.85 : ace == 1 ? 0.75 : 0.5);
        }

        if ((rules.Contains("同类强化") || rules.Contains("同类弱化")) && !string.IsNullOrEmpty(card.CardType))
            bonuses.Add(rules.Contains("同类强化") ? 0.7 : 0.5);

        if (rules.Contains("秩序") || rules.Contains("混乱")) bonuses.Add(!card.CanUse ? 0.3 : 0.6);

        if (bonuses.Count > 0)
        {
            double baseConfidence = bonuses.Max();
            double multiRuleBonus = Math.Min(0.05 * (bonuses.Count - 1), 0.15);
            confidence = Math.Min(0.95, baseConfidence + multiRuleBonus);
        }
        return confidence;
    }

    /// <summary><c>get_prediction_reasoning</c>.</summary>
    public string GetPredictionReasoning(LCard card, IReadOnlyList<string> rules)
    {
        var reasons = new List<string>();
        if (rules.Contains("选拔"))
        {
            int? star = _catalog.StarOf(card.CardId); // '?' (not in the table: every prediction id >= 1000) equals none of the numbers
            if (star == 5) reasons.Add("选拔模式下的5星王牌，战略价值极高");
            else if (star == 4) reasons.Add("选拔模式下的4星强力卡牌，关键时刻的选择");
            else if (star is 1 or 2) reasons.Add("选拔模式下的低星级卡牌，早期使用较安全");
            else reasons.Add("选拔模式下的中等星级卡牌，平衡性较好");
        }

        if (rules.Contains("同数") && HasSameNumberPotential(card))
        {
            if (rules.Contains("加算") && HasAdditionPotential(card)) reasons.Add("同时适合同数和加算连携的复合战术卡牌");
            else reasons.Add("同数规则下可能用于设置连携陷阱");
        }
        else if (rules.Contains("加算") && HasAdditionPotential(card)) reasons.Add("加算规则下适合数值组合连携");

        if (rules.Contains("逆转"))
        {
            double avg = Average(new[] { card.Modified(0), card.Modified(1), card.Modified(2), card.Modified(3) });
            reasons.Add(avg <= 5 ? "逆转规则下的优势低数值卡牌" : "虽然数值较高，但可能用于特殊战术");
        }

        if (rules.Contains("王牌杀手"))
        {
            bool has = Enumerable.Range(0, 4).Any(i => card.Modified(i) is 1 or 10);
            reasons.Add(has ? "王牌杀手规则下含1或A具有特殊效果" : "王牌杀手规则下的支援卡牌");
        }

        if (reasons.Count == 0)
        {
            int star = _catalog.StarOf(card.CardId) ?? 3;
            if (star <= 2) reasons.Add("低星级卡牌，可能用于早期控场");
            else if (star >= 4) reasons.Add("高星级卡牌，关键时刻的强力选择");
            else reasons.Add("中等星级卡牌，平衡性较好");
        }
        return string.Join("；", reasons);
    }

    /// <summary><c>analyze_draft_mode_constraints</c>. Throws KeyError(0) like Python for a board card of star 0.</summary>
    public string AnalyzeDraftModeConstraints(List<LCard> allHand, LBoard board)
    {
        var usage = new int[6];
        void Count(int? id)
        {
            int star = _catalog.StarOf(id) ?? 1;
            if (star is < 1 or > 5) throw new PyException(star.ToString(CultureInfo.InvariantCulture));
            usage[star]++;
        }
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var card = board.Get(r, c);
                if (card is { CardId: > 0 }) Count(card.CardId);
            }
        foreach (var card in allHand)
            if (card.CardId is > 0 && !(card.Up == 0 && card.Right == 0 && card.Down == 0 && card.Left == 0)) Count(card.CardId);

        var remaining = new int[6];
        for (int star = 1; star <= 5; star++) remaining[star] = Math.Max(0, 2 - usage[star]);

        var parts = new List<string>();
        var exhausted = Enumerable.Range(1, 5).Where(s => remaining[s] == 0).Select(s => s.ToString(CultureInfo.InvariantCulture)).ToList();
        if (exhausted.Count > 0) parts.Add($"{string.Join(",", exhausted)}星已用完");
        int highValueRemaining = remaining[4] + remaining[5];
        if (highValueRemaining > 0) parts.Add($"剩余{highValueRemaining}张高星卡牌");
        var availableStars = Enumerable.Range(1, 5).Where(s => remaining[s] > 0).Select(s => s.ToString(CultureInfo.InvariantCulture)).ToList();
        if (availableStars.Count > 0) parts.Add($"可用星级:{string.Join(",", availableStars)}");
        if (remaining[5] > 0) parts.Add("对手可能保留5星卡牌作为杀招");
        else if (remaining[4] > 0) parts.Add("对手可能依赖4星卡牌");
        return parts.Count > 0 ? string.Join(" | ", parts) : "星级分布正常";
    }

    /// <summary><c>analyze_opponent_strategy</c>.</summary>
    public static string AnalyzeOpponentStrategy(IReadOnlyList<string> rules)
    {
        var strategies = new List<string>();
        if (rules.Contains("选拔")) strategies.Add("严格控制星级配额，注意高星卡牌的战略性使用");
        if (rules.Contains("同数") && rules.Contains("加算")) strategies.Add("对手可能会同时利用同数和加算连携，设置复合陷阱");
        else if (rules.Contains("同数")) strategies.Add("对手可能会设置同数连携陷阱，注意重复数值卡牌");
        else if (rules.Contains("加算")) strategies.Add("对手可能会利用加算规则设置连携，注意互补数值组合");
        if (rules.Contains("逆转")) strategies.Add("偏好低数值卡牌");
        if (rules.Contains("王牌杀手")) strategies.Add("重点使用含1或A的卡牌");
        if (rules.Contains("同类强化")) strategies.Add("集中使用同类型卡牌以触发强化效果");
        if (rules.Contains("同类弱化")) strategies.Add("避免使用相同类型，采用多样化策略");
        return strategies.Count > 0 ? "对手策略：" + string.Join("，", strategies) : "对手采用常规策略，平衡攻防";
    }

    /// <summary><c>predict_unknown_cards</c>.</summary>
    public static List<Dictionary<string, object?>> PredictUnknownCards(IReadOnlyList<string> rules, int count)
    {
        var predictions = new List<Dictionary<string, object?>>();
        void Add(string card, double confidence, string reasoning) =>
            predictions.Add(new Dictionary<string, object?> { ["card"] = card, ["confidence"] = confidence, ["reasoning"] = reasoning });

        if (rules.Contains("选拔")) Add("根据星级配额的战略性卡牌", 0.85, "选拔模式下严格按照星级配额进行卡牌选择");
        if (rules.Contains("同数") && rules.Contains("加算")) Add("同数+加算复合连携卡牌", 0.8, "同时适合同数和加算规则的复合战术卡牌");
        else if (rules.Contains("同数")) Add("重复数值卡牌", 0.7, "同数规则下偏好设置连携陷阱");
        else if (rules.Contains("加算")) Add("互补数值卡牌", 0.6, "加算规则下偏好和数组合");
        if (rules.Contains("逆转") && predictions.Count < count) Add("低数值卡牌", 0.8, "逆转规则下低数值更有优势");
        if (rules.Contains("王牌杀手") && predictions.Count < count) Add("含1或A的卡牌", 0.75, "王牌杀手规则下1和A具有特殊效果");
        while (predictions.Count < count) Add("中等星级卡牌", 0.5, "玩家通常使用平衡的卡组配置");
        return predictions.Count > count ? predictions.GetRange(0, count) : predictions;
    }

    // =============================================================================================================== win probability

    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x / 5));

    /// <summary><c>evaluate_current_position</c>.</summary>
    public static double EvaluateCurrentPosition(LGameState state, string myOwner)
    {
        var (red, blue) = state.CountCards();
        int basic = myOwner == "blue" ? blue - red : red - blue;
        double bonus = 0.0;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var card = state.Board.Get(r, c);
                if (card is null) continue;
                bool corner = (r, c) is (0, 0) or (0, 2) or (2, 0) or (2, 2);
                double weight = corner ? 1.5 : (r, c) == (1, 1) ? 1.2 : 1.0;
                if ((card.Owner == "blue" && myOwner == "blue") || (card.Owner == "red" && myOwner == "red")) bonus += weight;
                else bonus -= weight;
            }
        return basic * 2 + bonus;
    }

    /// <summary><c>calculate_win_probability</c>: makes the move, evaluates, undoes it (the hand order changes, D-001).</summary>
    public Dictionary<string, object?> CalculateWinProbability(LGameState state, LMove move, string myOwner)
    {
        double currentEval = EvaluateCurrentPosition(state, myOwner);
        var record = state.MakeMove(move.Row, move.Col, move.Card);
        if (record is null)
            return new Dictionary<string, object?>
            {
                ["current"] = Py.Round(Sigmoid(currentEval), 3), ["after_move"] = Py.Round(Sigmoid(currentEval), 3), ["confidence"] = 0.5,
            };
        try
        {
            double afterEval = EvaluateCurrentPosition(state, myOwner);
            double currentProb = Sigmoid(currentEval);
            double afterProb = Sigmoid(afterEval);
            return new Dictionary<string, object?>
            {
                ["current"] = Py.Round(currentProb, 3),
                ["after_move"] = Py.Round(afterProb, 3),
                ["confidence"] = Math.Min(0.9, Math.Abs(afterEval) / 10 + 0.6),
            };
        }
        finally
        {
            state.UndoMove(record);
        }
    }

    // =============================================================================================================== recommendation

    /// <summary><c>generate_move_recommendation</c>.</summary>
    public Dictionary<string, object?> GenerateMoveRecommendation(LGameState state, LMove move, string myOwner)
    {
        string reasoning = AnalyzeMoveValue(state, move, myOwner);
        string strategic = AnalyzeStrategicValue(state, move);

        var alternatives = new List<(Dictionary<string, object?> Item, double Value)>();
        var current = state.Players[state.CurrentPlayerIdx];
        var playable = current.PlayableCards(state.Rules);
        var positions = state.Board.AvailablePositions();

        foreach (var altCard in playable.Take(2))
            foreach (var altPos in positions.Take(2))
            {
                if (altCard.CardId != move.Card.CardId || altPos != (move.Row, move.Col))
                {
                    double altEval = EvaluateAlternativeMove(state, new LMove(altCard, altPos.Row, altPos.Col), myOwner);
                    object altStar = StarOrQuestion(altCard.CardId);
                    double value = Py.Round(altEval, 3);
                    alternatives.Add((new Dictionary<string, object?>
                    {
                        ["card"] = $"U{altCard.Up} R{altCard.Right} D{altCard.Down} L{altCard.Left} 星级:{StarText(altStar)}",
                        ["pos"] = new List<object?> { altPos.Row, altPos.Col },
                        ["value"] = value,
                    }, value));
                }
            }

        // alternative_moves.sort(key=value, reverse=True): stable
        var sorted = alternatives.Select((a, i) => (a, i)).OrderByDescending(x => x.a.Value).ThenBy(x => x.i).Select(x => (object?)x.a.Item).Take(3).ToList();
        return new Dictionary<string, object?>
        {
            ["move_reasoning"] = reasoning,
            ["strategic_value"] = strategic,
            ["alternative_moves"] = sorted,
        };
    }

    private double EvaluateAlternativeMove(LGameState state, LMove move, string myOwner)
    {
        var record = state.MakeMove(move.Row, move.Col, move.Card);
        if (record is null) return EvaluateCurrentPosition(state, myOwner);
        try
        {
            return EvaluateCurrentPosition(state, myOwner);
        }
        finally
        {
            state.UndoMove(record);
        }
    }

    /// <summary><c>analyze_move_value</c>.</summary>
    public string AnalyzeMoveValue(LGameState state, LMove move, string myOwner)
    {
        int captured = CountCapturedCards(state, move, myOwner);
        string? combo = AnalyzeComboPotential(state, move);
        if (captured > 0) return combo is not null ? $"此移动可以吃掉{captured}张对手卡牌（{combo}）" : $"此移动可以吃掉{captured}张对手卡牌";
        if (combo is not null) return $"设置连携陷阱：{combo}";
        if ((move.Row, move.Col) is (0, 0) or (0, 2) or (2, 0) or (2, 2)) return "占据角落位置，具有防御优势";
        if ((move.Row, move.Col) == (1, 1)) return "控制中心位置，影响四个方向";
        return "稳定发展，保持场面控制";
    }

    /// <summary>
    /// <c>count_captured_cards</c>. PY-QUIRK (D-010): the move is made with a COPY of the card, so afterwards the hand holds that copy
    /// (appended at the end) instead of the original card.
    /// </summary>
    public int CountCapturedCards(LGameState state, LMove move, string myOwner)
    {
        var original = new Dictionary<(int, int), string?>();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var b = state.Board.Get(r, c);
                if (b is not null) original[(r, c)] = b.Owner;
            }
        var copy = move.Card.Copy();
        copy.Owner = myOwner == "red" ? "red" : "blue";
        var record = state.MakeMove(move.Row, move.Col, copy);
        if (record is null) return 0;
        try
        {
            int captured = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    if (!original.TryGetValue((r, c), out var originalOwner)) continue;
                    var b = state.Board.Get(r, c);
                    if (b is not null && originalOwner != copy.Owner && b.Owner == copy.Owner) captured++;
                }
            return captured;
        }
        finally
        {
            state.UndoMove(record);
        }
    }

    private static readonly (int Dr, int Dc, int My, int Opp)[] Directions =
    {
        (-1, 0, Dir.Up, Dir.Down), (1, 0, Dir.Down, Dir.Up), (0, -1, Dir.Left, Dir.Right), (0, 1, Dir.Right, Dir.Left),
    };

    /// <summary><c>analyze_combo_potential</c>.</summary>
    public string? AnalyzeComboPotential(LGameState state, LMove move)
    {
        var card = move.Card;
        var rules = state.Rules;
        var board = state.Board;
        var analysis = new List<string>();

        if (rules.Contains("同数"))
        {
            var same = new List<(int R, int C, int Value)>();
            foreach (var (dr, dc, my, opp) in Directions)
            {
                int nr = move.Row + dr, nc = move.Col + dc;
                if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
                var neighbor = board.Get(nr, nc);
                if (neighbor is null) continue;
                if (card.Base[my] == neighbor.Base[opp]) same.Add((nr, nc, card.Base[my]));
            }
            if (same.Count >= 2) analysis.Add($"同数连携：{same.Count}个数值{same[0].Value}的相邻卡牌");
            else if (same.Count == 1) analysis.Add($"部分同数匹配：与({same[0].R},{same[0].C})数值{same[0].Value}相同");
        }

        if (rules.Contains("加算"))
        {
            var groups = new List<(int Sum, List<(int R, int C)> Cells)>();
            foreach (var (dr, dc, my, opp) in Directions)
            {
                int nr = move.Row + dr, nc = move.Col + dc;
                if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
                var neighbor = board.Get(nr, nc);
                if (neighbor is null) continue;
                int sum = card.Base[my] + neighbor.Base[opp];
                int idx = groups.FindIndex(g => g.Sum == sum);
                if (idx < 0)
                {
                    groups.Add((sum, new List<(int, int)>()));
                    idx = groups.Count - 1;
                }
                groups[idx].Cells.Add((nr, nc));
            }
            foreach (var (sum, cells) in groups)
            {
                if (cells.Count >= 2) analysis.Add($"加算连携：{cells.Count}个和为{sum}的相邻卡牌");
                else if (cells.Count == 1) analysis.Add($"部分加算匹配：与({cells[0].R},{cells[0].C})和为{sum}");
            }
        }

        if (rules.Contains("逆转"))
        {
            double avg = Average(new[] { card.Modified(0), card.Modified(1), card.Modified(2), card.Modified(3) });
            if (avg <= 5) analysis.Add("逆转优势：低数值卡牌在逆转规则下更有竞争力");
            else if (avg >= 8) analysis.Add("逆转劣势：高数值卡牌在逆转规则下处于劣势");
        }

        if (rules.Contains("王牌杀手"))
        {
            int ace = Enumerable.Range(0, 4).Count(i => card.Modified(i) is 1 or 10);
            if (ace >= 2) analysis.Add($"王牌杀手强势：卡牌有{ace}个1或A，具有特殊攻击力");
            else if (ace == 1) analysis.Add("王牌杀手效果：卡牌包含1或A，部分方向具有特殊效果");
        }

        if ((rules.Contains("同类强化") || rules.Contains("同类弱化")) && !string.IsNullOrEmpty(card.CardType))
        {
            int sameType = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var b = board.Get(r, c);
                    if (b is not null && b.CardType == card.CardType) sameType++;
                }
            if (rules.Contains("同类强化") && sameType > 0) analysis.Add($"同类强化：棋盘上已有{sameType}张{card.CardType}类型卡牌，将获得数值提升");
            else if (rules.Contains("同类弱化") && sameType > 0) analysis.Add($"同类弱化：棋盘上已有{sameType}张{card.CardType}类型卡牌，将受到数值削弱");
        }

        analysis.AddRange(AnalyzeRuleCombinations(rules, analysis));
        return analysis.Count > 0 ? string.Join("；", analysis) : null;
    }

    /// <summary><c>analyze_rule_combinations</c>.</summary>
    public static List<string> AnalyzeRuleCombinations(IReadOnlyList<string> rules, List<string> existing)
    {
        var effects = new List<string>();
        if (rules.Contains("同数") && rules.Contains("加算"))
        {
            bool hasSame = existing.Any(a => a.Contains("同数连携"));
            bool hasAddition = existing.Any(a => a.Contains("加算连携"));
            if (hasSame && hasAddition) effects.Add("复合连携：同时触发同数和加算效果");
        }
        if (rules.Contains("逆转") && rules.Contains("王牌杀手")) effects.Add("逆转王牌杀手：1和A在逆转规则下具有双重优势");
        if (rules.Contains("逆转") && (rules.Contains("同数") || rules.Contains("加算"))) effects.Add("逆转连携：连携触发时使用逆转比较规则");
        if ((rules.Contains("同类强化") || rules.Contains("同类弱化")) && (rules.Contains("同数") || rules.Contains("加算"))) effects.Add("同类连携：卡牌数值变化影响连携计算");
        if (rules.Contains("秩序") && (rules.Contains("同数") || rules.Contains("加算"))) effects.Add("秩序连携：可用卡牌受限，连携策略需要重新评估");
        else if (rules.Contains("混乱") && (rules.Contains("同数") || rules.Contains("加算"))) effects.Add("混乱连携：可用卡牌受限，连携策略需要重新评估");
        return effects;
    }

    /// <summary><c>analyze_strategic_value</c> (reads the DISPLAY values, which may be stale).</summary>
    public string AnalyzeStrategicValue(LGameState state, LMove move)
    {
        var card = move.Card;
        var rules = state.Rules;
        var board = state.Board;
        int row = move.Row, col = move.Col;
        var strategies = new List<string>();

        if (rules.Contains("同数") && HasSameNumberPotential(card)) strategies.Add("为同数连携做准备");
        if (rules.Contains("加算") && HasAdditionPotential(card)) strategies.Add("为加算连携做准备");

        if (rules.Contains("逆转"))
        {
            double avg = (card.Up + card.Right + card.Down + card.Left) / 4.0;
            if (avg <= 5) strategies.Add("逆转优势策略：使用低数值卡牌");
            else if (avg >= 8) strategies.Add("逆转风险策略：高数值卡牌需谨慎使用");
        }

        if (rules.Contains("王牌杀手"))
        {
            int ace = new[] { card.Up, card.Right, card.Down, card.Left }.Count(v => v is 1 or 10);
            if (ace >= 1) strategies.Add($"王牌杀手战术：利用{ace}个特殊数值");
        }

        if ((rules.Contains("同类强化") || rules.Contains("同类弱化")) && !string.IsNullOrEmpty(card.CardType))
        {
            int onBoard = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    if (board.Get(r, c) is { } b && b.CardType == card.CardType) onBoard++;
            if (rules.Contains("同类强化"))
                strategies.Add(onBoard > 0 ? $"同类强化策略：与{onBoard}张同类卡牌协同" : "同类强化先锋：率先建立类型优势");
            else if (rules.Contains("同类弱化"))
                strategies.Add(onBoard > 0 ? $"同类弱化规避：避免与{onBoard}张同类卡牌聚集" : "同类弱化安全：首张同类卡牌不受影响");
        }

        if (rules.Contains("秩序")) strategies.Add(card.CanUse ? "秩序策略：珍惜可用卡牌机会" : "秩序限制：当前卡牌不可使用");
        else if (rules.Contains("混乱")) strategies.Add(card.CanUse ? "混乱策略：把握可用卡牌时机" : "混乱限制：当前卡牌不可使用");

        if ((row, col) == (1, 1)) strategies.Add("控制中心战略位置");
        else if ((row, col) is (0, 0) or (0, 2) or (2, 0) or (2, 2))
        {
            string? corner = AnalyzeCornerStrategy(card, row, col, board);
            strategies.Add(corner is not null ? $"占据防御要塞：{corner}" : "占据防御要塞");
        }
        else if ((row, col) is (0, 1) or (1, 0) or (1, 2) or (2, 1)) strategies.Add("控制边缘要道");

        strategies.AddRange(AnalyzeMultiRuleStrategies(rules, card, row, col));
        return strategies.Count > 0 ? string.Join("；", strategies) : "维持场面平衡，为后续发展铺路";
    }

    private static List<string> AnalyzeMultiRuleStrategies(IReadOnlyList<string> rules, LCard card, int row, int col)
    {
        var combo = new List<string>();
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        if (rules.Contains("逆转") && (rules.Contains("同数") || rules.Contains("加算")))
        {
            double avg = values.Sum() / 4.0;
            if (avg <= 5 && (HasSameNumberPotential(card) || HasAdditionPotential(card))) combo.Add("逆转连携双重优势：低数值+连携潜力");
        }
        if (rules.Contains("王牌杀手") && (rules.Contains("同数") || rules.Contains("加算")))
        {
            if (values.Any(v => v is 1 or 10) && (HasSameNumberPotential(card) || HasAdditionPotential(card))) combo.Add("王牌杀手连携：特殊数值+连携双重威胁");
        }
        if (rules.Contains("逆转") && rules.Contains("王牌杀手") && values.Any(v => v is 1 or 10)) combo.Add("逆转王牌杀手：1和A在逆转规则下无敌");
        if ((rules.Contains("同类强化") || rules.Contains("同类弱化")) && !string.IsNullOrEmpty(card.CardType))
        {
            if (rules.Contains("逆转")) combo.Add("同类逆转策略：数值变化影响逆转效果");
            if (rules.Contains("王牌杀手") && values.Any(v => v is 1 or 10)) combo.Add("同类王牌杀手：数值变化不影响特殊效果");
        }
        if ((row, col) == (1, 1) && rules.Count >= 3) combo.Add($"多规则中心：{rules.Count}条规则在中心位置发挥最大效果");
        return combo;
    }

    private static readonly string[] SideNames = { "上", "右", "下", "左" };

    /// <summary><c>analyze_corner_strategy</c> (display values).</summary>
    public static string? AnalyzeCornerStrategy(LCard card, int row, int col, LBoard board)
    {
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        var parts = new List<string>();
        int high = values.Count(v => v >= 8), veryHigh = values.Count(v => v >= 9), low = values.Count(v => v <= 4);
        var acePositions = Enumerable.Range(0, 4).Where(i => values[i] == 10).ToList();

        if (acePositions.Count >= 2)
        {
            if (row == 2 && col == 2 && acePositions.Contains(1) && acePositions.Contains(3)) parts.Add("最优AA右下角组合");
            else if (row == 2 && col == 0 && acePositions.Contains(1) && acePositions.Contains(0)) parts.Add("次优AA左下角组合");
            else if (row == 0 && col == 2 && acePositions.Contains(2) && acePositions.Contains(3)) parts.Add("优质AA右上角组合");
            else if (row == 0 && col == 0 && acePositions.Contains(1) && acePositions.Contains(2)) parts.Add("平衡AA左上角组合");
            else parts.Add($"{acePositions.Count}边AA优势组合");
        }

        if (high >= 3)
        {
            parts.Add(veryHigh >= 3 ? "三边9+超强角落控制" : "三边8+强力角落控制");
            if (low >= 1)
            {
                var weak = Enumerable.Range(0, 4).Where(i => values[i] <= 4).Select(i => SideNames[i]);
                parts.Add($"保护{string.Join("/", weak)}弱势边");
            }
        }
        else if (high >= 2)
        {
            var highPositions = Enumerable.Range(0, 4).Where(i => values[i] >= 8).ToList();
            if (AreAdjacentPositions(highPositions)) parts.Add($"相邻{string.Join("/", highPositions.Select(i => SideNames[i]))}高数值控制");
            else parts.Add("双边高数值防护");
        }

        var exposed = ExposedSides(row, col);
        if (exposed.Count > 0)
        {
            if (exposed.Any(i => values[i] <= 3))
                parts.Add($"警告：{string.Join("/", exposed.Where(i => values[i] <= 3).Select(i => SideNames[i]))}边存在弱点");
            else if (exposed.All(i => values[i] >= 8))
                parts.Add($"{string.Join("/", exposed.Select(i => SideNames[i]))}边强力防护");
        }

        int min = values.Min(), max = values.Max();
        if (max - min >= 7 && low == 1) parts.Add($"隐藏{SideNames[Array.IndexOf(values, min)]}边弱点(值{min})");

        string? synergy = AnalyzeAdjacentSynergy(card, row, col, board);
        if (synergy is not null) parts.Add(synergy);
        return parts.Count > 0 ? string.Join("，", parts) : null;
    }

    private static bool AreAdjacentPositions(List<int> positions)
    {
        if (positions.Count < 2) return false;
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

    private static List<int> ExposedSides(int row, int col)
    {
        var exposed = new List<int>();
        if (row > 0) exposed.Add(0);
        if (col < 2) exposed.Add(1);
        if (row < 2) exposed.Add(2);
        if (col > 0) exposed.Add(3);
        return exposed;
    }

    private static string? AnalyzeAdjacentSynergy(LCard card, int row, int col, LBoard board)
    {
        var effects = new List<string>();
        int[] values = { card.Up, card.Right, card.Down, card.Left };
        (int Dr, int Dc, int MySide, int AdjSide)[] dirs = { (-1, 0, 0, 2), (0, 1, 1, 3), (1, 0, 2, 0), (0, -1, 3, 1) };
        foreach (var (dr, dc, mySide, adjSide) in dirs)
        {
            int nr = row + dr, nc = col + dc;
            if (nr is < 0 or > 2 || nc is < 0 or > 2) continue;
            var adj = board.Get(nr, nc);
            if (adj is null) continue;
            int mine = values[mySide], theirs = adj.Disp[adjSide];
            if (mine == theirs && mine >= 8) effects.Add($"{SideNames[mySide]}边与邻牌高值匹配({mine})");
            else if (mine + theirs == 10 && Math.Min(mine, theirs) >= 4) effects.Add($"{SideNames[mySide]}边与邻牌互补({mine}+{theirs}=10)");
        }
        return effects.Count > 0 ? string.Join("，", effects) : null;
    }

    // =============================================================================================================== search summary

    /// <summary><c>generate_search_summary</c> over the progress entries (<c>search_progress</c>).</summary>
    public static object GenerateSearchSummary(List<Dictionary<string, object?>> progress)
    {
        if (progress.Count == 0) return "无搜索数据";
        var last = progress[^1];
        string trend = "稳定";
        if (progress.Count >= 2)
        {
            var changes = new List<double>();
            for (int i = 1; i < progress.Count; i++) changes.Add((double)progress[i]["best_score"]! - (double)progress[i - 1]["best_score"]!);
            double sum = 0.0;
            foreach (double d in changes) sum += d; // Python sum(): left to right
            double avg = sum / changes.Count;
            trend = avg > 0.1 ? "上升" : avg < -0.1 ? "下降" : "稳定";
        }
        double nps = (double)last["nodes_per_second"]!;
        string efficiency = nps > 1000 ? "高效" : nps > 500 ? "正常" : "较慢";
        return new Dictionary<string, object?>
        {
            ["max_depth_reached"] = last["depth"],
            ["total_nodes_searched"] = last["nodes_searched"],
            ["total_time_seconds"] = last["time_elapsed"],
            ["final_evaluation_score"] = last["best_score"],
            ["score_trend"] = trend,
            ["search_efficiency"] = efficiency,
            ["average_branching_factor"] = last["branching_factor"],
            ["transposition_table_hit_rate"] = Py.FloatRepr((double)last["tt_hit_rate"]!) + "%",
            ["alpha_beta_cutoff_rate"] = Py.FloatRepr((double)last["cutoff_rate"]!) + "%",
        };
    }
}
