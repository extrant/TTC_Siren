using System.Buffers;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Explain;
using TTC.Siren.Search;
using TTC.Siren.Server.GameLog;

namespace TTC.Siren.Server;

/// <summary>
/// Operating limits on a request's search (not part of the Python behaviour): <c>max_time</c> and <c>max_depth</c> are clamped to these. A search is not
/// interrupted when the proxy in front gives up, so a request that asks for an unbounded time would occupy a worker for nothing.
/// </summary>
public sealed record SearchCaps(double? MaxSeconds = null, int? MaxDepth = null)
{
    public static readonly SearchCaps None = new();

    /// <summary>The stricter of two caps.</summary>
    public SearchCaps Min(SearchCaps other) => new(
        MaxSeconds is { } a ? (other.MaxSeconds is { } b ? Math.Min(a, b) : a) : other.MaxSeconds,
        MaxDepth is { } d ? (other.MaxDepth is { } e ? Math.Min(d, e) : d) : other.MaxDepth);
}

/// <summary>What one <c>/ai_move</c> request produced: the HTTP status, the response object (before serialisation) and what the callers (game log, live events) need.</summary>
public sealed class AiMoveOutcome
{
    public int Status = 200;
    public Dictionary<string, object?> Body = new();
    public ulong Seed;
    public ParsedRequest? Parsed;
    public SolveRequest? Request;
    public SolveResult? Solve;
    public ExplanationFacts? Facts;
    public ExplanationText? Text;

    /// <summary>The language of <see cref="Text"/> and of the explanation in the answer (<c>zh-CN</c> or <c>en</c>).</summary>
    public string Language = TemplateSet.DefaultLanguage;

    /// <summary>The text of the same facts in every language (the game log keeps all of them, so a replay can be read in either).</summary>
    public Dictionary<string, ExplanationText>? Texts;
    public string? ExplainLevel;
    public int TtEntries;

    /// <summary>Milliseconds the request waited for a worker, and the class (0 cheap, 1 medium, 2 expensive) it was queued in.</summary>
    public double QueueWaitMs;
    public int RequestClass;

    /// <summary>Set when the server lowered the request's <c>max_time</c> or <c>max_depth</c>: what the limits are (reported in <c>engine_info.limits</c>).</summary>
    public Dictionary<string, object?>? Limits;

    /// <summary>The number of the worker grant this request got from the pool (1, 2, 3 ... in the order the pool served requests; 0: none).</summary>
    public long PoolGrant;

    /// <summary>Seconds a client should wait before asking again, when the answer is "busy" (503): becomes the <c>Retry-After</c> header; 0: none.</summary>
    public int RetryAfterSeconds;

    /// <summary>The access class the request was handled as.</summary>
    public AccessClass Access;

    /// <summary>The "generated unknown card" flag of every hand card at parse time (before the analysis reorders the hands), per player, in request order.</summary>
    public bool[][]? Generated;
    public LMove? Move;
    public string? EngineError;
}

/// <summary>
/// The native C# <c>/ai_move</c> (Phase 8A, Step B): what Python's <c>ai_move</c> did on the engine path, without Python. Request parsing and
/// unknown-card sampling are the ported Python logic (same random draws for the same seed); the search runs on the in-process engine
/// (worker pool, Fast iterative deepening, exact/robust endgame); the old response fields are produced by the faithful port of the analysis
/// functions; the order of those calls is Python's because they mutate the legacy state (hand order, display values).
/// Every response keeps the keys and value types of the Python server's response so that existing plugins keep working (API.md); anything
/// new is an additional key.
/// </summary>
public sealed class AiMoveService
{
    private readonly CardCatalog _catalog;
    private readonly SolveService _solve;
    private readonly WorkerPool _pool;
    private readonly LegacyAnalysis _analysis;
    private readonly TemplateSets _templates;

    private readonly int _ttEntries;

    /// <summary>
    /// Order the wait queue by estimated cost (review O8-1): cheap requests (endgames, the robust endgame) overtake openings, which age into the
    /// same class so that they cannot be overtaken for ever. Off: every request is queued in the same class, i.e. plain first come first served.
    /// </summary>
    public bool PriorityScheduling { get; set; } = true;

    /// <summary>Limits that apply to every request (the server's cap on <c>max_time</c>); none by default, so that the parity tests see the Python behaviour.</summary>
    public SearchCaps Caps { get; set; } = SearchCaps.None;

    /// <summary>Further limits for an anonymous request (decision A-27): the stricter of these and <see cref="Caps"/> applies. None by default.</summary>
    public SearchCaps AnonymousCaps { get; set; } = SearchCaps.None;

    public AiMoveService(CardCatalog catalog, SolveService solve, WorkerPool pool, TemplateSets? templates = null, int ttEntries = 0)
    {
        _ttEntries = ttEntries;
        _catalog = catalog;
        _solve = solve;
        _pool = pool;
        _analysis = new LegacyAnalysis(catalog);
        _templates = templates ?? TemplateSets.Default();
    }

    public CardCatalog Catalog => _catalog;

    /// <summary>The answer to an engine problem (A-20): <c>error</c> in the usual form plus <c>engine</c> and <c>engine_error</c>.</summary>
    private static AiMoveOutcome EngineError(string reason, string detail, int status, int retryAfterSeconds = 0)
    {
        string message = "C# engine: " + (detail.Length == 0 ? reason : $"{reason} ({detail})");
        return new AiMoveOutcome
        {
            Status = status,
            RetryAfterSeconds = retryAfterSeconds,
            EngineError = reason,
            Body = new Dictionary<string, object?> { ["error"] = message, ["engine"] = "csharp", ["engine_error"] = reason },
        };
    }

    private static AiMoveOutcome Error(int status, string message) =>
        new() { Status = status, Body = new Dictionary<string, object?> { ["error"] = message } };

    /// <summary>
    /// Handles one request. <paramref name="seed"/> seeds the unknown-card sampling (<see cref="PythonRandom"/>): production passes a fresh
    /// cryptographic seed and logs it, tests pass the seed Python was seeded with.
    /// </summary>
    public async Task<AiMoveOutcome> HandleAsync(JsonElement data, ulong seed, CancellationToken ct, IAiMoveObserver? observer = null, AccessClass access = AccessClass.Plugin, string? language = null)
    {
        var outcome = new AiMoveOutcome { Seed = seed, TtEntries = _ttEntries, Access = access };
        try
        {
            var rng = new PythonRandom(seed);
            var parser = new RequestParser(_catalog, rng);
            var parse = parser.TryParse(data);
            if (parse.Request is null)
            {
                outcome.Status = parse.Status;
                outcome.Body = new Dictionary<string, object?> { ["error"] = parse.Error };
                return outcome;
            }
            var req = parse.Request;
            outcome.Parsed = req;
            outcome.Generated = req.State.Players.Select(p => p.Hand.Select(RequestParser.IsGeneratedUnknownCard).ToArray()).ToArray();
            observer?.OnParsed(outcome);

            // max_depth / max_time as the Python client of the engine reads them (wire_max_depth, wire_max_time)
            if (!ReadNumber(data, "max_depth", 10, out var depthNumber, out string? depthError)) return Finish(outcome, EngineError("unsupported_parameter", depthError!, 400));
            if (!ReadNumber(data, "max_time", 10, out var timeNumber, out string? timeError)) return Finish(outcome, EngineError("unsupported_parameter", timeError!, 400));
            var caps = access == AccessClass.Anonymous ? Caps.Min(AnonymousCaps) : Caps;
            var requestedTime = timeNumber;
            depthNumber = CapDepth(depthNumber, caps.MaxDepth, out bool depthClamped);
            timeNumber = CapTime(timeNumber, caps.MaxSeconds, out bool timeClamped);
            if (depthClamped || timeClamped)
            {
                var limits = new Dictionary<string, object?>();
                if (timeClamped)
                {
                    limits["max_time"] = timeNumber.Boxed;
                    limits["requested_max_time"] = requestedTime.Value is null ? null : requestedTime.Boxed;
                }
                if (depthClamped) limits["max_depth"] = depthNumber.Boxed;
                outcome.Limits = limits;
            }
            object wireDepth = depthNumber.Value is { } dv && dv > 100 ? 100L : depthNumber.Boxed;

            // the request's explain level: "full" asks for the alternatives too; anything else is "brief"
            string explainLevel = data.TryGetProperty("explain", out var explainValue) && explainValue.ValueKind == JsonValueKind.String &&
                                  explainValue.GetString() == "full" ? "full" : "brief";
            outcome.ExplainLevel = explainLevel;
            // the language of the explanation: the caller's choice (?lang=), else the body's "lang", else Chinese; a tag we have no templates for is ignored
            outcome.Language = TemplateSet.Normalise(language)
                               ?? (data.TryGetProperty("lang", out var langValue) && langValue.ValueKind == JsonValueKind.String ? TemplateSet.Normalise(langValue.GetString()) : null)
                               ?? TemplateSet.DefaultLanguage;

            var (result, engineError, solveRequest, facts, explainError, explainMs) =
                await RunEngineAsync(parser, req, depthNumber, timeNumber, explainLevel, observer, ct, access, (w, c, g) => { outcome.QueueWaitMs = w; outcome.RequestClass = c; outcome.PoolGrant = g; }).ConfigureAwait(false);
            if (engineError is not null) return Finish(outcome, engineError);
            outcome.Solve = result;
            outcome.Request = solveRequest;

            // ---- _engine_move: the engine names a hand slot of the request order
            var state = req.State;
            LMove? ToMove(MoveOut? wire)
            {
                if (wire is null) return null;
                var w = wire.Value;
                var hand = state.CurrentPlayer.Hand;
                if (w.Slot < 0 || w.Slot >= hand.Count) throw new EngineProblem("illegal_move", $"hand slot {w.Slot} of {hand.Count}");
                var card = hand[w.Slot];
                if ((card.CardId ?? -1) != w.CardId) throw new EngineProblem("illegal_move", $"card id {card.CardId} at slot {w.Slot}, engine said {w.CardId}");
                if (w.Row is < 0 or > 2 || w.Col is < 0 or > 2 || !state.Board.IsEmpty(w.Row, w.Col)) throw new EngineProblem("illegal_move", $"cell ({w.Row}, {w.Col}) is not empty");
                if (!state.CurrentPlayer.PlayableCards(state.Rules).Any(c => ReferenceEquals(c, card)))
                    throw new EngineProblem("illegal_move", $"card at slot {w.Slot} is not playable under the active rules");
                return new LMove(card, w.Row, w.Col);
            }

            var progress = new List<Dictionary<string, object?>>();
            LMove? move;
            try
            {
                move = ToMove(result!.Final);
                if (req.ShowSearchProgress)
                {
                    double maxTimeSeconds = timeNumber.Value ?? 0.0;
                    foreach (var depth in result.Depths)
                    {
                        var best = ToMove(depth.Move);
                        double elapsed = depth.ElapsedSeconds;
                        long nodesTotal = depth.NodesTotal;
                        progress.Add(new Dictionary<string, object?>
                        {
                            ["depth"] = depth.Depth,
                            ["max_depth"] = wireDepth,
                            ["best_move"] = _analysis.FormatMoveForDisplay(best),
                            ["best_score"] = depth.Score,
                            ["nodes_searched"] = nodesTotal,
                            ["time_elapsed"] = Py.Round(elapsed, 3),
                            ["time_remaining"] = Py.Round(maxTimeSeconds - elapsed, 3),
                            ["nodes_per_second"] = Py.Round(nodesTotal / Math.Max(elapsed, 1e-9), 0),
                            ["tt_hit_rate"] = Py.Round((double)depth.TtHitsTotal / Math.Max(nodesTotal, 1) * 100, 1),
                            ["cutoff_rate"] = Py.Round((double)depth.CutoffsTotal / Math.Max(nodesTotal, 1) * 100, 1),
                            ["branching_factor"] = (long)result.RootMoves,
                        });
                    }
                }
            }
            catch (EngineProblem p)
            {
                return Finish(outcome, EngineError(p.Reason, p.Detail, 503));
            }

            if (move is null)
            {
                outcome.Body = new Dictionary<string, object?> { ["move"] = null, ["msg"] = "无可用动作", ["engine"] = "csharp" };
                if (data.TryGetProperty("solver", out var s0) && s0.ValueKind == JsonValueKind.String && s0.GetString() == "monte_carlo") AddSolverNote(outcome.Body);
                return outcome;
            }
            outcome.Move = move;

            // ---- the old analysis fields, in Python's order (they mutate the legacy state)
            var m = move.Value;
            var opponentAnalysis = _analysis.AnalyzeOpponentHand(req.OppHand, req.Rules, req.Board);
            var winProbability = _analysis.CalculateWinProbability(state, m, req.MyOwner);
            var recommendation = _analysis.GenerateMoveRecommendation(state, m, req.MyOwner);

            // Python: `new_state = game_state.copy(); new_state.play_move(row, col, card)` (the board shown on the console). It matters for the
            // response: `card` is the ORIGINAL card object, and playing it on the copy recalculates the type modifiers with it on the board,
            // so its type_modifier (shown in the card description) is the one it would have after the move.
            state.Copy().PlayMove(m.Row, m.Col, m.Card);

            long nodes = result.Nodes;
            var performance = new Dictionary<string, object?>
            {
                ["nodes_searched"] = nodes,
                ["search_depth"] = result.DepthCompleted,
                ["tt_hit_rate"] = (double)result.TtHits / Math.Max(nodes, 1) * 100,
                ["cutoff_rate"] = (double)result.Cutoffs / Math.Max(nodes, 1) * 100,
                ["unknown_cards_processed"] = opponentAnalysis["total_unknown"],
                ["performance_optimizations_active"] = true,
                ["engine"] = "csharp",
            };

            object star = _analysis.StarOrQuestion(m.Card.CardId);
            string cardDisplay = LegacyAnalysis.FormatCardDisplay(m.Card, star);

            var body = new Dictionary<string, object?>
            {
                ["card"] = cardDisplay,
                ["card_id"] = m.Card.CardId,
                ["pos"] = new List<object?> { m.Row, m.Col },
                ["opponent_hand_analysis"] = opponentAnalysis,
                ["win_probability"] = winProbability,
                ["recommendation"] = recommendation,
                ["performance_stats"] = performance,
                ["engine"] = "csharp",
                ["engine_info"] = new Dictionary<string, object?>
                {
                    ["protocol"] = Protocol.Version,
                    ["worker"] = result.WorkerId,
                    ["robust"] = result.Robust is not null,
                    ["overridden"] = result.Robust?.Overridden ?? false,
                    ["scenarios"] = result.Robust?.Scenarios,
                    ["timings"] = new Dictionary<string, object?>
                    {
                        ["buildMs"] = result.BuildMs, ["searchMs"] = result.SearchMs, ["robustMs"] = result.RobustMs, ["totalMs"] = result.TotalMs,
                        ["queueMs"] = outcome.QueueWaitMs,
                    },
                },
            };
            if (outcome.Limits is not null) ((Dictionary<string, object?>)body["engine_info"]!)["limits"] = outcome.Limits;
            if (req.ShowSearchProgress && progress.Count > 0)
            {
                body["search_progress"] = progress;
                body["search_summary"] = LegacyAnalysis.GenerateSearchSummary(progress);
            }
            // new, additive: the explanation (facts v2 and the text made from them)
            ((Dictionary<string, object?>)((Dictionary<string, object?>)body["engine_info"]!)["timings"]!)["explainMs"] = explainMs;
            if (facts is not null)
            {
                // the explanation never breaks the answer: a failure of the facts or of the wording is reported next to the move
                try
                {
                    outcome.Facts = facts;
                    outcome.Texts = _templates.RenderAll(facts);
                    outcome.Text = outcome.Texts[outcome.Language];
                    body["explanation"] = new Dictionary<string, object?>
                    {
                        ["version"] = ExplanationFacts.Version,
                        ["level"] = explainLevel,
                        ["language"] = outcome.Language,
                        ["facts"] = facts.ToJson(),
                        ["text"] = new Dictionary<string, object?> { ["brief"] = outcome.Text.Brief, ["full"] = outcome.Text.Full },
                    };
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    outcome.Facts = null;
                    outcome.Text = null;
                    body["explanation_error"] = "render: " + e.GetType().Name + ": " + e.Message;
                }
            }
            else if (explainError is not null) body["explanation_error"] = explainError;
            if (data.TryGetProperty("solver", out var solver) && solver.ValueKind == JsonValueKind.String && solver.GetString() == "monte_carlo") AddSolverNote(body);
            outcome.Body = body;
            return outcome;
        }
        catch (PyException e)
        {
            return Finish(outcome, Error(500, e.Message));
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception e)
        {
            return Finish(outcome, Error(500, e.Message));
        }
    }

    private static AiMoveOutcome Finish(AiMoveOutcome outcome, AiMoveOutcome answer)
    {
        outcome.Status = answer.Status;
        outcome.Body = answer.Body;
        outcome.EngineError = answer.EngineError;
        outcome.RetryAfterSeconds = answer.RetryAfterSeconds;
        return outcome;
    }

    /// <summary>Monte Carlo is retired (A-22): the request is answered by the minimax engine and the response says so.</summary>
    public static void AddSolverNote(Dictionary<string, object?> body)
    {
        body["solver_used"] = "minimax";
        body["solver_note"] = "蒙特卡洛求解器已下线：本次请求由 Minimax 引擎（含精确/信息集残局）求解，mc_simulations 参数被忽略。";
    }

    // ------------------------------------------------------------------------------------------------------------------ engine call

    private sealed class EngineProblem : Exception
    {
        public string Reason { get; }
        public string Detail { get; }

        public EngineProblem(string reason, string detail) : base(reason + ": " + detail)
        {
            Reason = reason;
            Detail = detail;
        }
    }

    /// <summary>A JSON number as Python holds it: an int or a float, plus its double value (null for +inf/NaN, which means "no limit").</summary>
    private readonly record struct Num(object Boxed, double? Value, bool IsInteger);

    private static bool ReadNumber(JsonElement data, string name, long defaultValue, out Num number, out string? error)
    {
        error = null;
        if (!data.TryGetProperty(name, out var e))
        {
            number = new Num(defaultValue, defaultValue, true);
            return true;
        }
        switch (e.ValueKind)
        {
            case JsonValueKind.True:
            case JsonValueKind.False:
                long b = e.ValueKind == JsonValueKind.True ? 1 : 0;
                number = new Num(b, b, true);
                return true;
            case JsonValueKind.Number:
                string raw = e.GetRawText();
                if (!raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E') && e.TryGetInt64(out long l))
                {
                    number = new Num(l, l, true);
                    return true;
                }
                double d = e.GetDouble();
                number = new Num(d, d, false);
                return true;
            default:
                number = default;
                error = $"{name}={Py.Repr(e)} is not a number";
                return false;
        }
    }

    /// <summary>The time limit lowered to <paramref name="cap"/> when the request asked for more, or for no limit at all.</summary>
    private static Num CapTime(Num time, double? cap, out bool clamped)
    {
        clamped = false;
        if (cap is not { } c) return time;
        if (time.Value is { } v && !double.IsNaN(v) && v <= c) return time;
        clamped = true;
        return c == Math.Floor(c) && c < long.MaxValue ? new Num((long)c, c, true) : new Num(c, c, false);
    }

    private static Num CapDepth(Num depth, int? cap, out bool clamped)
    {
        clamped = false;
        if (cap is not { } c) return depth;
        if (depth.Value is { } v && !double.IsNaN(v) && v <= c) return depth;
        clamped = true;
        return new Num((long)c, c, true);
    }

    private async Task<(SolveResult? Result, AiMoveOutcome? Error, SolveRequest? Request, ExplanationFacts? Facts, string? ExplainError, double ExplainMs)> RunEngineAsync(
        RequestParser parser, ParsedRequest req, Num depth, Num time, string explainLevel, IAiMoveObserver? observer, CancellationToken ct, AccessClass access,
        Action<double, int, long>? onQueued = null)
    {
        double queueWaitMs = 0;
        int usedClass = 0;
        SolveRequest request;
        try
        {
            request = RequestReader.Parse(BuildPayload(req, depth, time, null));
        }
        catch (RequestException e)
        {
            return (null, EngineError("invalid_request", e.Message, 503), null, null, null, 0);
        }

        int requestClass = PriorityScheduling ? RequestClass.Estimate(9 - req.Board.Occupancy(), req.UseEndgameRobust) : TTC.Siren.Engine.RequestClass.Medium;
        var acquired = await _pool.AcquireAsync(ct, requestClass, access).ConfigureAwait(false);
        var (worker, reason) = (acquired.Worker, acquired.Reason);
        queueWaitMs = acquired.WaitedMs;
        usedClass = requestClass;
        if (worker is null)
        {
            // 503 busy, and when to come back: a short wait for the classes that are served at once, longer for visitors who are not logged in
            string why = reason switch
            {
                BusyReason.QueueFull => "queue_full",
                BusyReason.QueueTimeout => "queue_timeout",
                BusyReason.AnonymousQueueFull => "anonymous_queue_full",
                BusyReason.AnonymousQueueTimeout => "anonymous_queue_timeout",
                _ => "aborted",
            };
            int retry = reason == BusyReason.Abandoned ? 0 : access == AccessClass.Anonymous ? 10 : 5;
            return (null, EngineError("busy", why, 503, retry), null, null, null, 0);
        }
        try
        {
            var result = _solve.Solve(request, worker, observer is null ? null : observer.OnDepth, observer is null ? null : observer.OnRobust);
            if (result.Status == SolveResult.NeedsFallbackCandidates)
            {
                // Python's legal enumeration of the unknown cards was empty: the random fallback candidates of the handler are drawn now,
                // with the arguments _build_endgame_scenarios would use (this consumes the request's random generator, as in Python)
                var oppHand = req.OppHand;
                int unknown = oppHand.Count(RequestParser.IsGeneratedUnknownCard);
                var knownOpp = oppHand.Where(c => !RequestParser.IsGeneratedUnknownCard(c)).ToList();
                var candidates = parser.Handler.GenerateOpponentCards(unknown, req.Rules, new HashSet<int>(req.UsedCards), req.Board, knownOpp, req.OppOwner, true);
                request = RequestReader.Parse(BuildPayload(req, depth, time, candidates));
                result = _solve.Solve(request, worker, observer is null ? null : observer.OnDepth, observer is null ? null : observer.OnRobust);
                if (result.Status != SolveResult.Ok) return (null, EngineError("unexpected_status", result.Status, 503), null, null, null, 0);
            }

            onQueued?.Invoke(queueWaitMs, usedClass, acquired.Grant);

            // the explanation, while this worker (and the table of the search it just ran) is still ours
            ExplanationFacts? facts = null;
            string? explainError = null;
            double explainMs = 0;
            if (result.Final is not null)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                try
                {
                    facts = Explainer.Build(BuildExplainInput(req, request, result, worker, explainLevel == "full"));
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    explainError = e.GetType().Name + ": " + e.Message;
                }
                explainMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            }
            return (result, null, request, facts, explainError, explainMs);
        }
        catch (RequestException e)
        {
            return (null, EngineError("invalid_request", e.Message, 503), null, null, null, 0);
        }
        catch (PyException)
        {
            throw;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            return (null, EngineError("engine_error", e.GetType().Name + ": " + e.Message, 503), null, null, null, 0);
        }
        finally
        {
            _pool.Release(worker);
        }
    }

    /// <summary>
    /// How the cards of the side NOT to move were assumed. Normally that is the request's opponent hand (the plugin asks on its own turn); if the
    /// side to move is the request's opponent, it is the plugin user's own hand, which has no NPC pool.
    /// </summary>
    private static UnknownInfo UnknownInfoOf(ParsedRequest req, int mover)
    {
        int moverIdx = mover;
        int ownIdx = req.MyOwner == "red" ? 0 : 1; // the request's own hand is players[ownIdx]
        bool nonMoverIsRequestOpponent = moverIdx == ownIdx;
        var sampling = nonMoverIsRequestOpponent ? req.OppSampling : req.OwnSampling;
        return nonMoverIsRequestOpponent
            ? new UnknownInfo(sampling.Strategy, sampling.PrimaryRule, req.NpcName, req.NpcIds.Count)
            : new UnknownInfo(sampling.Strategy, sampling.PrimaryRule, null, 0);
    }

    private ExplainInput BuildExplainInput(ParsedRequest req, SolveRequest request, SolveResult result, WorkerContext worker, bool full)
    {
        var final = result.Final!.Value;
        (int Slot, int Row, int Col)? standard = result.Standard is { } s ? (s.Slot, s.Row, s.Col) : null;
        var generated = req.State.Players.Select(p => p.Hand.Select(RequestParser.IsGeneratedUnknownCard).ToArray()).ToArray();
        bool sameMove = standard is { } sm && sm.Slot == final.Slot && sm.Row == final.Row && sm.Col == final.Col;

        RobustInput? robust = null;
        if (result.Robust is { } rb && request.Robust is { } rr)
        {
            robust = new RobustInput
            {
                Opponent = rr.Opponent, Generated = rr.Generated, SampleCount = rr.SampleCount, NpcIds = rr.NpcIds, FallbackCandidates = rr.FallbackCandidates,
                Scenarios = rb.Scenarios, Overridden = rb.Overridden,
                Candidates = rb.Candidates.Select(c => new RobustCandidate(c.Move.Slot, c.Move.Row, c.Move.Col, c.Move.CardId, c.FinalScore, c.SafetyRatio, c.RobustScore, c.CornerRisk)).ToList(),
                RobustMove = rb.RobustMove is { } rm ? (rm.Slot, rm.Row, rm.Col) : null,
            };
        }

        return new ExplainInput
        {
            Db = _catalog.Db,
            State = request.State,
            ChosenSlot = final.Slot, Row = final.Row, Col = final.Col,
            Aggressive = req.Aggressive,
            Full = full,
            DepthCompleted = result.DepthCompleted,
            DepthRequested = request.MaxDepth,
            Nodes = result.Nodes,
            SearchMs = result.SearchMs,
            StandardScore = result.StandardScore,
            StandardMove = standard,
            Tt = sameMove ? worker.Tt : null,
            Robust = robust,
            Unknown = UnknownInfoOf(req, request.State.Current),
            NpcIds = req.NpcIds.Where(i => i <= int.MaxValue).Select(i => (int)i).ToList(),
            Generated = generated,
        };
    }

    /// <summary>The engine request exactly as Python's client builds it (<c>csharp_engine.state_payload</c> and <c>_solve_minimax_with_engine</c>).</summary>
    private static byte[] BuildPayload(ParsedRequest req, Num depth, Num time, List<LCard>? fallback)
    {
        var buffer = new ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteNumber("protocol", Protocol.Version);
            w.WriteStartArray("rules");
            foreach (var rule in req.Rules) w.WriteStringValue(rule);
            w.WriteEndArray();

            var state = req.State;
            w.WriteStartObject("state");
            w.WriteNumber("cur", state.CurrentPlayerIdx);
            w.WriteStartArray("board");
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    var card = state.Board.Get(r, c);
                    if (card is null) w.WriteNullValue();
                    else WriteCard(w, card);
                }
            w.WriteEndArray();
            w.WriteStartArray("hands");
            foreach (var player in state.Players)
            {
                w.WriteStartArray();
                foreach (var card in player.Hand) WriteCard(w, card);
                w.WriteEndArray();
            }
            w.WriteEndArray();
            w.WriteEndObject();

            w.WriteBoolean("aggressive", req.Aggressive);
            if (depth.IsInteger) w.WriteNumber("maxDepth", Math.Min((long)depth.Value!.Value, 100L));
            else w.WriteNumber("maxDepth", Math.Min(depth.Value!.Value, 100.0));
            if (time.Value is null || double.IsNaN(time.Value.Value) || double.IsPositiveInfinity(time.Value.Value)) w.WriteNull("maxTime");
            else if (double.IsNegativeInfinity(time.Value.Value)) w.WriteNumber("maxTime", -1.0);
            else if (time.IsInteger) w.WriteNumber("maxTime", (long)time.Value.Value);
            else w.WriteNumber("maxTime", time.Value.Value);

            if (req.UseEndgameRobust)
            {
                w.WriteStartObject("robust");
                w.WriteNumber("aiPlayer", state.CurrentPlayerIdx);
                w.WriteNumber("opponent", req.MyOwner == "blue" ? 0 : 1);
                w.WriteStartArray("generated");
                foreach (var player in state.Players)
                {
                    w.WriteStartArray();
                    foreach (var card in player.Hand) w.WriteNumberValue(RequestParser.IsGeneratedUnknownCard(card) ? 1 : 0);
                    w.WriteEndArray();
                }
                w.WriteEndArray();
                w.WriteNumber("sampleCount", req.OppUnknownCount == 0 ? 1 : Math.Min(24, Math.Max(16, req.OppUnknownCount * 12)));
                w.WriteStartArray("npcIds");
                foreach (long id in req.NpcIds) w.WriteNumberValue(id);
                w.WriteEndArray();
                w.WriteBoolean("handler", true);
                if (fallback is null) w.WriteNull("fallbackCandidates");
                else
                {
                    w.WriteStartArray("fallbackCandidates");
                    foreach (var card in fallback) WriteCard(w, card);
                    w.WriteEndArray();
                }
                w.WriteEndObject();
            }
            else w.WriteNull("robust");
            w.WriteEndObject();
        }
        return buffer.WrittenSpan.ToArray();
    }

    /// <summary><c>csharp_engine.card_array</c>: <c>[id|null, type|null, "r"|"b"|null, mod, base4, display4, canUse]</c>.</summary>
    private static void WriteCard(Utf8JsonWriter w, LCard card)
    {
        w.WriteStartArray();
        if (card.CardId is { } id) w.WriteNumberValue(id);
        else w.WriteNullValue();
        if (card.CardType is null) w.WriteNullValue();
        else w.WriteStringValue(card.CardType);
        if (card.Owner == "red") w.WriteStringValue("r");
        else if (card.Owner == "blue") w.WriteStringValue("b");
        else w.WriteNullValue();
        w.WriteNumberValue(card.TypeModifier);
        for (int i = 0; i < 4; i++) w.WriteNumberValue(card.Base[i]);
        for (int i = 0; i < 4; i++) w.WriteNumberValue(card.Disp[i]);
        w.WriteNumberValue(card.CanUse ? 1 : 0);
        w.WriteEndArray();
    }
}
