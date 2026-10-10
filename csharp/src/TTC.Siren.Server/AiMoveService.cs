using System.Buffers;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;

namespace TTC.Siren.Server;

/// <summary>What one <c>/ai_move</c> request produced: the HTTP status, the response object (before serialisation) and what the callers (game log, live events) need.</summary>
public sealed class AiMoveOutcome
{
    public int Status = 200;
    public Dictionary<string, object?> Body = new();
    public ulong Seed;
    public ParsedRequest? Parsed;
    public SolveResult? Solve;
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

    public AiMoveService(CardCatalog catalog, SolveService solve, WorkerPool pool)
    {
        _catalog = catalog;
        _solve = solve;
        _pool = pool;
        _analysis = new LegacyAnalysis(catalog);
    }

    public CardCatalog Catalog => _catalog;

    /// <summary>The answer to an engine problem (A-20): <c>error</c> in the usual form plus <c>engine</c> and <c>engine_error</c>.</summary>
    private static AiMoveOutcome EngineError(string reason, string detail, int status)
    {
        string message = "C# engine: " + (detail.Length == 0 ? reason : $"{reason} ({detail})");
        return new AiMoveOutcome
        {
            Status = status,
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
    public async Task<AiMoveOutcome> HandleAsync(JsonElement data, ulong seed, CancellationToken ct)
    {
        var outcome = new AiMoveOutcome { Seed = seed };
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

            // max_depth / max_time as the Python client of the engine reads them (wire_max_depth, wire_max_time)
            if (!ReadNumber(data, "max_depth", 10, out var depthNumber, out string? depthError)) return Finish(outcome, EngineError("unsupported_parameter", depthError!, 400));
            if (!ReadNumber(data, "max_time", 10, out var timeNumber, out string? timeError)) return Finish(outcome, EngineError("unsupported_parameter", timeError!, 400));
            object wireDepth = depthNumber.Value is { } dv && dv > 100 ? 100L : depthNumber.Boxed;

            var (result, engineError) = await RunEngineAsync(parser, req, depthNumber, timeNumber, ct).ConfigureAwait(false);
            if (engineError is not null) return Finish(outcome, engineError);
            outcome.Solve = result;

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
                    },
                },
            };
            if (req.ShowSearchProgress && progress.Count > 0)
            {
                body["search_progress"] = progress;
                body["search_summary"] = LegacyAnalysis.GenerateSearchSummary(progress);
            }
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

    private async Task<(SolveResult? Result, AiMoveOutcome? Error)> RunEngineAsync(RequestParser parser, ParsedRequest req, Num depth, Num time, CancellationToken ct)
    {
        SolveRequest request;
        try
        {
            request = RequestReader.Parse(BuildPayload(req, depth, time, null));
        }
        catch (RequestException e)
        {
            return (null, EngineError("invalid_request", e.Message, 503));
        }

        var (worker, reason) = await _pool.AcquireAsync(ct).ConfigureAwait(false);
        if (worker is null)
            return (null, EngineError("busy", reason == BusyReason.QueueFull ? "queue_full" : reason == BusyReason.QueueTimeout ? "queue_timeout" : "aborted", 503));
        try
        {
            var result = _solve.Solve(request, worker);
            if (result.Status == SolveResult.NeedsFallbackCandidates)
            {
                // Python's legal enumeration of the unknown cards was empty: the random fallback candidates of the handler are drawn now,
                // with the arguments _build_endgame_scenarios would use (this consumes the request's random generator, as in Python)
                var oppHand = req.OppHand;
                int unknown = oppHand.Count(RequestParser.IsGeneratedUnknownCard);
                var knownOpp = oppHand.Where(c => !RequestParser.IsGeneratedUnknownCard(c)).ToList();
                var candidates = parser.Handler.GenerateOpponentCards(unknown, req.Rules, new HashSet<int>(req.UsedCards), req.Board, knownOpp, req.OppOwner, true);
                request = RequestReader.Parse(BuildPayload(req, depth, time, candidates));
                result = _solve.Solve(request, worker);
                if (result.Status != SolveResult.Ok) return (null, EngineError("unexpected_status", result.Status, 503));
            }
            return (result, null);
        }
        catch (RequestException e)
        {
            return (null, EngineError("invalid_request", e.Message, 503));
        }
        catch (PyException)
        {
            throw;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            return (null, EngineError("engine_error", e.GetType().Name + ": " + e.Message, 503));
        }
        finally
        {
            _pool.Release(worker);
        }
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
