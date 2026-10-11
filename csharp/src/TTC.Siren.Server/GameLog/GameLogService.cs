using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Engine;
using TTC.Siren.Server.Live;

namespace TTC.Siren.Server.GameLog;

/// <summary>Callbacks of <see cref="AiMoveService"/> while it handles one request (live events and the game log use them).</summary>
public interface IAiMoveObserver
{
    /// <summary>The request has been parsed (before any search).</summary>
    void OnParsed(AiMoveOutcome parsed);

    /// <summary>A search depth has completed. Called on the search thread: it must return quickly.</summary>
    void OnDepth(DepthOut depth);

    /// <summary>The robust endgame started (0, total) or finished (total, total).</summary>
    void OnRobust(int done, int total);
}

/// <summary>JSON conversion helpers shared by the log and the live events.</summary>
public static class JsonConv
{
    /// <summary>A JSON value as plain objects: long for integer tokens, double for the others (so a float stays a float).</summary>
    public static object? ToObject(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object: return e.EnumerateObject().ToDictionary(p => p.Name, p => ToObject(p.Value), StringComparer.Ordinal);
            case JsonValueKind.Array: return e.EnumerateArray().Select(ToObject).ToList();
            case JsonValueKind.String: return e.GetString();
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            case JsonValueKind.Number:
                // review L8-2: an if statement, not `cond ? l : e.GetDouble()` -- that expression has the type double, so every integer was boxed as a double
                // and written as 357.0
                string raw = e.GetRawText();
                if (!raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E') && e.TryGetInt64(out long l)) return l;
                return e.GetDouble();
            default: return null;
        }
    }

    /// <summary><c>[id|null, type|null, "r"|"b"|null, mod, base4, display4, canUse]</c> — the compact card form of the engine protocol.</summary>
    public static List<object?> Card(CardSnapshot c) => new()
    {
        c.Id < 0 ? null : c.Id, c.Type, c.Owner == OwnerId.Red ? "r" : c.Owner == OwnerId.Blue ? "b" : null, c.Mod,
        (int)c.Base[0], (int)c.Base[1], (int)c.Base[2], (int)c.Base[3], (int)c.Display[0], (int)c.Display[1], (int)c.Display[2], (int)c.Display[3], c.CanUse ? 1 : 0,
    };

    public static Dictionary<string, object?> State(StateSnapshot s, bool[][]? generated = null)
    {
        var d = new Dictionary<string, object?>
        {
            ["cur"] = s.Current,
            ["board"] = s.Board.Select(c => c is null ? null : (object)Card(c)).ToList(),
            ["hands"] = s.Hands.Select(h => (object?)h.Select(Card).Cast<object?>().ToList()).ToList(),
        };
        if (generated is not null) d["generated"] = generated.Select(h => (object?)h.Select(b => b ? 1 : 0).Cast<object?>().ToList()).ToList();
        return d;
    }

    /// <summary>The request state before the search, in the form the engine protocol uses.</summary>
    public static StateSnapshot Snapshot(ParsedRequest req)
    {
        var s = new StateSnapshot { Current = req.State.CurrentPlayerIdx, Rules = RuleParser.FromPythonList(req.Rules) };
        CardSnapshot Of(LCard c)
        {
            var snap = new CardSnapshot
            {
                Id = c.CardId ?? -1, Type = c.CardType, Owner = c.Owner == "red" ? OwnerId.Red : c.Owner == "blue" ? OwnerId.Blue : OwnerId.None,
                Mod = c.TypeModifier, CanUse = c.CanUse,
            };
            for (int i = 0; i < 4; i++)
            {
                snap.Base[i] = (byte)c.Base[i];
                snap.Display[i] = (byte)c.Disp[i];
            }
            return snap;
        }
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                if (req.State.Board.Get(r, c) is { } card) s.Board[r * 3 + c] = Of(card);
        for (int p = 0; p < 2; p++) s.Hands[p].AddRange(req.State.Players[p].Hand.Select(Of));
        return s;
    }

    public static string Utc(DateTimeOffset t) => t.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);
}

/// <summary>
/// The game log and the live events of one server (decision A-21, goal sections C2 and C3). For every request it decides which game the request
/// belongs to, writes the records of docs/design/GAME_LOG_SCHEMA.md (game start, the inferred opponent move, the request with its seed and
/// explanation, game end) and publishes the events of the live stream. A failure to write a log never fails the request.
/// </summary>
public sealed class GameLogService
{
    /// <summary>
    /// The schema version written into every record (<c>v</c>). Version 2 (Phase 8B): integers are written as integers, <c>request_raw</c> is the request as it was
    /// received, and an inferred opponent card carries <c>ambiguous</c> and <c>candidate_ids</c>. Files of version 1 are still read.
    /// </summary>
    public const int SchemaVersion = 2;

    internal sealed class ActiveGame
    {
        public required string Id;
        public required DateTimeOffset Started;
        public required string IdSource;
        public int Seq;
        public int Requests;
        public RequestInfo? Last;
        public bool Ended;
        public readonly object Gate = new();
    }

    private readonly GameLogStore _store;
    private readonly LiveHub _hub;
    private readonly TimeProvider _time;
    private readonly CardDb _db;
    private readonly ConcurrentDictionary<string, Dictionary<string, ActiveGame>> _active = new();
    private readonly string _serverVersion;
    private readonly CardCandidates _candidates;

    public GameLogService(GameLogStore store, LiveHub hub, CardDb db, string serverVersion, TimeProvider? time = null)
    {
        _store = store;
        _hub = hub;
        _db = db;
        _candidates = new CardCandidates(db);
        _serverVersion = serverVersion;
        _time = time ?? TimeProvider.System;
    }

    public GameLogStore Store => _store;
    public LiveHub Hub => _hub;
    public DateTimeOffset Now => _time.GetUtcNow();

    public RequestSession StartSession(string keyId, string rawRequest) => new(this, keyId, rawRequest, Now);

    // ------------------------------------------------------------------------------------------------------------- identity

    private Dictionary<string, ActiveGame> ActiveOf(string keyId) =>
        _active.GetOrAdd(keyId, k => LoadRecent(k));

    /// <summary>After a restart the games written recently are read back from their files, so inference still works.</summary>
    private Dictionary<string, ActiveGame> LoadRecent(string keyId)
    {
        var map = new Dictionary<string, ActiveGame>(StringComparer.Ordinal);
        if (!_store.Options.Enabled) return map;
        var limit = Now - TimeSpan.FromMinutes(_store.Options.GapMinutes);
        foreach (var (gameId, lastWrite) in _store.Files(keyId))
        {
            if (lastWrite < limit) break;
            var game = LoadGame(keyId, gameId);
            if (game is not null) map[gameId] = game;
        }
        return map;
    }

    private ActiveGame? LoadGame(string keyId, string gameId)
    {
        var lines = _store.ReadLines(keyId, gameId);
        if (lines.Count == 0) return null;
        try
        {
            ActiveGame? game = null;
            int seq = 0, requests = 0;
            bool ended = false;
            RequestInfo? last = null;
            RuleSet rules = RuleSet.None;
            string myOwner = "red";
            foreach (var line in lines)
            {
                using var doc = JsonDocument.Parse(line);
                var r = doc.RootElement;
                seq = r.GetProperty("seq").GetInt32() + 1;
                switch (r.GetProperty("type").GetString())
                {
                    case "game_start":
                        game = new ActiveGame
                        {
                            Id = gameId, IdSource = r.GetProperty("id_source").GetString()!,
                            Started = DateTimeOffset.Parse(r.GetProperty("t").GetString()!, CultureInfo.InvariantCulture),
                        };
                        rules = RuleParser.FromPythonList(r.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
                        myOwner = r.GetProperty("my_owner").GetString()!;
                        break;
                    case "request":
                        requests = r.GetProperty("request_no").GetInt32();
                        if (r.TryGetProperty("state", out var st) && st.ValueKind == JsonValueKind.Object)
                        {
                            var snapshot = StateSnapshot.FromJson(st, rules);
                            var gen = st.GetProperty("generated").EnumerateArray().Select(h => h.EnumerateArray().Select(x => x.GetInt32() == 1).ToArray()).ToArray();
                            (int, int, int)? move = null;
                            if (r.TryGetProperty("result", out var res) && res.ValueKind == JsonValueKind.Object && res.TryGetProperty("move", out var mv) && mv.ValueKind == JsonValueKind.Object)
                                move = (mv.GetProperty("slot").GetInt32(), mv.GetProperty("row").GetInt32(), mv.GetProperty("col").GetInt32());
                            last = new RequestInfo(snapshot, myOwner, move, DateTimeOffset.Parse(r.GetProperty("t").GetString()!, CultureInfo.InvariantCulture), gen, requests);
                        }
                        break;
                    case "game_end": ended = true; break;
                }
            }
            if (game is null) return null;
            game.Seq = seq;
            game.Requests = requests;
            game.Last = last;
            game.Ended = ended;
            return game;
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException or FormatException)
        {
            return null; // a damaged file is not a candidate for continuation
        }
    }

    private static string NewGameId(DateTimeOffset now)
    {
        Span<byte> random = stackalloc byte[4];
        System.Security.Cryptography.RandomNumberGenerator.Fill(random);
        return "g-" + now.UtcDateTime.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture) + "-" + Convert.ToHexString(random).ToLowerInvariant();
    }

    // ------------------------------------------------------------------------------------------------------------- session

    /// <summary>The handling of one request: identity and opponent move before the search, the request record after it.</summary>
    public sealed class RequestSession : IAiMoveObserver
    {
        private readonly GameLogService _svc;
        private readonly string _rawRequest;
        private readonly DateTimeOffset _at;
        private ActiveGame? _game;
        private RequestInfo? _info;
        private int _requestNo;
        private string? _clientGameId;
        private string? _inferenceNote;

        internal RequestSession(GameLogService svc, string keyId, string rawRequest, DateTimeOffset at)
        {
            _svc = svc;
            KeyId = keyId;
            _rawRequest = rawRequest;
            _at = at;
        }

        public string KeyId { get; }

        public DateTimeOffset At => _at;

        /// <summary>The <c>rules</c> text exactly as the plugin sent it (null when it was not a string).</summary>
        public string? RawRules { get; private set; }

        /// <summary>The game this request belongs to; null before the request has been parsed (or when it could not be).</summary>
        public string? GameId => _game?.Id;

        public int RequestNo => _requestNo;

        public void OnParsed(AiMoveOutcome parsed)
        {
            if (parsed.Parsed is null) return;
            try
            {
                using var raw = JsonDocument.Parse(_rawRequest);
                RawRules = raw.RootElement.TryGetProperty("rules", out var rulesValue) && rulesValue.ValueKind == JsonValueKind.String ? rulesValue.GetString() : null;
                _clientGameId = raw.RootElement.TryGetProperty("gameId", out var g) && g.ValueKind == JsonValueKind.String && GameLogStore.GameIdPattern.IsMatch(g.GetString()!)
                    ? g.GetString() : null;
                var snapshot = JsonConv.Snapshot(parsed.Parsed);
                var generated = parsed.Generated ?? snapshot.Hands.Select(h => new bool[h.Count]).ToArray();
                _info = new RequestInfo(snapshot, parsed.Parsed.MyOwner, null, _at, generated, 0);
                _svc.Begin(this, parsed, snapshot, generated);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                _svc.Warn("game log: " + e.GetType().Name + ": " + e.Message);
            }
        }

        public void OnDepth(DepthOut d)
        {
            if (_game is null) return;
            _svc._hub.Publish(KeyId, "search.depth", _game.Id, () => new Dictionary<string, object?>
            {
                ["request_no"] = _requestNo, ["depth"] = d.Depth, ["score"] = d.Score,
                ["move"] = d.Move is { } m ? MoveJson(m) : null,
                ["nodes"] = d.NodesTotal, ["tt_hits"] = d.TtHitsTotal, ["cutoffs"] = d.CutoffsTotal, ["elapsed_ms"] = d.ElapsedSeconds * 1000.0,
            });
        }

        public void OnRobust(int done, int total)
        {
            if (_game is null) return;
            _svc._hub.Publish(KeyId, "robust.progress", _game.Id, () => new Dictionary<string, object?> { ["request_no"] = _requestNo, ["done"] = done, ["total"] = total });
        }

        /// <summary>Writes the request record and the end of the game, publishes the decision. Adds <c>game_id</c> to the response.</summary>
        public void Complete(AiMoveOutcome outcome)
        {
            if (_game is null || _info is null) return;
            try
            {
                _svc.Finish(this, _game, _info, _requestNo, outcome, _rawRequest, _inferenceNote);
                if (outcome.Body is not null) outcome.Body["game_id"] = _game.Id;
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                _svc.Warn("game log: " + e.GetType().Name + ": " + e.Message);
                if (outcome.Body is not null) outcome.Body["game_id"] = _game.Id;
            }
        }

        internal void Bind(ActiveGame game, int requestNo, string? note)
        {
            _game = game;
            _requestNo = requestNo;
            _inferenceNote = note;
        }

        internal string? ClientGameId => _clientGameId;
    }

    private static Dictionary<string, object?> MoveJson(MoveOut m) => new() { ["slot"] = m.Slot, ["row"] = m.Row, ["col"] = m.Col, ["card_id"] = m.CardId };

    public Action<string>? WarningSink { get; set; }

    private void Warn(string message) => WarningSink?.Invoke(message);

    // ------------------------------------------------------------------------------------------------------------- records

    private Dictionary<string, object?> Record(ActiveGame game, string type, DateTimeOffset at)
    {
        var d = new Dictionary<string, object?>
        {
            ["v"] = SchemaVersion, ["seq"] = game.Seq++, ["type"] = type, ["t"] = JsonConv.Utc(at), ["elapsed_ms"] = (long)Math.Round((at - game.Started).TotalMilliseconds, 0),
        };
        return d;
    }

    private void Write(string keyId, ActiveGame game, Dictionary<string, object?> record)
    {
        if (_store.Options.Enabled) _store.Append(keyId, game.Id, PyJson.Serialize(record));
        _hub.Publish(keyId, "game.updated", game.Id, () => new Dictionary<string, object?> { ["record"] = record });
    }

    /// <summary>
    /// Called when a request has been parsed: finds or starts the game, publishes <c>request.received</c>, then writes the start of the game and the
    /// inferred opponent move (so the live stream shows the request first, then what the log learned from it).
    /// </summary>
    private void Begin(RequestSession session, AiMoveOutcome parsed, StateSnapshot snapshot, bool[][] generated)
    {
        string keyId = session.KeyId;
        var req = parsed.Parsed!;
        var active = ActiveOf(keyId);
        var at = session.At;
        lock (active)
        {
            ActiveGame? game = null;
            string idSource;
            string? note = null;
            bool isNew = false;
            RequestInfo? previous = null;

            string? client = session.ClientGameId;
            if (client is not null)
            {
                idSource = "client";
                if (!active.TryGetValue(client, out game))
                {
                    var loaded = _store.Exists(keyId, client) ? LoadGame(keyId, client) : null;
                    if (loaded is not null) { game = loaded; active[client] = game; }
                }
                if (game is null)
                {
                    game = new ActiveGame { Id = client, Started = at, IdSource = "client" };
                    active[client] = game;
                    isNew = true;
                }
                else previous = game.Last;
            }
            else
            {
                idSource = "inferred";
                var info = new RequestInfo(snapshot, req.MyOwner, null, at, generated, 0);
                var gap = TimeSpan.FromMinutes(_store.Options.GapMinutes);
                foreach (var candidate in active.Values.Where(g => !g.Ended && g.Last is not null).OrderByDescending(g => g.Last!.At))
                {
                    var check = GameTracker.CheckSuccessor(candidate.Last!, info, gap);
                    if (!check.IsSuccessor) continue;
                    game = candidate;
                    previous = candidate.Last;
                    note = check.Reason;
                    break;
                }
                if (game is null)
                {
                    game = new ActiveGame { Id = NewGameId(at), Started = at, IdSource = "new" };
                    active[game.Id] = game;
                    idSource = "new";
                    isNew = true;
                }
            }

            game.Requests++;
            session.Bind(game, game.Requests, note);
            var finalGame = game;
            int requestNo = game.Requests;
            _hub.Publish(keyId, "request.received", finalGame.Id, () => new Dictionary<string, object?>
            {
                ["request_no"] = requestNo, ["my_owner"] = req.MyOwner, ["rules"] = req.Rules, ["id_source"] = idSource,
                ["state"] = JsonConv.State(snapshot, generated),
            });

            if (isNew) StartRecord(session, keyId, game, req, snapshot, generated, idSource, null);
            if (previous is not null)
            {
                var current = new RequestInfo(snapshot, req.MyOwner, null, at, generated, requestNo);
                var inference = GameTracker.InferOpponentMove(_db, previous, current);
                if (inference is not null) WriteOpponentMove(keyId, game, previous, current, inference, at);
            }
        }
    }

    private void StartRecord(RequestSession session, string keyId, ActiveGame game, ParsedRequest req, StateSnapshot snapshot, bool[][] generated, string idSource, string? note)
    {
        var rec = Record(game, "game_start", session.At);
        rec["game_id"] = game.Id;
        rec["id_source"] = idSource;
        rec["key_id"] = keyId;
        rec["rules"] = req.Rules;
        rec["rules_text"] = RulesText(session.RawRules, req.Rules);
        rec["open_mode"] = req.OpenMode;
        rec["my_owner"] = req.MyOwner;
        rec["npc_name"] = req.NpcName == "unknown" ? null : req.NpcName;
        rec["npc_ids"] = req.NpcIds;
        int me = req.MyOwner == "red" ? 0 : 1;
        List<object?> Known(int player) => snapshot.Hands[player].Where((c, i) => !(i < generated[player].Length && generated[player][i]) && c.Id > 0 && c.Id < 1000).Select(c => (object?)c.Id).ToList();
        rec["known_decks"] = new Dictionary<string, object?> { ["mine"] = Known(me), ["opponent"] = Known(1 - me) };
        rec["server"] = _serverVersion;
        rec["schema"] = SchemaVersion;
        Write(keyId, game, rec);
    }

    private static string RulesText(string? raw, List<string> rules) => raw ?? string.Join(",", rules);

    private void WriteOpponentMove(string keyId, ActiveGame game, RequestInfo previous, RequestInfo current, OpponentInference inf, DateTimeOffset at)
    {
        var rec = Record(game, "opponent_move", at);
        rec["between"] = new List<object?> { previous.RequestNo, current.RequestNo };
        rec["consistent"] = inf.Consistent;
        rec["card_id"] = inf.OpponentCardId == 0 ? null : inf.OpponentCardId;
        rec["card_edges"] = inf.OpponentEdges.Length == 0 ? null : inf.OpponentEdges.Cast<object?>().ToList();
        // review L8-4: a request carries edges, not a card id. Several database cards can share the four edges; card_id is the first (as Python's
        // find_card_id_by_stats would pick), and the log says honestly that the card is one of candidate_ids
        var candidates = inf.OpponentEdges.Length == 4 ? _candidates.IdsWithEdges(inf.OpponentEdges[0], inf.OpponentEdges[1], inf.OpponentEdges[2], inf.OpponentEdges[3]) : new List<int>();
        rec["ambiguous"] = candidates.Count > 1;
        rec["candidate_ids"] = candidates.Cast<object?>().ToList();
        rec["row"] = inf.OpponentCell < 0 ? null : inf.OpponentCell / 3;
        rec["col"] = inf.OpponentCell < 0 ? null : inf.OpponentCell % 3;
        rec["owner"] = inf.OpponentOwner;
        rec["flips"] = inf.Flips.Select(f => (object?)new Dictionary<string, object?> { ["row"] = f.Row, ["col"] = f.Col, ["card_id"] = f.CardId, ["cause"] = f.Cause }).ToList();
        rec["my_move"] = inf.MyCell < 0 ? null : new Dictionary<string, object?> { ["row"] = inf.MyCell / 3, ["col"] = inf.MyCell % 3, ["as_recommended"] = inf.AsRecommended };
        rec["new_cells"] = inf.NewCells.Cast<object?>().ToList();
        rec["note"] = inf.Note;
        Write(keyId, game, rec);
    }

    private void Finish(RequestSession session, ActiveGame game, RequestInfo info, int requestNo, AiMoveOutcome outcome, string rawRequest, string? inferenceNote)
    {
        string keyId = session.KeyId;
        var at = Now;
        var rec = Record(game, "request", at);
        rec["request_no"] = requestNo;
        rec["seed"] = outcome.Seed.ToString(CultureInfo.InvariantCulture);
        using var rawDoc = JsonDocument.Parse(rawRequest);
        var raw = rawDoc.RootElement;
        rec["settings"] = new Dictionary<string, object?>
        {
            ["max_depth"] = raw.TryGetProperty("max_depth", out var md) ? JsonConv.ToObject(md) : 10L,
            ["max_time"] = raw.TryGetProperty("max_time", out var mt) ? JsonConv.ToObject(mt) : 10L,
            ["aggressive"] = outcome.Parsed!.Aggressive,
            ["solver"] = outcome.Parsed.SolverType == "\0non-string" ? null : outcome.Parsed.SolverType,
            ["explain"] = outcome.ExplainLevel,
            ["show_search_progress"] = outcome.Parsed.ShowSearchProgress,
            ["tt_entries"] = outcome.TtEntries,
        };
        rec["request_raw"] = RawJson.Compact(rawRequest); // the request as received: its numbers and members exactly as the plugin wrote them (review L8-2)
        rec["rules"] = outcome.Parsed.Rules;
        rec["state"] = JsonConv.State(info.State, info.Generated);
        rec["id_inference"] = inferenceNote;
        rec["http_status"] = outcome.Status;
        rec["error"] = outcome.Status == 200 ? null : outcome.Body.GetValueOrDefault("error");

        MoveOut? final = outcome.Solve?.Final;
        if (outcome.Solve is { } solve)
        {
            rec["result"] = new Dictionary<string, object?>
            {
                ["move"] = final is { } f ? MoveJson(f) : null,
                ["standard_move"] = solve.Standard is { } s ? MoveJson(s) : null,
                ["standard_score"] = solve.Standard is null ? null : solve.StandardScore,
                ["depth_reached"] = solve.DepthCompleted, ["nodes"] = solve.Nodes, ["tt_hits"] = solve.TtHits, ["cutoffs"] = solve.Cutoffs, ["root_moves"] = solve.RootMoves,
                ["search_ms"] = solve.SearchMs,
                ["depths"] = solve.Depths.Select(d => (object?)new Dictionary<string, object?>
                {
                    ["depth"] = d.Depth, ["score"] = d.Score, ["score_bits"] = BitConverter.DoubleToInt64Bits(d.Score).ToString("X16", CultureInfo.InvariantCulture),
                    ["move"] = d.Move is { } m ? MoveJson(m) : null, ["nodes_total"] = d.NodesTotal, ["elapsed_ms"] = d.ElapsedSeconds * 1000.0,
                }).ToList(),
                ["robust"] = solve.Robust is { } rb ? new Dictionary<string, object?> { ["scenarios"] = rb.Scenarios, ["overridden"] = rb.Overridden, ["elapsed_ms"] = rb.ElapsedSeconds * 1000.0 } : null,
                ["timings"] = new Dictionary<string, object?> { ["build_ms"] = solve.BuildMs, ["search_ms"] = solve.SearchMs, ["robust_ms"] = solve.RobustMs, ["total_ms"] = solve.TotalMs },
            };
        }
        if (outcome.Facts is not null && outcome.Text is not null)
            rec["explanation"] = new Dictionary<string, object?>
            {
                ["version"] = Explain.ExplanationFacts.Version, ["level"] = outcome.ExplainLevel, ["facts"] = outcome.Facts.ToJson(),
                ["text"] = new Dictionary<string, object?> { ["brief"] = outcome.Text.Brief, ["full"] = outcome.Text.Full },
            };
        game.Last = info with { Move = final is { } fm ? (fm.Slot, fm.Row, fm.Col) : null, At = at, RequestNo = requestNo };
        Write(keyId, game, rec);

        if (outcome.Status == 200 && final is { } move)
        {
            _hub.Publish(keyId, "decision", game.Id, () => new Dictionary<string, object?>
            {
                ["request_no"] = requestNo, ["move"] = MoveJson(move), ["brief"] = outcome.Text?.Brief,
                ["depth_reached"] = outcome.Solve!.DepthCompleted, ["nodes"] = outcome.Solve.Nodes,
            });
        }

        // the end of the game: observed (a request with a full board) or reached by the move just recommended
        var stateAfter = FinalState(info.State, final);
        if (stateAfter is not null && !game.Ended)
        {
            var (red, blue) = GameTracker.FinalCount(stateAfter);
            int mine = outcome.Parsed!.MyOwner == "red" ? red : blue;
            int theirs = outcome.Parsed.MyOwner == "red" ? blue : red;
            var end = Record(game, "game_end", at);
            end["source"] = final is null ? "observed" : "own_last_move";
            end["final"] = new Dictionary<string, object?> { ["mine"] = mine, ["theirs"] = theirs };
            end["winner"] = mine > theirs ? "mine" : mine < theirs ? "theirs" : "draw";
            end["board"] = JsonConv.State(stateAfter)["board"];
            game.Ended = true;
            Write(keyId, game, end);
            _hub.Publish(keyId, "game.ended", game.Id, () => new Dictionary<string, object?>
            {
                ["final"] = end["final"], ["winner"] = end["winner"], ["source"] = end["source"],
            });
        }
    }

    /// <summary>The finished position when the request shows a full board (no move) or the recommended move fills it; otherwise null.</summary>
    private static StateSnapshot? FinalState(StateSnapshot before, MoveOut? move)
    {
        int occupied = before.Board.Count(c => c is not null);
        if (move is null) return occupied == 9 ? before : null;
        if (occupied != 8) return null;
        var g = GameState.FromSnapshot(before);
        int inst = g.HandInstance(g.CurrentPlayer, move.Value.Slot);
        g.MakeMove(new Move(inst, move.Value.Row * 3 + move.Value.Col));
        return g.ToSnapshot();
    }
}

/// <summary>
/// The database cards that share four edges (review L8-4). The plugin's request names a card by its edges only, so a card inferred from two requests is one
/// of these; the all-zero placeholder rows are never candidates.
/// </summary>
public sealed class CardCandidates
{
    private readonly Dictionary<uint, List<int>> _byEdges = new();

    public CardCandidates(CardDb db)
    {
        foreach (var c in db.Cards)
        {
            if (c.Up == 0 && c.Right == 0 && c.Down == 0 && c.Left == 0) continue;
            uint key = Key(c.Up, c.Right, c.Down, c.Left);
            if (!_byEdges.TryGetValue(key, out var list)) _byEdges[key] = list = new List<int>();
            list.Add(c.Id);
        }
    }

    private static uint Key(int up, int right, int down, int left) => (uint)up | (uint)right << 8 | (uint)down << 16 | (uint)left << 24;

    /// <summary>The ids of the cards with these edges, in database order (the first is the one Python's lookup returns). Empty when none.</summary>
    public List<int> IdsWithEdges(int up, int right, int down, int left) =>
        _byEdges.TryGetValue(Key(up, right, down, left), out var list) ? new List<int>(list) : new List<int>();
}
