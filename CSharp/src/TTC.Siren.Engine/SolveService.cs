using System.Diagnostics;
using TTC.Siren.Core;
using TTC.Siren.Search;

namespace TTC.Siren.Engine;

/// <summary>A move expressed in the request's terms: the slot in the side to move's hand (request order), the cell, and the card id.</summary>
public readonly record struct MoveOut(int Slot, int Row, int Col, int CardId);

public sealed class DepthOut
{
    public int Depth;
    public MoveOut? Move;
    public double Score;
    public long Nodes;
    public long NodesTotal;
    public long TtHitsTotal;
    public long CutoffsTotal;
    public double ElapsedSeconds;
}

public sealed class CandidateOut
{
    public MoveOut Move;
    public double FinalScore, SafetyRatio, RobustScore, CornerRisk;
}

public sealed class RobustOut
{
    public int Scenarios;
    public List<CandidateOut> Candidates = new();
    public MoveOut? RobustMove;
    public bool Overridden;
    public long ExactNodes;
    public long ExactMakeMoves;
    public double ElapsedSeconds;
}

public sealed class SolveResult
{
    public const string Ok = "ok";
    public const string NeedsFallbackCandidates = "needs_fallback_candidates";

    public string Status = Ok;
    public int UnknownCount;
    public string? RequestId;

    public MoveOut? Final;
    public MoveOut? Standard;
    public double StandardScore;
    public List<DepthOut> Depths = new();

    public long Nodes, TtHits, Cutoffs;
    public int DepthCompleted;
    public int RootMoves;
    public double SearchSeconds;

    public RobustOut? Robust;

    public double BuildMs, SearchMs, RobustMs, TotalMs;
    public int WorkerId;
}

/// <summary>
/// One solve: what Python's <c>ai_move</c> does after the position is built, for the minimax solver. Fast iterative deepening
/// from the request state; then, when the request asks for it, the information-set robust endgame (scenarios, candidate
/// table, override decision) on a SECOND copy of the request state. The robust layer therefore sees the hand in the
/// request's order and the display edges the request carried, not the order and edges a search leaves behind (user decision
/// on D-015: Fast search + canonical hand order; it reads effective values only). Nothing is shared between requests except
/// the read-only card database; the worker's tables are bound to this request's state before use (review H4-1).
/// </summary>
public sealed class SolveService
{
    private readonly CardDb _db;

    public SolveService(CardDb db) => _db = db;

    /// <summary>
    /// <paramref name="onDepth"/> (live progress) is called once per completed search depth with that depth's numbers, on the search thread, so it must
    /// return quickly; <paramref name="onRobust"/> is called with (scenarios done, scenarios total) when the robust endgame starts (0, total) and when
    /// it has finished (total, total). Both observe only: the result is the same with or without them.
    /// </summary>
    public SolveResult Solve(SolveRequest req, WorkerContext worker, Action<DepthOut>? onDepth = null, Action<int, int>? onRobust = null)
    {
        long t0 = Stopwatch.GetTimestamp();
        var result = new SolveResult { RequestId = req.RequestId, WorkerId = worker.Id };

        GameState searchState;
        GameState? robustState = null;
        try
        {
            searchState = GameState.FromSnapshot(req.State);
            if (req.Robust is not null) robustState = GameState.FromSnapshot(req.State);
        }
        catch (InvalidGameStateException e)
        {
            throw new RequestException(e.Message);
        }

        // Instance indices are assigned in the same order for both states, so one slot table serves both.
        int cur = searchState.CurrentPlayer;
        var slotOfInst = new int[GameState.MaxInstances];
        for (int k = 0; k < searchState.HandCount(cur); k++) slotOfInst[searchState.HandInstance(cur, k)] = k;

        MoveOut Out(GameState g, Move m) => new(slotOfInst[m.Inst], m.Row, m.Col, g.InstanceId(m.Inst));

        // 1. Scenarios first: when Python's legal enumeration is empty the caller must supply the random fallback cards,
        //    and that is known before any search time is spent.
        List<GameState>? scenarios = null;
        if (robustState is not null)
        {
            var r = req.Robust!;
            int unknown = r.Generated[r.Opponent].Count(x => x);
            result.UnknownCount = unknown;
            if (unknown == 0 || !r.Handler)
            {
                scenarios = new List<GameState> { robustState }; // Python: `if not unknown_indices / not handler: return [base_state]`
            }
            else
            {
                var build = EndgameScenarios.Build(_db, robustState, r.Opponent, r.Generated, r.SampleCount, r.NpcIds, r.FallbackCandidates);
                if (build.RequiresFallback)
                {
                    result.Status = SolveResult.NeedsFallbackCandidates;
                    result.TotalMs = Ms(t0);
                    return result;
                }
                scenarios = build.Scenarios!;
            }
        }
        long t1 = Stopwatch.GetTimestamp();
        result.BuildMs = Ms(t0, t1);

        // 2. Standard search: Fast iterative deepening.
        result.RootMoves = CountRootMoves(searchState);
        var searcher = new FastSearcher(searchState, _db, worker.Tt);
        Action<LevelProgress>? levelCallback = null;
        if (onDepth is not null)
        {
            levelCallback = p => onDepth(new DepthOut
            {
                Depth = p.Depth,
                Move = p.BestInstance < 0 ? null : new MoveOut(slotOfInst[p.BestInstance], p.BestCell / 3, p.BestCell % 3, searchState.InstanceId(p.BestInstance)),
                Score = p.Score,
                Nodes = 0,
                NodesTotal = p.NodesTotal,
                TtHitsTotal = p.TtHitsTotal,
                CutoffsTotal = p.CutoffsTotal,
                ElapsedSeconds = Stopwatch.GetElapsedTime(0, p.ElapsedTimestampTicks).TotalSeconds,
            });
        }
        var found = searcher.RunIterativeDeepening(req.MaxDepth, req.Aggressive, new SearchLimits(req.MaxTime), onLevel: levelCallback);
        var stats = searcher.Stats;
        long t2 = Stopwatch.GetTimestamp();
        result.SearchMs = Ms(t1, t2);

        result.Nodes = stats.NodesSearched;
        result.TtHits = stats.TtHits;
        result.Cutoffs = stats.AlphaBetaCutoffs;
        result.DepthCompleted = stats.DepthCompleted;
        result.SearchSeconds = result.SearchMs / 1000.0;
        long nodesTotal = 0;
        for (int level = 0; level < stats.Levels; level++)
        {
            nodesTotal += stats.NodesPerDepth[level];
            int inst = stats.BestInstanceHistory[level];
            result.Depths.Add(new DepthOut
            {
                Depth = level + 1,
                Move = inst < 0 ? null : new MoveOut(slotOfInst[inst], stats.BestCellHistory[level] / 3, stats.BestCellHistory[level] % 3, searchState.InstanceId(inst)),
                Score = stats.BestScoreHistory[level],
                Nodes = stats.NodesPerDepth[level],
                NodesTotal = nodesTotal,
                TtHitsTotal = stats.LevelTtHitsTotal[level],
                CutoffsTotal = stats.LevelCutoffsTotal[level],
                ElapsedSeconds = Stopwatch.GetElapsedTime(stats.StartTimestamp, stats.LevelEndTimestamp[level]).TotalSeconds,
            });
        }

        if (!found.HasMove)
        {
            result.TotalMs = Ms(t0);
            return result; // Python: `if move is None: return {'move': None, 'msg': ...}`
        }

        var standardMove = new Move(found.BestInstance, found.BestCell);
        result.Standard = Out(searchState, standardMove);
        result.StandardScore = found.Score;
        result.Final = result.Standard;

        // 3. Robust endgame on the request-order state.
        if (robustState is not null)
        {
            var r = req.Robust!;
            onRobust?.Invoke(0, scenarios!.Count);
            var sel = EndgameRobust.Select(_db, robustState, scenarios!, robustState.CurrentPlayer, req.Aggressive, worker.Memo);
            var robust = new RobustOut { Scenarios = scenarios!.Count, ExactNodes = sel.ExactNodes, ExactMakeMoves = sel.ExactMakeMoves };
            var items = new List<OverrideItem>(sel.Scored.Count);
            foreach (var s in sel.Scored)
            {
                robust.Candidates.Add(new CandidateOut
                {
                    Move = Out(robustState, s.Move),
                    FinalScore = s.FinalScore,
                    SafetyRatio = s.SafetyRatio,
                    RobustScore = s.RobustScore,
                    CornerRisk = s.CornerRisk,
                });
                items.Add(new OverrideItem(s.Key, s.FinalScore, s.SafetyRatio));
            }

            var standardKey = new MoveKey(searchState.InstanceId(standardMove.Inst), standardMove.Row, standardMove.Col);
            MoveKey? robustKey = sel.HasBest ? sel.BestKey : null;
            if (sel.HasBest) robust.RobustMove = Out(robustState, sel.Best);
            var decided = EndgameRobust.DecideOverride(standardKey, robustKey, items, req.Aggressive);
            if (decided is not null && decided.Value != standardKey)
            {
                robust.Overridden = true;
                result.Final = Out(robustState, sel.Best);
            }
            result.Robust = robust;
            onRobust?.Invoke(scenarios!.Count, scenarios.Count);
            long t3 = Stopwatch.GetTimestamp();
            robust.ElapsedSeconds = Stopwatch.GetElapsedTime(t2, t3).TotalSeconds;
            result.RobustMs = Ms(t2, t3);
        }

        result.TotalMs = Ms(t0);
        return result;
    }

    private static int CountRootMoves(GameState g)
    {
        Span<Move> buffer = stackalloc Move[GameState.MaxMoves];
        return g.GenerateMoves(buffer);
    }

    private static double Ms(long start) => Stopwatch.GetElapsedTime(start).TotalMilliseconds;

    private static double Ms(long start, long end) => Stopwatch.GetElapsedTime(start, end).TotalMilliseconds;
}
