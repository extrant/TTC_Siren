using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using TTC.Siren.Core;
using TTC.Siren.Search;

// Benchmarks (Release, fixed inputs). Every table is labelled "warm" or "cold" and reports min / median / max of the
// timed runs.
//
//   Options (anywhere on the command line):
//     --warmup-ms <n>   warm mode only: repeat the WHOLE item set for n milliseconds (default 1500) before any timed
//                       run, so every item is timed with tier-1 JIT code. This is the default ("warm").
//     --cold            the previous behaviour ("cold"): one untimed warm-up run per item, then the timed runs. The
//                       first items of a process then run tier-0/OSR code (review A-1); use it to reproduce the
//                       Phase 5 report's configuration or to estimate a cold, per-request process.
//
//   TTC.Siren.Bench makeundo [pairsPerRun=2000000] [runs=5]   Make/Undo throughput (Phase 2)
//   TTC.Siren.Bench search   [runs=5]                         ReferenceCompat fixed-depth search on the 22 items of section 3.4 (Phase 3)
//   TTC.Siren.Bench fast     [runs=5]                         Phase 4: ReferenceCompat (TT on / TT off = oracle) vs Fast, rotating order
//   TTC.Siren.Bench server   [seconds=10] [runs=3]            Section 3.5: server-equivalent iterative deepening (max depth 10, wall clock)
//   TTC.Siren.Bench exact    [runs=5]                         Phase 5: exact endgame solver on the section 3.3 positions
//   TTC.Siren.Bench endgame  [positions.json] [runs=5]        Phase 5: scenario construction + robust selection vs Python
//                            (positions and Python timings from tools/bench/py_endgame_bench.py)
//   TTC.Siren.Bench endgame-work [positions.json]             Phase 5: make_move work of the robust pipeline (not timed)
//   TTC.Siren.Bench loop fast|compat|exact [seconds=12]       Phase 6: one variant only, repeated for a fixed time (profiling workload)
//
// Machine-readable summary lines start with "RESULT " (used by tools/bench/run_interleaved.py).
var opts = Opts.Parse(args);
string[] pos = opts.Pos;
string mode = pos.Length > 0 ? pos[0] : "makeundo";
var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
// Review A-7: on a hybrid CPU the core class a run lands on changes the result by 1.4-2x, so every table says which
// CPU set the process actually runs with (read back from the OS: 0xFFFFF = unpinned on this 20-thread machine).
string affinity = OperatingSystem.IsWindows() || OperatingSystem.IsLinux() ? "0x" + Process.GetCurrentProcess().ProcessorAffinity.ToInt64().ToString("X") : "n/a";
Console.WriteLine($"# runtime={Environment.Version} gc={(System.Runtime.GCSettings.IsServerGC ? "server" : "workstation")} mode={mode} timing={opts.Label} affinity={affinity} cpus={Environment.ProcessorCount}");

if (mode == "makeundo") MakeUndo(db, pos);
else if (mode == "search") Search(db, pos, opts);
else if (mode == "fast") Fast(db, pos, opts);
else if (mode == "exact") ExactBench(db, pos, opts);
else if (mode == "server") ServerBench(db, pos, opts);
else if (mode == "endgame") EndgameBench(db, pos, opts);
else if (mode == "endgame-work") EndgameWork(db, pos);
else if (mode == "loop") Loop(db, pos);
else Console.WriteLine("usage: [--cold] [--warmup-ms n] makeundo [pairs] [runs] | search [runs] | fast [runs] | exact [runs] | server [seconds] [runs] | endgame [positions.json] [runs] | endgame-work [positions.json] | loop fast|compat|exact [seconds]");
return;

static string F(double v, string fmt = "R") => v.ToString(fmt, CultureInfo.InvariantCulture);

// Phase 6 profiling workload: one variant only, repeated for a fixed time, so a sampling profile contains nothing else.
static void Loop(CardDb db, string[] args)
{
    string variant = args.Length > 1 ? args[1] : "fast";
    double seconds = args.Length > 2 ? double.Parse(args[2], CultureInfo.InvariantCulture) : 12.0;
    var bodies = new List<Func<long>>();
    if (variant == "fast")
    {
        // The Fast workload of the Phase 5 review profile (section 8): one table, as in Bench fast.
        var tt = new FastTranspositionTable();
        foreach (var (name, depth) in new[] { ("open_none", 5), ("open_all_rules", 5), ("mid3_ascension", 6), ("mid4_rev_ace", 6) })
        {
            string n = name;
            int d = depth;
            bodies.Add(() =>
            {
                var g = ReferenceScenarios.Create(n, db);
                var s = new FastSearcher(g, db, tt);
                s.RunFixedDepth(d, false, SearchLimits.Unlimited);
                return s.Stats.NodesSearched;
            });
        }
    }
    else if (variant == "compat")
    {
        var tt = new CompatTranspositionTable();
        foreach (var (name, depths) in new[] { ("open_none", new[] { 3, 4 }), ("open_same_plus", new[] { 3, 4 }), ("mid3_ascension", new[] { 4, 5 }), ("mid4_rev_ace", new[] { 4, 5 }), ("end5_same_plus", new[] { 3, 4 }) })
        {
            foreach (int depth in depths)
            {
                string n = name;
                int d = depth;
                bodies.Add(() =>
                {
                    var g = ReferenceScenarios.Create(n, db);
                    var s = new CompatSearcher(g, db, tt);
                    s.RunFixedDepth(d, false, SearchLimits.Unlimited);
                    return s.Stats.NodesSearched;
                });
            }
        }
    }
    else if (variant == "exact")
    {
        var memo = ExactEndgameSolver.CreateMemo(1 << 17);
        foreach (string n in new[] { "mid3_ascension" })
        {
            string name = n;
            bodies.Add(() =>
            {
                var g = ReferenceScenarios.Create(name, db);
                var solver = new ExactEndgameSolver(g, memo);
                solver.Bind(g.CurrentPlayer);
                solver.Value(g.CurrentPlayer);
                return solver.MakeMoves;
            });
        }
    }
    else
    {
        Console.WriteLine("usage: loop fast|compat|exact [seconds]");
        return;
    }

    foreach (var body in bodies) body(); // JIT warm-up round
    long t0 = Stopwatch.GetTimestamp();
    long work = 0;
    int rounds = 0;
    while (Stopwatch.GetElapsedTime(t0).TotalSeconds < seconds)
    {
        foreach (var body in bodies) work += body();
        rounds++;
    }
    Console.WriteLine($"loop {variant}: {rounds} rounds in {Stopwatch.GetElapsedTime(t0).TotalSeconds:F2} s, work units (nodes or make_moves) = {work:N0}");
}

static void MakeUndo(CardDb db, string[] args)
{
    long pairs = args.Length > 1 ? long.Parse(args[1]) : 2_000_000;
    int runs = args.Length > 2 ? int.Parse(args[2]) : 5;
    Console.WriteLine($"# make/undo throughput  pairs/run={pairs:N0} runs={runs} (one 200,000-pair warm-up per scenario)");
    Console.WriteLine($"{"scenario",-16} {"median s",9} {"pairs/s (median)",18} {"min s",8} {"max s",8} {"alloc bytes",12}");

    double sumRate = 0;
    foreach (string name in ReferenceScenarios.Names)
    {
        var g = ReferenceScenarios.Create(name, db);
        ReferenceScenarios.RunPlayouts(g, 1, 200_000); // warm-up

        var times = new double[runs];
        long alloc = 0, lastChecksum = 0;
        for (int i = 0; i < runs; i++)
        {
            long a0 = GC.GetAllocatedBytesForCurrentThread();
            long t0 = Stopwatch.GetTimestamp();
            long checksum = ReferenceScenarios.RunPlayouts(g, 987654321, pairs);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            alloc += GC.GetAllocatedBytesForCurrentThread() - a0;
            lastChecksum = checksum;
            if (checksum == long.MinValue) Console.WriteLine(); // keep the result observable
        }
        Array.Sort(times);
        double median = times[runs / 2];
        double rate = pairs / median;
        sumRate += rate;
        Console.WriteLine($"{name,-16} {median,9:F3} {rate,18:N0} {times[0],8:F3} {times[^1],8:F3} {alloc,12}");
        Console.WriteLine($"RESULT makeundo item={name} pairs={pairs} min_s={F(times[0])} median_s={F(median)} max_s={F(times[^1])} checksum={lastChecksum} alloc={alloc}");
    }
    Console.WriteLine($"{"mean of medians",-16} {"",9} {sumRate / ReferenceScenarios.Names.Length,18:N0}");
}

static void Search(CardDb db, string[] args, Opts opts)
{
    int runs = args.Length > 1 ? int.Parse(args[1]) : 5;
    (string Name, int[] Depths)[] items =
    {
        ("open_none", new[] { 1, 2, 3, 4 }),
        ("open_same_plus", new[] { 1, 2, 3, 4 }),
        ("mid3_ascension", new[] { 1, 2, 3, 4, 5 }),
        ("mid4_rev_ace", new[] { 1, 2, 3, 4, 5 }),
        ("end5_same_plus", new[] { 1, 2, 3, 4 }),
    };
    var tt = new CompatTranspositionTable();
    var bodies = new List<(string Name, int Depth, Func<Sample> Body)>();
    foreach (var (name, depths) in items)
    {
        foreach (int depth in depths)
        {
            string n = name;
            int d = depth;
            bodies.Add((n, d, () =>
            {
                var g = ReferenceScenarios.Create(n, db);
                var s = new CompatSearcher(g, db, tt);
                var sample = Harness.Timed(g, () =>
                {
                    var r = s.RunFixedDepth(d, false, SearchLimits.Unlimited);
                    return (s.Stats.NodesSearched, r.Score, r.BestInstance, r.BestCell);
                });
                return sample;
            }));
        }
    }

    Console.WriteLine($"# ReferenceCompat fixed-depth search [{opts.Tag}], runs={runs}; per-item times in microseconds (median, min, max); '*' = under 50 us: timer-limited");
    Console.WriteLine($"{"scenario",-16} {"depth",5} {"nodes",10} {"median us",12} {"min us",12} {"max us",12} {"NPS (median)",13} {"score(hex bits)",18} {"alloc B",8}");
    Harness.WarmUp(opts, bodies.Select(b => b.Body));

    long totalNodes = 0;
    var total = new Spread(0, 0, 0);
    foreach (var (name, depth, body) in bodies)
    {
        var (time, last) = Harness.TimeRuns(opts, runs, body);
        totalNodes += last.Nodes;
        total += time;
        string flag = time.Median < Harness.TimerLimitedSeconds ? "*" : " ";
        Console.WriteLine($"{name,-16} {depth,5} {last.Nodes,10:N0} {Harness.Us(time.Median),12}{flag}{Harness.Us(time.Min),12} {Harness.Us(time.Max),12} {last.Nodes / time.Median,13:N0} {last.ScoreBits,18:X16} {last.Alloc,8}");
        // Deterministic part of the row (no timing): the Phase 6 invariants compare these lines before and after a change.
        Console.WriteLine($"SIG search item={name}/{depth} nodes={last.Nodes} score={last.ScoreBits:X16} move={(last.Inst < 0 ? "none" : last.Game!.InstanceId(last.Inst) + "@" + last.Cell)} alloc={last.Alloc}");
    }
    Console.WriteLine($"{"total",-16} {"",5} {totalNodes,10:N0} {"",12} sum of per-item min {total.Min:F5} s, median {total.Median:F5} s, max {total.Max:F5} s; NPS (median) {totalNodes / total.Median:N0}");
    Console.WriteLine($"RESULT search mode={opts.Tag} nodes={totalNodes} total_min_s={F(total.Min)} total_median_s={F(total.Median)} total_max_s={F(total.Max)}");
}

static void Fast(CardDb db, string[] args, Opts opts)
{
    int runs = args.Length > 1 ? int.Parse(args[1]) : 5;
    (string Name, int[] Depths)[] items =
    {
        ("open_none", new[] { 1, 2, 3, 4, 5, 6, 7 }),
        ("open_same_plus", new[] { 1, 2, 3, 4, 5, 6, 7 }),
        ("mid3_ascension", new[] { 1, 2, 3, 4, 5, 6, 7, 8 }),
        ("mid4_rev_ace", new[] { 1, 2, 3, 4, 5, 6, 7 }),
        ("end5_same_plus", new[] { 1, 2, 3, 4, 5 }),
        ("open_all_rules", new[] { 1, 2, 3, 4, 5, 6 }),
    };
    var compatTt = new CompatTranspositionTable();
    var fastTt = new FastTranspositionTable();

    // The three variants of one item: 0 = ReferenceCompat TT on (Python semantics), 1 = TT off (oracle), 2 = Fast.
    var rows = new List<(string Name, int Depth, Func<Sample>[] V)>();
    foreach (var (name, depths) in items)
    {
        foreach (int depth in depths)
        {
            string n = name;
            int d = depth;
            Func<Sample> Compat(bool useTt) => () =>
            {
                var g = ReferenceScenarios.Create(n, db);
                var s = new CompatSearcher(g, db, compatTt) { UseTranspositionTable = useTt };
                return Harness.Timed(g, () =>
                {
                    var r = s.RunFixedDepth(d, false, SearchLimits.Unlimited);
                    return (s.Stats.NodesSearched, r.Score, r.BestInstance, r.BestCell);
                });
            };
            Func<Sample> fast = () =>
            {
                var g = ReferenceScenarios.Create(n, db);
                var s = new FastSearcher(g, db, fastTt);
                return Harness.Timed(g, () =>
                {
                    var r = s.RunFixedDepth(d, false, SearchLimits.Unlimited);
                    return (s.Stats.NodesSearched, r.Score, r.BestInstance, r.BestCell);
                });
            };
            rows.Add((n, d, new[] { Compat(true), Compat(false), fast }));
        }
    }

    Console.WriteLine($"# Phase 4 [{opts.Tag}]: ReferenceCompat TT on (Python semantics) | TT off (oracle) | Fast, runs={runs}; the three variants are measured in rotating order; Fast table={fastTt.Capacity:N0} entries ({fastTt.MemoryBytes / (1 << 20)} MiB)");
    Console.WriteLine("# times in microseconds: median [min-max]; '*' = median under 50 us (timer-limited)");
    Console.WriteLine($"{"scenario",-15} {"d",2} | {"on nodes",10} {"on us",26} | {"off nodes",10} {"off us",26} {"off NPS",10} | {"fast nodes",10} {"fast us",26} {"fast NPS",10} | {"score",5} {"off move",9} {"fast move",9} {"alloc B",7}");
    Harness.WarmUp(opts, rows.SelectMany(r => r.V));

    long[] nodes = new long[3];
    var tot = new Spread[3];
    int identical = 0, different = 0;
    foreach (var (name, depth, v) in rows)
    {
        var times = new double[3][];
        for (int k = 0; k < 3; k++) times[k] = new double[runs];
        var last = new Sample[3];
        if (opts.Cold)
        {
            for (int k = 0; k < 3; k++) last[k] = v[k](); // one untimed run per variant, all three before the timed rounds
        }
        for (int r = 0; r < runs; r++)
        {
            for (int j = 0; j < 3; j++)
            {
                int k = (j + r) % 3; // rotating order: no variant is always first (review A-6)
                last[k] = v[k]();
                times[k][r] = last[k].Sec;
            }
        }
        var sp = new[] { Spread.Of(times[0]), Spread.Of(times[1]), Spread.Of(times[2]) };
        bool same = last[1].ScoreBits == last[2].ScoreBits;
        if (same) identical++; else different++;
        string Cell(Spread s) => $"{Harness.Us(s.Median)}{(s.Median < Harness.TimerLimitedSeconds ? "*" : " ")}[{Harness.Us(s.Min)}-{Harness.Us(s.Max)}]";
        string Mv(Sample s) => s.Inst < 0 ? "none" : $"{s.Game!.InstanceId(s.Inst)}@{s.Cell}";
        Console.WriteLine($"{name,-15} {depth,2} | {last[0].Nodes,10:N0} {Cell(sp[0]),26} | {last[1].Nodes,10:N0} {Cell(sp[1]),26} {last[1].Nodes / sp[1].Median,10:N0} | {last[2].Nodes,10:N0} {Cell(sp[2]),26} {last[2].Nodes / sp[2].Median,10:N0} | {(same ? "same" : "DIFF"),5} {Mv(last[1]),9} {Mv(last[2]),9} {last[2].Alloc,7}");
        Console.WriteLine($"SIG fast item={name}/{depth} nodes={last[0].Nodes}/{last[1].Nodes}/{last[2].Nodes} score={last[0].ScoreBits:X16}/{last[1].ScoreBits:X16}/{last[2].ScoreBits:X16} move={Mv(last[0])}/{Mv(last[1])}/{Mv(last[2])} alloc={last[2].Alloc}");
        if (depth >= 4)
        {
            for (int k = 0; k < 3; k++)
            {
                nodes[k] += last[k].Nodes;
                tot[k] += sp[k];
            }
        }
    }
    Console.WriteLine($"{"total d>=4",-15} {"",2} | on {tot[0].Median:F4} s [{tot[0].Min:F4}-{tot[0].Max:F4}] ({nodes[0]:N0} nodes) | off {tot[1].Median:F4} s [{tot[1].Min:F4}-{tot[1].Max:F4}] ({nodes[1]:N0}) | fast {tot[2].Median:F4} s [{tot[2].Min:F4}-{tot[2].Max:F4}] ({nodes[2]:N0}); scores equal oracle: {identical}, different: {different}");
    Console.WriteLine($"Fast over TT-off (median) = {tot[1].Median / tot[2].Median:F2}x, over TT-on = {tot[0].Median / tot[2].Median:F2}x; node ratio Fast/TT-off = {(double)nodes[2] / nodes[1]:F3}. Node counts and NPS of Fast are NOT comparable with Python's or ReferenceCompat's (different tree, counts unsorted leaves).");
    Console.WriteLine($"RESULT fast mode={opts.Tag} on_min_s={F(tot[0].Min)} on_median_s={F(tot[0].Median)} on_max_s={F(tot[0].Max)} off_min_s={F(tot[1].Min)} off_median_s={F(tot[1].Median)} off_max_s={F(tot[1].Max)} " +
                      $"fast_min_s={F(tot[2].Min)} fast_median_s={F(tot[2].Median)} fast_max_s={F(tot[2].Max)} nodes_on={nodes[0]} nodes_off={nodes[1]} nodes_fast={nodes[2]} identical={identical} different={different}");
}

static void ExactBench(CardDb db, string[] args, Opts opts)
{
    int runs = args.Length > 1 ? int.Parse(args[1]) : 5;
    (string Name, int Value)[] items = { ("end5_same_plus", 1), ("mid4_rev_ace", -2), ("mid3_ascension", -1) };
    var memo = ExactEndgameSolver.CreateMemo(1 << 17);
    var bodies = items.Select(it => (it.Name, it.Value, Body: (Func<Sample>)(() =>
    {
        var g = ReferenceScenarios.Create(it.Name, db);
        var solver = new ExactEndgameSolver(g, memo);
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        long t0 = Stopwatch.GetTimestamp();
        solver.Bind(g.CurrentPlayer);
        int value = solver.Value(g.CurrentPlayer);
        double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        long alloc = GC.GetAllocatedBytesForCurrentThread() - a0;
        // Nodes carries the make_move count; the node and memo-hit counts are printed from the last run below.
        return new Sample(sec, alloc, solver.MakeMoves, solver.Nodes, (int)solver.MemoHits, 0, g, value);
    }))).ToList();

    Console.WriteLine($"# Phase 5 exact endgame solver (C#) [{opts.Tag}], runs={runs}; times in microseconds (median, min, max); '*' = under 50 us (timer-limited)");
    Console.WriteLine($"{"scenario",-16} {"value",5} {"expected",8} {"make_move",10} {"nodes",10} {"memo hits",10} {"median us",12} {"min us",12} {"max us",12} {"alloc B",8}");
    Harness.WarmUp(opts, bodies.Select(b => b.Body));
    foreach (var (name, expected, body) in bodies)
    {
        var (time, last) = Harness.TimeRuns(opts, runs, body);
        string flag = time.Median < Harness.TimerLimitedSeconds ? "*" : " ";
        Console.WriteLine($"{name,-16} {last.Value,5} {expected,8} {last.Nodes,10:N0} {last.ScoreBits,10:N0} {last.Inst,10:N0} {Harness.Us(time.Median),12}{flag}{Harness.Us(time.Min),12} {Harness.Us(time.Max),12} {last.Alloc,8}");
        Console.WriteLine($"RESULT exact mode={opts.Tag} item={name} value={last.Value} make_moves={last.Nodes} min_s={F(time.Min)} median_s={F(time.Median)} max_s={F(time.Max)}");
    }
}

static void EndgameBench(CardDb db, string[] args, Opts opts)
{
    string path = args.Length > 1 ? args[1] : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "Python", "tools", "bench", "endgame_bench_positions.json");
    int runs = args.Length > 2 ? int.Parse(args[2]) : 5;
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    var root = doc.RootElement;
    var positions = root.GetProperty("positions").EnumerateArray().ToList();

    // Prepare every position (outside the timed region).
    var prepared = new List<(JsonElement Pos, GameState G, bool[][] Gen, int Opp, int Sample)>();
    foreach (var pos in positions)
    {
        var rules = RuleParser.FromPythonList(pos.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
        var g = GameState.FromSnapshot(StateSnapshot.FromJson(pos.GetProperty("state"), rules));
        var gen = pos.GetProperty("gen").EnumerateArray().Select(h => h.EnumerateArray().Select(x => x.GetInt32() != 0).ToArray()).ToArray();
        prepared.Add((pos, g, gen, pos.GetProperty("opp").GetInt32(), pos.GetProperty("sample").GetInt32()));
    }

    // One "body" per (position, stage). Stage results are carried in a side table so that Sample stays small.
    var scenariosOf = new List<GameState>?[prepared.Count];
    var selected = new RobustSelection?[prepared.Count, 2];
    Sample Stage(int i, int stage)
    {
        var (_, g, gen, opp, sample) = prepared[i];
        long t0 = Stopwatch.GetTimestamp();
        if (stage == 0) scenariosOf[i] = EndgameScenarios.Build(db, g, opp, gen, sample).Scenarios;
        else selected[i, stage - 1] = EndgameRobust.Select(db, g, scenariosOf[i]!, g.CurrentPlayer, stage == 2);
        return new Sample(Stopwatch.GetElapsedTime(t0).TotalSeconds, 0, 0, 0, 0, 0, null);
    }

    Console.WriteLine($"# Phase 5 robust endgame pipeline [{opts.Tag}], C# vs Python {root.GetProperty("python").GetString()}, runs={runs}; Python figures are the medians stored in the positions file; positions: {path}");
    Console.WriteLine($"{"position",-14} {"scen",4} | {"py build",9} {"py normal",10} {"py aggr",9} | {"cs build us",11} {"cs normal us",13} {"cs aggr us",11} | {"result",6}");

    var allBodies = new List<Func<Sample>>();
    for (int i = 0; i < prepared.Count; i++)
    {
        int idx = i;
        allBodies.Add(() => Stage(idx, 0));
        allBodies.Add(() => Stage(idx, 1));
        allBodies.Add(() => Stage(idx, 2));
    }
    Harness.WarmUp(opts, allBodies);

    var cells = new SortedDictionary<(int Occ, int Unk), double[]>(); // py total, cs min, cs median, cs max, count
    var csTotal = new Spread(0, 0, 0);
    double totalPy = 0, pyMin = 0, pyMax = 0;
    bool havePySpread = true;
    int same = 0, different = 0;
    for (int i = 0; i < prepared.Count; i++)
    {
        var (pos, g, _, _, _) = prepared[i];
        int idx = i;
        var tb = Harness.TimeRuns(opts, runs, () => Stage(idx, 0)).Time;
        var tn = Harness.TimeRuns(opts, runs, () => Stage(idx, 1)).Time;
        var ta = Harness.TimeRuns(opts, runs, () => Stage(idx, 2)).Time;
        var scenarios = scenariosOf[i]!;

        int Code(RobustSelection r) => r.HasBest ? g.HandSlotOf(g.CurrentPlayer, r.Best.Inst) * 9 + r.Best.Cell : -1;
        var py = pos.GetProperty("py");
        int pyN = py.GetProperty("best_normal").ValueKind == JsonValueKind.Null ? -1 : py.GetProperty("best_normal").GetInt32();
        int pyA = py.GetProperty("best_aggressive").ValueKind == JsonValueKind.Null ? -1 : py.GetProperty("best_aggressive").GetInt32();
        bool ok = Code(selected[i, 0]!) == pyN && Code(selected[i, 1]!) == pyA && scenarios.Count == pos.GetProperty("scenarios").GetInt32();
        if (ok) same++; else different++;

        double pyB = py.GetProperty("build_s").GetDouble(), pyNs = py.GetProperty("select_normal_s").GetDouble(), pyAs = py.GetProperty("select_aggressive_s").GetDouble();
        Console.WriteLine($"{pos.GetProperty("label").GetString(),-14} {scenarios.Count,4} | {pyB,9:F4} {pyNs,10:F4} {pyAs,9:F4} | {Harness.Us(tb.Median),11} {Harness.Us(tn.Median),13} {Harness.Us(ta.Median),11} | {(ok ? "same" : "DIFF"),6}");

        var key = (pos.GetProperty("occupancy").GetInt32(), pos.GetProperty("unknown").GetInt32());
        if (!cells.TryGetValue(key, out var acc)) cells[key] = acc = new double[5];
        var sum = tb + tn + ta;
        acc[0] += pyB + pyNs + pyAs;
        acc[1] += sum.Min;
        acc[2] += sum.Median;
        acc[3] += sum.Max;
        acc[4]++;
        csTotal += sum;
        totalPy += pyB + pyNs + pyAs;
        if (py.TryGetProperty("build_min_s", out var bmin) && py.TryGetProperty("select_normal_min_s", out var nmin) && py.TryGetProperty("select_aggressive_min_s", out var amin) &&
            py.TryGetProperty("build_max_s", out var bmax) && py.TryGetProperty("select_normal_max_s", out var nmax) && py.TryGetProperty("select_aggressive_max_s", out var amax))
        {
            pyMin += bmin.GetDouble() + nmin.GetDouble() + amin.GetDouble();
            pyMax += bmax.GetDouble() + nmax.GetDouble() + amax.GetDouble();
        }
        else
        {
            havePySpread = false;
        }
    }

    Console.WriteLine();
    Console.WriteLine($"{"occupancy",9} {"unknown",7} {"positions",9} {"python s",10} {"c# median s",12} {"c# min-max s",20} {"speed-up (median)",18}");
    foreach (var (key, acc) in cells)
    {
        Console.WriteLine($"{key.Occ,9} {key.Unk,7} {acc[4],9:F0} {acc[0],10:F4} {acc[2],12:F5} {acc[1],9:F5}-{acc[3]:F5} {acc[0] / acc[2],17:F1}x");
    }
    string pySpread = havePySpread ? $"Python total min {pyMin:F4} / median {totalPy:F4} / max {pyMax:F4} s" : $"Python total (median of its runs) {totalPy:F4} s";
    Console.WriteLine($"{"total",9} {"",7} {"",9} {totalPy,10:F4} {csTotal.Median,12:F5} {csTotal.Min,9:F5}-{csTotal.Max:F5} {totalPy / csTotal.Median,17:F1}x   selections identical to Python: {same}, different: {different}");
    Console.WriteLine($"# C# totals are sums of per-position min / median / max of {runs} runs; {pySpread}");
    Console.WriteLine($"RESULT endgame mode={opts.Tag} py_total_s={F(totalPy)} cs_min_s={F(csTotal.Min)} cs_median_s={F(csTotal.Median)} cs_max_s={F(csTotal.Max)} same={same} different={different}");
}

static void EndgameWork(CardDb db, string[] args)
{
    string path = args.Length > 1 ? args[1] : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "Python", "tools", "bench", "endgame_bench_positions.json");
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    long solverMakes = 0, firstMoves = 0, pairs = 0, solverNodes = 0;
    int positions = 0;
    var byCell = new SortedDictionary<(int Occ, int Unk), long[]>(); // solver makes, first moves
    foreach (var pos in doc.RootElement.GetProperty("positions").EnumerateArray())
    {
        var rules = RuleParser.FromPythonList(pos.GetProperty("rules").EnumerateArray().Select(x => x.GetString()!));
        var g = GameState.FromSnapshot(StateSnapshot.FromJson(pos.GetProperty("state"), rules));
        var gen = pos.GetProperty("gen").EnumerateArray().Select(h => h.EnumerateArray().Select(x => x.GetInt32() != 0).ToArray()).ToArray();
        var scenarios = EndgameScenarios.Build(db, g, pos.GetProperty("opp").GetInt32(), gen, pos.GetProperty("sample").GetInt32()).Scenarios!;
        int moves = g.GenerateMoves(new Move[GameState.MaxMoves]); // legal moves of the side to move (can_use filter included)
        var key = (pos.GetProperty("occupancy").GetInt32(), pos.GetProperty("unknown").GetInt32());
        if (!byCell.TryGetValue(key, out var acc)) byCell[key] = acc = new long[2];
        foreach (bool aggressive in new[] { false, true })
        {
            var sel = EndgameRobust.Select(db, g, scenarios, g.CurrentPlayer, aggressive);
            // Every (move, scenario) pair makes one first move; Select skips a pair only if the card is missing from the
            // scenario hand, which the Python-side count below (top-level exact solves) rules out for these positions.
            long pairCount = (long)moves * scenarios.Count;
            solverMakes += sel.ExactMakeMoves;
            solverNodes += sel.ExactNodes;
            firstMoves += pairCount;
            pairs += pairCount;
            acc[0] += sel.ExactMakeMoves;
            acc[1] += pairCount;
        }
        positions++;
    }
    Console.WriteLine($"# Phase 5 robust pipeline work, C# (not timed): {positions} positions x 2 modes (normal + aggressive)");
    Console.WriteLine($"{"occupancy",9} {"unknown",7} {"solver make_move",17} {"first moves",12} {"total make_move",16}");
    foreach (var (key, acc) in byCell)
        Console.WriteLine($"{key.Occ,9} {key.Unk,7} {acc[0],17:N0} {acc[1],12:N0} {acc[0] + acc[1],16:N0}");
    Console.WriteLine($"{"total",9} {"",7} {solverMakes,17:N0} {firstMoves,12:N0} {solverMakes + firstMoves,16:N0}");
    Console.WriteLine($"RESULT endgame-work solver_make_moves={solverMakes} first_moves={firstMoves} total_make_moves={solverMakes + firstMoves} pairs={pairs} solver_nodes={solverNodes}");
}

static void ServerBench(CardDb db, string[] args, Opts opts)
{
    double seconds = args.Length > 1 ? double.Parse(args[1], CultureInfo.InvariantCulture) : 10;
    int runs = args.Length > 2 ? int.Parse(args[2]) : 3;
    Console.WriteLine($"# server-equivalent iterative deepening [{opts.Tag}], max depth 10, time limit {seconds} s, runs={runs} (wall clock: the depth reached depends on machine speed, so this is not a like-for-like work comparison)");
    Console.WriteLine($"{"scenario",-16} {"searcher",-14} {"depth done (min/med/max)",24} {"nodes (median)",15} {"seconds (median)",17} {"NPS min/median/max",34} {"move (card@cell)",18}");

    SearchStats Run(string name, bool fast, double limit, out double sec, out string move)
    {
        var g = ReferenceScenarios.Create(name, db);
        SearchStats stats;
        long t0 = Stopwatch.GetTimestamp();
        SearchResult r;
        if (fast)
        {
            var s = new FastSearcher(g, db, new FastTranspositionTable(1 << 22));
            r = s.RunIterativeDeepening(10, false, SearchLimits.WithTime(limit));
            stats = s.Stats;
        }
        else
        {
            var s = new CompatSearcher(g, db, new CompatTranspositionTable());
            r = s.RunIterativeDeepening(10, false, SearchLimits.WithTime(limit));
            stats = s.Stats;
        }
        sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        move = $"{g.InstanceId(r.BestInstance)}@{r.BestCell}";
        return stats;
    }

    var names = new[] { "open_none", "open_same_plus", "mid3_ascension" };
    if (!opts.Cold)
    {
        // Short wall-clock passes so that the search code is tier-1 before the timed runs.
        var sw = Stopwatch.StartNew();
        do
        {
            foreach (string name in names)
                foreach (bool fast in new[] { false, true }) Run(name, fast, Math.Min(0.25, seconds), out _, out _);
        } while (sw.ElapsedMilliseconds < Math.Min(opts.WarmupMs, 3000));
    }

    foreach (string name in names)
    {
        foreach (bool fast in new[] { false, true })
        {
            var depths = new int[runs];
            var secs = new double[runs];
            var nps = new double[runs];
            long[] nodes = new long[runs];
            string move = "";
            for (int i = 0; i < runs; i++)
            {
                var st = Run(name, fast, seconds, out double sec, out move);
                depths[i] = st.DepthCompleted;
                secs[i] = sec;
                nodes[i] = st.NodesSearched;
                nps[i] = st.NodesSearched / sec;
            }
            Array.Sort(depths);
            Array.Sort(nodes);
            var s = Spread.Of(secs);
            var n = Spread.Of(nps);
            string label = fast ? "fast" : "compat(Python)";
            Console.WriteLine($"{name,-16} {label,-14} {depths[0] + "/" + depths[runs / 2] + "/" + depths[^1],24} {nodes[runs / 2],15:N0} {s.Median,17:F3} {n.Min,11:N0}/{n.Median:N0}/{n.Max:N0} {move,18}");
            Console.WriteLine($"RESULT server mode={opts.Tag} item={name} searcher={(fast ? "fast" : "compat")} depth_min={depths[0]} depth_median={depths[runs / 2]} depth_max={depths[^1]} nodes_median={nodes[runs / 2]} sec_median={F(s.Median)} nps_min={F(n.Min)} nps_median={F(n.Median)} nps_max={F(n.Max)} move={move}");
        }
    }
}

/// <summary>Per-run timing summary.</summary>
readonly record struct Spread(double Min, double Median, double Max)
{
    public static Spread Of(double[] times)
    {
        var t = (double[])times.Clone();
        Array.Sort(t);
        return new Spread(t[0], t[t.Length / 2], t[^1]);
    }

    public static Spread operator +(Spread a, Spread b) => new(a.Min + b.Min, a.Median + b.Median, a.Max + b.Max);
}

/// <summary>One measured run: the timed region covers the search/solve call only (state and searcher are built outside).</summary>
readonly record struct Sample(double Sec, long Alloc, long Nodes, long ScoreBits, int Inst, int Cell, GameState? Game, int Value = 0);

sealed record Opts(bool Cold, int WarmupMs, string[] Pos)
{
    public string Tag => Cold ? "cold" : "warm";

    public string Label => Cold
        ? "cold (one untimed warm-up run per item; the first items run tier-0/OSR code)"
        : $"warm (the whole item set is repeated for {WarmupMs} ms before any timed run)";

    public static Opts Parse(string[] args)
    {
        bool cold = false;
        int warmup = 1500;
        var rest = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--cold") cold = true;
            else if (args[i] == "--warmup-ms" && i + 1 < args.Length) warmup = int.Parse(args[++i], CultureInfo.InvariantCulture);
            else rest.Add(args[i]);
        }
        return new Opts(cold, warmup, rest.ToArray());
    }
}

static class Harness
{
    /// <summary>Warm mode: run every body of the item set, over and over, for at least <paramref name="opts"/>.WarmupMs.</summary>
    public static void WarmUp(Opts opts, IEnumerable<Func<Sample>> bodies)
    {
        if (opts.Cold) return;
        var list = bodies.ToList();
        var sw = Stopwatch.StartNew();
        do
        {
            foreach (var body in list) body();
        } while (sw.ElapsedMilliseconds < opts.WarmupMs);
    }

    /// <summary>Times <paramref name="runs"/> runs of one item. In cold mode one untimed run comes first (the old behaviour).</summary>
    public static (Spread Time, Sample Last) TimeRuns(Opts opts, int runs, Func<Sample> body)
    {
        Sample last = default;
        if (opts.Cold) last = body();
        var t = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            last = body();
            t[i] = last.Sec;
        }
        return (Spread.Of(t), last);
    }

    /// <summary>Runs <paramref name="search"/> (already prepared) and times only the call.</summary>
    public static Sample Timed(GameState g, Func<(long Nodes, double Score, int Inst, int Cell)> search)
    {
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        long t0 = Stopwatch.GetTimestamp();
        var r = search();
        double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        long alloc = GC.GetAllocatedBytesForCurrentThread() - a0;
        return new Sample(sec, alloc, r.Nodes, BitConverter.DoubleToInt64Bits(r.Score), r.Inst, r.Cell, g);
    }

    public const double TimerLimitedSeconds = 50e-6;

    public static string Us(double sec) => (sec * 1e6).ToString("N1", CultureInfo.InvariantCulture);
}
