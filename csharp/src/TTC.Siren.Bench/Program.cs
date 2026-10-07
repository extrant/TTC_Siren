using System.Diagnostics;
using TTC.Siren.Core;
using TTC.Siren.Search;

// Benchmarks (Release, fixed inputs, N runs, median reported).
//   TTC.Siren.Bench makeundo [pairsPerRun=2000000] [runs=5]   Make/Undo throughput (Phase 2)
//   TTC.Siren.Bench search   [runs=5]                         ReferenceCompat fixed-depth search on the 22 items of section 3.4 (Phase 3)
string mode = args.Length > 0 ? args[0] : "makeundo";
var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
Console.WriteLine($"# runtime={Environment.Version} gc={(System.Runtime.GCSettings.IsServerGC ? "server" : "workstation")} mode={mode}");

if (mode == "makeundo") MakeUndo(db, args);
else if (mode == "search") Search(db, args);
else Console.WriteLine("usage: makeundo [pairs] [runs] | search [runs]");
return;

static void MakeUndo(CardDb db, string[] args)
{
    long pairs = args.Length > 1 ? long.Parse(args[1]) : 2_000_000;
    int runs = args.Length > 2 ? int.Parse(args[2]) : 5;
    Console.WriteLine($"# make/undo throughput  pairs/run={pairs:N0} runs={runs}");
    Console.WriteLine($"{"scenario",-16} {"median s",9} {"pairs/s (median)",18} {"min s",8} {"max s",8} {"alloc bytes",12}");

    double sumRate = 0;
    foreach (string name in ReferenceScenarios.Names)
    {
        var g = ReferenceScenarios.Create(name, db);
        ReferenceScenarios.RunPlayouts(g, 1, 200_000); // warm-up

        var times = new double[runs];
        long alloc = 0;
        for (int i = 0; i < runs; i++)
        {
            long a0 = GC.GetAllocatedBytesForCurrentThread();
            long t0 = Stopwatch.GetTimestamp();
            long checksum = ReferenceScenarios.RunPlayouts(g, 987654321, pairs);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            alloc += GC.GetAllocatedBytesForCurrentThread() - a0;
            if (checksum == long.MinValue) Console.WriteLine(); // keep the result observable
        }
        Array.Sort(times);
        double median = times[runs / 2];
        double rate = pairs / median;
        sumRate += rate;
        Console.WriteLine($"{name,-16} {median,9:F3} {rate,18:N0} {times[0],8:F3} {times[^1],8:F3} {alloc,12}");
    }
    Console.WriteLine($"{"mean of medians",-16} {"",9} {sumRate / ReferenceScenarios.Names.Length,18:N0}");
}

static void Search(CardDb db, string[] args)
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
    Console.WriteLine($"# ReferenceCompat fixed-depth search, runs={runs}, median time");
    Console.WriteLine($"{"scenario",-16} {"depth",5} {"nodes",10} {"median s",9} {"NPS",12} {"score(hex bits)",18} {"alloc B",8}");

    long totalNodes = 0;
    double totalTime = 0;
    foreach (var (name, depths) in items)
    {
        foreach (int depth in depths)
        {
            var times = new double[runs];
            long nodes = 0, alloc = 0;
            long scoreBits = 0;
            for (int i = 0; i < runs + 1; i++) // first run = warm-up
            {
                var g = ReferenceScenarios.Create(name, db);
                var searcher = new CompatSearcher(g, db, tt);
                long a0 = GC.GetAllocatedBytesForCurrentThread();
                long t0 = Stopwatch.GetTimestamp();
                var r = searcher.RunFixedDepth(depth, false, SearchLimits.Unlimited);
                double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                long a1 = GC.GetAllocatedBytesForCurrentThread();
                nodes = searcher.Stats.NodesSearched;
                scoreBits = BitConverter.DoubleToInt64Bits(r.Score);
                if (i == 0) continue;
                times[i - 1] = sec;
                alloc = a1 - a0;
            }
            Array.Sort(times);
            double median = times[runs / 2];
            totalNodes += nodes;
            totalTime += median;
            Console.WriteLine($"{name,-16} {depth,5} {nodes,10:N0} {median,9:F4} {nodes / median,12:N0} {scoreBits,18:X16} {alloc,8}");
        }
    }
    Console.WriteLine($"{"total",-16} {"",5} {totalNodes,10:N0} {totalTime,9:F4} {totalNodes / totalTime,12:N0}");
}
