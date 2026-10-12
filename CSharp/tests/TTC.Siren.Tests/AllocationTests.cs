using TTC.Siren.Core;
using Xunit.Abstractions;

namespace TTC.Siren.Tests;

public sealed class AllocationTests
{
    private readonly ITestOutputHelper _output;

    public AllocationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void One_million_make_undo_pairs_allocate_nothing()
    {
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var states = ReferenceScenarios.Names.Select(n => ReferenceScenarios.Create(n, db)).ToArray();

        // Warm up (JIT, tiering) so only steady-state allocation is measured.
        foreach (var g in states) ReferenceScenarios.RunPlayouts(g, 12345, 20_000);

        const long pairsPerScenario = 1_000_000 / 6 + 1;
        long total = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        long checksum = 0;
        foreach (var g in states)
        {
            checksum += ReferenceScenarios.RunPlayouts(g, 987654321, pairsPerScenario);
            total += pairsPerScenario;
        }
        long after = GC.GetAllocatedBytesForCurrentThread();

        _output.WriteLine($"{total} make/undo pairs over {states.Length} scenarios, checksum {checksum}, allocated {after - before} bytes");
        Assert.True(total >= 1_000_000);
        Assert.Equal(0, after - before);
        foreach (var g in states) Assert.Equal(0, g.UndoDepth);
    }

    [Fact]
    public void Playouts_exercise_type_rules_and_combo_in_allocation_scenarios()
    {
        // Guards the allocation test against silently degenerating: the type-rule scenarios must reach snapshot restores.
        var db = CardDb.LoadFromRepository(AppContext.BaseDirectory);
        var g = ReferenceScenarios.Create("open_all_rules", db);
        var cov = new CoverageCounters();
        g.Coverage = cov;
        ReferenceScenarios.RunPlayouts(g, 42, 50_000);
        _output.WriteLine(cov.ToString());
        Assert.True(cov.UndoWithSnapshot > 1000);
        Assert.True(cov.BaseFlip > 1000);
    }
}
