using System.Text;

namespace TTC.Siren.Core;

/// <summary>
/// Optional event counters used by the differential tests to prove that every rule path was exercised.
/// <see cref="GameState.Coverage"/> is <c>null</c> on the search path, so counting costs one null check.
/// </summary>
public sealed class CoverageCounters
{
    public long Moves;
    public long Undos;
    /// <summary>Moves in which at least one enemy card was flipped by the base comparison (reason 'base').</summary>
    public long BaseFlip;
    public long SameFlip;
    public long PlusFlip;
    /// <summary>Flips performed inside the one-level combo recursion (chain_only).</summary>
    public long ComboFlip;
    /// <summary>Comparisons (placed/chained card vs neighbour) won because of 逆转 (smaller number wins).</summary>
    public long ReverseWin;
    /// <summary>王牌杀手 special cases, one counter per case.</summary>
    public long Ace1v10;
    public long Ace10v1;
    public long Ace1v10Reverse;
    public long Ace10v1Reverse;
    /// <summary>Moves in which <c>recalculate_type_modifiers</c> ran (played card had a type).</summary>
    public long AscensionRecalc;
    public long DescensionRecalc;
    /// <summary>Undos that restored a snapshot (type rules active).</summary>
    public long UndoWithSnapshot;
    /// <summary>D-002 (fixed): undos in which at least one card's modifier had changed during the move and was restored by card identity.</summary>
    public long UndoModifierChanged;

    public void Reset()
    {
        Moves = Undos = BaseFlip = SameFlip = PlusFlip = ComboFlip = ReverseWin = 0;
        Ace1v10 = Ace10v1 = Ace1v10Reverse = Ace10v1Reverse = 0;
        AscensionRecalc = DescensionRecalc = UndoWithSnapshot = UndoModifierChanged = 0;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append($"moves={Moves} undos={Undos} | base={BaseFlip} same={SameFlip} plus={PlusFlip} combo={ComboFlip} ")
          .Append($"reverseWin={ReverseWin} | ace 1v10={Ace1v10} 10v1={Ace10v1} 1v10+rev={Ace1v10Reverse} 10v1+rev={Ace10v1Reverse} | ")
          .Append($"ascRecalc={AscensionRecalc} descRecalc={DescensionRecalc} | undoSnap={UndoWithSnapshot} undoModChanged={UndoModifierChanged}");
        return sb.ToString();
    }
}
