using System.Numerics;
using System.Runtime.CompilerServices;

namespace TTC.Siren.Core;

/// <summary>
/// Mutable Triple Triad state with allocation-free <see cref="MakeMove"/> / <see cref="UndoMove"/>.
///
/// Data-oriented layout: card instances live in parallel fixed arrays, the board stores instance indices,
/// hands are ordered instance-index arrays. Semantics follow the Python reference (core/game_state.py,
/// core/card.py) exactly, including the known quirks listed in CSHARP_MIGRATION.md §4 (marked PY-QUIRK).
/// Not thread-safe: one searcher owns one instance.
/// </summary>
public sealed class GameState
{
    public const int HandCapacity = 5;
    public const int MaxInstances = 9 + 2 * HandCapacity;
    public const int MaxUndoDepth = 16;
    public const int MaxMoves = 9 * HandCapacity;
    public const int NoWinner = -1;

    private const int SnapLen = MaxInstances;

    // Neighbours per cell in Python's direction order: up, down, left, right (out-of-board entries skipped).
    private static readonly byte[] NbCount = new byte[9];
    private static readonly byte[] NbCell = new byte[9 * 4];
    private static readonly byte[] NbMySide = new byte[9 * 4];
    private static readonly byte[] NbOppSide = new byte[9 * 4];

    /// <summary>Number of on-board neighbours of <paramref name="cell"/> (Python's direction order: up, down, left, right).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int NeighborCount(int cell) => NbCount[cell];

    /// <summary>The cell of neighbour <paramref name="k"/> (0..<see cref="NeighborCount"/>-1) of <paramref name="cell"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int NeighborCell(int cell, int k) => NbCell[cell * 4 + k];

    /// <summary>The side of the card on <paramref name="cell"/> that touches neighbour <paramref name="k"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int NeighborMySide(int cell, int k) => NbMySide[cell * 4 + k];

    /// <summary>The side of neighbour <paramref name="k"/> that touches the card on <paramref name="cell"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int NeighborOppSide(int cell, int k) => NbOppSide[cell * 4 + k];

    static GameState()
    {
        (int Dr, int Dc, int My, int Opp)[] dirs =
        {
            (-1, 0, Side.Up, Side.Down),
            (1, 0, Side.Down, Side.Up),
            (0, -1, Side.Left, Side.Right),
            (0, 1, Side.Right, Side.Left),
        };
        for (int cell = 0; cell < 9; cell++)
        {
            int r = cell / 3, c = cell % 3, n = 0;
            foreach (var (dr, dc, my, opp) in dirs)
            {
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= 3 || nc < 0 || nc >= 3) continue;
                NbCell[cell * 4 + n] = (byte)(nr * 3 + nc);
                NbMySide[cell * 4 + n] = (byte)my;
                NbOppSide[cell * 4 + n] = (byte)opp;
                n++;
            }
            NbCount[cell] = (byte)n;
        }
    }

    private struct UndoRecord
    {
        public byte Cell;
        public byte Player;
        public byte Inst;
        public byte PlacedOrigOwner;
        public bool HasSnapshot;
        public ushort FlipMask;
        public ushort FlipFromNoneMask;
        public ulong HashBefore; // Zobrist hash before the move: undo restores it instead of re-deriving it
    }

    // Card instance table (structure of arrays). Instance indices are stable for the life of the state.
    private readonly int[] _id = new int[MaxInstances];
    private readonly byte[] _type = new byte[MaxInstances];
    private readonly byte[] _owner = new byte[MaxInstances];
    private readonly sbyte[] _mod = new sbyte[MaxInstances];
    private readonly bool[] _canUse = new bool[MaxInstances];
    private readonly byte[] _base = new byte[MaxInstances * 4];
    private readonly byte[] _display = new byte[MaxInstances * 4]; // PY-QUIRK(D-003): Python's .up/.right/.down/.left
    private int _instCount;

    private readonly sbyte[] _cell = new sbyte[9];
    private ushort _occupied;
    private readonly byte[] _hand = new byte[2 * HandCapacity];
    private readonly byte[] _handCount = new byte[2];
    private int _current;
    private RuleSet _rules;
    private ulong _hash; // incremental Zobrist hash (see ZobristKeys for what it covers)

    private readonly UndoRecord[] _undo = new UndoRecord[MaxUndoDepth];
    private readonly sbyte[] _snapshot = new sbyte[MaxUndoDepth * SnapLen];
    private int _undoCount;

    // Scratch for the move being resolved.
    private ushort _flipMask;
    private ushort _flipFromNoneMask;

    /// <summary>Optional event counters (tests only). Leave <c>null</c> on the search path.</summary>
    public CoverageCounters? Coverage { get; set; }

    private GameState()
    {
        Array.Fill(_cell, (sbyte)-1);
    }

    // ------------------------------------------------------------------ read access

    public int CurrentPlayer => _current;

    /// <summary>Incremental Zobrist hash of the position (see <see cref="ZobristKeys"/>); updated by make, restored by undo.</summary>
    public ulong ZobristHash => _hash;

    /// <summary>The Zobrist hash derived from scratch from the current fields. Used by tests to verify the incremental one.</summary>
    public ulong ComputeZobristHash()
    {
        ulong h = _current == 1 ? ZobristKeys.Side : 0UL;
        for (int cell = 0; cell < 9; cell++)
        {
            if (_cell[cell] >= 0) h ^= ZobristKeys.Cell(cell, _cell[cell]);
        }
        for (int inst = 0; inst < _instCount; inst++)
            h ^= ZobristKeys.Owner(inst, _owner[inst]) ^ ZobristKeys.Mod(inst, _mod[inst]);
        return h;
    }
    public RuleSet Rules => _rules;
    public int UndoDepth => _undoCount;
    public ushort OccupiedMask => _occupied;
    public int HandCount(int player) => _handCount[player];
    public int HandInstance(int player, int slot) => _hand[player * HandCapacity + slot];
    public int BoardInstance(int cell) => _cell[cell];

    /// <summary>
    /// Current hand slot of a card instance, or -1 if it is not in that player's hand. A slot is only valid until
    /// the next make/undo (D-001 reorders the hand); <see cref="MakeMove"/> uses this to resolve
    /// <see cref="Move.Inst"/>, and tests use it to translate golden "slot" codes.
    /// </summary>
    public int HandSlotOf(int player, int instance)
    {
        int hc = _handCount[player];
        for (int i = 0; i < hc; i++)
        {
            if (_hand[player * HandCapacity + i] == instance) return i;
        }
        return -1;
    }

    /// <summary>
    /// Builds the move that plays the card currently at <paramref name="slot"/> of the side to move's hand. The slot is
    /// resolved against the hand as it is right now (golden files and the Python trace use slot codes).
    /// </summary>
    public Move MoveFromSlot(int slot, int cell)
    {
        if ((uint)slot >= _handCount[_current])
            throw new ArgumentOutOfRangeException(nameof(slot), $"slot {slot} but player {_current} holds {_handCount[_current]} cards");
        return new Move(_hand[_current * HandCapacity + slot], cell);
    }

    public bool IsEmpty(int cell) => (_occupied >> cell & 1) == 0;
    public int AvailableCellCount => 9 - BitOperations.PopCount(_occupied);

    /// <summary>Number of card instances (board + both hands at load time). Instances never change identity.</summary>
    public int InstanceCount => _instCount;

    /// <summary>Card id of an instance; -1 stands for Python <c>None</c>.</summary>
    public int InstanceId(int inst) => _id[inst];
    public byte InstanceOwner(int inst) => _owner[inst];
    public int InstanceModifier(int inst) => _mod[inst];
    public byte InstanceType(int inst) => _type[inst];
    public bool InstanceCanUse(int inst) => _canUse[inst];
    public int InstanceBase(int inst, int side) => _base[inst * 4 + side];
    public int InstanceDisplay(int inst, int side) => _display[inst * 4 + side];

    /// <summary>Python <c>Card.get_modified_value</c>: <c>max(1, min(10, base + type_modifier))</c>.</summary>
    public int InstanceEffective(int inst, int side) => Effective(inst, side);

    /// <summary>Cells flipped by the most recent un-undone move (bit per row-major cell).</summary>
    public ushort LastFlipMask => _undoCount > 0 ? _undo[_undoCount - 1].FlipMask : (ushort)0;

    // ------------------------------------------------------------------ construction / export

    public static GameState FromSnapshot(StateSnapshot s)
    {
        ArgumentNullException.ThrowIfNull(s);
        if (s.Current is not (0 or 1)) throw new InvalidGameStateException($"Invalid current player {s.Current}");

        var g = new GameState { _current = s.Current, _rules = s.Rules };
        for (int cell = 0; cell < 9; cell++)
        {
            var c = s.Board[cell];
            if (c is null) continue;
            int inst = g.AddInstance(c);
            g._cell[cell] = (sbyte)inst;
            g._occupied |= (ushort)(1 << cell);
        }
        for (int p = 0; p < 2; p++)
        {
            var hand = s.Hands[p];
            if (hand.Count > HandCapacity)
                throw new InvalidGameStateException($"Hand {p} has {hand.Count} cards (capacity {HandCapacity})");
            for (int i = 0; i < hand.Count; i++)
            {
                // D-004 (FIXED-BOTH): two cards of one hand may share a card id (identical copies of a real card, or two
                // None placeholders). Python's make_move now removes the hand card that is played (object identity first,
                // card_id only as a fallback for copies), and every C# card instance has its own identity, so such
                // hands are legal input. History and the Python TT key are keyed by id, as in Python.
                int inst = g.AddInstance(hand[i]);
                g._hand[p * HandCapacity + i] = (byte)inst;
            }
            g._handCount[p] = (byte)hand.Count;
        }
        g._hash = g.ComputeZobristHash();
        return g;
    }

    private int AddInstance(CardSnapshot c)
    {
        if (_instCount >= MaxInstances) throw new InvalidGameStateException("Too many cards");
        if (c.Base.Length != 4 || c.Display.Length != 4) throw new InvalidGameStateException("Card edges must have 4 entries");
        if (c.Mod is < sbyte.MinValue or > sbyte.MaxValue) throw new InvalidGameStateException($"Modifier {c.Mod} out of range");
        if (c.Owner > OwnerId.None) throw new InvalidGameStateException($"Invalid owner {c.Owner}");
        int i = _instCount++;
        _id[i] = c.Id;
        _type[i] = CardTypes.IdOf(c.Type);
        _owner[i] = c.Owner;
        _mod[i] = (sbyte)c.Mod;
        _canUse[i] = c.CanUse;
        for (int k = 0; k < 4; k++)
        {
            _base[i * 4 + k] = c.Base[k];
            _display[i * 4 + k] = c.Display[k];
        }
        return i;
    }

    public StateSnapshot ToSnapshot()
    {
        var s = new StateSnapshot { Current = _current, Rules = _rules };
        for (int cell = 0; cell < 9; cell++)
        {
            if (_cell[cell] >= 0) s.Board[cell] = SnapshotOf(_cell[cell]);
        }
        for (int p = 0; p < 2; p++)
        {
            for (int i = 0; i < _handCount[p]; i++) s.Hands[p].Add(SnapshotOf(_hand[p * HandCapacity + i]));
        }
        return s;
    }

    private CardSnapshot SnapshotOf(int inst)
    {
        var c = new CardSnapshot
        {
            Id = _id[inst],
            Type = CardTypes.NameOf(_type[inst]),
            Owner = _owner[inst],
            Mod = _mod[inst],
            CanUse = _canUse[inst],
        };
        for (int k = 0; k < 4; k++)
        {
            c.Base[k] = _base[inst * 4 + k];
            c.Display[k] = _display[inst * 4 + k];
        }
        return c;
    }

    // ------------------------------------------------------------------ queries

    /// <summary>Python <c>count_cards</c>: cards owned by red/blue on the board and in both hands (by card.owner).</summary>
    public void CountCards(out int red, out int blue)
    {
        red = 0;
        blue = 0;
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = _cell[cell];
            if (inst < 0) continue;
            if (_owner[inst] == OwnerId.Red) red++;
            else if (_owner[inst] == OwnerId.Blue) blue++;
        }
        for (int p = 0; p < 2; p++)
        {
            for (int i = 0; i < _handCount[p]; i++)
            {
                int inst = _hand[p * HandCapacity + i];
                if (_owner[inst] == OwnerId.Red) red++;
                else if (_owner[inst] == OwnerId.Blue) blue++;
            }
        }
    }

    public bool IsGameOver => _occupied == 0x1FF;

    /// <summary>Python <c>get_winner</c>: 0 red, 1 blue, <see cref="NoWinner"/> for draw or unfinished.</summary>
    public int GetWinner()
    {
        if (!IsGameOver) return NoWinner;
        CountCards(out int red, out int blue);
        return red > blue ? 0 : blue > red ? 1 : NoWinner;
    }

    /// <summary>
    /// Python <c>get_available_moves</c>: hand order (filtered by can_use under 秩序/混乱) times row-major empty cells.
    /// Returns the number of moves written; <paramref name="dst"/> must hold at least <see cref="MaxMoves"/>.
    /// </summary>
    public int GenerateMoves(Span<Move> dst)
    {
        int p = _current, hc = _handCount[p], n = 0;
        bool filter = (_rules & (RuleSet.Order | RuleSet.Chaos)) != 0;
        for (int slot = 0; slot < hc; slot++)
        {
            if (filter && !_canUse[_hand[p * HandCapacity + slot]]) continue;
            for (int cell = 0; cell < 9; cell++)
            {
                if ((_occupied >> cell & 1) == 0) dst[n++] = new Move(_hand[p * HandCapacity + slot], cell);
            }
        }
        return n;
    }

    // ------------------------------------------------------------------ rules

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Effective(int inst, int side)
    {
        int v = _base[inst * 4 + side] + _mod[inst];
        return v < 1 ? 1 : v > 10 ? 10 : v;
    }

    /// <summary>
    /// Python <c>Card.compare_values</c> on effective values: 1 = mover wins, -1 = loses, 0 = tie.
    /// <paramref name="info"/>: 1..4 = 王牌杀手 case (1v10, 10v1, 1v10+逆转, 10v1+逆转), 5 = decided by 逆转.
    /// </summary>
    public static int CompareValues(int my, int other, RuleSet rules, out int info)
    {
        info = 0;
        if ((rules & RuleSet.FallenAce) != 0)
        {
            bool myAce = my == 1 || my == 10;
            bool otherAce = other == 1 || other == 10;
            if (myAce && otherAce)
            {
                if (my == other) return 0;
                bool rev = (rules & RuleSet.Reverse) != 0;
                if (my == 1) // other == 10
                {
                    info = rev ? 3 : 1;
                    return rev ? -1 : 1;
                }
                info = rev ? 4 : 2; // my == 10, other == 1
                return rev ? 1 : -1;
            }
        }
        if ((rules & RuleSet.Reverse) != 0)
        {
            if (my < other)
            {
                info = 5;
                return 1;
            }
            return my > other ? -1 : 0;
        }
        return my > other ? 1 : my < other ? -1 : 0;
    }

    public static int CompareValues(int my, int other, RuleSet rules) => CompareValues(my, other, rules, out _);

    private int CompareInstances(int myInst, int mySide, int oppInst, int oppSide)
    {
        int result = CompareValues(Effective(myInst, mySide), Effective(oppInst, oppSide), _rules, out int info);
        var cov = Coverage;
        if (cov is not null && info != 0)
        {
            switch (info)
            {
                case 1: cov.Ace1v10++; break;
                case 2: cov.Ace10v1++; break;
                case 3: cov.Ace1v10Reverse++; break;
                case 4: cov.Ace10v1Reverse++; break;
                case 5: cov.ReverseWin++; break;
            }
        }
        return result;
    }

    /// <summary>
    /// Python <c>resolve_flip_with_record</c>. Directions are visited up, down, left, right.
    /// Reasons: 1 = base, 2 = plus, 3 = same; plus/same are "strong" (never overwritten, trigger the combo).
    /// </summary>
    private void ResolveFlips(int cell, int inst, bool chainOnly)
    {
        byte mover = _owner[inst];
        int n = NbCount[cell], nb0 = cell * 4;
        Span<byte> reason = stackalloc byte[4];
        Span<byte> order = stackalloc byte[4];
        int orderCount = 0;

        // Base comparison (includes 逆转 / 王牌杀手).
        for (int k = 0; k < n; k++)
        {
            int ni = _cell[NbCell[nb0 + k]];
            if (ni < 0 || _owner[ni] == mover) continue;
            if (CompareInstances(inst, NbMySide[nb0 + k], ni, NbOppSide[nb0 + k]) == 1)
                AddFlip(k, 1, reason, order, ref orderCount);
        }

        if (!chainOnly)
        {
            if ((_rules & RuleSet.Plus) != 0)
            {
                // Sums of (my edge + neighbour edge) over every neighbouring card, own cards included.
                Span<int> sums = stackalloc int[4];
                Span<byte> ks = stackalloc byte[4];
                int cnt = 0;
                for (int k = 0; k < n; k++)
                {
                    int ni = _cell[NbCell[nb0 + k]];
                    if (ni < 0) continue;
                    sums[cnt] = Effective(inst, NbMySide[nb0 + k]) + Effective(ni, NbOppSide[nb0 + k]);
                    ks[cnt++] = (byte)k;
                }
                for (int i = 0; i < cnt; i++)
                {
                    bool firstOfSum = true;
                    for (int j = 0; j < i; j++) if (sums[j] == sums[i]) firstOfSum = false;
                    if (!firstOfSum) continue;
                    int size = 0;
                    bool anyEnemy = false;
                    for (int j = 0; j < cnt; j++)
                    {
                        if (sums[j] != sums[i]) continue;
                        size++;
                        if (_owner[_cell[NbCell[nb0 + ks[j]]]] != mover) anyEnemy = true;
                    }
                    if (size < 2 || !anyEnemy) continue;
                    for (int j = 0; j < cnt; j++)
                    {
                        if (sums[j] == sums[i] && _owner[_cell[NbCell[nb0 + ks[j]]]] != mover)
                            AddFlip(ks[j], 2, reason, order, ref orderCount);
                    }
                }
            }

            if ((_rules & RuleSet.Same) != 0)
            {
                Span<byte> ks = stackalloc byte[4];
                int cnt = 0;
                bool anyEnemy = false;
                for (int k = 0; k < n; k++)
                {
                    int ni = _cell[NbCell[nb0 + k]];
                    if (ni < 0) continue;
                    if (Effective(inst, NbMySide[nb0 + k]) != Effective(ni, NbOppSide[nb0 + k])) continue;
                    ks[cnt++] = (byte)k;
                    if (_owner[ni] != mover) anyEnemy = true;
                }
                if (cnt >= 2 && anyEnemy)
                {
                    for (int j = 0; j < cnt; j++)
                    {
                        if (_owner[_cell[NbCell[nb0 + ks[j]]]] != mover)
                            AddFlip(ks[j], 3, reason, order, ref orderCount);
                    }
                }
            }
        }

        // Execute the flips in insertion order; same/plus flips start a one-level, base-only combo.
        // PY-QUIRK(D-013): combo flips never chain further.
        var cov = Coverage;
        for (int o = 0; o < orderCount; o++)
        {
            int k = order[o];
            int target = NbCell[nb0 + k];
            ushort bit = (ushort)(1 << target);
            if ((_flipMask & bit) != 0) continue;
            int ni = _cell[target];
            if (_owner[ni] == mover) continue;
            byte originalOwner = _owner[ni];
            _flipMask |= bit;
            if (originalOwner == OwnerId.None) _flipFromNoneMask |= bit;
            _owner[ni] = mover;
            _hash ^= ZobristKeys.Owner(ni, originalOwner) ^ ZobristKeys.Owner(ni, mover);
            if (cov is not null)
            {
                if (chainOnly) cov.ComboFlip++;
                else if (reason[k] == 1) cov.BaseFlip++;
                else if (reason[k] == 2) cov.PlusFlip++;
                else cov.SameFlip++;
            }
            if (reason[k] >= 2) ResolveFlips(target, ni, true);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddFlip(int k, byte why, Span<byte> reason, Span<byte> order, ref int orderCount)
    {
        if (reason[k] >= 2) return; // already same/plus: keep
        if (reason[k] == 0) order[orderCount++] = (byte)k;
        reason[k] = why;
    }

    /// <summary>Python <c>apply_type_modifier</c>: sets the modifier and refreshes the display edges.</summary>
    private void ApplyTypeModifier(int inst, int modifier)
    {
        _hash ^= ZobristKeys.Mod(inst, _mod[inst]) ^ ZobristKeys.Mod(inst, modifier);
        _mod[inst] = (sbyte)modifier;
        for (int side = 0; side < 4; side++) _display[inst * 4 + side] = (byte)Effective(inst, side);
    }

    /// <summary>
    /// Python <c>recalculate_type_modifiers</c>: per type present on the board, +count (同类强化 wins) or -count
    /// (同类弱化) is applied to every card of that type on the board and in both hands.
    /// </summary>
    public void RecalculateTypeModifiers()
    {
        bool ascension = (_rules & RuleSet.Ascension) != 0;
        if (!ascension && (_rules & RuleSet.Descension) == 0) return;

        Span<int> counts = stackalloc int[CardTypes.Count];
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = _cell[cell];
            if (inst >= 0 && _type[inst] != 0) counts[_type[inst]]++;
        }

        // Every card instance is on the board or in a hand, so one pass over the instance table visits exactly the cards
        // Python visits (board cards, then both hands, once per present type). Each card is touched at most once and its
        // new modifier depends only on its own type, so the order of the visits does not matter (the hash update is an XOR,
        // the display refresh is per card). A first typed placement therefore still re-applies EVERY present type, including
        // types whose count did not change (D-017 catch-up), and still refreshes their display edges (D-003).
        for (int inst = 0; inst < _instCount; inst++)
        {
            int t = _type[inst];
            if (t == 0) continue;
            int count = counts[t];
            if (count != 0) ApplyTypeModifier(inst, ascension ? count : -count);
        }
    }

    // ------------------------------------------------------------------ make / undo

    /// <summary>
    /// Python <c>make_move</c>. The move names a card instance (see <see cref="Move"/>); its current hand slot is
    /// resolved here. An invalid move throws <see cref="InvalidOperationException"/> (Python returned <c>None</c>): the
    /// instance is not in the side to move's hand, the cell is occupied or out of range, or the undo stack is full.
    /// The state is unchanged when it throws. The played card is removed from the hand by shifting later cards left.
    /// </summary>
    public void MakeMove(Move move)
    {
        int p = _current;
        int hc = _handCount[p];
        int inst = move.Inst;
        int cell = move.Cell;
        int slot = -1;
        for (int i = 0; i < hc; i++)
        {
            if (_hand[p * HandCapacity + i] == inst)
            {
                slot = i;
                break;
            }
        }
        if (slot < 0 || cell >= 9 || (_occupied >> cell & 1) != 0 || _undoCount >= MaxUndoDepth)
            ThrowInvalidMove(move);

        bool typeRules = (_rules & (RuleSet.Ascension | RuleSet.Descension)) != 0;
        ref UndoRecord rec = ref _undo[_undoCount];

        if (typeRules)
        {
            // D-002 (FIXED-BOTH): modifiers are saved per card instance, so undo restores each card's own value even
            // though D-001 reorders the hand. Every instance is on the board or in a hand, so this covers all cards.
            int sb = _undoCount * SnapLen;
            for (int i = 0; i < _instCount; i++) _snapshot[sb + i] = _mod[i];
        }

        rec.HashBefore = _hash;
        rec.PlacedOrigOwner = _owner[inst];
        _hash ^= ZobristKeys.Owner(inst, _owner[inst]) ^ ZobristKeys.Owner(inst, p) ^ ZobristKeys.Cell(cell, inst) ^ ZobristKeys.Side;
        _owner[inst] = (byte)p; // player 0 = red, 1 = blue
        _cell[cell] = (sbyte)inst;
        _occupied |= (ushort)(1 << cell);
        for (int i = slot; i < hc - 1; i++) _hand[p * HandCapacity + i] = _hand[p * HandCapacity + i + 1];
        _handCount[p] = (byte)(hc - 1);

        _flipMask = 0;
        _flipFromNoneMask = 0;
        ResolveFlips(cell, inst, false);

        if (typeRules && _type[inst] != 0)
        {
            RecalculateTypeModifiers();
            var cov = Coverage;
            if (cov is not null)
            {
                if ((_rules & RuleSet.Ascension) != 0) cov.AscensionRecalc++;
                else cov.DescensionRecalc++;
            }
        }

        rec.Cell = (byte)cell;
        rec.Player = (byte)p;
        rec.Inst = (byte)inst;
        rec.HasSnapshot = typeRules;
        rec.FlipMask = _flipMask;
        rec.FlipFromNoneMask = _flipFromNoneMask;
        _current = 1 - p;
        _undoCount++;
        if (Coverage is { } c2) c2.Moves++;
    }

    /// <summary>Python <c>undo_move()</c> without a record: reverts the last move. Returns false if none was made.</summary>
    public bool UndoMove()
    {
        if (_undoCount == 0) return false;
        ref UndoRecord rec = ref _undo[--_undoCount];
        int p = rec.Player;
        _current = p;

        // Restore flipped owners (originals: the other colour, or None).
        ushort fm = rec.FlipMask;
        while (fm != 0)
        {
            int c = BitOperations.TrailingZeroCount(fm);
            fm &= (ushort)(fm - 1);
            _owner[_cell[c]] = (rec.FlipFromNoneMask >> c & 1) != 0 ? OwnerId.None : (byte)(1 - p);
        }

        // Remove the placed card and put it back.
        _cell[rec.Cell] = -1;
        _occupied &= (ushort)~(1 << rec.Cell);
        int hc = _handCount[p];
        // PY-QUIRK(D-001): Python appends the card to the END of the hand, so make+undo reorders the hand.
        _hand[p * HandCapacity + hc] = rec.Inst;
        _handCount[p] = (byte)(hc + 1);
        _owner[rec.Inst] = rec.PlacedOrigOwner;

        if (rec.HasSnapshot)
        {
            int sb = _undoCount * SnapLen;
            var cov = Coverage;
            if (cov is not null)
            {
                cov.UndoWithSnapshot++;
                for (int i = 0; i < _instCount; i++)
                {
                    if (_mod[i] != _snapshot[sb + i])
                    {
                        cov.UndoModifierChanged++;
                        break;
                    }
                }
            }

            // D-002 (FIXED-BOTH): restore by card instance. Display edges are deliberately not restored (PY-QUIRK D-003).
            for (int i = 0; i < _instCount; i++) _mod[i] = _snapshot[sb + i];
        }

        _hash = rec.HashBefore; // every hashed field was just restored, so this is the pre-move hash
        if (Coverage is { } c2) c2.Undos++;
        return true;
    }

    private void ThrowInvalidMove(Move move) =>
        throw new InvalidOperationException(
            $"Invalid move {move}: player {_current} has {_handCount[_current]} cards, occupied mask 0x{_occupied:X}, undo depth {_undoCount}");
}
