# OPUS Phase 3 Review — TTC_Siren Python → C# solver migration

Reviewer: independent senior reviewer / architecture auditor (Opus), review session only (no implementation, no Phase 4 work).
Date: 2026-10-07.

---

## 1. Review scope

Phase 2 and Phase 3 were reviewed together, because search correctness depends on state-mutation correctness.

- **Phase 2**: `Card`/`CardDb`, `Move`, `RuleSet`/`RuleParser`, `GameState` (`MakeMove`/`UndoMove`/`ResolveFlips`/`RecalculateTypeModifiers`/`GenerateMoves`/`CountCards`), snapshot import/export, rule-trace parity tooling.
- **Phase 3**: `Evaluator` (`evaluate_state`, `evaluate_move`, placement-edge score, endgame exposure penalty), `HistoryTable`, `CompatTranspositionTable` + `PythonStateKey`, `CompatSearcher` (minimax/alpha-beta, iterative deepening, time/node limits, `SearchStats`), search parity tooling.
- **Reference**: `core/card.py`, `core/board.py`, `core/player.py`, `core/game_state.py` (including the approved D-002 diff), `ai/ai.py`.
- **Process documents**: `CLAUDE.md`, `CSHARP_MIGRATION.md`, `.claude/phase_goal.md`.

Out of scope: `ai_server.py` request parsing, robust endgame, unknown-card sampling (Phase 5/7/8).

## 2. Repository state reviewed

- Branch `csharp-solver-migration`, HEAD `eb60a02`. Everything from Phase 2 and Phase 3 is **uncommitted**: `csharp/`, `tools/`, `.claude/`, `CLAUDE.md`, `CSHARP_MIGRATION.md` and `tests/test_undo_type_modifiers.py` are untracked, and `core/game_state.py` is modified (the approved D-002 fix, +13/−22, checked against the diff).
- `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` gives 0 warnings and 0 errors.
- `dotnet test` gives **33/33 passed**, 0 skipped.
- `python -m pytest -q` gives **29 passed**: the 26 baseline tests plus 3 D-002 regression tests.
- Cached golden files in `tools/parity/golden/` are newer than the D-002 change to `core/game_state.py`, and `ai/ai.py` is unchanged since 2026-09-03.

### 2.1 Independent verification performed by this review

The review did not rely on "tests pass". It did the following (all scratch work stayed outside the repository):

| # | Check | Result |
|---|---|---|
| V1 | Regenerated **all five** golden sets from the Python reference with **new seeds** (rules 31415927, search 27182818), then ran the unchanged C# differential tests against them | 5/5 pass. Rules: 20,000 sequences / 182,850 steps, all 256 rule subsets, coverage counters similar to the documented ones. Eval: 100,000 positions, 400,000 + 1,588,782 bit-exact checks. Fixed depth: 900 positions + 22 bench items, 4,513 runs, 1,251,726 nodes. ID: 203 unlimited + 600 budget runs (587 interrupted), 200 direct-budget runs. TT key: 38,562 states, 16,807 ↔ 16,807 bijection |
| V2 | **Adversarial differential** from a new Python generator that targets cases the stock generator under-weights: (a) openings with typed decks under 同类强化/弱化 searched to depth 5 (node cap 8,000); (b) **the same card id in both hands** plus board cards that share ids with hand cards; (c) server-style raw `Card(...)` construction (display = raw base), the all-zero id 474, and predicted 1000+ ids. Compared fixed depth 1..5 and ID to depth 4 (score bits, move, nodes, cutoffs, TT hits, per-level data) | 90 positions, 507 runs, 2,296,000 nodes, **0 mismatches** |
| V3 | Drift probe: 2,000,000 random make/undo pairs per reference scenario (6 scenarios), then full-state comparison (ignoring only D-001 hand order and D-003 display edges) | No drift in any scenario |
| V4 | Exception probe: a search that runs into `TranspositionTableFullException` | **State left mutated** (`UndoDepth = 3`). A follow-up search on the same object silently searched the corrupted root. See H-1 |
| V5 | Production-equivalent call: ID with `max_depth = 10` and a 10 s wall clock (the `find_best_move_parallel` defaults used by `ai_server`) on `open_none` and `open_all_rules`, default TT | **Throws `TranspositionTableFullException` after 2.7 s / 5.2 s** (depth 8, 4.2M / 5.0M nodes), leaving `UndoDepth = 7 / 6`. See H-2 |
| V6 | ID with `maxDepth = 200`, unlimited time, on `end5_same_plus` | `IndexOutOfRangeException` in `SearchStats` (L-1) |
| V7 | Line-by-line semantic comparison of rule resolution, evaluator, ordering, history, TT key, minimax and ID against Python | No semantic divergence found (see §11) |

## 3. Gate result

# **PASS_TO_PHASE_4** (after remediation verification, 2026-10-07; see §15)

> History: the initial review gate was **BLOCK_PHASE_4**. All three blocking findings (H-1, H-2, H-3) were independently verified as **RESOLVED** in §15, and no regression was found. The MEDIUM, LOW and OPTIONAL findings remain open and nonblocking. The rest of this section is the original rationale, kept for the record.

Original rationale: there were 3 HIGH findings and no CRITICAL findings. Rules, evaluation and search parity with Python is **real and independently confirmed** (V1, V2). The blockers concern robustness and architecture that Phase 4 builds on directly: exception safety of the mutable state, the reference TT failing under the production configuration, and a move-identity type that the Phase 4 TT and move-ordering work would misuse. All three are small, local fixes. No redesign is needed.

## 4. Executive summary

- **State representation and make/undo are correct.** `GameState` holds everything that affects future play: board instances, three-valued owners, ordered hands, modifiers, display edges (D-003), `can_use`, rules and side to move. Undo restores owners (including `None` through a dedicated mask), hand membership, modifiers by identity (D-002 fix) and side to move. It reproduces D-001 (append to the end of the hand) and D-003 (display edges are not restored) on purpose. No aliasing, struct-copy or stale-cache defect was found. The only derived values (`_occupied` and the display edges) are maintained, or deliberately stale as in Python.
- **The evaluator is bit-exact.** Coefficients, accumulation order, `min(…, 1000.0)`, display-edge versus effective-edge reads, `star` lookups (None / 1000+ / 474 → 0), aggressive-mode phases and the use of a hand card's own `owner` attribute all match Python. The set-iteration order in `_calculate_placement_edge_score` is harmless because every term is a multiple of 0.5.
- **Search semantics match Python** node for node: a fixed AI perspective, maximizing toggled per ply (equivalent to side to move, because players strictly alternate), the time check before the terminal check, TT probe before move generation, the no-move leaf with no TT store, `>`/`<` strict best-move updates, `alpha = max`, `beta = min`, a cutoff at `beta <= alpha`, a history bonus of `min(old + 2^depth, 10^6)` only when a best move exists, an unconditional TT overwrite (D-006), and the ID discard rule `time_up && depth > 1`. Note that the algorithm is **fail-soft**, not fail-hard as documented (L-3).
- **The parity infrastructure is substantive.** It compares the full state after every make **and** undo, the ordered legal moves, counts and winner, flip sets, bit-exact scores, best move, all counters and the final history table. Golden data covers all 256 rule subsets, `None` owners, wrong-colour hand owners, stale display edges, predicted ids and node-budget "time-up" paths. Gaps are listed in §10. The most important ones: golden files are not tied to the reference source (M-1), and the post-search hand order and display edges are never compared with Python (M-2).
- **The hot path has no allocation.** There is no `GameState` cloning, LINQ, `List`, `Dictionary`, closures, boxing or strings in search; a test asserts 0 bytes. The remaining performance observations are optional (§9).

## 5. CRITICAL findings

None.

## 6. HIGH findings

### H-1 — Recursive search is not exception-safe: an exception leaves `GameState` mutated

| Field | Value |
|---|---|
| Finding ID | H-1 |
| Affected file | `csharp/src/TTC.Siren.Search/CompatSearcher.cs` |
| Affected method | `Node` (`MakeMove` at :217, `UndoMove` at :219), `RunFixedDepth`, `RunIterativeDeepening` |
| Observed behavior | `MakeMove → Node(...) → UndoMove` has no `try/finally` and no unwind at the entry point. Any exception thrown below a `MakeMove` (for example `TranspositionTableFullException` from `_tt.Store` at :252, `InvalidOperationException` from `MakeMove`, or `IndexOutOfRangeException`) escapes with up to 9 moves still applied. Probe V4: `UndoDepth = 3`, side to move, hands and occupancy all changed. A second search on the same `GameState` ran without any error on the corrupted root. Probe V5: a production-equivalent call left `UndoDepth = 7`. |
| Expected behavior | Python wraps every child in `try: ... finally: state.undo_move(move_record)` (`ai/ai.py:520-533`, `:571-584`), so the root state is restored whenever an exception propagates. CLAUDE.md §5 requires GenerateMoves → MakeMove → AlphaBeta → UndoMove pairing. |
| Why this matters | The `GameState` is a single shared mutable object (ADR A-3). After any failure, every later use of it (an ID retry, a fallback, Phase 5's robust endgame calling the searcher repeatedly, Phase 7's long-lived engine process) silently computes on a wrong position. Phase 4 adds more code under `MakeMove` (Zobrist updates, a new TT, ordering), which adds more ways to throw, and Fast mode would copy this pattern. |
| Evidence | Review probe V4 and V5 (§2.1). Code inspection of `CompatSearcher.Node` lines 210-249. |
| Recommended fix | Guard at the entry points, so the per-node hot path is untouched. In `RunFixedDepth` and `RunIterativeDeepening`, record `int rootUndo = _g.UndoDepth;` and wrap the search in `try { ... } finally { while (_g.UndoDepth > rootUndo) _g.UndoMove(); }`. LIFO unwinding through `UndoMove` performs exactly the same mutations as normal recursive unwinding, including D-001 appends in the same order. That keeps the post-exception state identical to "search returned early". The exception must still propagate (or be handled as H-2 specifies); it must not be swallowed silently. A per-node `try/finally` (Python-like) is also acceptable if the allocation and benchmark checks show no regression. |
| Required regression test | `Search_restores_state_when_an_exception_escapes`: build `open_none` and a deliberately tiny `CompatTranspositionTable` (or a test hook that throws at the Nth node) and run `RunFixedDepth(4)` and `RunIterativeDeepening(…)`. Assert that the exception surfaces, that `UndoDepth == 0`, and that `SearchGolden.NormalizedState(g)` equals the pre-search value. Then build `g2 = GameState.FromSnapshot(g.ToSnapshot())`, run `RunFixedDepth(3)` on both `g` and `g2` with fresh TTs, and assert that score bits, move, nodes and cutoffs are identical. This proves no hidden undo or scratch state survived. Also add one case for `RunIterativeDeepening`. |
| Human decision required | No |

### H-2 — The ReferenceCompat transposition table overflows under the production configuration

| Field | Value |
|---|---|
| Finding ID | H-2 |
| Affected file | `csharp/src/TTC.Siren.Search/CompatTranspositionTable.cs` |
| Affected method | `Store` (:86 throws `TranspositionTableFullException`), constructor default `1 << 21` (:41) |
| Observed behavior | Python's `TRANSPOSITION_TABLE` is an unbounded `dict`. The C# emulation is a fixed table with a 75% load limit (≈1.57M entries, 64 MB) that throws when full. The server-equivalent call `RunIterativeDeepening(10, false, WithTime(10))`, which mirrors `find_best_move_parallel(max_depth=10, max_time=10)` (§3.5), throws after **2.7 s** on `open_none` and **5.2 s** on `open_all_rules`, while searching depth 8 (Python completes depth 5 in the same 10 s). Combined with H-1, the state is also left corrupted. |
| Expected behavior | The ReferenceCompat searcher reproduces Python's control flow for every input Python accepts, including the standard server call (the same time budget lets C# reach a deeper level than Python, which is fine). Under the same wall-clock budget it must either complete (Python semantics: the dict never refuses an insert) or, if a hard memory cap is ever hit, fail cleanly with the state restored and the cause made explicit to the caller. |
| Why this matters | `CSHARP_MIGRATION.md` §11.4 makes ReferenceCompat the oracle for Phase 4 (Fast versus TT-off compat), and Phase 5 requires a "full Python-vs-C# benchmark" that includes the §3.5 server-equivalent calls. Today the oracle cannot run the most important real configuration for even 3 seconds. The limitation is documented ("大搜索…会超过默认 TT 容量") but its reach was underestimated ("Phase 3 的测试规模不会触发"). The production entry point triggers it immediately. |
| Evidence | Probe V5: `prod open_none: TranspositionTableFullException after 2.7s, levels=7, nodes=4,222,118, UndoDepth left=7`; `prod open_all_rules: … after 5.2s, levels=7, nodes=4,974,035, UndoDepth left=6`. |
| Recommended fix | Let the table **grow on demand**: when the load limit is reached in `Store`, allocate a table twice the size and reinsert the live entries of the current generation. Lookups compare the full 128-bit key, so reinsertion order does not change semantics. Steady-state allocation stays at 0 (CLAUDE.md §5 forbids steady-state allocation, not rare amortized growth outside the per-node loop). Keep a configurable hard cap. If the cap is reached, throw a dedicated exception (H-1 guarantees state restoration) whose message reports the cap and entry count. Expose `Capacity` and the number of grow events (`Grows`) for benchmarks. Keep the existing allocation test pre-sized so it still measures 0 bytes. |
| Required regression test | (1) `Compat_TT_grows_without_changing_results`: run a fixed-depth golden subset (for example all 22 bench items plus 50 golden positions) with an initial capacity of `1 << 6`, so many grow events happen. Assert bit-identical scores, moves, nodes, cutoffs, `tt_hits` and `tt_cutoffs` against the golden data, and assert `Grows > 0`. (2) `Server_equivalent_call_completes`: `RunIterativeDeepening(10, false, WithTime(10))` on `open_none`, `open_same_plus`, `mid3_ascension` and `open_all_rules` completes without an exception, `UndoDepth == 0`, the state is restored (normalized), and `HasMove`. Record levels, nodes, peak TT entries and memory in §10. This is a behavioural test because the wall clock is nondeterministic; it may be marked as a long-running test but must not be skipped. (3) The hard-cap path: with a tiny cap, the dedicated exception propagates and the state is restored (shared with the H-1 test). |
| Human decision required | Only if meeting test (2) needs a TT larger than **1 GB** (entries are 32 bytes, so a 10 s run at about 1.5M NPS may need about 2^24 entries, or 512 MB). In that case, stop and ask the user whether the compat oracle may use that memory or whether the server-equivalent configuration should be excluded from compat-mode requirements. |

### H-3 — `Move` identity is a hand slot, which every make/undo invalidates (D-001); Phase 4 would persist it

| Field | Value |
|---|---|
| Finding ID | H-3 |
| Affected file | `csharp/src/TTC.Siren.Core/Primitives.cs` (`Move`, :120), `csharp/src/TTC.Siren.Core/GameState.cs` (`GenerateMoves`, `MakeMove`, `HandSlotOf`), `csharp/src/TTC.Siren.Search/Evaluator.cs` (`EvaluateMove`, :209), `csharp/src/TTC.Siren.Search/CompatSearcher.cs` (:202-216 workaround) |
| Observed behavior | `Move = (HandSlot, Cell)`. Because undo appends the card to the end of the hand (D-001), a `Move` produced before any make/undo at the same node, or in another node, or from a transposition, refers to **a different card** afterwards. `CompatSearcher` works around this by snapshotting an `instances[]` array per node after ordering and rebuilding each `Move` with `HandSlotOf` before `MakeMove`. `Evaluator.EvaluateMove` also interprets the slot against the *current* hand. Python's moves hold `Card` objects (identity), and Python's TT stores `(card, pos)` (`ai/ai.py:545-549`). |
| Expected behavior | The canonical move identity is stable for the lifetime of the `GameState`: (card instance, cell), exactly like Python's `(card object, (row, col))`. |
| Why this matters | This bug class already caused a real Phase 3 parity failure (open_none depth 2: Python −4.56, C# −1.375; design decision 1 in §7). Phase 4 introduces the TT best move, move ordering and the history heuristic, which store moves and replay them in **other nodes and transposed positions whose hand order differs**. Unlike Phase 3, the Fast mode has **no node-level Python oracle**. Its acceptance compares only the root score and move against TT-off compat, so a TT move replayed as the wrong card would silently degrade ordering and could corrupt reported moves without failing any planned test. Fixing the type now makes the error impossible by construction, rather than relying on every future call site remembering the workaround. |
| Evidence | Code: `CompatSearcher.cs:202-216` (explicit comment), `Evaluator.cs:209`, the `D001_undo_appends_card_to_end_of_hand` test, and `CSHARP_MIGRATION.md` Phase 3 design decision 1 (the original bug and its fix). |
| Recommended fix | A narrow API change with no rule or search-semantics change. Make the public move identity `(Inst, Cell)`: `GenerateMoves` emits instance-based moves (same order: hand order × row-major empties, with the `can_use` filter). `MakeMove` resolves the slot internally (`HandSlotOf`) and throws if the instance is not in the side-to-move's hand or the cell is occupied. `EvaluateMove` reads the instance directly. Remove the per-node `instances[]` workaround. If a slot-based helper is still useful for tests, keep it explicitly named (for example `MoveFromSlot(slot, cell)`). The golden move codes (`slot*9+cell`) are translated to instances **at the point of use** (the slot is resolved against the hand at that exact step), so every golden comparison stays unchanged. |
| Required regression test | (1) `Move_identity_survives_hand_reordering`: generate the moves at a node, make and undo the second move (which reorders the hand), then make the *first* generated move and assert that the played instance is the one originally generated. (2) `MakeMove_rejects_instance_not_in_hand` (an opponent's card, a card already on the board). (3) All existing differential tests (rules trace, eval, fixed depth, ID, TT) pass unchanged, and the allocation tests stay at 0 bytes. (4) Update `Move_order_is_recomputed_by_card_identity_after_hand_reordering` so it keeps asserting the open_none depth-2 golden values; it must not be deleted. |
| Human decision required | No |

## 7. MEDIUM findings

| ID | Location | Finding | Recommendation |
|---|---|---|---|
| M-1 | `tests/TTC.Siren.Tests/GoldenFiles.cs:70`, both generators | **Golden data is not bound to the reference source.** Golden files are regenerated only when missing, and validity rests on a manually bumped `GENERATOR_VERSION`. If `core/*.py` or `ai/ai.py` changes again (for example another approved D-fix), the C# tests keep passing against stale Python behaviour. The header records the Python version, but the tests do not check it. | Write a SHA-256 of the reference sources (`core/*.py`, `ai/ai.py`, `csv_compat.py`, the generator script, the CSV) into each header. Tests recompute it and regenerate, or fail, on mismatch. Verify `python` major/minor. |
| M-2 | `SearchGolden.NormalizedState`, `py_search_trace.py` | **The post-search state is not compared with Python.** After each search, C# checks only its own normalized state, which deliberately ignores hand order (D-001) and display edges (D-003). Python's post-search hand order and display edges are never exported. These are real state that changes later move ordering, so later searches on the same object (ID levels, Phase 5 robust endgame loops, a long-lived engine) depend on them. Today ID per-level parity covers this only indirectly. | Export `R.export_state(state)` after every `run_fixed_depth` and `run_id` in the generator. In C#, compare the full canonical string (including hand order and display edges) after each search. |
| M-3 | `py_search_trace.py` (`make_position`, `search_depths`) | **Search-golden diversity is skewed toward late positions**: 85% of positions have occupancy ≥ 4, a 5,000-node cap stops deepening, and ID runs use only positions with occupancy ≥ 4. Opening and middlegame searches under 同类 rules, and cross-hand duplicate ids, are covered only by the 5 bench positions and by chance. This review's adversarial set (V2) passed with 0 mismatches, but it is not part of the permanent suite. | Add a permanent golden kind or section with (a) typed-deck openings under 同类强化/弱化 to depth ≥ 4, (b) the same card id in both hands and on the board, and (c) server-style raw `Card(...)` construction including id 474/475 and 1000+ ids. (V2's generator can be adapted.) |

## 8. LOW findings

| ID | Location | Finding | Recommendation |
|---|---|---|---|
| L-1 | `CompatSearcher.cs:9`, `:147` | `SearchStats.MaxLevels = 128`. `RunIterativeDeepening(maxDepth > 128)` with unlimited time throws `IndexOutOfRangeException` (probe V6). Python accepts any `max_depth` (`find_best_move_parallel` clamps it to 100). | Validate `maxDepth` (clamp to 100 like the Python entry point, or reject > `MaxLevels` with `ArgumentOutOfRangeException`) and document it. |
| L-2 | `SearchLimits` | `default(SearchLimits)` means "time already up" (`NodeBudget = 0`, `TimeLimitSeconds = 0`). A NaN or negative time limit becomes an undefined deadline. | Validate in the constructor, and consider making `default` mean unlimited or rejecting it. |
| L-3 | `CompatSearcher.cs:75`, `CSHARP_MIGRATION.md` §1.7 | The search is documented as "fail-hard". Python and C# actually return the unclamped best child score, so they are **fail-soft**. This matters for Phase 4 bound classification (an upper bound when `score <= alpha_orig`, a lower bound when `score >= beta`). | Correct the comments and the documentation. |
| L-4 | `CompatTranspositionTable.cs:100-110` (`PythonStateKey`) | The 128-bit digest is a custom multiply-xor mixer, not a vetted hash. The "2^-128" claim is optimistic: each lane can collapse when `a ^ w == 0`, which is negligible in practice, and a collision would silently produce a wrong TT hit. The doc comment cites `PythonStateKeyTests`, which does not exist; the real test is `Transposition_key_partition_equals_pythons_get_state_hash`. The pre-hash encoding itself is injective (checked). | Fix the comment and soften the claim. Optionally use XxHash128 (`System.IO.Hashing`) on the stackalloc buffer if it is allocation-free. The compat TT is a test oracle, so this is not urgent. |
| L-5 | `AllocationTests` | The 1M make/undo allocation test checks only `UndoDepth == 0`, not that the state is restored. Probe V3 found no drift, but nothing asserts it. | Add a normalized-state equality assertion after the playouts. |
| L-6 | Docs | Stale or inconsistent statements: `CSHARP_MIGRATION.md` §0 line 24 says no tracked file was modified, which the D-002 fix contradicts; §5.2 still describes the mod snapshot as using "Python key semantics", which changed with D-002; `.claude/phase_goal.md:26` and `CLAUDE.md` §7 still cite "26 passed" (now 29; §7 calls 26 the baseline, but phase_goal's wording "stays at 26" is wrong); `CLAUDE.md` has no "Human Decision Protocol" section, although the review brief refers to one. | Update the wording only. Add a short human-decision protocol to `CLAUDE.md` if the user wants one. |
| L-7 | `GameState.FromSnapshot` (D-004 / Q-2) | C# throws on duplicate card ids within a hand (including two `None` ids), where Python silently corrupts state. This is documented and correct for parity, but a Phase 7 request with duplicate predicted or unknown ids would fail where Python "works". | Q-2 must be decided before Phase 7, as already scheduled. Phase 7 needs an input-normalisation step. |
| L-8 | Git | All Phase 2/3 work, including the approved Python fix, is uncommitted. One accidental `git clean` or `checkout` would lose it. | User decision: commit locally, split into logical commits as `CSHARP_MIGRATION.md` §11.2 proposes. |
| L-9 | `Snapshot.cs:63` | `CardSnapshot.FromJson` maps every owner string other than `"r"` to Blue, so a malformed golden owner would be masked (test infrastructure only). | Accept only `"r"`, `"b"` or `null`; throw otherwise. |

## 9. OPTIONAL OPTIMIZATION findings

**Do not implement any of these without the CLAUDE.md §6 baseline/benchmark process.** They are listed only so Phase 6 has a starting list. None of them blocks anything.

| ID | Observation |
|---|---|
| O-1 | `Node` and `PythonStateKey.Compute` `stackalloc` about 90 + 360 + 45 + 256 bytes per node, and the JIT zero-initialises them because there is no `[SkipLocalsInit]`. That is a memset per node. |
| O-2 | In timed mode, `IsTimeUp()` reads `Stopwatch.GetTimestamp()` at node entry **and** before every child. Checking every N nodes would be cheaper, but the node-budget (test) path must stay exactly per-node. |
| O-3 | Under 同类 rules, every `MakeMove` snapshots all 19 instance modifiers and recalculates the whole board (make/undo throughput is about 3M versus 8–17M pairs/s). A delta undo log would be smaller. |
| O-4 | `EvaluateState` recomputes `CountCards` and every board card's `PlacementEdgeScore` at each leaf. Incremental maintenance is possible. |
| O-5 | `PythonStateKey` builds a buffer of about 220 bytes and hashes it at every interior node. This is inherent to compat mode; Fast mode's incremental Zobrist (Phase 4) replaces it. |
| O-6 | Ordering calls `EvaluateMove` for every move, including at depth-1 nodes. Compat mode requires this, but Fast mode may not, since it changes only node counts and not values. This must be measured and decided under Q-5. |

## 10. Python/C# semantic differences

### 10.1 Verified equal (no difference)

Directional order (up, down, left, right); base, Plus and Same insertion order and reason precedence (same/plus never overwritten; base can be upgraded); exactly one level of base-only Combo started by same/plus flips (D-013); Reverse and Fallen Ace, including the reverse inversion and 1/10 ties; Plus and Same ignoring Reverse/Ace and counting own neighbours; flips of `owner=None` cards and their undo restoring `None`; 同类 recalculation triggered only by typed played cards, Ascension taking precedence over Descension, only types present on the board touched, hand cards affected; D-002 restore by identity (FIXED-BOTH, Python diff inspected); D-001 append on undo; D-003 display-only updates; Order and Chaos `can_use` filtering; `count_cards` counting board and hand cards by `card.owner`; the winner; every evaluator term and its order; the stable descending sort; history keyed by card id (shared across both hands and across `None`); the TT key information set (board: owner, id/None, base edges, modifier; hand id strings concatenated with no separator (D-005); side to move; and **not** `can_use`, type, hand modifiers or display edges); the TT write on every exit (D-006); terminal evaluation by heuristic (D-007); ID never stopping early (D-014).

### 10.2 Intentional and documented differences (acceptable)

| Area | Python | C# |
|---|---|---|
| D-004 duplicate in-hand ids | state corruption | `InvalidGameStateException` at load (Q-2 open until Phase 7) |
| Invalid move | `make_move` returns `None`, and minimax `continue`s | `InvalidOperationException` (unreachable from search) |
| Capacity | unbounded | hand ≤ 5, instances ≤ 19, undo ≤ 16 (rejected, never truncated) |
| TT key | exact string | 128-bit digest (probabilistic, L-4) |
| TT capacity | unbounded dict | fixed and throwing; **H-2** |
| Outputs | `path`/PV, `branching_factor`, progress callback, verbose output | not produced; `ai_server` uses only `best_move` |
| Global state (D-008) | module globals | per-searcher context |
| `used_cards` | appended, never undone | not modelled (search-irrelevant) |

### 10.3 Unintentional difference

| Area | Python | C# |
|---|---|---|
| Exception during search | `try/finally` restores the state | state left mutated: **H-1** |

## 11. Detailed audit notes (by requested area)

**State representation.** Complete for gameplay: the board (`_cell`, `_occupied`), card identity (instance index plus `_id`), the owner (three-valued), ordered hands, `can_use`, the rule set, side to move, `type_modifier` and display edges. There is no separate "type count" cache; counts are recomputed, so they cannot go stale. Terminal status is derived from `_occupied`. Score-relevant state is the owners of board **and hand** cards. Hand-card owners are constant per instance except while the card is on the board.

**Make/undo transaction.** Every mutation is represented in the undo record: the placed instance and its original owner; the cell; the side to move; the flip set together with the "was None" mask (flipped originals can only be the other colour or `None`, because the mover's own cards are never flipped, combo included); and the modifier snapshot over all instances (every instance is on the board or in a hand, so this equals Python's snapshot). The hand removal (shift left) and undo append reproduce Python's list behaviour. Scratch flip masks are reset per move. The undo stack is a value-type array accessed by `ref`, so there are no struct-copy surprises. `HistoryTable` and the `Evaluator` star table capture only instance-invariant data. No aliasing was found. Repeated make/undo shows no drift (V3 and the golden data).

**Move generation.** Hand order × row-major empties, with the Order/Chaos `can_use` filter; occupied cells and cards not in hand are excluded; at most 45 moves, guaranteed by the capacity limits. Duplicate ids across hands or between board and hand are handled correctly (V2), and duplicates within a hand are rejected (D-004). Move identity is unstable: H-3.

**Evaluator.** Bit-exact (V1: 2M checks; V2). Terminal positions use the heuristic `evaluate_state`, as in Python (D-007). **For exact endgame search (Phase 5), neither `EvaluateState` nor `EvaluateMove` may be used as a value.** The terminal value there must be the AI-perspective card-count difference (`_solve_endgame_exact`). `EvaluateMove` reads stale display edges (D-003), so it is path-dependent and must only ever be used for ordering.

**Search semantics.** The root is maximizing with `aiPlayer = side to move` and a fixed-perspective leaf. Maximizing alternates per ply, which is consistent because every move switches the side. Depth decrements by 1 per ply. Termination conditions are checked in this order: time, `depth == 0` or a full board, TT, no moves. The window update and cutoff conditions are correct (`alpha = max`, `beta = min`, `beta <= alpha`). The best move propagates from strict improvements. Bound handling is fail-soft (L-3). The no-move and root-without-move cases return the leaf and `BestInstance = -1`, matching Python's `None`. Time-up inside a node breaks the loop, stores the partial value (D-006) and returns it. None of the classic errors was found.

**Iterative deepening.** The completed depth is recorded only after the `time_up && depth > 1` check, so an interrupted level can never overwrite an accepted result for depth ≥ 2. The time-up predicate is monotonic, so an interrupted search can never look complete. Depth-1 partial acceptance is a Python quirk and is reproduced. Partial TT entries written during an aborted level have no later reader, because the whole search stops. Statistics stay consistent (`Levels == DepthCompleted`). Timeout leaves no mutated state, because every path unwinds normally. **Exceptions do leave mutated state (H-1).** Clock-check frequency is per node and per child, which is correct and only slightly costly (O-2).

**Exception and early return.** All early returns (time, terminal, TT cutoff, no moves) happen **before** any `MakeMove` in that frame. `break` out of the loop happens after `UndoMove`. The only unsafe path is an exception: H-1.

**Parity infrastructure.** Both implementations receive identical inputs: the golden state includes the ordered hands, owners (including `None` and wrong colours), modifiers, display edges, `can_use`, rules, side to move, the history table (eval golden), depth, node budgets and an aggressive flag. Search golden data is built from the same `R.export_state` encoding that the C# loader parses, so there is no conversion step that could hide a difference. The tests compare more than happy paths: the full state after every make **and** undo, the legal-move list and its order, counts and winner, flip sets, scores (bit-exact), best move, nodes, cutoffs, TT hits and cutoffs, move evaluations, per-level ID data, and the final history. Gaps are M-1, M-2 and M-3, plus L-5. The claim of "thousands of cases" is accurate and meaningfully diverse (random rule subsets with every subset represented, random owners, stale edges, predicted ids, budgets). Mutation spot-checks reported by the implementer are plausible, and this review's fresh-seed and adversarial runs confirm the suite is not tuned to one cached data set.

**Allocations and data layout.** None of the anti-patterns listed in the brief appears in the hot path. `SearchResult` is a `readonly struct`; `Coverage` is a nullable reference checked once; `HistoryTable` and `Evaluator` allocate only at construction. Interface dispatch and LINQ appear in tests only.

## 12. Phase 4 readiness (design constraints, not findings)

After H-1 to H-3 are fixed, the state/search architecture is a sound base for Zobrist, the TT, ordering and history. Constraints for the Phase 4 design:

1. **Zobrist contents.** Within one search from a fixed root, the position is fully determined by the cell → instance map, the owners of board instances, the side to move and (implicitly) hand membership. Type modifiers are a function of the root and the placed set: absent types keep their root values, and present types equal ±count once any typed card has been placed. Hashing board-cell modifiers anyway is cheap insurance. **Display edges (D-003) are path-dependent and must not be hashed.** They affect ordering only, not values.
2. TT scores are from the **fixed AI perspective**. Clear the TT per search, or include the AI player in the key.
3. Never store TT entries from an aborted (timed-out) node in Fast mode. Classify bounds against the original window (fail-soft returns, L-3).
4. The comparison oracle is **compat with the TT turned off**, because D-006 means that compat with the TT on may legitimately differ from exact minimax. That switch does not exist yet (`CSHARP_MIGRATION.md` §11.4).
5. Moves stored in the TT, killer or history tables must use instance identity (H-3).
6. **Human decision still pending: Q-5** (may Fast mode choose a different move among equally scored moves?). Options: (a) *No* (the current default), so Fast mode must reproduce Python's tie-breaking, which practically forbids TT-move-first and history reordering from changing the chosen root move. (b) *Yes, but report every case*: the root score must be bit-identical, and the move may differ only among moves with exactly equal scores, each listed. (c) *Yes, unrestricted among ties.* **Reviewer recommendation: (b).** It keeps the result value exact and verifiable while allowing the main Phase 4 benefits. Consequence: the root move can differ from Python's in tie cases, and Phase 7 end-to-end parity must then compare scores and tie sets instead of exact moves.

## 13. Test coverage gaps (summary)

1. No exception-safety test (H-1).
2. No test with TT growth or the server-equivalent configuration (H-2).
3. No test of move identity across reordering at the API level (H-3).
4. Golden data is not bound to the reference source (M-1).
5. The post-search hand order and display edges are not compared with Python (M-2).
6. Opening, typed-deck, duplicate-id and server-style coverage is not permanent (M-3).
7. The long-sequence drift test does not assert the state (L-5).
8. There is no input-validation test for `maxDepth > 128` or for invalid `SearchLimits` (L-1, L-2).
9. The wall-clock path is behaviour-tested only. This is acceptable and cannot be made deterministic.

## 14. Recommended next action

1. Run the Sonnet remediation goal `.claude/sonnet_review_fix_goal.md` (H-1, H-2, H-3 only, with regression tests).
2. Optionally, in the same or a following session, address M-1 to M-3 (parity hardening). These are recommended before Phase 4 begins but do not block it.
3. Re-review only the remediation diff, then change the gate.
4. Before Phase 4 starts, the user decides **Q-5**.
5. The user decides whether to commit the Phase 2/3 work locally (L-8).

---

## 15. Remediation Verification (2026-10-07)

Scope: verify independently that the blocking findings H-1, H-2 and H-3 are fixed, using the remediation that `.claude/sonnet_review_fix_goal.md` requested, and check for regressions. Sonnet's claims in `CSHARP_MIGRATION.md` ("Review remediation (H-1/H-2/H-3)") were treated as unverified until reproduced.

### 15.1 What was inspected

- Code: `CompatSearcher.cs` (entry points, `UnwindTo`, `Node`), `CompatTranspositionTable.cs` (`Store`, `Grow`, the exception, the constructor), `Primitives.cs` (`Move`), `GameState.cs` (`GenerateMoves`, `MakeMove`, `MoveFromSlot`, `HandSlotOf`), `Evaluator.EvaluateMove`.
- Tests: the new `SearchExceptionSafetyTests`, `CompatTtGrowthTests` (4) and `MoveIdentityTests` (5). Every modified test file was diffed against the pre-remediation copy kept by the first review. The only changes were moves now built with `g.MoveFromSlot(slot, cell)` and slots read back with `g.HandSlotOf(...)`. **No asserted value was changed or weakened**, and `SearchBehaviourTests.Move_order_is_recomputed_…` is byte-identical.
- `git diff --stat` shows only the approved D-002 change to `core/game_state.py` (+13/−22, unchanged). **No Python file was touched by the remediation.**

### 15.2 Commands and evidence (run by the reviewer)

| # | Check | Result |
|---|---|---|
| R1 | `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| R2 | `dotnet test` (the full suite, detailed logger) | **43/43 passed**, 0 skipped. Every golden counter is identical to the pre-remediation run: fixed depth 4,507 runs / 1,152,695 nodes; ID 203 + 600 (586 interrupted) + 200 direct; eval 100,000 positions / 400,000 + 1,592,088 checks; rules 20,000 sequences / 182,910 steps; TT 16,733 ↔ 16,733. The search allocation test ran 371,212 nodes with **0 bytes** allocated |
| R3 | `python -m pytest -q` | 29 passed |
| R4 | The new code against the **fresh-seed golden data** from the first review (rules 31415927, search 27182818). Differential and growth tests were copied to a scratch project with only the seed and path changed | 6/6 passed. The numbers are identical to the first review's fresh-seed run (for example 4,513 runs / 1,251,726 nodes; TT 16,807 ↔ 16,807). The growth test on fresh data had 75 runs grow, 197 grows in total |
| R5 | The adversarial set from the first review (V2: typed openings under 同类, the same id in both hands, server-style raw cards, ids 474 and 1000+) | 90 positions, 507 runs, 2,296,000 nodes, **0 mismatches** |
| R6 | H-1 probe: the caller makes 2 moves itself, then a search hits the TT hard cap | The exception propagates. `UndoDepth == 2` (unwinding stops at the caller's depth and does not undo the caller's moves). The normalized state is restored, and the caller can still undo its own 2 moves |
| R7 | H-1/H-2 probe: a hard cap is reached **inside an ID level > 1** (caps 256, 1024 and 4096 on `open_same_plus`, aggressive mode) | Throws after 2, 3 and 3 completed levels. `UndoDepth == 0` and the state is restored in all three cases |
| R8 | H-2 server-equivalent call (`RunIterativeDeepening(10, false, WithTime(10))`, default table) | All four scenarios complete. `open_none`: 9 levels / 15.0M nodes / 6.84M entries / 3 grows / 512 MiB. `open_same_plus`: 9 / 15.96M / 7.91M / 3 / 512 MiB. `mid3_ascension`: 10 / 30,500 / 0 grows. `open_all_rules`: 7 / 10.72M / 3.83M / 2 / 256 MiB. Before remediation these threw after 2.7 s and 5.2 s. The peak of 512 MiB is under the 1 GiB stop line, so no human decision was needed |
| R9 | H-3 fuzz: in 6 scenarios × 20,000 trials, take a random prefix, generate moves, perform random sibling make/undo sequences (which reorder the hand through D-001), then play a **stale** pre-generated move | 115,955 checks, **0 wrong**: the generated instance always lands on the cell and leaves the mover's hand |
| R10 | Drift: 2,000,000 random make/undo pairs per scenario, then a full normalized-state comparison | All 6 restored |

### 15.3 Per-finding verdicts

| Finding | Verdict | Basis |
|---|---|---|
| **H-1** Exception safety | **RESOLVED** | `RunFixedDepth` and `RunIterativeDeepening` record `rootUndo` and call `UnwindTo(rootUndo)` in `finally`. LIFO `UndoMove` performs exactly the mutations a normal return would, including D-001 append order. The exception propagates and is not swallowed. The per-node hot path is unchanged. `MakeMove` validates before mutating, so a rejected move can never leave a half-applied state. Verified by the new test, R6 and R7. These are the only public recursive entry points. |
| **H-2** TT overflow | **RESOLVED** | Grow-on-demand doubling inside `Store` only, when inserting a *new* key. That key is provably absent: probing reached an empty slot and the table never deletes entries. Only current-generation entries are reinserted. The 75% load limit keeps every probe loop terminating. The 1 GiB hard cap throws a dedicated exception, after which H-1 restores the state. The capacity diagnostics are present. Results are identical with an initial capacity of 64 on both the original and the fresh-seed golden data (R2, R4), and with one shared table reused across generations while it grows. The server-equivalent call completes (R8), and steady-state allocation is still 0 bytes. |
| **H-3** Move identity | **RESOLVED** | `Move = (Inst, Cell)`. `GenerateMoves` keeps exactly the previous order (hand order × row-major empties, `can_use` filter), as shown by the unchanged golden move-list comparisons and a dedicated test. `MakeMove` resolves the slot internally and rejects an instance not in the mover's hand, an occupied cell, a cell out of range or a full undo stack without mutating anything. `EvaluateMove` reads `move.Inst`. The per-node `instances[]` workaround is gone. The only remaining slot-based APIs are `MoveFromSlot` and `HandSlotOf` (used by tests and diagnostics) and the Python TT key builder, which legitimately iterates hand order. Verified by the new tests and R9. Reported benchmark impact: `Bench search` +0.6% and `Bench makeundo` +0.1%, both within noise. The reviewer did not re-benchmark, but node counts are unchanged. |

### 15.4 Regression check

**No regression was introduced.** All parity counters are unchanged on both golden data sets, the allocation tests show 0 bytes, Python is untouched, and rules, evaluation, ordering, TT and ID semantics are unchanged (the diff review agrees with R2, R4 and R5).

New nonblocking observations (LOW, no action required before Phase 4):

| ID | Observation |
|---|---|
| L-10 | `MoveIdentityTests.MakeMove_rejects_a_full_undo_stack_without_changing_state` actually exercises the *occupied cell* rejection. The undo-stack-full branch (16 > 9 plies, so unreachable from legal play) has no direct test. Rename the test or add a direct test when convenient. |
| L-11 | During growth, the old and new arrays coexist, so a transient peak of about 1.5× the new table occurs. The server-equivalent compat run uses up to 512 MiB, plus about 256 MiB during the last grow. This is acceptable for a reference oracle and is documented. Phase 4/5 benchmark harnesses running compat side by side should budget for it. |
| L-12 | `Server_equivalent_iterative_deepening_completes` is wall-clock dependent (about 30 s; asserts ≥ 3 levels and < 20 s). It is robust on this machine but could be flaky on a heavily loaded CI agent. Keep it, but watch it. |

The open items from the original review are unchanged and **remain nonblocking**: M-1 to M-3, L-1 to L-9 (L-1 re-confirmed: `maxDepth = 200` still throws `IndexOutOfRangeException`, now with the state restored), and O-1 to O-6. **M-1 to M-3 are still recommended before Phase 4's parity work relies on new golden data.**

### 15.5 Gate after remediation

**PASS_TO_PHASE_4.** Phase 4 is authorized from a technical review perspective. The user still has to authorize Phase 4 explicitly (CLAUDE.md: one phase per authorization) and should decide Q-5 (§15.6). If Q-5 is not answered, §9 applies the default ("not allowed unless each case is reported").

### 15.6 Q-5 advice (refines §12.6)

**Question:** in Fast mode, may the chosen root move differ from Python's when several moves have exactly the same score?

**Facts that constrain the answer:**
1. Fast mode is already **not** compared with Python's TT-on behaviour. D-006 means Python's own TT can produce non-minimax root scores, so ADR A-2 and the Phase 4 acceptance compare Fast with **ReferenceCompat with the TT turned off**. "Identical to Python" has already been narrowed to "identical to Python's minimax value".
2. Exact tie-move parity is **practically unachievable** for any searcher that traverses the tree differently. Root ordering depends on the history table, which TT cutoffs change, and on D-003 display edges, which depend on the *entire* search path, including undone siblings. Requiring exact moves in ties (option a) would force Fast mode to replicate compat's traversal node for node, which defeats Phase 4.
3. With strict `>` best-move updates and fail-soft alpha-beta, a later move with an equal value receives a fail-low bound and is never selected. So the root move is "the first move in the current root ordering that attains the exact maximum". Ties are therefore broken by ordering, which carries no game meaning: by the engine's own evaluation, tied moves are equally good.
4. In production, ID is time-limited, so the reached depth (and therefore the move) is already nondeterministic between runs and machines. Exact move parity is not observable there anyway.

**Recommendation: option (b), "Yes, among exactly tied moves only, verified and reported"**, with this precise acceptance rule for Phase 4:
- At fixed depth with no time limit, Fast's root score must be **bit-identical** to ReferenceCompat with the TT turned off.
- Fast's root move must belong to the **exact tie set**: every root move whose full-window compat-TT-off score is bit-equal to the root score. The test computes this set explicitly; it is cheap at the golden depths.
- Each case where Fast's move ≠ compat's move is counted and listed, and the frequency is recorded in `CSHARP_MIGRATION.md`. Any move outside the tie set is a bug (BLOCKED), not a tie.
- Fast mode must be **deterministic**: the same input and the same depth give the same move.
- Phase 7 end-to-end comparisons with the Python server then compare scores and tie-set membership, not literal moves.

**Consequences:** (b) keeps the value-level result exactly verifiable and allows TT-move-first ordering, the history heuristic and other ordering improvements, which are where most of Phase 4's speed comes from. The cost is that, on tied positions, users may see a different but equally rated move than the Python build would show. Option (c) (unrestricted, unreported) loses the audit trail. Option (a) (the current default) blocks the main Phase 4 optimisations and in practice would make Fast mode a copy of compat.
