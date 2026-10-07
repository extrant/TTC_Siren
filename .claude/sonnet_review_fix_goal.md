# TTC_Siren — Review Remediation Goal (Phase 2/3 review → fix H-1, H-2, H-3 only)

> **STATUS: COMPLETED AND VERIFIED (2026-10-07).** The re-review found H-1, H-2 and H-3 all RESOLVED, with no regressions (`docs/reviews/OPUS_PHASE3_REVIEW.md` §15). The gate is now `PASS_TO_PHASE_4`. **Do not run this goal again.** There is no remaining blocking work. It is kept for the record only.

Run with: `/goal @.claude/sonnet_review_fix_goal.md`

This goal authorizes **only** the remediation of the three blocking findings of the independent Phase 2/3 review.
It does **not** authorize Phase 4, or any part of it (no Zobrist, no Fast mode, no bound-aware TT, no TT-off switch, no new move-ordering heuristics).

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely. Its rules are permanent and win over anything here if they seem to conflict.
2. Read `CSHARP_MIGRATION.md` completely, especially §0 (review gate), §1.4–§1.7, §4 (D-001, D-003, D-006), §7 Phase 3 and §11.
3. Read `docs/reviews/OPUS_PHASE3_REVIEW.md` completely. Sections §6 (H-1, H-2, H-3) are your specification.
4. Run the baseline and record the results in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (expect 0 warnings, 0 errors)
   - `dotnet test csharp/TTC.Siren.sln -c Release` (expect 33/33)
   - `python -m pytest -q` (expect 29 passed)
   - `dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search` and `... -- makeundo` (keep the numbers for the comparison in §5)
5. In one short message, state the three findings you will fix and their acceptance criteria. Then begin.

## 2. Scope discipline

- Fix **only** H-1, H-2 and H-3. Do not fix MEDIUM, LOW or OPTIONAL findings (M-*, L-*, O-*), even if they look trivial. The one exception is a change those fixes strictly require (for example, updating a doc comment you touch).
- **Do not change any Python file** (`core/`, `ai/`, `ai_server.py`, `tests/`). These fixes do not change game or search semantics. The Python reference stays as it is.
- **Do not change search or rule semantics.** Every existing golden comparison (scores bit-for-bit, best moves, nodes, cutoffs, TT hits/cutoffs, move evaluations, history tables, per-level ID data, full state after every make/undo) must stay identical. If any golden comparison changes, you have changed semantics: stop and diagnose.
- Do not skip, delete, `[Skip]`, weaken or comment out any test. An existing test may be *adapted* to a new API only if its asserted values stay the same.
- No unbenchmarked performance work, no `unsafe`, no SIMD, no parallelism.
- Git: do not commit unless the user asks (CLAUDE.md §8). Never push. Never modify `main`. No destructive git operations.

## 3. Findings and acceptance criteria

Fix them one at a time, as separate logical changes, in this order: **H-1 → H-2 → H-3**. After each fix, build and run the full C# test suite before starting the next.

### H-1 — Exception safety of the recursive search

Problem: `CompatSearcher.Node` performs `MakeMove → Node → UndoMove` without protection. Any exception below a `MakeMove` (for example `TranspositionTableFullException`) leaves the shared `GameState` mutated (the review measured `UndoDepth` 3 and 7). Python uses `try/finally` (`ai/ai.py:520-533, 571-584`).

Required change: in `RunFixedDepth` and `RunIterativeDeepening` (and any other public entry point that recurses), record `int rootUndo = _g.UndoDepth` and guarantee `while (_g.UndoDepth > rootUndo) _g.UndoMove();` in a `finally`. The exception must still propagate (unless H-2 handles it explicitly), and it must not be swallowed. A per-node `try/finally` is acceptable instead, but only if the allocation tests stay at 0 bytes and the `Bench search` median does not regress by more than 5%. The entry-point guard is preferred.

Acceptance criteria:
- [ ] New test `Search_restores_state_when_an_exception_escapes` covers both `RunFixedDepth` and `RunIterativeDeepening`. It forces an exception deep in the search (a tiny `CompatTranspositionTable` with the H-2 hard cap set very low, or an internal test hook that throws at node N), then asserts that the exception surfaces, that `g.UndoDepth == 0`, and that `SearchGolden.NormalizedState(g)` equals the value before the search.
- [ ] In the same test, after the exception, `g2 = GameState.FromSnapshot(g.ToSnapshot())`. `RunFixedDepth(3)` on `g` and on `g2`, each with a fresh TT, gives identical score bits, move, nodes and cutoffs.
- [ ] All 33 existing tests still pass with unchanged assertions.

### H-2 — ReferenceCompat TT overflow under the production configuration

Problem: `CompatTranspositionTable` is fixed at `1 << 21` entries (75% load limit) and throws when it is full. Python's dict is unbounded. The server-equivalent call `RunIterativeDeepening(10, false, SearchLimits.WithTime(10))` (= `find_best_move_parallel(max_depth=10, max_time=10)`) throws after about 2.7 s on `open_none` and about 5.2 s on `open_all_rules`.

Required change: grow on demand. When `Store` would exceed the load limit, allocate a table twice the size and reinsert all entries of the current generation. Lookups compare the full 128-bit key, so semantics are unchanged. Keep a configurable hard cap (constructor parameter). Reaching it throws a dedicated exception with a clear message (cap, entries); H-1 guarantees the state is restored. Expose read-only `Capacity`, `Used` and `Grows` for diagnostics. Steady-state allocation must remain 0: growth may only happen inside `Store` when the limit is reached, never per node. The existing allocation tests must keep a pre-sized table and still measure 0 bytes.

Acceptance criteria:
- [ ] New test `Compat_TT_grows_without_changing_results`: starting capacity `1 << 6`. Replays at least the 22 bench items plus at least 50 golden positions from `search_search_g1_s20261008_n900.jsonl.gz` at every recorded depth. All golden values are bit-identical and `Grows > 0`.
- [ ] New behavioural test `Server_equivalent_iterative_deepening_completes`: `RunIterativeDeepening(10, false, SearchLimits.WithTime(10))` on `open_none`, `open_same_plus`, `mid3_ascension` and `open_all_rules`. No exception, `UndoDepth == 0`, normalized state restored, `HasMove`, `Levels == DepthCompleted`. It may be slow; do not skip it.
- [ ] Hard-cap test: a tiny cap makes the dedicated exception propagate with the state restored (may share code with the H-1 test).
- [ ] `SearchAllocationTests` and `AllocationTests` still report 0 bytes.
- [ ] `CSHARP_MIGRATION.md` §10 records the server-equivalent C# runs (levels completed, total nodes, peak TT entries, `Grows`, approximate peak TT memory, wall time). Measured numbers only.
- [ ] **Stop condition:** if passing the server-equivalent test needs a TT larger than **1 GB**, stop, record the measurement and ask the user whether the compat oracle may use that memory or whether the server-equivalent configuration should be excluded from compat-mode requirements. Do not pick on your own.

### H-3 — Move identity must be the card instance, not the hand slot

Problem: `Move = (HandSlot, Cell)`. Undo appends the card to the end of the hand (D-001, which must stay reproduced), so a previously generated `Move` names a different card after any make/undo. `CompatSearcher` works around this with a per-node `instances[]` array plus `HandSlotOf`, and `Evaluator.EvaluateMove` reads the slot against the current hand. Python moves hold card objects. This caused a real Phase 3 parity bug, and Phase 4 (TT move, history and ordering) would persist moves across nodes and positions.

Required change (an API and identity change only; no change to rules, ordering or search semantics):
- `Move` carries `(Inst, Cell)` (the card instance index and the row-major cell). Keep `Row` and `Col` helpers and equality.
- `GameState.GenerateMoves` emits instance-based moves in exactly the same order as today: current hand order × row-major empty cells, with the Order/Chaos `can_use` filter.
- `GameState.MakeMove(Move)` resolves the hand slot internally (via `HandSlotOf` or equivalent). It throws `InvalidOperationException` if the instance is not in the side-to-move's hand, the cell is occupied or out of range, or the undo stack is full. Hand removal (shift left), undo (append to the end, D-001), D-003 and the D-002 fix all stay exactly as they are.
- `Evaluator.EvaluateMove` reads `move.Inst` directly.
- Remove the `instances[]` / `HandSlotOf` workaround in `CompatSearcher.Node`. `bestInst` comes from `move.Inst`.
- Tests that read golden `slot*9+cell` codes resolve the slot to the instance **against the hand at that exact step** (for example a small helper `Move FromGoldenCode(GameState g, int code)` in the test project, or an explicitly named `GameState.MoveFromSlot(slot, cell)`). Golden files and generators are not changed.
- Update the `// PY-QUIRK(D-001)` comments and the `HandSlotOf` doc comment where they describe the old workaround.

Acceptance criteria:
- [ ] New test `Move_identity_survives_hand_reordering`: generate the moves at a position, make then undo the second generated move (the hand order changes), then `MakeMove` the **first** generated move. The instance now on that cell is the instance originally generated, and the hand order follows D-001.
- [ ] New test `MakeMove_rejects_instance_not_in_hand`: an opponent's hand instance, an instance already on the board, and an occupied cell all throw, and the state is unchanged after each attempt.
- [ ] `Move_order_is_recomputed_by_card_identity_after_hand_reordering` still asserts the same values (open_none depth 2: score bits `0xC0123D70A3D70A3E`, card 64 at cell 7, 214 nodes, 42 cutoffs). It may be renamed or adapted to the new API but must not be deleted or weakened.
- [ ] Every existing differential test (rules trace 20,000 sequences, eval 100,000 positions, fixed depth 900 + 22, ID 203 + 600 + 200, TT key 3,000 families) passes with all assertions unchanged.
- [ ] `grep` shows no remaining search-path code that interprets a `Move` as a hand slot.
- [ ] Allocation tests still report 0 bytes.
- [ ] Hot-path impact is measured: `Bench search` and `Bench makeundo` medians before and after, same machine, Release, at least 3 runs, recorded in `CSHARP_MIGRATION.md` §10 together with node counts, which must be unchanged. A regression above 10% in either median must be reported to the user (do not optimise it away in this goal).

## 4. Verification (all must hold at the end)

- [ ] `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` gives 0 warnings and 0 errors.
- [ ] `dotnet test csharp/TTC.Siren.sln -c Release` passes everything: 33 original tests plus the new ones, 0 skipped. Report the exact counts.
- [ ] `python -m pytest -q` gives 29 passed (no Python file changed: confirm with `git status` / `git diff --stat` that only C#, test and doc files changed).
- [ ] All Python-generated golden differential tests pass, using the existing cached golden files. Do not regenerate them unless they are missing.
- [ ] Every new test fixes its inputs and seeds and prints them in failure messages (CLAUDE.md §7).

## 5. Documentation

Update `CSHARP_MIGRATION.md`:
- In §0 "审查门禁", note that the H-1, H-2 and H-3 remediation is complete and **awaiting re-review**. Do **not** change the gate to PASS yourself, and do not change the Phase 4 status to anything but "NOT STARTED".
- Add a short "Review remediation (H-1/H-2/H-3)" record under §7 Phase 3: files changed, design decisions (for example the entry-point unwind, grow-on-demand, `Move(Inst, Cell)`), tests added with their results, benchmark before/after, and known limitations.
- §10: add the H-2 server-equivalent measurements and the H-3 benchmark comparison.
- Do not rewrite existing history. Append and adjust only.

## 6. Stop conditions

Stop immediately and report to the user (record the issue in `CSHARP_MIGRATION.md` §9 BLOCKED with reproduction input, Python output, C# output and the hypotheses ruled out) if:
- any golden comparison changes after a change that was supposed to be semantics-neutral, and you cannot explain it;
- a fix would require changing Python behaviour, rules semantics, evaluation, ordering, TT semantics (D-005/D-006) or ID control flow;
- the H-2 memory stop condition above is hit;
- a destructive or remote git operation seems necessary.

Also stop when all of §3–§5 are satisfied. Do not continue into Phase 4.

## 7. Final report

Report briefly:
- for each of H-1, H-2 and H-3: what changed, the tests added, and the evidence;
- build, C# test, Python test and parity results (exact counts);
- the benchmark before/after table;
- any new discrepancy or BLOCKED item;
- that the review gate awaits re-review and Phase 4 has **not** been started.
