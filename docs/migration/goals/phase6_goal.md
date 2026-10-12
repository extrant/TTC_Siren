# TTC_Siren — Goal: commit Phase 5, fix D-004 in both implementations, then Phase 6 (profiler-driven optimisation)

> **STATUS: COMPLETED AND REVIEWED (2026-10-10).** Gate `PASS_WITH_NONBLOCKING_FINDINGS` (`docs/migration/reviews/OPUS_PHASE6_REVIEW.md`). **Do not run this goal again.** If a session still has this goal active (for example a Stop hook asks for a Phase 6 report), the goal is stale: clear it or start a new session; never redo Phase 6 work. Kept for the record only.

Run with: `/goal @docs/migration/goals/phase6_goal.md`

The user authorized all of the following on 2026-10-09 (recorded in `CSHARP_MIGRATION.md` §9 Q-2 and §0 row 6). Do them **in this order**, as separate logical changes:
- **Step A:** commit Phase 5 and the benchmark work, then push the branch.
- **Step B:** the D-004 fix in Python and C# (Q-2 = fix both).
- **Step C:** Phase 6, profiler-driven single-thread optimisation.

Nothing beyond Phase 6 is authorized: no Phase 7 integration work, no root parallelism, no Lazy SMP.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely. Its rules win if anything here seems to conflict. Note §1.4 (semantic fixes), §5 (zero allocation), §6 (performance process), §7 (tests), §8 (git).
2. Read `CSHARP_MIGRATION.md` completely, especially §0, §4 (D-001, D-003, D-004, D-015, D-017), §7 Phase 5 and Phase 6, §9, §10 and §12.
3. Read `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` §8 (profile evidence), §9 (Phase 6 targets) and §12–§13 (pinning rule A-7). Also read `docs/migration/reports/PHASE5_BENCHMARK.md`, the "Corrections" section and the A-7 section.
4. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (expect 0 warnings, 0 errors)
   - `dotnet test csharp/TTC.Siren.sln -c Release` (expect 78/78)
   - `python -m pytest -q` (expect 29 passed)
   - `git status -sb` (the branch tracks `origin/csharp-solver-migration` at `710a70e`; all Phase 5 work is uncommitted)
5. In one short message, state Steps A, B and C with their acceptance criteria, then begin.

## 2. Global rules

- **Git:**
  - Never touch `main`. Never force-push, rebase pushed history, tag, release or open a PR.
  - The user authorized **one push**: Step A's commits to `origin/csharp-solver-migration`.
  - Step B and Step C commits stay **local**. Do not push them; the user decides after the independent review.
  - One logical change per commit, with messages stating what and why. Never commit `csharp/.vs/`, `bin/`, `obj/`, `.pytest_cache/` or golden data (`tools/parity/golden/*.jsonl.gz` is git-ignored).
- **Python:** apart from the approved D-004 fix in Step B (`core/game_state.py` plus a new regression test file), do not change Python business code. Parity generators under `tools/parity/` may be extended for Step B coverage.
- **Tests:** never skip, delete or weaken tests. The single exception is `D004_duplicate_hand_ids_are_rejected`. It asserts the behaviour that Q-2 deliberately changes, so replace it with a test of the new behaviour and record the reason (CLAUDE.md §7).
- **Timing:** every timing measurement uses the pinned harness. Use `python tools/bench/run_interleaved.py --affinity auto …`. For C#-only runs, use `tools/bench/pin.py` (`pin.run_pinned([...Bench.exe...], pin.performance_core_mask())`) or `cmd /c start /wait /b /affinity <mask> …`. Every recorded table states the mask, warm or cold, and min/median/max over at least 3 independent process runs. Unpinned numbers are not evidence.
- **Stop conditions:** if any parity test fails, or any node count, score bit or selected move changes where it must not, stop and diagnose. If the cause is not explained, record it in `CSHARP_MIGRATION.md` §9 BLOCKED (reproduction input, Python output, C# output, hypotheses ruled out) and report to the user.

## 3. Step A — commit and push Phase 5 (authorized push)

- [ ] Split the current uncommitted work into logical local commits, for example:
  1. docs: `CSHARP_MIGRATION.md`, `docs/migration/reports/PHASE5_BENCHMARK.md`, `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md`, `docs/migration/goals/sonnet_review_fix_goal.md`, `docs/migration/goals/phase6_goal.md`;
  2. golden binding (M4-1/M4-2): `tools/parity/golden_sources.py`, `py_rules_trace.py`, `py_search_trace.py`, `GoldenFiles.cs`, `GoldenSources.cs`, `GoldenBindingTests.cs`, `ZobristSufficiencyTests.cs`;
  3. C# endgame: `ExactEndgameSolver.cs`, `EndgameScenarios.cs`, `EndgameRobust.cs`, `SearchContext.cs`;
  4. endgame parity: `tools/parity/py_endgame_trace.py`, `EndgameDifferentialTests.cs`, `ExactEndgameSolverTests.cs`;
  5. benchmark tooling: `tools/bench/*` (including `endgame_bench_positions.json`, `pin.py`), `csharp/src/TTC.Siren.Bench/Program.cs`, `csharp/.gitignore`.

  Adjust the grouping if a file belongs elsewhere, but every commit must build. Run `dotnet build` (0 warnings) and `dotnet test` at least on the final commit.
- [ ] Push to `origin/csharp-solver-migration` (fast-forward only). Confirm with `git status -sb` that the branch is in sync, and report the commit hashes.

## 4. Step B — D-004 fixed in Python and C# (Q-2, approved semantic change)

**Problem.** Python `make_move` takes the hand card by `card_id`: `next(c for c in hand if c.card_id == card.card_id)`. With two cards of the same id in one hand it removes the first such card but places the card that was passed, so after undo there is a duplicate reference and the original card is lost (D-004). C# currently rejects such input in `GameState.FromSnapshot` (`InvalidGameStateException`).

**Required change.**
- **Python `core/game_state.py`, `make_move`:** look up the hand card by **object identity first** (`next((c for c in hand if c is card), None)`). Only if that finds nothing, fall back to the old `card_id` match.
  - This keeps every existing caller's behaviour. For example, `ai_server.count_captured_cards` passes a `card.copy()` and still uses the fallback, and the robust layer passes the first hand card with that id, which identity finds.
  - The only change is that a card that *is* in the hand is the one removed, which fixes D-004.
  - Do not change `undo_move` (D-001 append and D-003 stay).
- **C#:** `GameState.FromSnapshot` accepts duplicate ids within a hand. Remove the D-004 rejection and update its `PY-QUIRK(D-004)` comment. Moves are already instance-based (`Move(Inst, Cell)`), so `MakeMove`/`UndoMove` need no change.
  - Check `HistoryTable` (keyed by id; duplicates share entries, as in Python), `PythonStateKey` (id concatenation, as in Python), the Zobrist hash (per instance) and the endgame code.
  - `EndgameRobust`/`EndgameScenarios` map moves to scenario cards **by card id, first match** (Python's `next(c for c in scenario hand if c.card_id == …)`). Keep that exactly, and record it in §4 as a new entry **D-019 (OPEN-EMULATE)**: "with duplicate ids the robust layer maps every duplicate to the first card of that id; harmless for real data because equal ids are equal card definitions".
- **Golden coverage:**
  - Extend `py_rules_trace.py` and `py_search_trace.py` so a deterministic share of records (at least 5%) has duplicate ids in one hand. Use both identical cards with the same real id and two `None`-id placeholders.
  - Extend `py_endgame_trace.py` with at least 50 robust-selection records where the side to move or a scenario hand has duplicate ids.
  - Bump each generator version. The golden-source binding (M4-2) will regenerate the files because `core/game_state.py` changes.
- **Tests:**
  - Python: `tests/test_duplicate_hand_ids.py`. It covers two same-id cards and two `None` cards in one hand. `make_move` with the second card places exactly that object and removes exactly that object. Undo restores both cards, with no duplicate reference and no lost card (check by identity), across a seeded random make/undo sequence. It must fail on the old code (state that you checked this). The existing 29 tests still pass.
  - C#: replace `D004_duplicate_hand_ids_are_rejected` with `Duplicate_hand_ids_are_supported`: state round-trip, make/undo by instance, Zobrist equals recomputation, and search on such a state equals the oracle. The full differential suite passes on the regenerated golden data and reports how many duplicate-id records and steps it covered.
- [ ] `D-004` becomes **FIXED-BOTH** in §4. D-019 is added. One local commit "fix: D-004 …" (not pushed).

## 5. Step C — Phase 6, profiler-driven single-thread optimisation

**Baseline first (no code change).** Use `run_interleaved.py --rounds 3 --affinity auto` (warm and `--cold`) and pinned C#-only runs of `Bench search`, `fast`, `exact`, `endgame` and `makeundo` (at least 3 process runs each). Profile the C# side with `dotnet-trace collect --profile dotnet-sampled-thread-time,dotnet-common --format Speedscope` on a warm, pinned workload (Fast search, compat search, exact solver), and record the top inclusive costs. Record everything in §10 and a new `docs/migration/reports/PHASE6_OPTIMIZATION.md`.

**Candidates**, in the order the evidence suggests (review §8/§9). Measure each one; do not assume:

1. **同类 modifier maintenance in `MakeMove`/`UndoMove`.** This is `RecalculateTypeModifiers` + `ApplyTypeModifier` + the 19-entry modifier snapshot; about 29% of Fast and 57% of exact-solver time on 同类 workloads. **Semantic constraints, which golden data must keep proving:**
   - Python recalculates **every type present on the board** on **every typed placement**. A first typed placement can therefore change modifiers of types whose count did not change (D-017 scenario cards start at 0).
   - `ApplyTypeModifier` also refreshes the **display edges** of every card of every present type, and undo does **not** restore them (D-003). Display edges feed `EvaluateMove`, so they change move ordering and therefore compat node counts.
   - Any incremental scheme must produce exactly the same modifiers, display edges and Zobrist hash after every make and undo.
2. **`PlacementEdgeScore` / `EvaluateState`.** For example a precomputed score per (instance, cell, modifier), or incremental leaf evaluation. Results must stay **bit-identical** (double arithmetic order unchanged; see A-4 in §8).
3. **`EvaluateMove`/ordering** on the compat path.
4. **Small items:** `[SkipLocalsInit]` on the recursive search methods (O-1); avoiding the per-call memo allocation in `EndgameRobust.Select` (L5-2); the per-node clock read in compat (O-2).

**Process for every candidate** (CLAUDE.md §6): Baseline → Modify (one kind of change) → Build (0 warnings) → Test (all tests) → Benchmark (pinned, warm and cold, at least 3 runs, median) → Compare. **Keep** a change only if it gives a repeatable improvement of at least 3% on the workload it targets and no regression above 3% elsewhere. Otherwise **revert** it and record the measurement anyway. Each kept change is its own local commit.

**Invariants (acceptance for every kept change):**
- [ ] All tests pass (78 plus Step B's), with no assertion weakened.
- [ ] **ReferenceCompat stays node-for-node identical to Python.** Identical node counts, cutoffs, TT counters, history tables, score bits and moves on all golden data, including the 22 fixed-depth items.
- [ ] **Fast:** root scores stay bit-identical to the oracle. Fast node counts and moves should stay identical as well. If a kept change alters them (it should not, as no ordering semantics may change in this phase), list every changed row and prove tie-set membership (Q-5 (b)).
- [ ] **Exact solver:** values identical. A `make_move` count may change only if memo sizing or layout changed, and the reason must be recorded.
- [ ] **Endgame:** 72/72 selections identical in `Bench endgame`; robust parity tests pass.
- [ ] Zero managed allocation in the hot paths, asserted by the existing tests.
- [ ] No `unsafe`, SIMD or multithreading unless profiler evidence is recorded first. **Root parallelism is not authorized** in this goal.

**Scope limit:** work through the candidates until the top three measured hotspots have each been attempted, or until no remaining candidate shows at least 3% in a quick pinned measurement, whichever comes first. Then stop.

## 6. Documentation

- `docs/migration/reports/PHASE6_OPTIMIZATION.md`: baseline tables with mask and warm/cold, the profile before and after, each candidate (kept or reverted, with before/after medians and min/max), the final Python-vs-C# pinned ratios, and the remaining hotspots.
- `CSHARP_MIGRATION.md`:
  - §0: Phase 6 status **DONE, awaiting independent review**. Do not set any review gate yourself.
  - §4: D-004 FIXED-BOTH and D-019.
  - §7 Phase 6: a delivery record covering design decisions, tests and known limits.
  - §10: measurements only.
  - §12: handoff.
  - Append and adjust only; do not rewrite history.

## 7. Final report

Report briefly:
- Step A commit hashes and push confirmation.
- Step B change, tests that failed before and pass after, golden coverage counts, and the local commit hash.
- Step C baseline, every candidate with its result (kept or reverted), final pinned ratios, and invariant evidence.
- Exact build, test and pytest counts, and `git status -sb`.
- That Phase 7 has not been started, and that Step B and Step C commits are local and not pushed.
