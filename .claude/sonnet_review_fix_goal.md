# TTC_Siren — Review Remediation Goal (Phase 4 review → fix H4-1 only)

> **STATUS: COMPLETED AND VERIFIED (2026-10-07).** The re-review found H4-1 **RESOLVED**, with no regression (`docs/reviews/OPUS_PHASE4_REVIEW.md` §13). The Phase 4 gate is now `PASS_TO_PHASE_5`. **Do not run this goal again.** There is no remaining blocking work. It is kept for the record only.


Run with: `/goal @.claude/sonnet_review_fix_goal.md`

This goal replaces the previous (completed) Phase 3 remediation goal. It authorizes **only** the remediation of finding **H4-1** of `docs/reviews/OPUS_PHASE4_REVIEW.md`. It does **not** authorize Phase 5, or any part of it: no exact endgame solver, no robust endgame, no scenario construction.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely. Its rules win over anything here if they seem to conflict.
2. Read `CSHARP_MIGRATION.md` completely, especially §0, §7 Phase 4 (design decisions 1–12) and §12.
3. Read `docs/reviews/OPUS_PHASE4_REVIEW.md` completely. §6 (H4-1) is your specification; §4.1 explains why the hash domain is "one `GameState`, one AI side, one evaluation mode".
4. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (expect 0 warnings, 0 errors)
   - `dotnet test csharp/TTC.Siren.sln -c Release` (expect 58/58)
   - `python -m pytest -q` (expect 29 passed)
   - `dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- fast` (keep the totals line for §4)
5. In one short message, state the finding and its acceptance criteria, then begin.

## 2. Scope discipline

- Fix only H4-1. Do not fix M4-*, L4-*, O4-* or any open Phase 3 item. The one exception is a change H4-1 strictly requires, such as a doc comment you touch.
- Do not change any Python file. Do not change rules, evaluation, ordering, bound semantics, the equal-depth reuse rule, interruption handling or ID control flow. Every existing test must pass with unchanged assertions, and every Fast-versus-oracle result (scores bit-for-bit, tie-set membership, node counts of the existing tests) must stay identical.
- No change to the per-node hot path: the context check belongs in `Begin` (or equivalent), not in `Node` or `TryGet`/`Store`. The Fast allocation test must stay at 0 bytes.
- Do not skip, delete or weaken tests. Git: no commit unless the user asks, never push, never touch `main`.

## 3. Finding H4-1 and acceptance criteria

**Problem.** `FastTranspositionTable` entries carry no validity context. Stored scores are from the fixed AI perspective and depend on the evaluation mode (`aggressive`). The Zobrist key covers only instance-level state of one `GameState`, so equal hashes in another state mean different positions. `clearTable:false` (and sharing one table between searchers) keeps old entries with no check. The reviewer's probe found these silently wrong root scores:
- (a) search D, the caller plays a move, then `RunFixedDepth(D − 1, …, clearTable:false)`: wrong in 6/6 reference scenarios (for example `end5_same_plus` gave +2.76 where the exact value is −2.76);
- (b) same root and depth with `aggressive` flipped: wrong in 6/6;
- (c) one table shared by `open_none` and then `open_same_plus` with `clearTable:false`: wrong in 3 of 32 depth/mode combinations.

Reuse with the same `GameState`, the same AI side and the same evaluation mode is **correct**, even from a different root (reviewer: 90/90 runs, root advanced by two plies).

**Required change.**
- Define the **search context** = (identity of the `GameState` object, AI player = side to move at the root, evaluation mode). Today the evaluation mode is heuristic-normal or heuristic-aggressive. Introduce it as a small enum (for example `EvalMode { Heuristic, HeuristicAggressive }`) so that Phase 5 must add its own value (an exact mode) rather than reuse the heuristic one. Do not implement anything for Phase 5.
- `FastTranspositionTable` records the context of its current generation. Starting a new generation sets it (for example `NewGeneration(in SearchContext ctx)`; keep a context-free overload only if it marks the table as "unbound"). A freshly constructed table is unbound.
- `FastSearcher.Begin`: with `clearTable:true`, start a new generation with the current context. With `clearTable:false`, if the table is unbound or its context differs from the current one, **throw `InvalidOperationException` before changing anything** (state, hash, statistics, history, table, clock). The message names the mismatching component (state, AI side or evaluation mode).
- Use the object reference for state identity (`ReferenceEquals`). A `GameState`'s cards, rules and `can_use` are immutable for its lifetime. Do not hash card contents into the per-node key.
- Update the doc comments of `RunFixedDepth`, `RunIterativeDeepening`, `FastTranspositionTable` and `ZobristKeys` to state the hash domain and the reuse rule precisely ("same GameState, same AI side, same evaluation mode; any root").
- `CompatSearcher` and `CompatTranspositionTable` are out of scope. Compat clears per search and emulates Python's per-request dict.

**Acceptance criteria.**
- [ ] `Table_reuse_with_a_different_ai_side_is_rejected`: probe (a) on all 6 `ReferenceScenarios` and on both `RunFixedDepth` and `RunIterativeDeepening`. It throws `InvalidOperationException`. Afterwards `UndoDepth`, `ZobristHash == ComputeZobristHash()`, the canonical snapshot (`ToSnapshot().ToCanonicalString()`, including hand order and display edges) and `Stats` are unchanged. A following `clearTable:true` search equals the oracle bit for bit.
- [ ] `Table_reuse_with_a_different_evaluation_mode_is_rejected`: probe (b), with the same post-conditions.
- [ ] `Table_shared_between_game_states_is_rejected`: probe (c) (`open_none` → `open_same_plus`, and `open_none` → `open_all_rules`), with the same post-conditions. An unbound (new) table with `clearTable:false` is also rejected.
- [ ] `Valid_table_reuse_matches_the_oracle`: over all 6 reference scenarios and at least 100 positions from `search_id_g1_s20261008_n200.jsonl.gz` (those with ≥ 4 empty cells), for each of normal and aggressive mode: search D = min(5, empties), play Fast's move and then one reply (generated order, fixed choice such as the middle move), and call `RunFixedDepth(D − 2, …, clearTable:false)`. The score is bit-equal to the oracle (`CompatSearcher`, `UseTranspositionTable = false`) and the move is in the exact tie set (reuse `Oracle.RootMoves` and `Oracle.InTieSet`). Also re-search the original root at depth D − 1 with `clearTable:false`, with the same assertions. Fixed seeds and positions are printed in failure messages.
- [ ] `An_interrupted_search_stores_nothing_that_later_searches_could_reuse` is unchanged and passes (it is valid reuse).
- [ ] All existing tests pass (58 plus the new ones, 0 skipped); report exact counts. `python -m pytest -q` gives 29 passed, and `git diff --stat -- '*.py'` is empty.
- [ ] The Fast allocation test still reports 0 bytes. `Bench fast`: the totals line before and after (3 runs, median) is recorded in `CSHARP_MIGRATION.md` §10, with node counts unchanged. Report any regression above 5% to the user; do not optimise it in this goal.

## 4. Documentation

- `CSHARP_MIGRATION.md`: under §7 Phase 4 add "Review remediation (H4-1)" (files, design decision, tests and results, benchmark). In §0, note that H4-1 is fixed and **awaiting re-review**. **Do not** set the gate to PASS yourself, and do not change Phase 5's status from NOT STARTED.
- Add a Phase 5 design constraint to §12 (or the Phase 5 section): the exact endgame scoring must use its own `EvalMode` value, and scenario `GameState`s never share a table generation.
- Do not rewrite history. Append and adjust only.

## 5. Stop conditions

Stop and report (and record it in `CSHARP_MIGRATION.md` §9 when relevant) if:
- any existing Fast-versus-oracle result changes;
- a fix would require changing search semantics, Python behaviour or the hot path;
- a destructive or remote git operation seems necessary.

Stop when §3 and §4 are satisfied. Do not continue into Phase 5.

## 6. Final report

Report briefly: the change, the tests added with their results, exact build/test/pytest counts, the benchmark before and after, and that the gate awaits re-review and Phase 5 has not been started.
