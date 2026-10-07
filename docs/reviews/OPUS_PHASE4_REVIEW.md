# OPUS Phase 4 Review — Zobrist / Transposition Table / Move Ordering / History (Fast search)

Reviewer: independent senior reviewer / architecture auditor (Opus), review session only (no implementation, no Phase 5 work).
Date: 2026-10-07. Severity scale and gate vocabulary are the same as in `OPUS_PHASE3_REVIEW.md`.

---

## 1. Review scope

- `csharp/src/TTC.Siren.Core/ZobristKeys.cs` (new) and the Zobrist changes in `GameState.cs` (`ZobristHash`, `ComputeZobristHash`, `UndoRecord.HashBefore`, XOR updates in `MakeMove`, `ResolveFlips` and `ApplyTypeModifier`, restore in `UndoMove`).
- `csharp/src/TTC.Siren.Search/FastTranspositionTable.cs` and `FastSearcher.cs` (new).
- `CompatSearcher.cs` changes: the `UseTranspositionTable` oracle switch and `RunFixedDepthAs`.
- Tests: `ZobristTests.cs` (4), `FastSearchTests.cs` (8 Fast tests, 3 table tests, and the `Oracle` helper).
- The Phase 4 record in `CSHARP_MIGRATION.md` §7 (acceptance, design decisions 1–12, known limitations) and Q-5 = option (b).

## 2. Repository state reviewed

- Branch `csharp-solver-migration`. Phases 1–3 are committed (`aa3fc23`…`d0ef721`). Phase 4 is **uncommitted**: 5 modified files and 5 new ones (`ZobristKeys.cs`, `FastSearcher.cs`, `FastTranspositionTable.cs`, `FastSearchTests.cs`, `ZobristTests.cs`), so `git diff` isolates Phase 4 exactly.
- `git diff --stat -- '*.py'` is empty, so Phase 4 changed no Python.
- Build (`--no-incremental`): 0 warnings, 0 errors.
- `dotnet test`: **58/58 passed** (0 failed, 0 skipped).
- `python -m pytest -q`: 29 passed.

### 2.1 Independent verification performed by this review

| # | Check | Result |
|---|---|---|
| V1 | All 15 Phase 4 tests re-run against the **fresh-seed golden data** from the Phase 3 review (rules 31415927, search 27182818), never seen by the implementer | 15/15 passed. Fixed depth: 4,513 runs (3,358 with 同类 rules, 2,058 searched to the end of the game, 135 with no root move); **every root score is bit-equal to the oracle and every move is in the exact tie set**, also with the 16-entry table. ID: 699 levels match the oracle, 4 levels choose a tied move. Brute force equals the oracle on 4,475 runs (7.5M leaves). Zobrist: 202,850 incremental-versus-recomputed checks and 115,189 distinct positions, with no collision. Interrupted searches: 240 interrupted, 210 continuations hit the table, all exact |
| V2 | **Adversarial stress**: the reviewer's 90 positions (typed openings under 同类, duplicate ids across hands, server-style cards, ids 474 and 1000+) at depth 1..min(5, empties), both evaluation modes, with table sizes **4** (minimum) and 2^16, against the oracle (compat with TT off). Every move that differed from the oracle's was re-verified by an exact full-window search of that move | **1,756 runs, 0 score mismatches, 0 non-tie moves** |
| V3 | Table reuse that is *legitimate* under the analysis in §4.1: same `GameState`, same AI side, same evaluation mode, root advanced by two plies, `clearTable:false` | 90 runs, 0 wrong |
| V4 | Table reuse through the public `clearTable:false` option in three *unchecked* situations: (a) the caller plays one move and searches again (the AI side flips); (b) same root and depth but a different `aggressive` flag; (c) one table shared by two `GameState`s with the same instance layout | **(a) 6/6 scenarios wrong, (b) 6/6 wrong, (c) 3/32 wrong** (for example `end5_same_plus`: Fast +2.76, exact −2.76). See H4-1 |
| V5 | `Bench fast` (Release) | Fast is 1.52× faster than compat TT-off and 1.58× faster than compat TT-on on d ≥ 4 items; node ratio Fast/TT-off is 0.961. All 47 rows have the same score, and every move matches the oracle's (column "same") |

## 3. Gate result

# **PASS_TO_PHASE_5** (after H4-1 remediation verification, 2026-10-07; see §13)

> History: the initial Phase 4 gate was **BLOCK_PHASE_5**. H4-1 was independently verified as **RESOLVED** in §13, with no regression. The MEDIUM, LOW and OPTIONAL findings remain open and nonblocking. The paragraph below is the original rationale, kept for the record.

Original rationale: there was one HIGH finding and no CRITICAL one. The search algorithm itself is correct: Zobrist coverage, incremental updates, bound semantics, interruption handling, the depth rule, TT-move validation and history are all verified analytically and empirically (V1, V2). The blocker is that transposition-table entries carry **no validity context**: the hash only distinguishes positions within one `GameState` searched for one AI side with one evaluation mode, and the public `clearTable:false` path lets those entries be reused outside that domain, returning wrong scores silently (V4). Phase 5 (exact endgame scoring, many scenario `GameState`s that share an instance layout, repeated searches inside the robust selector) is exactly where this reuse is tempting. The fix is small and local.

## 4. Executive summary

### 4.1 Zobrist

- **Coverage.** The hash covers the instance on each cell, the owner of **every** instance (board and hand), the modifier of every instance, and the side to move. Within one `GameState` this is **sufficient**:
  - Hand membership is implied: instances never change hands, and an instance is either on the board or in its original hand.
  - `can_use`, card identity and base edges, type, rules and star are per-instance or per-state invariants.
  - Future modifiers are determined by the board types and the current modifiers, which are hashed.
  - Legal moves depend on occupancy, hand membership, `can_use` and the side to move, all covered or invariant.
  - `EvaluateState` reads only owners and effective edges (base plus modifier).
- **Exclusions.** Display edges (D-003, path-dependent) and hand order (D-001) are correctly excluded: they influence `EvaluateMove` (ordering) only.
- **Domain limit.** Rules, card identity, base edges, `can_use`, type, the AI perspective and the evaluation mode are **not** in the hash. That is correct by design, but it means the hash domain is "one `GameState`, one AI side, one evaluator". See H4-1.
- **Incremental updates.** `MakeMove` XORs the placement (cell key, owner change, side). `ResolveFlips` XORs each owner change, including flips from `None` and combo flips; Same and Plus flips use the same code path. `ApplyTypeModifier` XORs each modifier change, which also covers the hand cards affected by 同类 rules and the public `RecalculateTypeModifiers`. `FromSnapshot` computes from scratch. `UndoMove` restores the saved pre-move hash after restoring every hashed field. That is O(1), cannot drift, and is asserted after every golden undo.
- **Mutation sites.** All mutation sites of hashed fields were enumerated (grep of `_owner[…] =`, `_mod[…] =`, `_cell[…] =`, `_current =`); each one is covered.
- **Test coverage.** Tests compare the incremental hash with a full recomputation over 20,000 Python golden sequences (and over the fresh seed in V1), plus transposition and modifier tests. **The requirement in the review brief is met.**

### 4.2 Transposition table

- **Key validation.** The full 64-bit key is compared inside a two-entry bucket, with no secondary check. The computed false-hit probability is about 10^-11 per search. A stored move is validated against the generated move list before use, so a collision can at worst cost a score, never an illegal move (L4-2).
- **Bounds (verified mathematically).** The search is fail-soft. Bounds are classified against the **original** window (`alphaIn`, `betaIn`, captured before the loop), not the post-loop alpha/beta: `best ≤ alphaIn` → Upper, `best ≥ betaIn` → Lower, otherwise Exact. Retrieval returns Exact unconditionally, Lower only if `s ≥ β`, and Upper only if `s ≤ α`. Each returned value therefore satisfies the fail-soft contract of the caller's window, so a TT cutoff cannot return an invalid score.
- **Depth.** Scores are reused only at **equal** depth (`hit.Depth == depth`). Within one search the remaining depth of a position is fixed by its ply, because ply equals the number of placements, so equal depth is the only case that arises. Shallower or deeper entries are used for their move only, so a shallow entry can never override deeper information as a value. Same-key stores overwrite; a full bucket evicts the shallower entry, which affects performance only.
- **Interruption.** `_stopped` is sticky, and no node stores once it is set. Every ancestor of an interrupted node therefore skips its store, and partial values never enter the table. Verified by the continuation test (V1) and by the mutation the implementer reports.
- **Root.** The root is never cut off by the table, and its move is always produced by its own loop.
- **Score types.** Only heuristic scores exist today (D-007). Exact endgame scores will exist from Phase 5 and are **not distinguishable** in the table (H4-1).

### 4.3 Move ordering

- The order is: the stored move first (only if it is found in the generated list, matched on `(Inst, Cell)`), then the same stable `evaluate_move` + history sort as compat. Depth-1 non-root nodes are not sorted (the O-6 change): their children are leaves, so the node value is order-independent, and stored bounds remain valid bounds for the window used.
- Ordering cannot change legality (moves are only permuted) or the value (alpha-beta with valid bounds; V1, V2, and Fast equal to brute force through the oracle).
- "Ordering disabled versus enabled": the oracle (stable-ordered alpha-beta) equals the **unordered** full-width brute force on 4,475 runs, and Fast equals the oracle with a 4-entry table (V2) and a 16-entry table (V1).

### 4.4 History heuristic

- Keyed by `(card id, cell)` with Python semantics. Instances that share an id (both players holding the same card, `None` ids) share entries; that is a harmless collision because it affects ordering only.
- Updated only at alpha-beta cutoffs, with `min(old + 2^depth, 10^6)`, and cleared per search.
- It cannot affect values (V1, V2).

### 4.5 Performance (recorded, not judged)

Fast is 1.52×/1.58× faster than compat TT-off/TT-on for d ≥ 4. The table saves few nodes (Fast/TT-off node ratio 0.961; on the adversarial set Fast visited 8.0M nodes versus 7.6M for the oracle, because Fast also counts the unsorted depth-1 children). Most of the speed-up comes from the cheaper key (Zobrist versus the Python-string digest) and from skipping depth-1 ordering, not from transposition cutoffs. This is an observation for Phase 6 (O4-1), not a defect.

## 5. CRITICAL findings

None.

## 6. HIGH findings

### H4-1 — Transposition-table entries are not bound to the context in which they are valid; `clearTable:false` silently returns wrong scores

| Field | Value |
|---|---|
| Finding ID | H4-1 |
| Affected file | `csharp/src/TTC.Siren.Search/FastSearcher.cs`, `csharp/src/TTC.Siren.Search/FastTranspositionTable.cs` (and the hash-domain statement in `ZobristKeys.cs`) |
| Affected method | `FastSearcher.RunFixedDepth(…, clearTable)`, `RunIterativeDeepening(…, clearTable)`, `Begin`; `FastTranspositionTable` (no context) |
| Observed behavior | Stored scores are from the **fixed AI perspective** and depend on the `aggressive` flag. The key covers neither, nor the card contents or rules of the `GameState`, because instance indices are reused between states. `clearTable:false` keeps all entries with no check, and so does sharing one table between two searchers or states. Probe V4: (a) after the caller plays the best move and searches `depth − 1` with `clearTable:false`, the score is wrong in **6 of 6** reference scenarios (for example `end5_same_plus` Fast +2.76 versus exact −2.76, and `mid3_ascension` +10.16 versus −6.96), because values stored for the other side are reused at equal depth. (b) Same root and depth with `aggressive` flipped: wrong in **6 of 6**. (c) One table shared between `open_none` and `open_same_plus` (same cards and layout, different rules): the root hashes are equal and the result is wrong in **3 of 32** runs. |
| Expected behavior | A transposition table may only return entries that were computed under the same conditions: the same `GameState` (card set, rules, `can_use`), the same AI perspective, the same evaluation function (heuristic `aggressive` on or off; from Phase 5 also exact card-difference scoring). Otherwise the searcher must refuse the reuse, or start from an empty table. The XML doc of `RunFixedDepth` already states the precondition ("the previous search of the same root") but nothing enforces it, and even "same root" is not the right condition (see V3: a different root is fine; a different AI side is not). |
| Why this matters | The failure is silent: the root score and move are wrong and no test notices, because all current tests use reuse only in the valid way. Phase 5 is where the invalid patterns arise naturally: (1) the robust endgame evaluates many scenario `GameState`s that share the same instance layout and differ only in hidden cards, so their positions hash identically; reusing a table across scenarios is the obvious optimisation and is exactly case (c). (2) The exact endgame solver produces card-difference scores; sharing a table with the heuristic search mixes two score types that the table cannot tell apart. (3) Phase 7's long-lived engine will be tempted to keep the table between moves, which is case (a) whenever the side to move changes. The Phase 4 parity tests cannot catch any of this, because they only exercise valid reuse. |
| Evidence | Probe V4 (reviewer scratch program; outputs quoted above). Code: `FastSearcher.Begin` calls `_tt.NewGeneration()` only `if (clearTable)` and has no other check; `Node` reuses at `hit.Depth == depth` with no context; `FastTranspositionTable.Entry` has no context field. V3 shows the condition that is actually safe: the same state, AI side and evaluation mode, with any root reachable in that state. |
| Recommended fix | Bind the table contents to a **search context** and enforce it in `Begin` (not on the hot path). The context = (identity of the `GameState` object; AI player = root side to move; evaluation mode = heuristic normal, heuristic aggressive, or from Phase 5 an exact mode). (a) `FastTranspositionTable` records the context of its current generation; `NewGeneration(context)` sets it. (b) `Begin` with `clearTable:true` starts a new generation with the current context. (c) With `clearTable:false`, if the recorded context differs from the current one (or the table has never been used), **throw `InvalidOperationException` before touching state, statistics or table**, with a message that names the mismatching component. (d) Document that the Zobrist hash domain is a single `GameState` instance, and that sharing a table between `GameState`s requires a new generation. Using the object reference for state identity is acceptable, because a `GameState`'s card set, rules and `can_use` are immutable for its lifetime. Do **not** add context to the per-node key (no hot-path cost is needed). Optionally expose an evaluation-mode enum now so that Phase 5 must choose one. |
| Required regression test | (1) `Table_reuse_with_a_different_ai_side_is_rejected`: search D, make one move from outside, call `RunFixedDepth(D − 1, …, clearTable:false)`; expect `InvalidOperationException`, with state, hash, `UndoDepth` and table contents unchanged (a subsequent `clearTable:true` search equals the oracle). (2) `Table_reuse_with_a_different_evaluation_mode_is_rejected` (flip `aggressive`). (3) `Table_shared_between_game_states_is_rejected` (`open_none` → `open_same_plus` with `clearTable:false`). The same for `RunIterativeDeepening`. (4) `Valid_table_reuse_matches_the_oracle`: the same state and AI side with the root advanced by two plies (own move plus reply) and `clearTable:false`, plus re-searching the same root at a different depth, over at least 100 golden positions and all 6 reference scenarios; scores are bit-equal to the oracle and moves are in the exact tie set. (5) All existing tests (58) still pass, including `An_interrupted_search_stores_nothing_…` (valid reuse), and the Fast allocation test stays at 0 bytes. |
| Human decision required | No. Throwing is the conservative default; silently auto-clearing would also be correct but hides caller bugs. If Sonnet prefers auto-clear, it must record why and expose the event in the statistics. |

## 7. MEDIUM findings

| ID | Location | Finding | Recommendation |
|---|---|---|---|
| M4-1 | `ZobristTests`, `FastSearchTests` | Hash **sufficiency** is argued analytically but tested only against the same component list the hash uses (`HashRelevantKey` mirrors the hash), so it cannot detect a value-relevant field missing from both. The analysis in §4.1 holds today, but Phase 5 may add state (for example scenario-specific data). | When `GameState` gains a field, add a check: within a search, positions with equal hashes must have equal `ToSnapshot()` outside the documented exclusions (hand order, display edges). Can be a debug-only assertion mode in a test. |
| M4-2 | carried over from M-1 (Phase 3) | Golden files are still not bound to the Python source. Phase 5 adds new golden kinds (exact endgame, scenarios), which widens the exposure. | Address M-1 before or at the start of Phase 5 (source hash in the golden header). |
| M4-3 | carried over from M-2 and M-3 | Unchanged: the post-search hand order and display edges are not compared with Python, and search golden data is skewed toward late positions. Fast does not depend on them, but Phase 5's exact solver will be compared with Python's `_solve_endgame_exact`, whose memo key includes hand order. | Address M-2 before Phase 5's exact-solver parity. |

## 8. LOW findings

| ID | Location | Finding |
|---|---|---|
| L4-1 | `FastTranspositionTable.NewGeneration` | When the generation byte wraps, the whole array is cleared (96 MiB at the default size) every 255 new generations. Phase 5 may run thousands of short searches (scenarios × moves). Harmless for correctness; measure in Phase 6. |
| L4-2 | `FastTranspositionTable` | There is no secondary key check (64-bit key only). The estimated false-hit rate of about 10^-11 per search is computed, not measured. The stored move is validated, so the worst case is one wrong score. Acceptable; revisit if Phase 5 search volume grows by orders of magnitude. |
| L4-3 | `SearchStats` | `TtHits`, `TtCutoffs` and `NodesSearched` mean different things in Fast and compat (Fast counts unsorted depth-1 leaves, and never cuts off at the root). That is documented in §7, but the shared type invites cross-mode NPS comparisons. Label the mode in benchmark output. |
| L4-4 | `FastSearchTests` | The tests write `phase4_q5_*_report.txt` into `tools/parity/golden/` (git-ignored) as a side effect of every run. Fine, but a test-output directory would be cleaner. |
| L4-5 | Process | The Phase 4 changes are uncommitted. The doc says Phases 1–3 were pushed; Phase 4 is only in the working tree. The commit is the user's decision. |

Phase 3 items still open (unchanged, nonblocking): L-1 to L-12, O-1 to O-6 (O-6 is now partly implemented in Fast as design decision 9).

## 9. OPTIONAL OPTIMIZATION findings

Do not implement any of these without the CLAUDE.md §6 process.

| ID | Observation |
|---|---|
| O4-1 | The table cuts few nodes (node ratio 0.96). Candidates to measure in Phase 6: a TT-move-first gain is already there; aspiration windows or PVS; storing the root's best child to improve ID ordering; a smaller or larger table and its cache behaviour. |
| O4-2 | Values are reused only at equal depth. Within one search this loses nothing (equal depth is the only case), so a `>=` rule would gain nothing there and only risks inexact root values across ID levels. Keep equal depth. Noted only so that nobody "optimises" it. |
| O4-3 | The wall clock is read every 1024 nodes; the node-budget path checks every node. Fine. |

## 10. Python/C# semantic differences introduced by Phase 4

| Area | Python | Fast (C#) | Status |
|---|---|---|---|
| TT key | string with hand-id concatenation (D-005) | 64-bit Zobrist over instances | intentional (A-2) |
| TT bounds | none; partial results stored as exact (D-006) | Exact/Lower/Upper; interrupted nodes never stored | intentional; Fast equals the exact minimax value, Python can differ (3 of 4,513 runs on the fresh seed) |
| Tie-breaking | Python ordering plus its TT | stored move first, then the same ordering | Q-5 option (b): only exact-tie moves may differ, verified (V1, V2) |
| ID termination | continues to `max_depth` (D-014) | stops at depth ≥ empties or with no root move | intentional; deeper levels cannot change the value |
| Depth-1 ordering | always sorts | does not sort at non-root depth 1 | value-neutral (O-6), measured |
| TT reuse across searches | the global dict is cleared per request | `clearTable:false` with no context check | **H4-1** |

## 11. Test coverage gaps

1. No test of table reuse across AI sides, evaluation modes or `GameState`s (H4-1).
2. Hash sufficiency is not tested independently of the hash's own component list (M4-1).
3. The wall-clock path is behavioural only (inherent).
4. Fast has no Python golden data of its own, by design. Its oracle is compat TT-off, which is itself verified against Python with the TT on through Phase 3 parity and against brute force. This chain is sound.

## 12. Recommended next action

1. Run `/goal @.claude/sonnet_review_fix_goal.md` (H4-1 only).
2. Re-review the H4-1 diff, then change the gate to PASS_TO_PHASE_5.
3. Before Phase 5's parity work, address M-1 (golden bound to the Python source) and M-2 (post-search state compared with Python). They are recommended, not blocking.
4. Phase 5 design constraint: the exact endgame solver must use its own evaluation mode in the table context (or its own table), and scenario `GameState`s must never share a table generation.
5. Committing Phase 4 is the user's decision.

---

## 13. H4-1 Remediation Verification (2026-10-07)

### 13.1 The hazard H4-1 was meant to prevent

A Fast table entry is a statement "position with Zobrist key K, searched to remaining depth d, has this value or bound". The value is only true **(i)** within the `GameState` whose instance layout, card contents, rules and `can_use` give K its meaning; **(ii)** from the AI perspective of the search that stored it; and **(iii)** under the evaluation function that valued its leaves. None of the three is in the key. Before the fix, `clearTable:false`, or a table shared between searchers, reused entries with no check, and the reviewer measured silently wrong root scores: AI side flipped 6/6, `aggressive` flipped 6/6, cross-state 3/32. The fix had to make reuse outside (state, AI side, evaluation mode) **impossible**, without touching the hot path or the exact-value semantics of valid reuse.

### 13.2 Files inspected

`csharp/src/TTC.Siren.Search/SearchContext.cs` (new), `FastTranspositionTable.cs`, `FastSearcher.cs` (entry points, `Begin`, `Node`), `csharp/src/TTC.Siren.Core/ZobristKeys.cs` (doc only), `GameState.cs` (every mutation site of fields outside the hash), `tests/TTC.Siren.Tests/FastTableContextTests.cs` (new), the remediation record in `CSHARP_MIGRATION.md` §7 Phase 4 "Review remediation (H4-1)" and §10, and a grep for every user of `FastTranspositionTable`. Phase 4 is still uncommitted, so the H4-1 delta was established by comparing with the versions read in the original Phase 4 review (§2).

### 13.3 Implementation verified

| Requirement | Verified behaviour |
|---|---|
| Context domain | `SearchContext = (GameState object, AiPlayer = side to move at the root, EvalMode)`. **Completeness:** grep shows that `_rules`, `_canUse`, `_base`, `_id` and `_type` are written only in `FromSnapshot`/`AddInstance`, so a `GameState` object cannot change anything outside the Zobrist key during its lifetime. Its only other mutators are `MakeMove`/`UndoMove` and `RecalculateTypeModifiers`, which change only hashed fields or ordering-only fields (display edges, hand order). Search depth is handled by the equal-depth rule, and limits by "interrupted nodes never store". The `CardDb` star table affects ordering only. **No value-relevant dimension is missing.** |
| State identity by reference | Intentional and safe, for the reason above. A logically identical clone (`FromSnapshot(ToSnapshot())`) is **rejected**: conservative and correct, with no hidden reuse (probe R4). |
| EvalMode | `Heuristic` / `HeuristicAggressive` are exactly the two leaf functions `EvaluateState(ai, aggressive)` can produce. There is no other evaluator path in Fast today. |
| `clearTable:true` | `Begin` calls `NewGeneration(context)`, which starts a fresh generation **and** binds the context. It cannot fail. |
| `clearTable:false` | `Begin` calls `RequireContext(context)` **as its first statement** (depth and `maxDepth` are validated before `Begin`). It throws if the table is unbound or any component differs, and it only reads. Statistics, history, the clock, `_stopped`, the root fields, the table and the `GameState` are untouched, so no partial binding is possible. The context-free `NewGeneration()` unbinds, so it cannot be used to slip past the check. |
| Valid reuse | Same generation, same entries, and the equal-depth rule is unchanged. |
| No bypass | `FastSearcher.Begin` is the only path into `Node`, and both public entry points call it. `FastTranspositionTable` has no other user in `src/` (only Compat, which uses its own table). See L4-6 for low-level `TryGet`/`Store`. |
| Hot path | `Node`, `TryGet` and `Store` are unchanged apart from line offsets. The context is built and compared once per top-level search: no per-node allocation, comparison, virtual call, dictionary, string or exception path. The Fast allocation test reports 0 bytes (R1). |
| Docs | `SearchContext`, `FastTranspositionTable`, `RunFixedDepth`/`RunIterativeDeepening` and `ZobristKeys` state the hash domain and the reuse rule ("same GameState, same AI side, same evaluation mode; any root"). |

### 13.4 Tests inspected and run (by the reviewer)

| # | Check | Result |
|---|---|---|
| R1 | Build `--no-incremental`; full `dotnet test`; `pytest`; `git diff --stat -- '*.py'` | 0 warnings, 0 errors; **62/62 passed**, 0 skipped; 29 passed; no Python change. Allocation tests: Make/Undo 0 bytes, compat search 0 bytes, Fast search 0 bytes (1,483,703 nodes; this count includes a wall-clock-limited segment, so it varies between machines and runs) |
| R2 | Content of the 4 new tests | Rejection tests: all 6 reference scenarios × `RunFixedDepth`/`RunIterativeDeepening` (and both mode directions). Each asserts `InvalidOperationException` with the right component named, then an unchanged `UndoDepth`, Zobrist hash (and recomputation), the canonical snapshot including hand order and display edges, all `Stats` fields, the history table, `table.Stores`, `IsBound` and (cross-state) `table.Context`. Then it asserts that the valid same-context continuation is still exact, and that a following `clearTable:true` search equals the oracle. The unbound cases (a new table, and after the context-free `NewGeneration()`) are covered. Valid reuse: 448 searches (6 scenarios + 112 golden positions × 2 modes × {root advanced two plies at D−2, original root at D−1}), each asserting a **bit-equal root score first** and then tie-set membership of the move. A wrong score cannot pass by being a tie, because the score assertion comes first and `InTieSet` compares the move's exact full-window value with the oracle root score. The oracle (compat with TT off) is independent of the Fast table: it uses a different searcher, no table, and was verified against brute force in the Phase 4 review. |
| R3 | **Mutation tests** (scratch copy of `csharp/`, run only `FastTableContextTests`): M0 none; M1 `RequireContext` call removed (= pre-fix); M2 AI side not compared; M3 `EvalMode` not compared; M4 `GameState` identity not compared | M0: 4/4 pass. **M1: 3 rejection tests fail**, which confirms Sonnet's claim and shows the tests fail against the pre-fix behaviour. **M2: only the AI-side test fails. M3: only the evaluation-mode test fails. M4: only the shared-state test fails.** Every context component is guarded by its own test. |
| R4 | The reviewer's original H4-1 probes (§2.1 V4), re-run on the fixed code, plus extra variants: a second `FastSearcher` object on the same table, ID instead of fixed depth, and a logically identical clone | **51/51 rejected, 0 silently wrong, 0 with any state, hash, `UndoDepth`, `Stores` or `IsBound` change after the rejection.** Before the fix the same probes gave 6/6, 6/6 and 3/32 wrong scores |

**Performance claim.** Structurally, no hot-path change exists (§13.3). The reported same-batch Fast/TT-off ratios (1.53× → 1.54×) and the unchanged node counts (3,431,185 / 3,771,758 / 3,625,620) agree with that. Raw timings across batches are not interpreted, as instructed. There is no allocation regression.

### 13.5 H4-1 status

# **RESOLVED**

There is no regression, and every pre-existing test passes unchanged.

### 13.6 New findings (all LOW, nonblocking)

| ID | Finding | Recommendation |
|---|---|---|
| L4-6 | `FastTranspositionTable.TryGet`/`Store` remain public and context-free. The Phase 5 constraint ("exact scoring adds its own `EvalMode`; scenario states never share a generation") is **enforced** for every search that goes through `FastSearcher.Begin`, but a new Phase 5 solver that drives the table directly could still store exact scores under a heuristic-bound generation. Such a solver cannot *reuse* a heuristic generation silently only if it binds with `NewGeneration(ctx)` using its own mode. The context-free `NewGeneration()` leaves the table unbound, so a `FastSearcher` reuse is then rejected, which is safe. The table also keeps a strong reference to the bound `GameState` until it is rebound (negligible memory). | The Phase 5 review must check that any new searcher binds the table with its own `EvalMode`, or uses its own table. Optionally make `TryGet`/`Store` `internal` when Phase 5 adds its solver. |
| L4-7 | `Valid_table_reuse_matches_the_oracle` counts "table consulted" from `Stats.TtHits`, which also counts hits on entries stored by the same search, so the metric (396 of 448) overstates how many cross-search reuses happened. Correctness is unaffected: the D−2 two-ply case provably produces cross-search equal-depth entries (a position at new ply *p* is old ply *p*+2, with remaining depth D−2−*p* in both). The reviewer's earlier probe V3 confirmed 90/90 exact results. | Optionally count cutoffs on entries from the previous generation's search (for example, snapshot `Stores` before the reuse search) if the metric is ever used as evidence. |

### 13.7 Remaining Phase 4 findings

- CRITICAL: **0**. HIGH: **0** (H4-1 resolved).
- Open and nonblocking: M4-1 to M4-3; L4-1 to L4-5, plus the new L4-6 and L4-7; O4-1 to O4-3; and the Phase 3 carry-overs. As recommended in §12, M-1 (golden data bound to the Python source) and M-2 (post-search state compared with Python) should be done before Phase 5's exact-solver parity relies on new golden kinds. They are not blockers.

### 13.8 Final Phase 4 gate

# **PASS_TO_PHASE_5**

Phase 5 is technically safe to start. It still needs the user's explicit authorization (CLAUDE.md: one phase per authorization). The Phase 5 design constraint in `CSHARP_MIGRATION.md` §12.8 (its own `EvalMode` for exact scoring; never share a table generation across scenario `GameState`s) is sufficient and is enforced for every `FastSearcher`-based search. L4-6 covers the remaining discipline item for any new low-level user of the table.
