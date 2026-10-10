# Opus independent review — Phase 6 goal (Step A push, Step B D-004 fix, Step C optimisation)

Date: 2026-10-10. Reviewer: Opus (independent; nothing below is taken from the implementer's summary without re-running it).
Goal under review: `.claude/phase6_goal.md`. Implementer report: `docs/reports/PHASE6_OPTIMIZATION.md`, `CSHARP_MIGRATION.md` §4, §7 Phase 6, §10, §11, §12.

## 1. Gate

**`PASS_WITH_NONBLOCKING_FINDINGS`**

- CRITICAL: 0. HIGH: 0. MEDIUM: 0. LOW: 5 (L6-1 … L6-5). OPTIONAL OPTIMIZATION: 3 (O6-1 … O6-3).
- Step A: **PASS** (logical commits, every code commit builds, only Step A was pushed, fast-forward, `main` untouched, no new PR, no build output or golden data committed).
- Step B: **PASS** (D-004 fixed in Python and C#, the new Python regression test fails on the old code, D-019 emulated and covered, golden versions bumped, coverage thresholds asserted and met on a fresh seed).
- Step C: **PASS** (four kept changes, each its own commit, each semantically verified; deterministic outputs byte-identical to the pre-optimisation build; pinned re-measurement within 15% of the report).
- Phase 7 may start **once the user authorises it**. No remediation goal is needed; the LOW items can be folded into the Phase 7 goal (see §9).

## 2. Scope and repository state

- Branch `csharp-solver-migration`, HEAD `dce736f`, working tree clean, 16 commits ahead of `origin/csharp-solver-migration`.
- `git ls-remote origin`: `refs/heads/csharp-solver-migration = b6e4ea4`, `refs/heads/main = eb60a02` (= local `main`, unchanged), `refs/pull/1/head = d0ef721` (PR #1, **closed 2026-10-07**, i.e. before this goal; no new PR).
- Commits under review: Step A `e10574a 994d47a da54372 4b46047 b6e4ea4` (pushed); Step B `a0eb193`; Step C `48e38cd 3a77dbe 6b61630 2c226bb 782a3b5 c6d3115 17c07c7 290b15e 427d67f d7566e3 fd0819c 34086ac 7878d1f a4d86ad dce736f` (local only).
- Net code at HEAD = `782a3b5` (C1 + C3 + C4a + C7): `git diff 782a3b5 HEAD -- csharp/src` is empty; the two C3 revert/re-apply pairs cancel out (`290b15e` = `a4d86ad` = `6b61630` for `Evaluator.cs`).
- Python business code changed only in `core/game_state.py` (`make_move`, +8/−2 lines); plus the new `tests/test_duplicate_hand_ids.py`. `git diff 710a70e..HEAD` outside `csharp/ docs/ tools/ .claude/ CSHARP_MIGRATION.md` touches nothing else.

## 3. Step A — commit and push of Phase 5

| check | result |
|---|---|
| logical split | `e10574a` docs, `994d47a` M4-1/M4-2 golden binding + Zobrist sufficiency tests, `da54372` Phase 5 solver, `4b46047` Phase 5 golden generator + tests, `b6e4ea4` benchmark tooling. One logical change each. |
| every commit builds | Built **each code commit independently** from `git archive` (`994d47a da54372 4b46047 b6e4ea4 a0eb193 48e38cd 3a77dbe 6b61630 2c226bb 17c07c7 782a3b5`): `dotnet build -c Release --no-incremental` → 0 warnings / 0 errors for all 11. `e10574a` is docs-only on `710a70e`. |
| push scope | remote branch = `b6e4ea4`; `710a70e` is an ancestor (fast-forward); B/C commits are not on the remote. |
| `main` / force / PR | `main` = `eb60a02` locally and remotely; no forced update; no PR opened by this goal. |
| artifacts | `git ls-files` contains no `bin/`, `obj/`, `.vs/`, `.pytest_cache/`, `*.jsonl.gz`; `tools/parity/golden/` only has its `.gitignore`. `tools/bench/endgame_bench_positions.json` (52 KB) is a benchmark fixture, intended. |

## 4. Step B — D-004 fixed in Python and C#

**Python** (`core/game_state.py:make_move`): identity first `next((c for c in hand if c is card), None)`, then the old `card_id` match. `undo_move` unchanged (D-001 append, D-003 display edges not restored). `Player.play_card` removes by identity (`Card` defines no `__eq__`), so the removed card is the found one.

Callers re-checked (unchanged list): `ai/ai.py:516,567`, `ai/monte_carlo.py:249,307`, `ai_server.py:571,581,607,633,1398,1773,1812`, `app.py` (5), `draft_analyzer_app.py` (2), `npc_deck_recommender_demo.py` (2), `core/game_state.py:296`. Every caller passes either a card that is in the current hand (identity finds it; identical to before unless the hand had a duplicate id — the bug) or a copy (`ai_server.py:1773 card.copy()`; falls back to the old id match). The robust layer (`607/633`) passes `next(c for c in hand if c.card_id == …)`, the first card of that id, which identity finds. **No behaviour change outside the duplicate-id case.**

**Regression test verified to fail on the old code.** I extracted HEAD's `core/` and `tests/` into a scratch directory, replaced `core/game_state.py` with `a0eb193^`'s version and ran `pytest tests/test_duplicate_hand_ids.py`: **3 failed, 2 passed** (the two passing ones are the copy fallback and the rejection case, which must pass on both). On HEAD: `python -m pytest -q` → **34 passed** (29 + 5).

**C#.** `GameState.FromSnapshot` no longer rejects duplicate ids; comment updated. `Duplicate_hand_ids_are_supported` replaces `D004_duplicate_hand_ids_are_rejected` with the reason in the commit message and §7 (the old test asserted the behaviour that was deliberately reversed) — round trip, make/undo by instance (second same-id card placed, D-001 order after undo), 200 seeded random walks with incremental == recomputed Zobrist, and Fast == oracle bit-for-bit at depth 1–4. The Zobrist hash is keyed per instance (`ZobristKeys.Cell(cell, inst)`, `Owner(inst)`, `Mod(inst)`), so two same-id cards with different edges (the `None` placeholders) never collide; `HistoryTable` and `PythonStateKey` stay id-keyed as in Python.

**D-019** (`EndgameRobust.BestImmediateReplyScore` and `Select`): moves are mapped to the first card of that id in the hand **as it is on entry**, resolved for all moves before the first `Make` — this matches Python, which copies the state per reply/move and never sees the D-001 reordering. Without duplicates the mapping is unique, so pre-fix results are unchanged. `PY-QUIRK(D-019)` comments present; §4 row present (OPEN-EMULATE); D-004 row says FIXED-BOTH.

**Golden coverage** (versions bumped: `py_rules_trace/4`, `py_search_trace/3`, `py_endgame_trace/2`). The tests assert the thresholds (≥5% per golden kind, ≥50 robust records with duplicates, ≥20 with duplicates in the mover's hand and ≥20 in a scenario hand), so the coverage is enforced, not only reported. Numbers I observed on the **fresh seeds** (§6): rules 2,500 / 20,000 sequences (12.5%) with 3,367 makes of a duplicated-id card; eval 12,500 / 100,000 positions; search 112 / 900 positions (559 runs); ID 25 / 200; corner and reply 187 / 1,500 each; robust 77 / 400 records (mover 50, scenario 77) on both Python-built and C#-built scenarios. Known, documented limit: the `exact` and `scenario` golden kinds carry no duplicate ids (L6-5).

## 5. Step C — Phase 6 optimisation

### 5.1 Kept changes, semantics checked line by line

| commit | change | semantic check |
|---|---|---|
| C1 `3a77dbe` | `RecalculateTypeModifiers`: one pass over the instance table instead of per-type board + hand scans | Instances are created only in `FromSnapshot` from the board and the hands, and hands change only in `MakeMove`/`UndoMove`, so "every instance is on the board or in a hand" holds. Each card's new modifier depends only on its own type's board count; every present type is still re-applied on every typed placement (D-017 catch-up), every touched card still refreshes its display edges, undo still does not restore them (D-003); hash update is XOR (order-free). **Probe (below): byte-identical.** |
| C3 `6b61630` (= `290b15e` = `a4d86ad`) | table-driven `PlacementEdgeScore` | Effective values are clamped to 1..10 (`Effective()`; Python `max(1, min(10, …))`), so the 11-entry tables are never indexed out of range or at 0. Terms are the same constants, `score -= x` ≡ `score += -x` in IEEE; all terms are small multiples of 0.5, so every partial sum is exact and the order of additions cannot change the bits; no `-0.0` can arise. Corner pair bonus reads the two exposed values in the old side order (and is symmetric anyway). |
| C4a `2c226bb` | `EvaluateMove` walks the precomputed neighbour table | Old order `Dr={-1,1,0,0}, Dc={0,0,-1,1}` = up, down, left, right; `GameState` builds `NbCell/NbMySide/NbOppSide` in the same order, skipping off-board neighbours exactly as the old `continue`. Accumulation order unchanged. |
| C7 `782a3b5` | `[ThreadStatic]` exact-solver memo reused across `Select` calls | Every scenario still calls `Bind` → `NewGeneration(SearchContext)`; the generation byte wraps with a full `Array.Clear` at 255 (`FastTranspositionTable.NewGeneration`), so stale entries cannot be read after many calls. Per-thread, so no sharing between threads. |

**Independent full-state probe for C1.** A console probe built twice, once against the original Core (`48e38cd`, pre-optimisation, D-004 already fixed) and once against HEAD's Core: 4,000 seeded random positions (同类强化 / 同类弱化 / both / combined with 同数, 加算, 逆转, 王牌杀手; typed and untyped cards; 30% of cards with a stale starting modifier as D-017 scenario cards; same-id copies), random make/undo walks, and after **every** step a dump of the incremental Zobrist hash, the recomputed hash, every instance's modifier, owner and four display edges, the board and both hands in order. 180,298 steps; modifiers −3..+4 all exercised. Both outputs have SHA-256 `640F3B81…E67614` — **byte-identical**; incremental hash == recomputed hash at every step.

**Deterministic outputs, original build vs final build** (both run by me): `Bench search` 22 `SIG` lines (nodes, score bits, move, alloc) identical; `Bench fast` 40 `SIG` lines identical, 40/40 `same`, 0 `DIFF` — so Fast node counts, scores and moves did not change and no tie-set analysis is needed; `Bench exact` values 1 / −2 / −1 and `make_move` 217 / 4,707 / 241,264 identical; `Bench endgame` `same=72 different=0`; `Bench makeundo` checksums identical (signature `575d814747eec483`), `alloc=0`.

**Process.** Every kept change is its own commit with pinned (`0xFFF`) before/after medians in the message and in the report; reverted candidates C2, C5a, C6, C8a, C9 are recorded with their numbers (`docs/reports/phase6/candidates_log.txt`). The 3% keep rule was applied conservatively (C9 at −2.8% median was reverted). No `unsafe`, SIMD, threads or root parallelism (`grep` of `csharp/src`: none). The C3 cold question was handled by reverting, re-measuring under the goal's protocol and re-applying; my own alternating measurement (§6) settles it: C3 makes the cold 22-item suite **faster**, not slower.

### 5.2 Correctness runs (mine)

| command | result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test csharp/TTC.Siren.sln -c Release` (cached golden) | **78 passed**, 0 failed, 0 skipped |
| `python -m pytest -q` | **34 passed** |
| fresh golden: HEAD tree in a scratch copy, seeds changed to rules 31337101 / search 31337102 / endgame 31337103, all golden regenerated by the Python generators, `dotnet test` | **78 passed**, 0 failed (rules 20,000 sequences / 184,082 steps; search 4,519 fixed-depth runs incl. the 22 items, 1.19M nodes; eval 1.6M `evaluate_move` + 400k `evaluate_state`; ID 200 positions; robust 400 records × 2 scenario sources; all bit-for-bit) |

## 6. Benchmarks re-measured (pinned `0xFFF` on both sides, CPython 3.11.4)

**Python vs C#, `run_interleaved.py --rounds 3 --affinity auto`** (every child reported `0xFFF`; node counts, exact values and 72/72 asserted by the runner), min / median / max:

| suite | warm: report (`427d67f`) | warm: reviewer | cold: report | cold: reviewer |
|---|---|---|---|---|
| fixed-depth compat (22 items) | 156.0 / 162.4 / 165.7× | 157.2 / 158.6 / 160.7× | 62.9 / 63.4 / 66.6× | 61.7 / 63.9 / 68.4× |
| exact solver | 249.4 / 255.2 / 263.4× | 254.6 / 255.5 / 263.6× | 241.8 / 256.6 / 259.1× | 250.3 / 252.8 / 257.8× |
| robust pipeline (72) | 235.9 / 239.7 / 254.8× | 239.8 / 243.5 / 248.0× | 140.9 / 142.7 / 150.3× | 139.2 / 150.0 / 150.1× |

All within 4% of the report. Python was ~6% faster today (fixed 2.58 vs 2.76 s), C# likewise; the ratios agree.

**C# original (`48e38cd`) vs final, alternating sample by sample (`ab_interleaved.py`, pinned):**

| workload | original | final | change | report |
|---|---|---|---|---|
| compat 22 items, warm (7) | 0.02288 s | 0.01690 s | −26.1% | −24.6% |
| compat 22 items, **cold** (21; plus a no-C3 build `17c07c7`) | 0.04793 s | 0.04021 s | **−16.1%** (no-C3: 0.04883 s, so C3 = −17.6% cold) | −7.5% |
| Fast d≥4, warm (7) / cold (5) | 1.3395 / 1.3658 s | 0.9713 / 0.9627 s | −27.5% / −29.5% | −28.9% / −28% |
| exact `mid3_ascension`, warm (7) | 0.06247 s | 0.03886 s | −37.8% | −38.0% |
| robust pipeline, warm (7) | 0.04937 s | 0.03682 s | −25.4% | −28.0% |
| make/undo, type rules (`run_cs_pinned.py`, 3 runs) | mid3 0.6726 / all-rules 0.9623 s | 0.3438 / 0.5605 s | −48.9% / −41.8% | −49.9% / −44.3% |
| make/undo, no type rules | 0.225–0.287 s | 0.237–0.311 s | +8% in one block order, −0..+1.6% in the reversed order: noise (spread up to 12%; code path unchanged) | −1.0..−1.4% |

No figure differs from the report by more than 15% except the cold compat suite, where my alternating figure is **better** than the report's (the report's −7.5% came from a different sampling window; both show no regression).

### Trusted speed-up range (both sides on the P-core set, CPython 3.11.4, single thread)

- ReferenceCompat fixed-depth search (identical nodes): **warm ≈ 156–166×**, cold ≈ 62–68× (JIT-dominated 40 ms micro-suite; bimodal; earlier range 52–91× still describes cold start).
- Exact solver: **≈ 250–264×** warm, ≈ 242–259× cold; of this, 1.235× is the memo-key work reduction (≈ 205–214× per `make_move`).
- Robust pipeline (72 positions): **warm ≈ 236–255×**, cold ≈ 139–150×; 1.074× of it is algorithmic.
- Phase 6 itself (C# vs C#): compat −25%, Fast −28%, exact −38%, robust −25..−28%, make/undo with 同类 rules −42..−50%.
- PyPy unmeasured (Q-6).

## 7. Findings

No CRITICAL, HIGH or MEDIUM findings.

**L6-1 (LOW) — stale statements in `CSHARP_MIGRATION.md` contradict the D-004 fix and the push.** §4/§7 are correct, but: line 126 (Python reference description: "第一张 card_id 相同的卡（不是按对象身份！见 D-004）"), line 427 (design: duplicate ids throw), line 975 (§12 quirk list: "D-004 … C# 在载入时抛异常"), lines 886 and 1001 (Q-2 still listed as an open question), line 927 (Phase 5 "改动未提交"). §12 is what a new session reads first. *Fix:* amend those lines (append "已修复/已推送" notes; do not rewrite history). Not blocking.

**L6-2 (LOW) — `PHASE6_OPTIMIZATION.md` §5 contains a garbled paragraph**: the sentence "C3 was re-applied as `290b15e`" runs into "Reverting C3 does not restore …" and the parenthesis "(code identical to `6b61630` …)" ends up after it; the header still says C3 "was reverted once", it was reverted twice (`17c07c7`/`290b15e`, `7878d1f`/`a4d86ad`). *Fix:* reflow the paragraph and the header; add this review's alternating cold numbers (§6) as the deciding evidence.

**L6-3 (LOW) — commit history noise.** Two revert/re-apply pairs (four commits with zero net code) and five follow-up doc commits are local and unpushed. Each commit is still one logical change and builds, so CLAUDE.md §8 is respected. Squashing unpushed local commits before a push is allowed but is a history rewrite: **user decision**, default = keep as is.

**L6-4 (LOW) — C7 retains a ~768 KiB memo per thread for the life of the thread.** Harmless for the single-threaded benchmark and for one request thread; under a Phase 7 server with a thread pool (or later root parallelism) memory grows with the number of threads that ever ran `Select`. *Recommendation for Phase 7:* keep the memo owned by the engine/session object (or keep `[ThreadStatic]` and document the bound).

**L6-5 (LOW) — duplicate-id coverage does not reach the `exact` and `scenario` golden kinds** (documented in §7 "已知限制"). D-019 is covered through `robust`, `reply` and `corner`; the exact solver walks instances by slot and has no id mapping, so the risk is low. Add duplicates there if Phase 7 feeds real duplicate-id hands to the exact solver.

**O6-1 (OPTIONAL)** — remaining hotspots as listed in the report §8 (同类 maintenance 42% of exact / 18% of Fast; `EvaluateState` 29% of Fast; `ResolveFlips` 11–16%). Any further work needs a structural design that reproduces D-017/D-003 exactly; the C1 probe of §5.1 is a ready-made acceptance harness for it.
**O6-2 (OPTIONAL)** — `ab_interleaved.py` should become the standard for cold A/B comparisons (block-order cold comparisons produced two false regressions in this phase); `run_cs_pinned.py` could gain an alternating mode.
**O6-3 (OPTIONAL)** — the full-state probe used in this review could be added as a unit test comparing `RecalculateTypeModifiers` against a naive reference implementation kept in the test project.

## 8. Test coverage gaps (non-blocking)

- Duplicate ids in `exact`/`scenario` golden kinds (L6-5).
- No test compares the new `RecalculateTypeModifiers` against the old per-type algorithm directly; equivalence rests on golden rule traces (25k ascension + 12.7k descension recalculations on the fresh seed) and on this review's probe (O6-3).
- M4-3 remains open (post-search hand order / display edges not compared with Python; D-015 makes hand order matter for robust selection) — `CSHARP_MIGRATION.md` §11 item 3 says it must be handled before Phase 7.

## 9. Next steps

1. Phase 7 (integration) may start after explicit user authorisation. Its goal should include: M4-3 (required before Phase 7 by §11), the D-015 decision (Fast + canonical hand order before robust selection; compare score and tie set per Q-5 (b)), L6-1/L6-2 doc fixes, and L6-4 (memo ownership).
2. Pushing `a0eb193..dce736f` needs separate user authorisation (only Step A was authorised); optional squash per L6-3 is a user decision.
3. Root parallelism remains unauthorised.
4. No remediation goal: `.claude/sonnet_review_fix_goal.md` is left as the completed A-7 record.

## 10. Commands run by the reviewer

```
git status -sb; git log --oneline; git ls-remote origin; gh pr list --state all
git show --stat <each commit>; git show a0eb193 / 3a77dbe / 6b61630 / 2c226bb / 782a3b5
git archive <commit> csharp | tar -x; dotnet build csharp/TTC.Siren.sln -c Release --no-incremental   (11 commits)
scratch copy with core/game_state.py from a0eb193^: python -m pytest -q tests/test_duplicate_hand_ids.py   -> 3 failed, 2 passed
python -m pytest -q   -> 34 passed
dotnet build csharp/TTC.Siren.sln -c Release --no-incremental; dotnet test csharp/TTC.Siren.sln -c Release   -> 0/0; 78/78
scratch copy of HEAD with seeds 31337101/31337102/31337103: dotnet test (golden regenerated)   -> 78/78
Bench search|fast|exact|endgame|makeundo on 48e38cd vs HEAD: SIG / RESULT fields diffed   -> identical
full-state make/undo probe vs Core of 48e38cd and HEAD (4,000 positions, 180,298 steps)   -> SHA-256 identical
python tools/bench/run_interleaved.py --rounds 3 --affinity auto [--cold]
python tools/bench/ab_interleaved.py --affinity auto --mode search --cold --samples 21 orig noC3 head; --mode search|fast|exact|endgame --samples 7; --mode fast --cold --samples 5
python tools/bench/run_cs_pinned.py --modes makeundo --runs 3 --affinity auto --exe <orig|head>   (both orders)
```
