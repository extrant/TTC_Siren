# TTC_Siren — Goal: Phase 6 wrap-up and push, then Phase 7 (Python service calls the C# engine)

> **STATUS: COMPLETED AND REVIEWED (2026-10-10).** Gate `PASS_WITH_NONBLOCKING_FINDINGS` (`docs/reviews/OPUS_PHASE7_REVIEW.md`). **Do not run this goal again.** The follow-up work is `.claude/phase7_hardening_goal.md`. Note that its automatic Python fallback has since been withdrawn by the user (`CSHARP_MIGRATION.md` A-20). Kept for the record only.

Run with: `/goal @.claude/phase7_goal.md`

The user authorized all of the following on 2026-10-10 (recorded in `CSHARP_MIGRATION.md` §0 row 7, §8 A-8 and §9 Q-7). Do them **in this order**, as separate logical changes:
- **Step A:** Phase 6 wrap-up (review documents, the non-blocking doc fixes L6-1/L6-2), then push all local commits.
- **Step B:** Phase 7: a local C# HTTP engine service, the Python client with automatic fallback, request-level parallelism, end-to-end verification, a load test and a Windows Server deployment guide.

Not authorized: Phase 8 (replacing Flask, porting `parse_hand`/unknown-card sampling/analysis text/Monte Carlo), root parallelism or Lazy SMP, any change to search, rule or evaluation semantics, any further optimisation work, pushing Step B commits.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely. Its rules win if anything here seems to conflict (§1 correctness, §2 BLOCKED, §5 zero allocation in the search hot path, §7 tests, §8 git).
2. Read `CSHARP_MIGRATION.md` completely, especially §0, §1.2 (the real `/ai_move` call chain), §4 (D-001, D-003, D-004, D-010, D-011, D-014, D-015, D-019), §7 Phase 4–7, §8 (A-8, A-15, A-16, A-18), §9 (Q-5, Q-7), §11 and §12.
3. Read `docs/reviews/OPUS_PHASE6_REVIEW.md` (§7 findings L6-1 … L6-5, §9 next steps) and `docs/reports/PHASE6_OPTIMIZATION.md`.
4. Read `ai_server.py` `ai_move()` (≈ lines 1999–2273) and the functions it calls for the minimax path: `parse_hand`, `_should_use_endgame_robust_mode`, `_build_endgame_scenarios` (including `handler.generate_opponent_cards` and the `if not handler` early return), `select_endgame_robust_move`, the override block, `calculate_win_probability`, `generate_move_recommendation`; and `ai/ai.py` `find_best_move_parallel` / `iterative_deepening_search`.
5. Read the C# entry points you will call: `FastSearcher.RunIterativeDeepening`, `SearchLimits`, `GameState.FromSnapshot`, `StateSnapshot`/`CardSnapshot.FromJson`, `EndgameScenarios.Build`, `EndgameRobust.ShouldUse/Select/DecideOverride`, `ExactEndgameSolver.CreateMemo`.
6. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (expect 0 warnings, 0 errors)
   - `dotnet test csharp/TTC.Siren.sln -c Release` (expect 78/78)
   - `python -m pytest -q` (expect 34 passed)
   - `git status -sb` (expect `ahead 16` of `origin/csharp-solver-migration` at `b6e4ea4`, with uncommitted `CSHARP_MIGRATION.md`, `docs/reviews/OPUS_PHASE6_REVIEW.md` and `.claude/phase7_goal.md` written by the reviewer)
7. In one short message, state Steps A and B with their acceptance criteria and the protocol you intend to implement, then begin.

## 2. Global rules

- **Git:**
  - Never touch `main`. Never force-push, rewrite pushed history, tag, release or open a PR.
  - The user authorized **one push**: at the end of Step A, push every local commit up to and including the Step A commits to `origin/csharp-solver-migration`, fast-forward only, **without squashing** (the C3 revert/re-apply commits stay as they are).
  - Step B commits stay **local**. Do not push them; the user decides after the independent review.
  - One logical change per commit, with messages stating what and why. Never commit `csharp/.vs/`, `bin/`, `obj/`, `publish/` output, `.pytest_cache/`, golden data or recorded request logs.
- **Python business code:** the only allowed changes are
  - `ai_server.py`: dispatching the minimax solver path of `ai_move()` to the C# engine with fallback (move the existing solver block into a function without changing what it does), filling `performance_stats`/`search_progress` from the engine's statistics, an additive `engine` field in the response, and an opt-in request recorder (off by default);
  - a new module for the engine client (for example `ai/csharp_engine.py`);
  - new tests under `tests/`.
  - When the engine is off, unavailable or falls back, the Python path must behave **exactly** as before (same calls, same order, same random-number consumption). Nothing else in `core/`, `ai/ai.py`, `ai/monte_carlo.py`, `app.py` or the analysis/text functions may change. `solver=monte_carlo` stays pure Python.
- **C# search code:** `TTC.Siren.Core` and `TTC.Siren.Search` are reused as they are. Allowed there: small input-validation helpers and the findings listed in §4.6. No change to rules, evaluation, ordering or search semantics; the existing parity tests must stay green unchanged.
- **Dependencies:** C# may use the ASP.NET Core shared framework (`Microsoft.AspNetCore.App`, part of the .NET 10 runtime install) and, if you add Windows-service hosting, `Microsoft.Extensions.Hosting.WindowsServices`; tests may use `Microsoft.AspNetCore.Mvc.Testing`. Python uses the **standard library only** for the client (`http.client`/`urllib`, `json`), because the production interpreter may be PyPy. Nothing else.
- **Tests:** never skip, delete or weaken tests. Random tests use fixed seeds that appear in failure messages.
- **Timing:** single-request latency tables follow the pinned protocol (`--affinity auto`, mask recorded, warm/cold, ≥ 3 independent runs, min/median/max). The load test is a concurrency measurement, so run it with the full CPU set **and** with the P-core set `0xFFF`, and record which.
- **Stop conditions:** if any parity test fails, or the end-to-end comparison (§4.4) shows a difference that is not explained by Q-5 (b) ties, D-006 or D-015, stop. Record it in `CSHARP_MIGRATION.md` §9 BLOCKED (reproduction request, Python output, C# output, hypotheses ruled out) and report to the user. Also stop and report if a destructive or remote git operation other than the authorized push seems necessary.

## 3. Step A — Phase 6 wrap-up and the authorized push

- [ ] Commit the reviewer's files as they are: `docs/reviews/OPUS_PHASE6_REVIEW.md`, the gate and decision edits in `CSHARP_MIGRATION.md`, and `.claude/phase7_goal.md` ("docs: Phase 6 review, gate and Phase 7 authorization").
- [ ] **L6-1:** fix the stale statements in `CSHARP_MIGRATION.md` that contradict the D-004 fix and the push. The review lists them (§1.4 ≈ line 126 "第一张 card_id 相同的卡（不是按对象身份！见 D-004）", §7 Phase 2 design item 8 ≈ line 427 "同一手牌内 card_id 重复（D-004）时抛出明确的异常", §12.5 ≈ line 976 "D-004（手牌内重复 id → C# 在载入时抛异常）", §11 item 2 and §12.7 still listing Q-2 as open, §12.2 Phase 5 row "改动未提交"; line numbers are approximate, search for the quoted text). Append a correction marker instead of deleting history (for example "（已修复，见 D-004 FIXED-BOTH）").
- [ ] **L6-2:** in `docs/reports/PHASE6_OPTIMIZATION.md`, reflow the garbled paragraph of §5 (the sentence "C3 was re-applied as `290b15e`" runs into the next one), correct the header ("reverted once" → twice: `17c07c7`/`290b15e` and `7878d1f`/`a4d86ad`), and add the reviewer's alternating cold measurement (`OPUS_PHASE6_REVIEW.md` §6: original 0.04793 s, without C3 0.04883 s, final 0.04021 s) as the deciding evidence. One commit "docs: Phase 6 report and migration doc corrections (L6-1, L6-2)".
- [ ] Build (0 warnings), `dotnet test` 78/78, pytest 34.
- [ ] Push to `origin/csharp-solver-migration`, fast-forward only. Confirm with `git status -sb` and `git ls-remote origin` that the remote branch equals your last Step A commit and `main` is unchanged. Report the hashes.

## 4. Step B — Phase 7

### 4.1 Architecture (decided by the user; do not change it)

```
client ──HTTP──> Flask ai_server (Python: parse board/hands/rules, unknown-card sampling, analysis text)
                     │  minimax path only
                     ├──HTTP 127.0.0.1──> TTC.Siren.Engine (C#, one process, worker pool)
                     │                      Fast iterative deepening (max_depth, max_time as sent)
                     │                      + robust endgame: scenarios, Select, override (if requested)
                     └── on any engine problem: the unchanged Python solver path (fallback)
```

- New project `csharp/src/TTC.Siren.Engine` (ASP.NET Core minimal API, Kestrel, `net10.0`, `TreatWarningsAsErrors`), added to `TTC.Siren.sln`. **Framework-dependent** publish (the server installs the .NET 10 / ASP.NET Core runtime); no self-contained publish.
- Listens on **127.0.0.1 only** (configurable port; refuse to start on a non-loopback address unless an explicit `--allow-remote` flag is given). Configuration by command line and/or `appsettings.json`: port, worker count, queue limit, queue wait timeout, card-database path.
- **Request-level parallelism:** a fixed pool of N worker contexts (default `Environment.ProcessorCount`, configurable). Each worker owns its `FastTranspositionTable` (default capacity), its exact-solver memo (`ExactEndgameSolver.CreateMemo()`, passed explicitly to `EndgameRobust.Select` instead of relying on the `[ThreadStatic]` table — review L6-4), and its scratch buffers. A request builds its own `GameState` (`FromSnapshot`), acquires a worker, searches synchronously, releases it. Nothing mutable is shared between requests. Requests beyond N wait in a bounded queue; when the queue is full or the wait times out, answer HTTP 503 with reason `busy`.
- The card database is loaded once at start-up. `GET /v1/health` returns the protocol version, the engine build (assembly version and git commit if available), worker/queue state and the **SHA-256 of the card database file** it loaded.

### 4.2 Protocol (`POST /v1/solve`, JSON)

Design it so that a request is a fully determined position; the engine never samples anything. Required content:
- `protocol` (integer version), the state in the existing golden snapshot format (`{"cur", "board", "hands"}` with the 13-field card arrays that `CardSnapshot.FromJson` reads, hands **in the request's hand order**), the rule set, `aggressive`, `maxDepth`, `maxTime`;
- the robust-endgame inputs when Python's `_should_use_endgame_robust_mode` is true: `aiPlayer`, the opponent index, `generated[player][slot]` (Python `_is_generated_unknown_card`), `sampleCount` (Python's formula), `npcIds`, whether Python's unknown-card handler exists (`if not handler: return [base_state]`), and optional `fallbackCandidates`.
- Response: `status` (`ok`, `needs_fallback_candidates`, `invalid_request`, `busy`, `error`), the chosen move as **(hand slot index in the request's hand order, row, col)** plus its `cardId` for a cross-check, the standard search move and score (score as a double and as hex bits), per-depth statistics (depth, best move, score, nodes, elapsed seconds), totals (nodes, depth completed, TT hits, cutoffs, elapsed), whether the robust layer ran, the robust candidate table (move, `final_score`, `safety_ratio`) and whether it overrode the standard move, and the engine's own timings.
- **Fallback candidates:** when the robust layer is requested and `EndgameScenarios.Build` reports `RequiresFallback`, the engine answers `needs_fallback_candidates` **before** searching. Python then calls its handler exactly as `_build_endgame_scenarios` would, and resends the request once with `fallbackCandidates`. Random fallback is outside parity (A-7).
- **Input validation (strict):** reject unknown owners instead of treating them as blue (review L-9), reject unknown rule names, card values outside the format, hands over capacity, a side to move that has no hand, etc., with `invalid_request` and a message. `maxDepth` and `maxTime` must behave like Python for every value Python accepts: Python uses `min(max_depth, 100)`; `max_time <= 0` makes Python's ID stop before depth 1 completes. Clamp `maxDepth` to the C# maximum without changing results (review L-1: `maxDepth > 128` must not throw) and reproduce Python's behaviour for zero, negative and NaN `maxTime` (review L-2). Write tests for each edge case and document the mapping.

### 4.3 Python client and dispatch

- Engine client module (standard library only): configuration by environment variables, for example `TTC_CS_ENGINE` = `on` (**default**) | `off`, `TTC_CS_ENGINE_URL` (default `http://127.0.0.1:<port>`), connect timeout (short, for example 0.5 s), read timeout = `max_time` + margin (configurable). One keep-alive connection per thread.
- At first use (and after a failure), check `GET /v1/health`: protocol version must match and the engine's card-database SHA-256 must equal the hash of the CSV the Python server loaded; otherwise fall back.
- **Automatic fallback** to the unchanged Python solver path on: connection refused, timeout, HTTP error, `busy`, `error`, `invalid_request`, malformed JSON, protocol/card-database mismatch, or a returned move that is not legal in Python's `game_state` (card at that slot does not exist, card id differs, cell occupied, or the card is not playable under 秩序/混乱). After a connection-level failure, skip the engine for a short back-off period (circuit breaker, configurable, for example 30 s) so every request does not pay the timeout. Log every fallback with its reason.
- In `ai_move()` (minimax path only): build the request from `game_state` **before** any Python search runs, so the engine sees the request's hand order. This is the user's D-015 decision: Fast search + robust selection on the canonical (request-order) hand. Map the returned slot to the Python card object `game_state.current_player.hand[slot]`, verify it, then continue with the existing analysis code unchanged (`analyze_opponent_hand`, `calculate_win_probability`, `generate_move_recommendation`, the predicted board).
- `performance_stats` is filled from the engine statistics when the engine answered (keep the same keys; add `engine: "csharp" | "python"` and, on fallback, `fallback_reason`). `search_progress`/`search_summary` (when `show_search_progress` is set) are filled from the per-depth statistics. Python's global `SEARCH_STATS` is not touched on the engine path.
- **Opt-in request recorder** (off by default, for example `TTC_RECORD_REQUESTS=<directory>`): writes the raw `/ai_move` JSON and the Python random seed state needed to replay it, so the user can collect real traffic on the server for later replay. Document that it stores user requests.
- Known, expected difference to document (not a bug): on the engine path, Python never runs its own search, so the D-001 hand reordering and the D-003 display edges left by Python's search do not occur. Display edges feed the move-description text (`format_card_display`), so the text can differ from the Python path for the same move.

### 4.4 Verification

- [ ] **Engine-off equivalence.** With `TTC_CS_ENGINE=off`, `/ai_move` responses must equal those of the pre-Phase-7 code. Generate a request corpus (§4.5), capture the responses of the Step A commit (in a separate worktree) through Flask's test client with fixed random seeds and **deterministic search settings** (`max_depth` ≤ 4, `max_time` = 10^6), then compare with the new code. Exclude only fields that are timing-dependent by nature (elapsed times, nodes per second) and list them. Everything else must be identical.
- [ ] **Engine-on end-to-end comparison** on the corpus at fixed depth (no time limit), for depths 1–4 and for the full depth of positions with ≤ 6 empty cells. Per request:
  1. Standard search: the engine's root score must be bit-identical to the C# oracle (ReferenceCompat, TT off) on the same state. The engine's move must lie in the exact tie set (all root moves whose exact value has the same bits as the best). Compute the tie set in the test harness, not in production code.
  2. Python's standard move (Python search on a **copy** of the state, so the original keeps the request order) must lie in the same tie set. Otherwise check whether D-006 explains it (Python without TT, as in the Phase 4 tie reports). Count the cases per category. An unexplained case is BLOCKED.
  3. Robust layer (when active): run Python's `_build_endgame_scenarios` + `select_endgame_robust_move` on the **request-order** state (with the same seed for any fallback sampling) and compare its candidate table with the engine's, bit for bit (`final_score`, `safety_ratio`, per move). Then the override decision computed with Python's table **and the engine's standard move** must equal the engine's final move.
  4. Record per category: identical move, different move in the tie set, D-006, robust override cases.
  With this, M4-3 is closed by the D-015 decision: record that the post-search hand order and display edges are no longer an input of the engine path, and that the robust layer reads only effective values (`evaluate_state`), not display edges.
- [ ] **Fallback tests (pytest):** a fake engine (local stub server or monkeypatched client) for every fallback reason in §4.3. Each one must return exactly the Python-path result with `engine: "python"` and the right `fallback_reason`. Also cover the circuit breaker and the `needs_fallback_candidates` round trip.
- [ ] **Engine tests (xUnit):** protocol round trip; strict validation and the `maxDepth`/`maxTime` edge cases; duplicate card ids in a hand (D-004/D-019) through the protocol; **determinism under concurrency** (at least 200 different requests sent concurrently with more requests than workers: every response equals the response of a sequential run on a fresh engine, field by field except timings); no cross-request leakage (TT/memo generation binding, H4-1 rule); `busy` behaviour when the queue is full; health endpoint content. The existing zero-allocation tests stay green (the HTTP layer may allocate; the search hot path must not).

### 4.5 Request corpus

At least 300 `/ai_move` requests, generated deterministically (seeded) and stored as test data (small JSON, committed if under about 1 MB, otherwise generated on demand like the golden data). Cover: both owners; `currentPlayer` 0 (inferred), 1 and 2; every rule, including 同类强化/同类弱化 (≥ 30 requests), 秩序/混乱, 选拔, 全明牌/三明牌; unknown opponent cards (0, 1, 2 and 3+); NPC id lists; duplicate card ids in one hand; aggressive and standard mode; openings (empty board, both sides to move) and mid-games; **≥ 100 requests that trigger the robust endgame**, of which **≥ 5 take the fallback-candidates path**.

### 4.6 Load test and latency

- [ ] `tools/bench/load_test.py` (standard library only): replays the corpus against (a) the engine directly and (b) the full Flask `ai_server` with the engine on, with a configurable concurrency. It reports throughput (requests/s), latency p50/p95/p99/max and the fallback count, and checks that no response changes with concurrency.
- [ ] Run it on this machine with production settings (`max_depth` 10, `max_time` 10) at concurrency 1, 2, 4, 8, 12, 16 and engine workers 1, 4, 8, 12, 20. Use the full CPU set and the P-core set `0xFFF`, and record both. Report separately: opening requests (empty board), mid-game, and robust-endgame requests. Also report the Python-side time per request on the engine path (parsing, sampling, analysis text), because Flask's `threaded=True` server and the GIL may become the bottleneck.
- [ ] Single-request latency table (pinned protocol): Python solver vs engine, same requests, production settings.
- [ ] Give the user a short procedure to run the same load test on the Windows Server (E5-2698B v3, 16 cores / 32 threads, 16 GB) and to choose the worker count. State the expected memory use (workers × TT size + runtime). Do not extrapolate numbers to that machine as if measured.

### 4.7 Deployment guide

`docs/DEPLOY_WINDOWS.md`: install the .NET 10 runtime with ASP.NET Core (the Windows Hosting Bundle or the ASP.NET Core Runtime installer), `dotnet publish csharp/src/TTC.Siren.Engine -c Release` (framework-dependent), configuration (port, workers, queue, card-database path — it must be the same CSV the Python server loads), running it as a Windows service (with `sc.exe`, or `UseWindowsService` if you added that package), firewall (loopback only), the Python environment variables, the health check, start order, how to switch the engine off (`TTC_CS_ENGINE=off`), and how to roll back. Note that Flask's built-in server (`app.run(threaded=True)`) is a development server; you may recommend a production WSGI server for Windows (for example `waitress`), but do not change how the Python server is started in this phase.

## 5. Documentation

- `docs/reports/PHASE7_INTEGRATION.md`: architecture, protocol (with an example request and response), the fallback matrix, the D-015/M4-3 treatment, the end-to-end comparison counts per category, the engine-off equivalence result, the concurrency determinism result, the load-test and latency tables (with masks), memory, known limits and remaining work for Phase 8.
- `CSHARP_MIGRATION.md`: §0 row 7 **DONE, awaiting independent review** (do not set any review gate yourself); §7 Phase 7 delivery record; D-015 and M4-3 status; §10 measurements only; §11/§12 handoff (how to build, run, test and deploy the engine). Append and adjust only; do not rewrite history.
- `CLAUDE.md` §10: add the `TTC.Siren.Engine` project to the directory list (no other rule change).

## 6. Acceptance criteria (all required)

- [ ] Step A done and pushed; Step B commits local only; `main` unchanged; no PR.
- [ ] `dotnet build -c Release --no-incremental`: 0 warnings, 0 errors. `dotnet test`: all pass (78 + new). `python -m pytest -q`: all pass (34 + new). No test skipped or weakened.
- [ ] Engine-off equivalence: identical responses on the whole corpus (excluded timing fields listed).
- [ ] Engine-on comparison: 0 unexplained differences; counts per category reported; M4-3 recorded as closed by D-015.
- [ ] Every fallback reason tested; concurrency determinism proven; strict validation tested.
- [ ] Load test and latency tables recorded with CPU sets; Windows Server procedure and memory estimate documented.
- [ ] `docs/DEPLOY_WINDOWS.md` and `docs/reports/PHASE7_INTEGRATION.md` written; `CSHARP_MIGRATION.md` updated (no self-set gate).

## 7. Final report

Report briefly: Step A hashes and push confirmation; the engine design (project, endpoints, protocol version, worker pool); every Python change (files, functions); the fallback matrix and its tests; the end-to-end comparison counts per category; the engine-off equivalence result; the load-test and latency headline numbers with CPU sets; exact build, test and pytest counts; `git status -sb`; and that Phase 8 has not been started and Step B commits are not pushed.
