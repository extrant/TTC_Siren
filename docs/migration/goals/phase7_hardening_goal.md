# TTC_Siren — Goal: push Phase 7, then Phase 7 hardening (engine-only minimax path, connection pool, harness deadlock)

> **STATUS: COMPLETED AND REVIEWED (2026-10-10).** Gate `PASS_WITH_NONBLOCKING_FINDINGS` (`docs/migration/reviews/OPUS_PHASE7_HARDENING_REVIEW.md`). **Do not run this goal again.** If a session still has this goal active (for example a Stop hook asks for its final report), the goal is stale: clear it or start a new session, and never redo this work. Kept for the record only.

Run with: `/goal @docs/migration/goals/phase7_hardening_goal.md` — **in a new session**. If an older goal (Phase 6 or Phase 7) is still active in a session, clear it first or open a new one. Those goals are finished, and their Stop-hook condition would block this session (that happened at the end of Phase 7).

The user authorized the following on 2026-10-10 (recorded in `CSHARP_MIGRATION.md` §0 row 7, §8 A-20 and §9 Q-8). Do them **in this order**, as separate logical changes:
- **Step A:** commit the reviewer's files, then push all local commits.
- **Step B:** Phase 7 hardening: the findings M7-1, M7-2, M7-3, L7-1, L7-2 and L7-3 of `docs/migration/reviews/OPUS_PHASE7_REVIEW.md`, with the user's decision **A-20: no automatic Python fallback**.

Not authorized: Phase 8 (replacing Flask, porting parsing/sampling/analysis text/Monte Carlo), root parallelism, search optimisation, any change to rule, evaluation, ordering or search semantics, deleting Python solver code, pushing Step B commits.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely; its rules win if anything here seems to conflict.
2. Read `CSHARP_MIGRATION.md`: §0, §7 Phase 7 (delivery record), §8 A-8 and **A-20**, §9 Q-7 and **Q-8**, §11, §12.
3. Read `docs/migration/reviews/OPUS_PHASE7_REVIEW.md` completely, especially §6.4, §6.6, §7 (M7-1 … M7-3, L7-1 … L7-3) and §9. Also read `docs/migration/reports/PHASE7_INTEGRATION.md` and `docs/DEPLOY_WINDOWS.md`.
4. Read the code you will change: `ai_server.py` (`ai_move`, `_solve_minimax_with_engine`, `_engine_move`, `_engine_outcome`, `_solve_minimax_python`); `ai/csharp_engine.py`; `csharp/src/TTC.Siren.Engine/*`; `csharp/tests/TTC.Siren.Tests/GoldenFiles.cs`; `tools/phase7/e2e_reference.py`; the tests `tests/test_csharp_engine_client.py`, `tests/test_engine_integration.py`, `tests/test_engine_off_equivalence.py`; and `app.py` (how the server is started today).
5. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (expect 0 warnings, 0 errors)
   - `dotnet test csharp/TTC.Siren.sln -c Release` (expect 149/149)
   - `python -m pytest -q` (expect 126 passed)
   - `git status -sb`: expect `ahead 11` of `origin/csharp-solver-migration` at `03b4704`. Expect uncommitted reviewer changes in `CSHARP_MIGRATION.md`, `docs/migration/reviews/OPUS_PHASE7_REVIEW.md`, `docs/migration/goals/phase_goal.md`, `docs/migration/goals/phase6_goal.md`, `docs/migration/goals/phase7_goal.md` and `docs/migration/goals/phase7_hardening_goal.md`.
6. In one short message, state Steps A and B with their acceptance criteria, then begin.

## 2. Global rules

- **Git:**
  - Never touch `main`. Never force-push, rewrite pushed history, tag, release or open a PR.
  - The user authorized **one push**: at the end of Step A, every local commit up to and including the Step A commit goes to `origin/csharp-solver-migration`, fast-forward, not squashed.
  - Step B commits stay **local**.
  - One logical change per commit. Never commit `bin/`, `obj/`, `.vs/`, `publish/` output, `.pytest_cache/`, golden data, corpora or recorded requests.
- **Python:**
  - Allowed changes: `ai_server.py` (the minimax dispatch and the error responses only), `ai/csharp_engine.py`, `tools/phase7/*`, tests, and one new optional launcher for waitress (§4.2).
  - Do **not** delete or change the Python solver (`ai/ai.py`, `core/`, `_solve_minimax_python`, the robust/endgame functions). It stays the parity reference (CLAUDE.md §1.2) and the manual `TTC_CS_ENGINE=off` path.
  - `app.py`, `solver=monte_carlo` and the analysis/text functions do not change.
- **C#:**
  - `TTC.Siren.Core` and `TTC.Siren.Search` do not change.
  - The engine (`TTC.Siren.Engine`) and the tests may change.
  - No change in the search hot path: abandoned requests are dropped **before** a search starts, never by adding checks inside the search.
- **Tests:** never skip or weaken tests. The tests that assert the **automatic fallback** assert behaviour the user has withdrawn (A-20). Replace them with tests of the new behaviour, one for each former fallback reason, and record the reason in the commit message and in the report (CLAUDE.md §7). The engine-off equivalence tests stay unchanged and must still pass.
- **Timing:** measurements follow the pinned protocol (mask, warm/cold, ≥ 3 independent runs, min/median/max); concurrency measurements record the CPU set. Compare only within one session (review L7-2).
- **Stop conditions:**
  - A parity, end-to-end or equivalence test fails without an explanation: stop, record it in `CSHARP_MIGRATION.md` §9 BLOCKED, report to the user.
  - A destructive or remote git operation other than the authorized push seems necessary: stop and report.

## 3. Step A — reviewer files and the authorized push

- [ ] One docs commit with the reviewer's files exactly as they are: `docs/migration/reviews/OPUS_PHASE7_REVIEW.md`; the `CSHARP_MIGRATION.md` edits (Phase 7 gate, A-20, Q-8, row 7); the status banners in `docs/migration/goals/phase_goal.md`, `docs/migration/goals/phase6_goal.md` and `docs/migration/goals/phase7_goal.md`; and this file.
- [ ] Build (0 warnings), `dotnet test` 149/149, pytest 126.
- [ ] Push to `origin/csharp-solver-migration`, fast-forward only. Confirm with `git status -sb` and `git ls-remote origin` that the remote branch equals the Step A commit and that `main` is still `eb60a02`. Report the hashes.

## 4. Step B — hardening

### 4.1 A-20 and M7-1: the engine is the only minimax solver; overload waits instead of cascading

- **No automatic fallback.** When the engine is on (the default), every engine problem makes `/ai_move` answer **HTTP 503** with a JSON body. The problems are: connection refused/reset, timeout, `busy`, `error`, `invalid_request`, malformed response, protocol or card-database mismatch, or an illegal move.
  - The body keeps the existing error form (`{"error": "<readable message>"}`) and adds `"engine": "csharp"` and `"engine_error": "<reason code>"`.
  - The Python solver is **never** run on that path.
  - Log the failure at error level. When `TTC_RECORD_REQUESTS` is set, record the request.
  - The fallback-candidates round trip (`needs_fallback_candidates` → `handler.generate_opponent_cards` → second call) is **not** a solver fallback; keep it unchanged.
- **Manual switch:** `TTC_CS_ENGINE=off` (set by the operator, never automatically) runs the unchanged Python solver, exactly as now. The engine-off equivalence tests must still pass.
- **Remove the circuit breaker.** Without a fallback, a breaker would turn a short glitch into 30 s of errors. Keep the health verification (protocol version and card-database SHA-256): verify on first use and again after any connection-level failure. If verification fails, answer 503 with `engine_error` (`protocol_mismatch` or `card_db_mismatch`).
- **Waiting instead of timing out:**
  - `GET /v1/health` reports the engine's queue wait (`queueWaitMs`). The client's read timeout becomes `max_time` + the engine's queue wait + a margin (the margin stays configurable).
  - A request that queues in the engine is therefore answered by the engine and never times out on the client first.
  - `busy` (503 from the engine) happens only when the queue is full. It becomes a 503 `busy` to the game client.
- **Abandoned requests:** when a worker becomes free, the engine skips a queued request whose HTTP connection has already been aborted (`HttpContext.RequestAborted`), without searching. Running searches are not interrupted.
- **Default queue settings:** choose defaults that fit the new behaviour (a longer queue limit is reasonable now) and document them. **Correct `docs/DEPLOY_WINDOWS.md` §8 step 5**, whose advice "a request that waits longer … is better served by the fallback" no longer applies.
- **Tests (replace the fallback tests):**
  - For every former fallback reason: `/ai_move` answers 503 with the right `engine_error`, and the Python solver is not called (assert that `find_best_move_parallel` and `select_endgame_robust_move` are never invoked).
  - The `needs_fallback_candidates` round trip still works.
  - Engine-off still equals the pre-Phase-7 responses.
  - **An automated reproduction of the review's overload probe:** a real engine with 1 worker and the real Flask server, more concurrent opening requests than the old read timeout could absorb. Every request must be answered by the engine, with 0 client timeouts and 0 Python solver calls. When the queue limit is exceeded, the excess requests get 503 `busy`.
  - The test may use small search limits so it runs in reasonable time, but it must still create a queue longer than `max_time` + the old margin.

### 4.2 M7-2: a process-wide connection pool (plus an optional waitress launcher)

- Replace the per-thread connection (`threading.local`) in `ai/csharp_engine.py` with a **process-wide, thread-safe pool** of keep-alive `HTTPConnection`s. Use a LIFO with a configurable maximum size, for example `TTC_CS_ENGINE_POOL_SIZE`. It must also work when every request runs on a new thread (Flask's `threaded=True`) and under PyPy (close connections explicitly; never rely on GC).
  - A connection that fails is discarded.
  - A reused idle connection that the server has closed is retried once on a fresh connection.
  - If the pool is empty, open a new connection; return it afterwards unless the pool is full.
- **Tests:**
  - A stub server that counts accepted connections: N requests on N short-lived threads accept at most the pool size.
  - **Repeat the review's probe:** 60 sequential `/ai_move` requests through the real threaded Flask server leave at most the pool size of client connections to the engine (count with `netstat`).
- **Optional launcher:** a new script, for example `serve_waitress.py` at the repository root. It starts `ai_server.app` with waitress (fixed thread pool, configurable threads, host and port) after the **same initialisation `app.py` performs** (read `app.py`; reuse its code paths, do not duplicate business logic).
  - If waitress is not installed, it exits with a clear message.
  - `app.py` and its behaviour do not change.
  - Document it in `docs/DEPLOY_WINDOWS.md` (`pip install waitress`) as the recommended production front end.

### 4.3 M7-3: golden regeneration must not deadlock

- In `GoldenFiles.cs`, `Ensure` and `RunPython` must read stdout and stderr **concurrently** (asynchronous reads of both streams, then `WaitForExit`), so a generator that writes a lot to either stream can never block.
- Regression test: run, through the same helper, a Python one-liner that writes at least 1 MB to stdout and 1 MB to stderr. It must complete.
- Adapt `tools/phase7/e2e_reference.py` to the new dispatch. It recorded the payload by replacing the engine call with a recorder and relying on the fallback; without a fallback it needs another way to capture the payload the server sends. Keep its records equivalent.
- **Proof:** delete the cached e2e reference and the corpus, then run `dotnet test`. The e2e golden must regenerate through the harness and pass. Report the wall time.

### 4.4 L7-1: the GC setting

- `TTC.Siren.Engine.csproj` has `<ServerGarbageCollector>`, which MSBuild ignores. The property is `ServerGarbageCollection`, so Server GC is active today.
- Decide with data, following the CLAUDE.md §6 process:
  - Measure Server GC against workstation GC: the engine's load matrix subset (`--workers 12` and the expected server count 16, concurrency 1/4/8/16, openings, mid-games and robust endgames), throughput, p95 and engine memory, ≥ 3 runs, the P-core set and all CPUs.
  - Set the property explicitly to the better choice.
  - Record the table in the report and `CSHARP_MIGRATION.md` §10.
- The published `runtimeconfig.json` must show the chosen value.

### 4.5 L7-2 and L7-3: documentation

- `docs/migration/reports/PHASE7_INTEGRATION.md` (append a dated note, do not rewrite): absolute latencies vary between sessions (the reviewer measured 35–45% lower on both sides); only ratios and comparisons within one session are meaningful.
- `docs/DEPLOY_WINDOWS.md`:
  - the engine is now **required** for the minimax solver;
  - what the game client receives on engine errors (503 + `engine_error`);
  - the monitoring hint (log lines, health endpoint);
  - on a machine without the engine, set `TTC_CS_ENGINE=off`;
  - the new queue/timeout defaults;
  - the waitress launcher and the GC choice.

## 5. Documentation

- New `docs/migration/reports/PHASE7_HARDENING.md`:
  - each finding with what changed and its tests;
  - the new error matrix (replacing the fallback matrix);
  - the overload result before and after;
  - the connection count before and after;
  - the harness regeneration proof;
  - the GC measurements;
  - known limits.
- `CSHARP_MIGRATION.md` (append and adjust only; no self-set gate):
  - §0 row 7: "Phase 7 hardening DONE, awaiting independent review";
  - §7 Phase 7: a hardening delivery record;
  - §10: measurements only;
  - §11/§12: handoff (the engine is required; the manual switch; how to start with waitress).

## 6. Acceptance criteria (all required)

- [ ] Step A committed and pushed (fast-forward); Step B commits local; `main` unchanged; no PR.
- [ ] Build: 0 warnings, 0 errors. `dotnet test`: all pass. pytest: all pass. Every replaced fallback test is listed with its replacement and the reason (A-20). No other test was weakened.
- [ ] With the engine on, no code path runs the Python solver. Every engine problem gives 503 + `engine_error`. `TTC_CS_ENGINE=off` still equals the pre-Phase-7 responses on the whole corpus.
- [ ] Overload reproduction: every request is answered by the engine, with 0 client timeouts; queue overflow gives 503 `busy`. Abandoned queued requests are not searched (tested).
- [ ] The connection count with the threaded Flask server is at most the pool size.
- [ ] The e2e golden regenerates through `dotnet test` (cache deleted) and passes with 0 unexplained differences.
- [ ] The GC choice is backed by a measured table; `runtimeconfig.json` matches it.
- [ ] Docs updated (`PHASE7_HARDENING.md`, `PHASE7_INTEGRATION.md` note, `DEPLOY_WINDOWS.md`, `CSHARP_MIGRATION.md`).

## 7. Final report

Report briefly:
- the Step A hash and push confirmation;
- every change per finding, with its commit hash;
- the replaced tests and their replacements;
- the error matrix;
- the overload, connection-count and harness-regeneration results;
- the GC table and choice;
- exact build, test and pytest counts, and `git status -sb`;
- that Phase 8 has not been started and that Step B commits are not pushed.
