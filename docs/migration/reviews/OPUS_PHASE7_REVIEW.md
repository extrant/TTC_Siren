# Opus independent review — Phase 7 goal (Step A wrap-up and push, Step B Python ↔ C# engine integration)

Date: 2026-10-10. Reviewer: Opus (independent; every claim below was re-run or re-read by the reviewer, not taken from the implementer's summary).
Goal under review: `docs/migration/goals/phase7_goal.md`. Implementer documents: `docs/migration/reports/PHASE7_INTEGRATION.md`, `docs/DEPLOY_WINDOWS.md`, `CSHARP_MIGRATION.md` §7 Phase 7.

## 1. Gate

**`PASS_WITH_NONBLOCKING_FINDINGS`**

- CRITICAL: 0. HIGH: 0. MEDIUM: 3 (M7-1, M7-2, M7-3). LOW: 3 (L7-1 … L7-3). OPTIONAL: 2 (O7-1, O7-2).
- Step A: **PASS.** Step B: **PASS**. Correctness, fallback, determinism and engine-off equivalence were all verified independently. The MEDIUM findings concern production behaviour under overload and the test harness. They do not change any result.
- Phase 8 may start **once the user authorises it**.
- **Production deployment: conditional.** Before real multi-user traffic, fix M7-1, or at least apply its configuration workaround (§9). M7-2 is only relevant at sustained high request rates.
- No remediation goal is written, because nothing blocks Phase 8. A short "Phase 7 hardening" goal for M7-1 … M7-3 and L7-1 is recommended before deployment if the user wants it (§9).

### Note on the implementer session's end

The implementer session ended with a Stop-hook loop. That session still had the **Phase 6** `/goal` condition active, so the hook kept checking a Phase 6 report against a Phase 7 session ("A hook blocked the turn from ending 9 consecutive times"). This was a stale goal, not a failure of the Phase 7 work. Start each phase's goal in a fresh session, or clear the previous goal first.

## 2. Scope and repository state

- Branch `csharp-solver-migration`, HEAD `014a2ed`, working tree clean, 11 commits ahead of `origin`.
- `git ls-remote origin`:
  - `refs/heads/csharp-solver-migration = 03b4704`
  - `refs/heads/main = eb60a02`, unchanged
  - `refs/pull/1/head = d0ef721`: PR #1, closed 2026-10-07; no new PR.
- Step A commits: `d719a59`, `03b4704` (pushed). Step B commits (local only): `3e232ec 725b917 285efde 7fa5131 82237f4 167163b 55cf4fc 5038080 9dab505 42fb7ae 014a2ed`.

## 3. Step A — PASS

- **Reviewer files committed as written** (`d719a59`). `docs/migration/reviews/OPUS_PHASE6_REVIEW.md` and `docs/migration/goals/phase7_goal.md` are unchanged since that commit. The `CSHARP_MIGRATION.md` hunks are exactly the reviewer's gate and decision edits.
- **L6-1 fixed by appended corrections** (`03b4704`), not by rewriting: §1.4 make_move description, Phase 2 design item 8, §11 item 2, §12.2 Phase 4/5 rows, §12.5 quirk list, §12.7. **L6-2 fixed** in `PHASE6_OPTIMIZATION.md`.
- **Push:** `b6e4ea4..03b4704`, fast-forward, not squashed (the C3 revert/re-apply commits are still present). `main` untouched, no force-push, no PR.
- **Every code commit builds:** I built `3e232ec 725b917 82237f4 167163b` from `git archive`, each with `dotnet build -c Release --no-incremental`: 0 warnings, 0 errors. `HEAD` was built as well.
- **No build output, golden data or recorded requests committed:**
  - The corpus is git-ignored and generated.
  - `tests/data/engine_off_expected.json` (57 KB, a regression sample) and `docs/migration/reports/phase7/*.jsonl` (raw measurements) are intended.

## 4. Step B — Python changes

- **The solver block is moved, not changed.** I compared the removed inline block with `_solve_minimax_python` line by line: identical statements, same arguments, same order. The implementer also states AST identity.
- **Only allowed Python files changed:** `ai_server.py`, the new `ai/csharp_engine.py`, the new tests, and `tools/parity/py_endgame_trace.py`. The last is a parity generator: it now locates the override block in its new function. `git diff 03b4704..HEAD` touches nothing in `core/`, `ai/ai.py`, `ai/monte_carlo.py` or `app.py`, and no analysis/text function. `solver=monte_carlo` is untouched.
- **Engine-off and fallback paths:**
  - With `TTC_CS_ENGINE=off` the client is never called and no response field is added.
  - On fallback the old function runs with the old arguments.
  - The only random-number consumer on the engine path is the fallback-candidates round trip. It saves `random.getstate()` and restores it on any later failure, so the Python path draws the same numbers. This is tested.
- **Client uses the standard library only** (`http.client`, `json`, `hashlib`, `socket`, `threading`, `urllib.parse`), so it is PyPy-compatible.
- **D-015:** the request is built from `game_state` before any Python search, in request hand order. The engine runs the Fast search on one copy of the state and the robust layer on a second copy.
- **Returned move is validated:**
  - the slot exists;
  - the card id matches (`None` maps to −1);
  - the cell is on the board and empty;
  - the card is in `get_playable_cards(rules)`, which covers 秩序/混乱 `can_use`.

  Any failure falls back with `illegal_move`.
- **M4-3 closed by D-015: agreed.** On the engine path the robust layer sees the request's hand order and display edges. It reads effective values only (`evaluate_state`, corner risk, candidate threat scoring). The display-text difference (`format_card_display`) is documented as expected.

## 5. Step B — C# engine

- **Loopback only:** `EngineOptions.Validate` rejects non-loopback addresses unless `--allow-remote` is given (tested). Kestrel runs HTTP/1 only, with a body-size limit.
- **Strict validation:**
  - owners must be `"r"`, `"b"` or null (L-9, also fixed in the shared `CardSnapshot.FromJson`);
  - unknown rule names are rejected, and card fields are range-checked;
  - `maxDepth` uses the floor, clamped to 100; below 1 or NaN means no level (L-1);
  - `maxTime`: null means no limit, ≤ 0 means already over, and a huge value cannot overflow (L-2 via `SearchLimits.DeadlineFrom`).

  This matches Python's `min(max_depth, 100)` and `elapsed > max_time`. Non-numeric values never reach the engine; the client leaves them to Python.
- **Worker pool:**
  - Each `WorkerContext` owns its Fast TT and its exact memo, and the memo is passed explicitly to `EndgameRobust.Select` (L6-4).
  - Each request builds fresh `GameState`s, so `NewGeneration(SearchContext)` binds the table to the request's own state object (H4-1 holds per request).
  - No mutable static state exists in Core/Search/Engine other than the unused Phase 6 `[ThreadStatic]` memo (checked by grep).
  - A bounded queue answers `busy` (503) when full or after the wait timeout.
- **`needs_fallback_candidates` and `handler`:**
  - The engine asks for fallback candidates only when there are unknown opponent slots, the handler exists and `EndgameScenarios.Build` reports no legal candidate. This is exactly Python's `_build_endgame_scenarios` order (`if not unknown_indices` → `if not handler` → legal enumeration → random fallback).
  - It asks before searching.
  - `handler: false` yields the base state as the only scenario (tested).
- **Health checks are effective.** In the real fault tests (§6.4), a card-database copy with one extra byte was refused with `card_db_mismatch`. The protocol-mismatch check is covered by tests.
- **Core/Search changes are limited to the allowed set:**
  - L-1, L-2, L-9;
  - per-level statistics (one clock read per completed level, never per node).

  The 78 original tests pass unchanged, including the zero-allocation tests. No `unsafe`, SIMD or root parallelism.

## 6. Verification by the reviewer

### 6.1 Build and tests

| command | result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test csharp/TTC.Siren.sln -c Release` | **149 passed**, 0 failed, 0 skipped (78 previous + 71 new) |
| `python -m pytest -q` | **126 passed** (34 previous + 92 new) |

### 6.2 Engine-off equivalence (independent)

- **Method:**
  - Pre-Phase-7 tree `03b4704`, extracted with `git archive` into a scratch directory.
  - Responses captured with Flask's test client, using the same corpus (360 requests), the same per-request `random.seed`, `PYTHONHASHSEED=0`, `max_depth` 4 and `max_time` 10^6.
  - The same capture run on HEAD with `TTC_CS_ENGINE=off`.
- **My own strict field-by-field diff:**
  - 360/360 status codes equal; 360/360 moves equal.
  - **347/360 bodies byte-identical.**
  - The other 13 (the requests with `search_progress`) differ only in `time_elapsed`, `time_remaining`, `nodes_per_second`, `total_time_seconds` and `search_efficiency`. One also differs in an intermediate wall-clock progress entry (count and `nodes_searched`/`cutoff_rate` of entries without a best move).
- **The implementer's normalised comparison:** `IDENTICAL`. **The excluded fields are timing fields only: confirmed.**

### 6.3 End-to-end comparison on a fresh corpus (independent)

I changed `E2eSeed` to 31337104 in a scratch copy, regenerated the corpus and the Python reference, and ran `EngineEndToEndTests`. It passed.

| | implementer (seed 20261010) | reviewer (seed 31337104) |
|---|---|---|
| records (robust / fallback-candidates path / no move) | 404 (209 / 16 / 13) | 415 (207 / 16 / 18) |
| search levels: engine score bit-identical to the oracle / engine move in the exact tie set | 1,392 / 1,392 of 1,392 | **1,419 / 1,419 of 1,419** |
| Python move identical / other move in the tie set / outside, explained by D-006 / **unexplained** | 1,381 / 6 / 5 / **0** | 1,410 / 4 / 5 / **0** |
| robust tables bit-identical (candidates) / scenario counts equal | 209 (1,274) / 209 | **207 (1,272) / 207** |
| final move = override decision (overridden / kept) | 96 / 113 | 77 / 130 |

While doing this I found **M7-3**: through `dotnet test` the regeneration hangs. I had to run the generator directly; it took 40 s.

### 6.4 Real faults (published engine, real Flask server code, `TTC_CS_ENGINE_BACKOFF=2`)

| scenario | response | same move as engine off |
|---|---|---|
| engine not started | `python` / `timeout` (0.86 s: on Windows a refused connect exceeds the 0.5 s connect timeout), next request `circuit_open` (0.35 s) | yes / yes |
| engine started | `csharp`, robust layer ran, move (127,(2,2)) = engine-off move | yes |
| engine killed 0.4 s into an opening request | `python` / `connection_failed` (the user waited 10.4 s: the full Python search) | — |
| engine started with a card-database copy plus one byte | `python` / `card_db_mismatch` | yes |
| good engine again after the back-off | `csharp` | — |

The breaker opened and closed as documented.

### 6.5 Concurrency determinism (independent)

`tools/phase7/determinism_check.py --workers 2,12 --concurrency 2,8,24` (the implementer used 1,8,20 workers and concurrency 1,4,16):

- engine directly: openings, mid-games and robust endgames, 346 distinct requests;
- the Flask server in front: 9 + 50 + 93 requests without unknown cards.

**27 cells, 0 differences.**

### 6.6 Latency and load (re-measured; this machine only)

**Single-request latency, pinned `0xFFF`, 3 independent runs, warm medians:**

| request | Python, report → reviewer | engine path, report → reviewer |
|---|---|---|
| openings | 10.0 s → 10.0 s (time-limited) | 0.59 / 2.49 / 0.82 s → **0.35 / 1.55 / 0.49 s** |
| mid-games 7 / 14 | 3.93 / 3.82 s → 2.35 / 2.34 s | 0.0089 / 0.0045 s → 0.0050 / 0.0030 s |

All of my numbers are 35–45% lower than the report's, **on both sides and in the same proportion** (ratios: request 7 ≈ 440× in the report vs ≈ 470× here). Python's time-limited depth on opening 27 rose from 5 to 6. This is a machine-state difference between the two sessions, not a measurement error. The report's figures are the conservative ones; this is noted as L7-2.

**Load spot check** (`run_load_matrix.py --cpu auto --workers 12 --concurrency 1,4,8,16`):

| category | report: requests/s at c = 4 / 8 / 16 | reviewer |
|---|---|---|
| openings | 2.2 / 4.1 / 5.6 | 4.7 at c = 8, 6.3 at c = 16 |
| mid-games | 93.8 / 129.6 / 157.5 | 109.9 / 151.5 / 175.7 |
| robust | 4,802 / 5,052 / 4,695 | 4,801 / 5,115 / 4,768 |

- Memory: 1,231 MiB (report) vs 1,235 MiB (reviewer).
- No errors, no `busy`, and the determinism check was identical in every cell.
- Mid-game throughput is 12–17% higher here, in line with the faster machine state above.

### 6.7 Deployment guide walk-through

- `dotnet publish csharp/src/TTC.Siren.Engine -c Release -o <dir>` gives a framework-dependent build. `runtimeconfig.json` requires `Microsoft.NETCore.App` and `Microsoft.AspNetCore.App` 10.0.
- The published `TTC.Siren.Engine.exe --port … --workers 4 --card-db …` started, `/v1/health` answered, and Flask used it. The steps in `docs/DEPLOY_WINDOWS.md` §2–§3 and §7 work as written.
- The Windows-service part (`sc.exe`) was not executed: it needs an elevated prompt and changes system configuration. Its content is consistent with `UseWindowsService`.

## 7. Findings

### M7-1 (MEDIUM) — overload turns into a cascade of Python fallbacks

- **Files / methods:**
  - `csharp/src/TTC.Siren.Engine/EngineOptions.cs` (`QueueWaitMs` = 30,000) and `WorkerPool.AcquireAsync`;
  - `ai/csharp_engine.py` `read_timeout_for` (`max_time` + 5 + 1 s) and `solve` (a `timeout` opens the circuit breaker);
  - `SolveService.Solve` (no cancellation);
  - `docs/DEPLOY_WINDOWS.md` §8 step 5.
- **Observed** (reviewer probe: published engine with `--workers 1` and the default queue wait, Flask in front, 12 concurrent openings at production settings, then 5 cheap robust requests 20 s later):
  - 6 openings were answered by the engine (4.4–15.7 s).
  - 6 hit the client read timeout (16 s) while still queued, because the engine's own `busy` would come only after 30 s. They fell back to Python and finished after **17.6–34.3 s**.
  - The timeouts opened the 30 s breaker, so **all 5 cheap requests were also answered by Python** (`circuit_open`).
  - The engine kept computing the abandoned requests.
- **Expected:**
  - Under load, a request waits for the engine, which even when queued is far faster than the Python solver.
  - A slow-but-alive engine does not trip the breaker.
  - `busy` arrives before the client gives up.
- **Impact:**
  - Exactly when the server is busiest, work moves to Python. Python is GIL-bound and about 100× slower per search, and the extra load then hits every request for 30 s.
  - Deployment-guide §8 step 5 ("a request that waits longer than it would take Python to answer is better served by the fallback") recommends the wrong direction for this solver.
  - No result is wrong; every answer is still a legal, correctly computed move.
- **Fix:**
  - Make the client read timeout exceed `QueueWaitMs` + `max_time` + margin, or derive both from one setting. The engine can also clamp the queue wait to the request's own time budget.
  - Do not open the breaker on a read timeout when the connection was established and health is OK; open it only on connection-level failures.
  - Optionally cancel abandoned requests (pass `HttpContext.RequestAborted` into the search as a stop flag).
  - Correct deployment-guide §8 step 5.
- **Regression test:**
  - A client test with a stub engine that answers after `read timeout − ε`: no breaker opens.
  - An engine test: queue wait ≤ the configured client budget.
  - The overload probe above (1 worker, 12 openings) must end with 0 `timeout` / 0 `circuit_open` fallbacks.
- **Human decision:** no. The configuration workaround is in §9.

### M7-2 (MEDIUM) — one new TCP connection per request behind Flask's threaded server

- **File / method:** `ai/csharp_engine.py`, `_local = threading.local()` / `_connection`.
- **Observed:** Flask's development server (`app.run(threaded=True)`, as in production) starts a new thread per request, so the per-thread keep-alive connection is never reused. Reviewer probe: 60 sequential `/ai_move` requests through a real Flask server left **61 client-side connections to the engine in `TIME_WAIT`**.
- **Impact:**
  - Windows keeps `TIME_WAIT` for about 120 s and has 16,384 dynamic ports by default. Sustained rates above roughly 130 requests/s can exhaust ephemeral ports.
  - That causes `connection_failed`, then the breaker (see M7-1), then everything falls back to Python.
  - It also adds a TCP handshake per request. Under PyPy, sockets of finished threads are closed only when the GC runs.
- **Fix:**
  - A process-wide pool of keep-alive `HTTPConnection`s (a lock-protected LIFO list with a maximum size) instead of `threading.local`; or
  - document a WSGI server with a fixed thread pool (for example `waitress --threads=N`) as required for production, where per-thread connections are reused.
- **Regression test:** a pytest that runs N requests on N short-lived threads against a stub server and asserts that the number of accepted connections stays ≤ the pool size.
- **Human decision:** no.

### M7-3 (MEDIUM) — regenerating the e2e golden through `dotnet test` deadlocks

- **File / method:** `csharp/tests/TTC.Siren.Tests/GoldenFiles.cs`, `Ensure` / `RunPython`. They call `proc.StandardOutput.ReadToEnd()` and only then `StandardError.ReadToEnd()`.
- **Observed:**
  - `tools/phase7/e2e_reference.py` writes about 14 KB to stderr (a `C# engine fallback (e2e_capture)` warning per request) plus its pool workers' output.
  - Under the test harness the stderr pipe fills, the generator blocks, and the harness waits on stdout forever. The reviewer's fresh-seed run hung for 35 min with idle pool workers (≈ 8 s CPU each); I killed it.
  - The same generator run directly finished in 40 s.
  - The implementer's run did not hit this, because the cached reference file already existed.
- **Impact:** any change to a bound Python source (`ai_server.py`, `core/*`, `ai/csharp_engine.py`, the generators) invalidates the e2e golden, and the next `dotnet test` hangs instead of regenerating. The other golden kinds share the code but write little to stderr.
- **Fix:**
  - Read stdout and stderr concurrently (`BeginErrorReadLine`/`BeginOutputReadLine`, or `ReadToEndAsync` on both, then `WaitForExit`), in both `Ensure` and `RunPython`.
  - Optionally silence the `ttc.engine` logger in `e2e_reference.py`.
- **Regression test:** a test-helper test that runs a Python one-liner writing 1 MB to stderr and 1 MB to stdout through the same helper and completes.
- **Human decision:** no.

### LOW

- **L7-1 — the GC setting in the engine project file has no effect.**
  - `TTC.Siren.Engine.csproj` sets `<ServerGarbageCollector>false</ServerGarbageCollector>`. The MSBuild property is `ServerGarbageCollection`, so the Web SDK default applies: the published `runtimeconfig.json` says `"System.GC.Server": true`.
  - The memory figures in the report were measured with Server GC, so they describe what actually runs.
  - The setting should be renamed (or removed) and the choice measured and documented. Server GC creates one heap per logical CPU (32 on the E5 server).
- **L7-2 — the latency figures depend on the session.** My pinned re-run is 35–45% faster on both sides with the same ratios (§6.6). The report should state that absolute latencies vary between sessions and that only ratios within one session are comparable.
- **L7-3 — engine not installed while the default is on.** On Windows a refused connection takes about 0.9 s (reported as `timeout`), so with no engine running one request every `TTC_CS_ENGINE_BACKOFF` seconds (30 s) pays about 1 s extra. This is documented, but the deployment guide should say explicitly: if the engine is not deployed, set `TTC_CS_ENGINE=off`.

### OPTIONAL

- **O7-1:** expose fallback counters per reason (and the breaker state) in the Python server's logs or a small status endpoint, so a production operator sees fallbacks without reading every response.
- **O7-2:** the engine could return `busy` immediately when the queue is predicted to exceed the request's budget (queue length × recent median service time), instead of a fixed wait.

## 8. Test-coverage gaps

- No test reproduces overload through the real Flask server and the real engine (M7-1). The load matrix used 20 workers with Flask, so it never queued long.
- No test covers connection reuse under the production threading model (M7-2).
- No test exercises golden regeneration through the harness when a generator writes a lot to stderr (M7-3).
- The Windows-service installation (`sc.exe`) was not executed by either side.
- PyPy was not tested. The client is standard-library only, but connection lifetime under PyPy's GC is unverified (relates to M7-2).

## 9. Checklist for deploying to the Windows Server (E5-2698B v3, 16 cores / 32 threads, 16 GB)

1. Install the **ASP.NET Core Runtime 10.x** (or the Hosting Bundle). `dotnet --list-runtimes` must show `Microsoft.AspNetCore.App 10.x`.
2. Publish the engine as described in `docs/DEPLOY_WINDOWS.md` §2. Use the **same** `幻卡数据库.csv` as the Python server (`--card-db` with an absolute path). Check that `/v1/health` `cardDbSha256` equals `Get-FileHash`.
3. **Until M7-1 is fixed:**
   - Make the client wait longer than the engine queues. For example, start the engine with `--queue-wait-ms 20000` and set `TTC_CS_ENGINE_READ_MARGIN=25` for Python: read timeout = `max_time` + 26 s > 20 s queue + 10 s search.
   - Do **not** follow deployment-guide §8 step 5 ("keep the queue small … fallback is better").
4. **Because of M7-2:**
   - Prefer a WSGI server with a fixed thread pool (for example `waitress-serve --threads=16 ai_server:app`) over `app.run(threaded=True)`; or
   - keep sustained traffic well below ~130 requests/s.
5. Start the engine with `--workers 16`. Then run the load matrix on the server itself (`docs/DEPLOY_WINDOWS.md` §8) and adjust; expect about 1.6 GiB of engine memory for 16 workers.
6. If the engine is not deployed on a machine, set `TTC_CS_ENGINE=off` there (L7-3).
7. Rollback is `TTC_CS_ENGINE=off` plus a restart of the Python server. Verified: the engine-off responses equal the pre-Phase-7 code.

## 10. Next steps

1. **Recommended before production traffic:** a small "Phase 7 hardening" goal (M7-1, M7-2, M7-3, L7-1, L7-3; about a day of work), reviewed like the other phases. The user decides; it is not a prerequisite for Phase 8.
2. **Phase 8** (full service port) may start after explicit authorisation. M7-1/M7-2 disappear once Flask is replaced; M7-3 should be fixed first anyway, because Phase 8 will change the bound Python sources.
3. Pushing the Step B commits needs separate authorisation.

## 11. Commands run by the reviewer

```
git status -sb; git log; git ls-remote origin; gh pr list --state all; git show --stat <each commit>; git diff 03b4704 HEAD -- ai_server.py ...
git archive <3e232ec|725b917|82237f4|167163b> csharp | tar -x; dotnet build -c Release --no-incremental          -> 0/0 each
dotnet build csharp/TTC.Siren.sln -c Release --no-incremental; dotnet test csharp/TTC.Siren.sln -c Release; python -m pytest -q   -> 0/0, 149/149, 126
git archive 03b4704 | tar -x (scratch); PYTHONHASHSEED=0 python tools/phase7/equivalence.py capture --root <old> ...; TTC_CS_ENGINE=off ... capture --root <HEAD>
strict field-by-field diff (reviewer script) + python tools/phase7/equivalence.py compare      -> 347/360 byte-identical, rest timing-only; IDENTICAL
scratch copy with E2eSeed=31337104: python tools/phase7/e2e_reference.py --corpus ... (direct, 40 s); dotnet test --filter EngineEndToEndTests   -> passed (counts §6.3)
dotnet publish csharp/src/TTC.Siren.Engine -c Release -o <scratch>; reviewer fault probe (not started / running / killed mid-request / card-db mismatch / recovery)
reviewer probes: overload (1 worker, 12 openings through Flask) -> M7-1; connection count through the threaded Flask server -> M7-2
python tools/phase7/determinism_check.py --workers 2,12 --concurrency 2,8,24      -> 27 cells, 0 differences
python tools/phase7/latency.py --affinity auto --runs 3; python tools/phase7/run_load_matrix.py --cpu auto --workers 12 --concurrency 1,4,8,16
grep of csharp/src for mutable statics, [ThreadStatic], unsafe, Parallel, Vector<
```
