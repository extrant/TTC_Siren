# Opus independent review — Phase 7 hardening (A-20 engine-only minimax path, M7-1 … M7-3, L7-1 … L7-3)

Date: 2026-10-10. Reviewer: Opus. This review is independent: every claim was re-run or re-read by the reviewer.
Goal under review: `docs/migration/goals/phase7_hardening_goal.md`. Implementer documents: `docs/migration/reports/PHASE7_HARDENING.md`, `docs/DEPLOY_WINDOWS.md`, `CSHARP_MIGRATION.md` §7 "Phase 7 加固交付记录".

## 1. Gate

**`PASS_WITH_NONBLOCKING_FINDINGS`**

- **Counts:** CRITICAL 0, HIGH 0, MEDIUM 0, LOW 3 (L8-1 … L8-3), OPTIONAL 2 (O8-1, O8-2).
- **Previous findings:** M7-1, M7-2, M7-3, L7-1, L7-2 and L7-3 are **RESOLVED**, each verified independently (§4–§6). M7-2 carries one caveat: the real waitress server was not run (§9).
- **A-20 is implemented faithfully.** With the engine on, no code path runs the Python solver. Every engine problem is HTTP 503 (400/500 for request or client defects) with `engine_error`. The manual `TTC_CS_ENGINE=off` still reproduces the pre-Phase-7 responses.
- **Phase 8** may start once the user authorises it.
- **Production deployment:** YES, under the conditions in §8.
- No remediation goal is needed.

## 2. Scope and repository state

- **Branch:** `csharp-solver-migration`, HEAD `5ad12f8`, working tree clean, 7 commits ahead of `origin`.
- **`git ls-remote origin`:** `refs/heads/csharp-solver-migration = 45a9b28` (the Step A commit), `main = eb60a02` (unchanged). The only PR is PR #1, which is closed; there is no new one.
- **Step A:** `45a9b28` was pushed as a fast-forward of `03b4704..45a9b28`. The intermediate Phase 7 commits were kept, not squashed. It contains the reviewer's files exactly as written: `docs/migration/reviews/OPUS_PHASE7_REVIEW.md`, the `CSHARP_MIGRATION.md` edits, the three goal-file banners and `docs/migration/goals/phase7_hardening_goal.md`. None of these files changed afterwards (`git diff 45a9b28 HEAD -- .claude docs/migration/reviews` is empty).
- **Step B (local only):**
  - `e35e8c3` (M7-3)
  - `262ba77` (engine, M7-1)
  - `76eafcc` (Python, A-20/M7-1/M7-2)
  - `9e60222` (pool tests)
  - `93b0da1` (waitress launcher)
  - `db8bab8` (GC, L7-1)
  - `5ad12f8` (docs)
- **Every code commit builds:** `e35e8c3`, `262ba77`, `76eafcc` and `db8bab8` were each built from `git archive` with `dotnet build -c Release --no-incremental`: 0 warnings, 0 errors.
- **No artifacts committed:** no build output, golden data, corpus or recorded request is tracked. `docs/migration/reports/phase7/gc/*.jsonl` is raw measurement data, as intended.
- **Unchanged code:** `TTC.Siren.Core`, `TTC.Siren.Search`, `app.py`, `core/` and `ai/ai.py` are unchanged (`git diff 45a9b28 HEAD --stat` on them is empty). `_solve_minimax_python` is unchanged apart from a comment above it.

## 3. Correctness (reviewer runs)

| command | result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test csharp/TTC.Siren.sln -c Release` | **155 passed**, 0 failed, 0 skipped |
| `python -m pytest -q` | **152 passed** |
| Clean `git archive HEAD` copy with **no golden data and no corpus**, `E2eSeed` changed to 31337105, full `dotnet test` | **155 passed in 741 s**. Every golden kind and the e2e reference were regenerated through the harness; nothing hung. |
| End-to-end counts on that fresh seed | 415 records (robust 210, fallback-candidates path 16, no move 12). **1,427/1,427 search levels: engine score bit-identical to the oracle, engine move in the exact tie set.** Python: 1,419 identical moves, 7 other tie-set moves, 1 explained by D-006, **0 unexplained**. Robust tables bit-identical 210/210 (1,255 candidates). Override decision 88 overridden / 122 kept. |
| Engine-off equivalence (`TTC_CS_ENGINE=off`, HEAD vs the pre-Phase-7 capture of `03b4704`, same 360 requests and seeds, `PYTHONHASHSEED=0`) | **347/360 byte-identical**. The other 13 differ only in timing fields; normalised: **IDENTICAL**. No `engine` field is present in any response. |
| Concurrency determinism on the fresh-seed corpus (`determinism_check.py --workers 3,10 --concurrency 3,6,20`) | **27 cells, 0 differences** (engine directly, and Flask in front) |

## 4. A-20 and M7-1 — RESOLVED

**Code review.** I read `ai_server.py` and `ai/csharp_engine.py` line by line.

- **The Python solver runs only on the manual path.** `ai_move` calls `_solve_minimax_python` only in the `else` branch, when `csharp_engine.is_enabled()` is false, which only `TTC_CS_ENGINE=off` causes. Every engine problem raises `EngineError` and is answered by `_engine_error_response`. That response logs at error level and returns 503, or 400/500 for `unsupported_parameter`/`client_error`.
- **The circuit breaker is gone.** There is no silent degradation. A failed health verification is not cached.
- **The read timeout covers the engine's queue.** It is `max_time` + the engine's `queueWaitMs` (read from `/v1/health` and required to be present) + a margin. Engine defaults are now `--queue 256` and `--queue-wait-ms 60000`, so the client read timeout is 75 s at `max_time` 10.
- **Abandoned queued requests are dropped in `WorkerPool.AcquireAsync` (`RequestAborted`).** There is no check inside the search, and Core/Search are untouched.
- **The `needs_fallback_candidates` round trip is unchanged.** On failure it restores the random state.
- **Every replaced test maps one to one.** Each former fallback test now asserts the new behaviour with a spy proving that `find_best_move_parallel`, `select_endgame_robust_move` and `_solve_minimax_python` are not called (`PHASE7_HARDENING.md` §2). No other test was weakened.

**Real-fault probe** (published engine of HEAD, real `ai_server` code, spies on the three Python solver functions):

| scenario | response | Python solver calls |
|---|---|---|
| engine not started | 503 `connection_failed` (0.52 s); again 503 (0.51 s, no breaker) | 0 |
| engine running | 200 `csharp`, move (127,(2,2)) | 0 |
| engine killed 0.4 s into an opening | 503 `connection_failed` (0.91 s); the next request 503 | 0 |
| engine with a card-database copy plus one byte | 503 `card_db_mismatch` (0.02 s) | 0 |
| good engine started again | 200 `csharp` **at once** (no back-off) | 0 |

**Overload probe.** This is the Phase 7 review's probe: 1 engine worker, the real threaded Flask server, 12 concurrent openings at production settings (`max_depth` 10, `max_time` 10), then 5 cheap requests.

| | Phase 7 (review) | hardening, default queue | hardening, `--queue 2` |
|---|---|---|---|
| openings answered by the engine | 6 of 12 | **12 of 12** | 3 of 12; the other **9 got 503 `busy` at once** |
| client timeouts / Python fallbacks | 6 / 6 (up to 34.3 s) | **0 / 0**; slowest 19.3 s | 0 / 0 |
| cheap requests | 5 of 5 answered by Python (`circuit_open`) | 5 of 5 by the engine (1 waited 14.4 s in the queue, see L8-2) | 5 of 5 by the engine at once |
| engine `served` / `busyAnswers` | — | 17 / 0 | 8 / 9 |

`docs/DEPLOY_WINDOWS.md` §8 step 5 is corrected ("there is no fallback … keep the defaults"). The guide now states that the engine is **required**, explains what the game client receives (503 + `engine_error`), and says to set `TTC_CS_ENGINE=off` on a machine without the engine.

## 5. M7-2 — RESOLVED (one part not executable here)

- **The pool works as specified.** `ai/csharp_engine.py` uses one process-wide, lock-protected LIFO of `(url, HTTPConnection)`, default size 32 (`TTC_CS_ENGINE_POOL_SIZE`).
  - Failed connections are closed and discarded.
  - A stale idle connection triggers re-verification and a retry on a fresh connection.
  - `Connection: close` answers and pool overflow close the connection explicitly; `atexit` closes the pool.
  - Nothing relies on the garbage collector. Nine pool unit tests cover this.
- **Reviewer probe, connections:** 60 sequential `/ai_move` requests through the real threaded Flask server leave **2** client-side connections to the engine: 1 `ESTABLISHED` pooled and 1 `TIME_WAIT`. The Phase 7 review measured **61** for the same probe.
- **`serve_waitress.py` launcher:**
  - It mirrors `python app.py`: it imports `app` (which registers the unified routes), silences the werkzeug log, sets up and starts the console panel (`--no-console` skips it) and changes to the repository root.
  - It duplicates no business logic, and `app.py` is unchanged.
  - Without waitress it prints `pip install waitress` and exits with 2 (verified).
  - **The real waitress server was not run:** waitress is not installed on this machine, and installing it needs a download that the user has not authorised. Its wiring is covered only by a stand-in-module test (§9).

## 6. M7-3, L7-1, L7-2, L7-3 — RESOLVED

- **M7-3:** `GoldenFiles.RunPythonProcess` reads both streams with `ReadToEndAsync`, then `WaitForExit`; `Ensure` and `RunPython` use it.
  - The regression test (a child writing 1 MiB to each stream) passes; the implementer also reports a mutation check.
  - **Proof:** the full regeneration from an empty golden directory on a new seed passed in 741 s (§3).
  - `e2e_reference.py` still captures the real payloads: a recorder replaces `csharp_engine.solve` and raises `EngineError('e2e_capture')`, and the Python solver is stubbed to fail if called. Its records reproduce the earlier counts.
- **L7-1:** `<ServerGarbageCollection>true</ServerGarbageCollection>` and `<ConcurrentGarbageCollection>false</ConcurrentGarbageCollection>` are now explicit. The published `runtimeconfig.json` says `"System.GC.Server": true, "System.GC.Concurrent": false`, and `/v1/health` reports `serverGc`.
  - **Reviewer spot check:** `gc_compare.py --runs 3 --workers 12 --concurrency 4 --cpus auto`, the implementer's strongest cell.

    | category | workstation/server throughput ratio |
    |---|---|
    | openings | 1.024 |
    | mid-games | 1.019 |
    | robust | **0.978**; the report gives 0.795 for this cell |

    The choice is harmless (the modes are equal, and memory is equal), but the report's claim that Server GC is faster for robust requests on the P-cores is not reproduced. See L8-1.
- **L7-2, L7-3:**
  - A dated note in `PHASE7_INTEGRATION.md` records that absolute latencies vary between sessions.
  - `DEPLOY_WINDOWS.md` covers the required engine, the manual switch, the error matrix, monitoring (`queued`, `busyAnswers`, `abandonedSkipped`, `served`), the new defaults, waitress and GC.

## 7. Findings

No CRITICAL, HIGH or MEDIUM findings.

- **L8-1 (LOW) — the GC conclusion is overstated.**
  - `PHASE7_HARDENING.md` §7 and the `db8bab8` message say that Server GC is ahead for robust endgames on the P-cores (median ratio 0.80–1.01). The reviewer's 3-run repeat of the strongest cell gives 0.978.
  - The data support "equal within noise", not "faster". The decision itself (Server GC, written explicitly) is fine.
  - **Fix:** reword it as "no measurable difference; Server GC kept explicitly". There is no code change.
- **L8-2 (LOW) — queued cheap requests wait behind openings.**
  - When every worker is busy, the FIFO queue makes a ~1 ms endgame request wait for the full searches ahead of it. In the 1-worker probe it waited 14.4 s.
  - With 16 workers on the server this happens only at saturation. It is worth knowing when choosing `--workers`, and see O8-1.
- **L8-3 (LOW) — the game client's own HTTP timeout is unknown.**
  - The server now waits up to `max_time` + 60 s queue + 5 s (75 s) instead of falling back. If the game-side plugin gives up earlier, the user sees its timeout rather than a 503 `busy`.
  - A running search still finishes for nothing; a queued one is dropped.
  - **Fix:** check the plugin's timeout and set `--queue-wait-ms` below it (minus `max_time`). This is a deployment checklist item (§8).

**Optional**

- **O8-1:** two queues or shortest-job-first ordering, for example by the number of empty cells, so endgame and mid-game requests are not stuck behind openings at saturation.
- **O8-2:** `solve()` retries a POST once when a *reused* idle connection turns out stale. If the server had in fact received the request, the engine computes it twice. That is harmless (deterministic) but costs CPU. An idempotency key or a GET-free health probe before reuse would avoid it.

## 8. Windows Server go-live checklist (E5-2698B v3, 16 cores / 32 threads, 16 GB)

1. **Engine runtime:** install the **ASP.NET Core Runtime 10.x** (or the Hosting Bundle). `dotnet --list-runtimes` must show `Microsoft.AspNetCore.App 10.x`.
2. **Publish:** `dotnet publish csharp/src/TTC.Siren.Engine -c Release -o <dir>`, framework-dependent.
3. **Card database:** pass `--card-db` with the absolute path of the **same** `幻卡数据库.csv` the Python server loads. Check `/v1/health` `cardDbSha256` against `Get-FileHash`.
4. **Engine service:** start it with `--workers 16`, keeping the defaults `--queue 256 --queue-wait-ms 60000` unless L8-3 requires a shorter wait. Register it as a service (`DEPLOY_WINDOWS.md` §4; not executed in review). The engine must be running before the Python server takes traffic: there is no fallback.
5. **Python front end:**
   - Install waitress on the server (`pip install waitress`) and start the Python server with `python serve_waitress.py --host 127.0.0.1 --port 5000 --threads 16 --no-console` when it runs as a service.
   - Send one `/ai_move` request and check that the response has `"engine": "csharp"`. This is the step that could not be run here.
   - `python app.py` also works: the connection pool fixed M7-2.
6. **Game-client timeout:** confirm it is at least 75 s, or lower `--queue-wait-ms` accordingly (L8-3).
7. **Measure on the server:** run the load matrix there (`DEPLOY_WINDOWS.md` §8) and adjust `--workers`. Expect about 1.6 GiB of engine memory for 16 workers.
8. **Monitoring:** watch the `ttc.engine` error log and `/v1/health` (`queued`, `busyAnswers`, `abandonedSkipped`).
9. **Rollback:** `TTC_CS_ENGINE=off` and a restart of the Python server restores the pre-Phase-7 behaviour exactly (verified on 360 requests). The Python solver is then about 100× slower.
10. **PyPy:** if the production interpreter is PyPy, run the same smoke test under PyPy. PyPy was never executed in this project.

## 9. Test-coverage gaps

- The real waitress server has not run, and PyPy has not run: neither is installed here.
- The Windows-service installation (`sc.exe`) is not executed.
- Head-of-line waiting (L8-2) is observed but not covered by any test or measurement table.
- No test covers a game-client timeout shorter than the engine's queue wait (L8-3).

## 10. Next steps

1. Optional: the documentation fix for L8-1 can be folded into the next goal.
2. Pushing the 7 local Step B commits needs the user's authorisation.
3. Phase 8 may start after explicit authorisation. Note that A-20 already makes the engine the only solver on the server path, so Phase 8 mainly removes Flask/Python from the request path and ports parsing, sampling, the analysis text and Monte Carlo.
4. Deploy using the §8 checklist.

## 11. Commands run by the reviewer

```
git status -sb; git log; git ls-remote origin; gh pr list --state all; git show --stat <each commit>; git diff 45a9b28 HEAD -- ai_server.py (full read); ai/csharp_engine.py (full read)
git archive <e35e8c3|262ba77|76eafcc|db8bab8> csharp | tar -x; dotnet build -c Release --no-incremental    -> 0/0 each
dotnet build ... --no-incremental; dotnet test csharp/TTC.Siren.sln -c Release; python -m pytest -q    -> 0/0, 155/155, 152
git archive HEAD (scratch, empty golden/corpus), E2eSeed=31337105; dotnet test     -> 155/155 in 741 s, e2e counts §3
PYTHONHASHSEED=0 TTC_CS_ENGINE=off python tools/phase7/equivalence.py capture ...; compare vs the 03b4704 capture; strict diff script    -> IDENTICAL, 347/360 byte-identical
dotnet publish csharp/src/TTC.Siren.Engine -c Release -o <scratch>; runtimeconfig.json inspected
reviewer probes: real faults (5 scenarios, solver spies), overload (1 worker, 12 openings; default queue and --queue 2), connection count (60 requests, netstat)
python serve_waitress.py --no-console (no waitress installed)    -> clear message, exit 2
python tools/phase7/determinism_check.py --workers 3,10 --concurrency 3,6,20 (fresh-seed corpus)    -> 27 cells, 0 differences
python tools/phase7/gc_compare.py --runs 3 --workers 12 --concurrency 4 --cpus auto; gc_table.py    -> §6
```

## 12. Addendum (2026-10-11)

- **waitress verified** (closes the §5 / §9 gap):
  - The user installed waitress 3.0.2. The reviewer started `python serve_waitress.py --port 5091 --threads 8 --no-console` together with the published engine (4 workers).
  - Result: **30 concurrent `/ai_move` requests → 30 × HTTP 200 `engine: csharp`**.
  - `/api/recommend`, `/api/draft/analyze` and `/api/winrate` are all served.
  - 9 client connections to the engine: 8 waitress threads + the health check, below the pool size.
- **Repository visibility:** `extrant/TTC_Siren` was **public**: no license, 12 stars, 0 forks, and the C# solver was pushed to its `csharp-solver-migration` branch. The user decided to make the whole repository private (Q-9), and the reviewer ran `gh repo edit --visibility private` on 2026-10-11 (verified `isPrivate: true`).
  - Content that was public before cannot be recalled.
  - The README image that points to `raw.githubusercontent.com` no longer loads publicly.
