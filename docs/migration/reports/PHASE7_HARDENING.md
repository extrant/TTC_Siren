# Phase 7 hardening (2026-10-10)

Goal file `docs/migration/goals/phase7_hardening_goal.md`; findings of `docs/migration/reviews/OPUS_PHASE7_REVIEW.md` (M7-1, M7-2, M7-3, L7-1, L7-2, L7-3) and the user's decision **A-20**
(`CSHARP_MIGRATION.md` section 8): the C# engine is the only minimax solver, no automatic fallback to Python. **Status: DONE, awaiting independent review.** No review gate is set
here. Step A (the reviewer's files) was committed (`45a9b28`) and pushed (fast-forward `03b4704..45a9b28`); **the Step B commits listed below are local, not pushed.**

All timings are from one session on one machine (i7-13650HX, Windows 11). Absolute numbers differ between sessions (review L7-2); only the comparisons inside a table are meaningful.

## 1. What changed, per finding

| Finding | Change | Commit | Tests |
|---|---|---|---|
| **M7-3** golden regeneration deadlocks | `GoldenFiles.RunPythonProcess`: stdout and stderr are read asynchronously and concurrently, then `WaitForExit`; `Ensure` and `RunPython` use it. `tools/phase7/e2e_reference.py` is adapted to the new dispatch (section 5). | `e35e8c3`, `76eafcc` | `GoldenHarnessTests`: a child writing 1 MiB to stdout and 1 MiB to stderr completes; a failing child reports exit code and both streams. Mutation check: with the old sequential reads the first test fails after its 60 s limit. |
| **M7-1** overload cascade, **A-20** | Engine: queued requests whose HTTP connection was aborted are dropped without a search (`WorkerPool.AcquireAsync` + `RequestAborted`; running searches are never interrupted, nothing is checked inside the search); `/v1/health` reports `queueWaitMs`, `abandonedSkipped`, `served`, `serverGc`; defaults `--queue 256`, `--queue-wait-ms 60000`. Python: no fallback, no circuit breaker; every engine problem is an HTTP error with `engine_error`; read timeout = `max_time` + the engine's queue wait + margin. | `262ba77` (engine), `76eafcc` (Python) | section 3 and the list of replaced tests (section 2) |
| **M7-2** one connection per request | Process-wide, thread-safe LIFO pool of keep-alive `HTTPConnection`s in `ai/csharp_engine.py` (`TTC_CS_ENGINE_POOL_SIZE`, default 32); optional `serve_waitress.py` | `76eafcc`, `9e60222`, `93b0da1` | `tests/test_csharp_engine_pool.py` (9), netstat probe in `tests/test_engine_overload.py`, `tests/test_serve_waitress.py` (4) |
| **L7-1** GC setting | `<ServerGarbageCollection>true</ServerGarbageCollection>` written explicitly after a measured comparison (section 7) | `db8bab8` | `The_engine_runtime_config_states_the_chosen_gc_mode_explicitly` |
| **L7-2** session-dependent latencies | Dated note in `docs/migration/reports/PHASE7_INTEGRATION.md` | docs commit | — |
| **L7-3** engine not installed | `docs/DEPLOY_WINDOWS.md`: required engine, `TTC_CS_ENGINE=off` on a machine without it, error matrix, monitoring, defaults, waitress, GC | docs commit | — |

Not changed (as required): `TTC.Siren.Core`, `TTC.Siren.Search`, `ai/ai.py`, `core/`, `_solve_minimax_python` (byte-identical, one comment above it), the endgame functions, `app.py`,
`solver=monte_carlo`, the analysis text, `main`. No search-hot-path change.

## 2. Replaced tests (A-20) and the reason

The tests below asserted the **automatic fallback** (`engine == "python"`, `fallback_reason`, the circuit breaker, a response equal to the Python path). The user withdrew that behaviour
(A-20), so they were replaced one by one by tests of the new behaviour; the new tests additionally assert that **none of `find_best_move_parallel`, `select_endgame_robust_move` and
`_solve_minimax_python` is called** (a spy that counts calls). Nothing else was skipped, removed or loosened.

| Old test (Phase 7) | Replacement |
|---|---|
| `test_fallback_busy` | `test_error_busy` |
| `test_fallback_invalid_request` | `test_error_invalid_request` |
| `test_fallback_engine_error` | `test_error_engine_error` |
| `test_fallback_malformed_json_opens_the_breaker` | `test_error_malformed_json` |
| `test_fallback_malformed_fields` | `test_error_malformed_fields` |
| `test_fallback_unknown_status` | `test_error_unknown_status` |
| `test_fallback_timeout_opens_the_breaker` | `test_error_timeout`, `test_a_timeout_does_not_disable_the_engine` |
| `test_fallback_connection_refused_opens_the_breaker_and_skips_the_engine_afterwards`, `test_the_breaker_closes_after_the_backoff` | `test_error_connection_refused_then_the_engine_comes_back`, `test_nothing_switches_the_engine_off_automatically` |
| `test_fallback_protocol_mismatch`, `test_fallback_card_database_mismatch`, `test_fallback_health_failure` | `test_error_protocol_mismatch`, `test_error_card_database_mismatch`, `test_error_health_failure` (+ `test_error_health_without_queue_wait`, `test_a_failed_verification_is_not_cached_so_a_corrected_engine_is_accepted_at_once`) |
| `test_fallback_illegal_move[5 kinds]`, `test_fallback_a_card_that_the_order_rule_forbids` | `test_error_illegal_move[5 kinds]`, `test_error_a_card_that_the_order_rule_forbids` |
| `test_unsupported_parameters_are_left_to_the_python_path[4]` | `test_a_parameter_that_is_not_a_number_is_a_400_and_the_engine_is_not_asked[4]`, `test_a_bool_is_the_integer_it_is_in_python` |
| `test_a_failure_after_the_fallback_candidates_restores_the_random_state` | `test_a_failure_after_the_fallback_candidates_leaves_no_trace_in_the_random_generator` |
| `test_fallback_unexpected_status_after_the_fallback_candidates` | `test_error_unexpected_status_after_the_fallback_candidates` |
| `test_fallback_client_error_never_fails_the_request` | `test_a_defect_of_the_client_code_is_a_500_with_a_reason` |
| `test_a_stopped_engine_falls_back_to_the_python_path` (real engine) | `test_a_stopped_engine_is_a_503_and_the_python_solver_does_not_run` |

Unchanged and passing: `tests/test_engine_off_equivalence.py` (43 requests), `test_engine_off_never_contacts_the_engine_and_adds_no_fields`, the `needs_fallback_candidates` round-trip test,
the recorder test, every other client test. New: `test_engine_off_runs_the_python_solver`, `test_every_error_is_logged_at_error_level`, `test_a_failed_request_is_recorded_when_recording_is_on`,
`test_the_read_timeout_is_max_time_plus_the_engines_queue_wait_plus_a_margin`, `test_an_engine_that_answers_after_the_old_timeout_is_still_waited_for`, the pool tests, the overload tests.

## 3. The error matrix (replaces the fallback matrix of `PHASE7_INTEGRATION.md` section 4)

While the engine is on (the default) the Python solver never runs. The body is `{"error": "<readable message>", "engine": "csharp", "engine_error": "<reason>"}`; every failure is logged at
error level by the logger `ttc.engine`, and `TTC_RECORD_REQUESTS` records the request.

| `engine_error` | HTTP | Situation | Test |
|---|---|---|---|
| `connection_failed` | 503 | nothing listens / reset / connect timeout (Windows: a refused connection can exceed the 0.5 s connect timeout) | `test_error_connection_refused_then_the_engine_comes_back`, real engine: `test_a_stopped_engine_is_a_503_and_the_python_solver_does_not_run` |
| `timeout` | 503 | no answer within `max_time` + queue wait + margin | `test_error_timeout`, `test_a_timeout_does_not_disable_the_engine` |
| `busy` | 503 | engine queue full / wait expired | `test_error_busy`; real engine: `test_a_full_queue_answers_503_busy_and_nothing_times_out`; engine: `A_full_queue_answers_busy_and_the_engine_recovers` |
| `engine_error` | 503 | engine HTTP 500 | `test_error_engine_error` |
| `invalid_request` | 503 | engine HTTP 400 | `test_error_invalid_request` |
| `malformed_response`, `http_error`, `unexpected_status` | 503 | answer not as the protocol says | `test_error_malformed_json`, `_malformed_fields`, `_unknown_status`, `_unexpected_status_after_the_fallback_candidates`, `test_error_illegal_move[bad_types]` |
| `protocol_mismatch`, `card_db_mismatch`, `health_failed` | 503 | health check failed; not cached | `test_error_protocol_mismatch`, `_card_database_mismatch`, `_health_failure`, `_health_without_queue_wait` |
| `illegal_move` | 503 | slot / card id / cell / playability (秩序·混乱) | `test_error_illegal_move[4]`, `test_error_a_card_that_the_order_rule_forbids` |
| `unsupported_parameter` | 400 | `max_depth` / `max_time` is not a number (Python would raise a TypeError); a bool is the integer it is in Python | `test_a_parameter_that_is_not_a_number_...[4]`, `test_a_bool_is_the_integer_it_is_in_python` |
| `client_error` | 500 | a defect in the Python client code itself | `test_a_defect_of_the_client_code_is_a_500_with_a_reason` |

The manual switch `TTC_CS_ENGINE=off` runs the unchanged Python solver and adds no field; the whole 360-request corpus was compared with the pre-Phase-7 commit in Phase 7 and the
43-request regression sample (`tests/test_engine_off_equivalence.py`) still passes unchanged.

The `needs_fallback_candidates` round trip (Python's `handler.generate_opponent_cards` feeding the robust layer) is not a solver fallback and is unchanged; a request that fails after it
leaves the random generator exactly as a request that failed before it (tested).

## 4. Overload: before and after (review M7-1)

The review's probe, automated in `tests/test_engine_overload.py`: engine with **1 worker**, the real Flask app behind werkzeug's threaded server, **12 concurrent opening requests** with
`max_time` 1 s and a read margin of 1 s (so the old client gave up after 3 s). "Before" is the pre-hardening tree (`262ba77`) against the same engine process with the old 30 s queue wait,
measured with a scratch probe; "after" is the test.

| | before (`262ba77`) | after |
|---|---|---|
| answered by the engine | 4 of 12 | **12 of 12** |
| client timeouts (fell back to Python) | 8 | **0** |
| Python solver calls | 8 searches | **0** (spy) |
| 5 cheap requests right after | all 5 `circuit_open` -> Python | (not needed: no breaker; `test_a_timeout_does_not_disable_the_engine`) |
| latencies (s) | 1.0 1.4 2.4 2.4 4.5 x 8 (the 4.5 s ones are the timeouts + Python) | 1.0 1.3 1.3 1.6 2.6 2.6 3.2 4.2 5.2 5.3 6.1 6.6 |
| slowest request vs the old read timeout (3.0 s) | — | 6.6 s: the queue really was longer than the old timeout |
| engine `served` / `busyAnswers` | 5 / 0 | 12 / 0 |

Queue overflow: engine with `--queue 2`, 8 concurrent openings: the excess gets **503 `busy`** (`engine_error: "busy"`, `queue_full`), nothing else happens (no timeout, no 500), the
engine's `busyAnswers` equals the number of busy responses (`test_a_full_queue_answers_503_busy_and_nothing_times_out`).

Abandoned requests: three queued HTTP requests are aborted while a test-held lease occupies the only worker; `abandonedSkipped` becomes 3, `queued` returns to 0, `served` stays 1 (the
lease), nothing was searched for them, and the next request is served (`Queued_requests_that_were_aborted_are_dropped_without_a_search`; pool-level test
`A_request_whose_caller_has_gone_is_not_given_a_worker`).

## 5. Connections (review M7-2)

60 sequential `/ai_move` requests through the real threaded Flask server (a new thread per request), pool size 4 in the test; client-side TCP connections to the engine port counted with
`netstat -an -p TCP`, every state:

| | before (`262ba77`) | after |
|---|---|---|
| connections to the engine port left after 60 requests | **60** (delta over the start) | **1** (delta; 2 absolute including one `TIME_WAIT` of the test's start-up health poll) |
| limit | — | pool size (4 in the test, 32 by default) |

Pool unit tests (stub engine counting accepted connections): 40 requests on 40 short-lived threads use **one** connection; a 12-thread burst opens more than the pool size but keeps exactly the
pool size afterwards and closes the rest (checked by the `sock is None` state of every `HTTPConnection` created, not by the garbage collector); an idle connection closed by the server is replaced
once and the engine is verified again before the retry; a failed connection is discarded; `Connection: close` answers are not pooled; a pool of size 0 closes after use.
**PyPy was not run** (not installed here); the client uses the standard library only and closes every connection explicitly (`close()` on failure, overflow, `reset()` and at exit), never relying on the GC.

`serve_waitress.py` (optional launcher): fixed thread pool, same initialisation as `app.py` (import `app`, log levels, console panel, `--no-console`); `app.py` unchanged. **waitress itself is
not installed on this machine**, so the real server was not started: the wiring and the missing-waitress message are tested with a stand-in module.

## 6. Harness regeneration (review M7-3)

`tools/parity/golden/e2e_reference_g1_s20261010_n360.jsonl.gz`, `e2e_counts.txt` and `tools/phase7/corpus/corpus_s20261010_n360.json` were deleted, then
`dotnet test csharp/TTC.Siren.sln -c Release` was run (the generator `e2e_reference.py` logs one error line per request to stderr, so it exercises exactly the pipe that used to deadlock):
the corpus and the reference were **regenerated through the harness and the whole suite passed: 155/155, 88 s wall time** (the other golden files were already current; changing
`ai_server.py` earlier had made the endgame goldens stale, and their regeneration through the same helper had already run in an earlier `dotnet test` without hanging).
The regenerated reference is **record-for-record identical** to the cached one made before the change: 404/404 records equal, the corpus equal, only the three bound source hashes
(`ai/csharp_engine.py`, `ai_server.py`, `tools/phase7/e2e_reference.py`) differ in the header. The counts are unchanged: 1,392 levels, engine score == oracle 1,392, engine move in the exact tie set 1,392,
Python move identical 1,381 / other move in the tie set 6 / explained by D-006 5 / **unexplained 0**, robust tables bit-identical 209 (1,274 candidates).

How the generator captures the payload now: it still drives `ai_move` through Flask's test client and replaces `csharp_engine.solve` by a recorder, so `_solve_minimax_with_engine` builds the
payload (and the second payload with the random fallback candidates) exactly as in production; the recorder raises `EngineError('e2e_capture')`, which `ai_move` answers with its 503. The
Python solver is stubbed to fail loudly if it is ever called on this path. The reference computations on copies of the pristine state are unchanged.

## 7. GC (review L7-1)

`<ServerGarbageCollector>` was ignored by MSBuild; Server GC was active by accident. Measured with CLAUDE.md section 6: the same build is started with `DOTNET_gcServer=1` and `=0` (`/v1/health`
`serverGc` verified for both), `tools/phase7/gc_compare.py` runs the engine load matrix (workers **12 and 16**, concurrency **1/4/8/16**, openings / mid-games / robust endgames) **3 times** per GC mode and CPU
set, alternating the order of the two modes each repetition; CPU sets: **P-cores `0xFFF`** (detected, `--cpu auto`) and **all 20 logical CPUs `0xFFFFF`**. The load generator runs unpinned (as in Phase 7). Raw data:
`docs/migration/reports/phase7/gc/*.jsonl`; full table `docs/migration/reports/phase7/gc/gc_table.txt` (median [min-max] over the 3 runs, throughput in requests/s, p95 in seconds). No cell had an error or a `busy` answer.
(A first attempt was discarded: two copies of the driver ran at the same time by my mistake and the cells showed errors; all files were deleted and the measurement was repeated with one driver.)

Summary (workstation / server ratio of the median throughput; 1.0 = equal):

| CPU set | category | workers 12 (c = 1 / 4 / 8 / 16) | workers 16 (c = 1 / 4 / 8 / 16) |
|---|---|---|---|
| P-cores | openings | 1.03 / 1.03 / 1.04 / 0.99 | 0.98 / 0.97 / 0.99 / 0.99 |
| P-cores | mid-games | 0.99 / 0.94 / 0.97 / 0.97 | 1.00 / 1.05 / 1.00 / 0.97 |
| P-cores | robust endgames | 0.85 / 0.80 / 0.95 / 0.96 | 0.96 / 0.90 / 1.01 / 0.90 |
| all CPUs | openings | 0.98 / 1.01 / 1.02 / 1.02 | 1.02 / 0.98 / 1.02 / 0.99 |
| all CPUs | mid-games | 1.00 / 1.02 / 0.99 / 1.03 | 1.00 / 1.00 / 0.96 / 1.02 |
| all CPUs | robust endgames | 1.24 / 1.04 / 1.00 / 0.97 | 1.06 / 0.98 / 0.94 / 1.02 |

Engine working set after the load (MiB, median of 3): 12 workers 1,235 (server) / 1,234 (workstation) on the P-cores and 1,238 / 1,238 on all CPUs; 16 workers 1,623 / 1,621 and 1,626 / 1,621.
p95 latencies are equal within the run-to-run ranges (openings 1.7-7.6 s, mid-games 0.18-0.57 s, robust 1.2-7.1 ms).

Reading: for openings and mid-games the two modes are indistinguishable (ratios 0.94-1.05, inside the min-max ranges); for the very short robust requests Server GC is ahead on the P-cores (median ratio below 1.0
in 7 of 8 cells), while on all CPUs the picture is mixed and the ranges overlap (the one cell with 1.24 has ranges [1256-1780] against [1229-1843]); memory is identical because the 96 MiB tables dominate.
There is no cell where workstation GC is better outside the run-to-run range. **Decision: Server GC, non-concurrent, written explicitly**; the published `runtimeconfig.json` contains
`"System.GC.Server": true` and `"System.GC.Concurrent": false`. Limits: min-max over 3 runs is not a confidence interval; the 32-thread server has more heaps (not measured there); one machine.

## 8. Build and test results

Final state of the working tree before the documentation commit (sources frozen):

| command | result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test csharp/TTC.Siren.sln -c Release` | **155 passed**, 0 failed, 0 skipped (149 before + 2 harness + 2 abandoned-request + 1 default-settings + 1 GC-config tests) |
| `python -m pytest -q` | **152 passed** (126 before: 17 fallback tests replaced one for one, plus the new client, pool, overload, launcher and engine-process tests) |

## 9. Known limits

* PyPy and the real waitress were not run (not installed here); the pool and the launcher are written for them but only tested on CPython with a stand-in module.
* The Windows service installation (`sc.exe`) is still unexecuted (as in the review).
* A request that waits in the engine queue can wait up to `queue-wait-ms` (60 s) + `max_time`: a game client with a shorter timeout will give up first; the abandoned request is then dropped without a search if it
  was still queued, but a search that has already started finishes (nothing is checked inside the search by design).
* `TTC_CS_ENGINE=off` is the only way to the Python solver; a machine without the engine that forgets to set it answers 503 `connection_failed` to every minimax request.
* Server GC on the 32-thread production server was not measured there.
* The reviewer's optional O7-1 (fallback counters) is moot with no fallback; O7-2 (predictive `busy`) was not done.
