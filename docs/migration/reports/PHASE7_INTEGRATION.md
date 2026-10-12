# Phase 7 — the Python server calls the C# engine (2026-10-10)

Goal file `docs/migration/goals/phase7_goal.md`, Step B. **Status: DONE, awaiting independent review.** No review gate is set here. Commits are local (not pushed).
Deployment: `docs/DEPLOY_WINDOWS.md`.

## 1. Architecture

```
client --HTTP--> Flask ai_server (Python: parse board/hands/rules, unknown-card sampling, analysis text)
                     | minimax path only
                     |--HTTP 127.0.0.1:5300--> TTC.Siren.Engine (C#, one process, worker pool)
                     |                          Fast iterative deepening (max_depth / max_time as sent)
                     |                          + robust endgame: scenarios, Select, override (when Python's rule says so)
                     `-- on any engine problem: the unchanged Python solver path (fallback)
```

* Project `csharp/src/TTC.Siren.Engine` (ASP.NET Core minimal API, Kestrel, `net10.0`, warnings are errors, in `TTC.Siren.sln`). **Framework-dependent** publish.
  It reuses `TTC.Siren.Core` and `TTC.Siren.Search` as they are (the only changes there are the review findings L-1, L-2, L-9 and one statistics addition, section 9).
* Listens on **one loopback address** (default `127.0.0.1:5300`); `EngineOptions.Validate` refuses any other address unless `--allow-remote` is passed (tested).
  Configuration: `appsettings.json` section `Engine` and the command line (`--port --workers --queue --queue-wait-ms --card-db --tt-entries --memo-entries --host --allow-remote`).
* **Request-level parallelism:** `WorkerPool` holds `N` worker contexts (default = number of logical CPUs). A `WorkerContext` owns its **Fast transposition table**
  (4 Mi entries = 96 MiB by default) and its **exact-endgame memo** (32 Ki entries); the memo is passed explicitly to `EndgameRobust.Select` (review L6-4: nothing relies on
  the `[ThreadStatic]` memo of Phase 6). A request builds its own `GameState` objects, leases a worker, starts a new table generation bound to its own `SearchContext`
  (review H4-1), searches synchronously and releases the worker. Contexts are created on first use. Requests beyond `N` wait in a **bounded queue** (`--queue`, default 64)
  for at most `--queue-wait-ms` (default 30 s); when the queue is full or the wait times out the answer is **HTTP 503 `busy`**. The card database is loaded once and is read-only;
  nothing else is shared between requests.
* `GET /v1/health`: protocol version, assembly version with git commit, card count, **SHA-256 of the card database file**, workers / created / idle / queued, busy answers,
  table memory, uptime.

### D-015 / M4-3 treatment (user decision: Fast search + canonical hand order)

The engine request is built from Python's `game_state` **before any Python search**, so it carries the hand order of the request. The engine runs the standard Fast search on one copy
of that state and the robust layer (scenarios, `Select`, override) on a **second copy**: the robust selection therefore sees the request's hand order and the display edges the request
carried, never the order and edges a search leaves behind (D-001, D-003). **M4-3 is closed by this decision:** the post-search hand order and display edges are no longer an input of the
engine path. The robust layer reads effective values only (`evaluate_state`, `_calculate_corner_safety_risk` and `_score_unknown_candidate` use `get_effective_value`/effective
comparisons on board and hand cards; the only display reads are of freshly created candidate cards, whose display edges equal their base edges). The end-to-end comparison of section 5 compares the
robust tables against Python run **on the request-order state**, bit for bit.
Expected, documented difference: because Python does not search on the engine path, its `game_state` keeps the request hand order and unrefreshed display edges, and the move-description text
(`format_card_display`, which shows display edges) can differ from the Python path for the same move.

## 2. Protocol (`POST /v1/solve`, JSON, version 1)

Request fields: `protocol` (1), `rules` (Python rule names; unknown names are rejected), `state` (`{"cur", "board"[9], "hands"[2][]}` in the golden card format
`[id|null, type|null, "r"|"b"|null, mod, base u/r/d/l, display u/r/d/l, canUse 0/1]`, hands **in the request's order**), `aggressive`, `maxDepth`, `maxTime`, and `robust`
(`null` or `{aiPlayer, opponent, generated[player][slot], sampleCount, npcIds, handler, fallbackCandidates}`: exactly the inputs of Python's `_build_endgame_scenarios`).
Python's formula for `sampleCount`, `_is_generated_unknown_card` for `generated` and `if not handler: return [base_state]` for `handler` are evaluated by the client.

`maxDepth` / `maxTime` mapping (tested): `maxDepth` is a number; its floor, at most 100 (Python's `min(max_depth, 100)`); below 1 or NaN means no level (no move). `maxTime` is a number or `null`:
`null` = no limit (Python's NaN or +inf: `elapsed > NaN` is never true); zero or negative = already over (no level completes, no move); a huge value = no limit.
Non-numeric values and booleans are never sent: the client leaves them to the Python path.

Response (`status` `ok`): `move` (final: `{slot, row, col, cardId}` with `slot` = index in the side to move's hand **in the request's order**, `cardId` = -1 for Python's `None`),
`standard` (the standard search's move, score and `scoreBits`), `depths[]` (per completed depth: move, score and bits, nodes, running nodes, TT hits, cutoffs, elapsed seconds),
`totals` (nodes, depthCompleted, ttHits, cutoffs, rootMoves, elapsed), `robust` (`null` or `ran`, `scenarios`, `robustMove`, `overridden`, exact-solver counters, and the candidate table in
Python's returned order: move, `finalScore`/`finalBits`, `safetyRatio`/`safetyBits`, `robustBits`, `cornerBits`), `timings` (build / search / robust / total ms), `worker`.
Non-finite doubles are JSON `null` with the exact value in the `...Bits` field (16 hex digits of the IEEE-754 pattern). Other statuses: `needs_fallback_candidates` (HTTP 200, `unknownCount`; sent
**before** any search when Python's legal enumeration is empty — the client then calls `handler.generate_opponent_cards` exactly as `_build_endgame_scenarios` would and resends once with
`fallbackCandidates`), `invalid_request` (400, `message`), `busy` (503, `reason` `queue_full`|`queue_timeout`), `error` (500).

An actual exchange (the request is the 4-empty-cell position of corpus entry 17; duplicate card ids in the side to move's hand; `overridden` is true: the robust layer replaced the
standard move). The full files are `docs/migration/reports/phase7/example_exchange.json`:

```
POST /v1/solve  {"protocol":1,"rules":["逆转"],"state":{"cur":1,"board":[[35,"兽人","r",0,6,2,6,6,6,2,6,6,1],[289,null,"b",0,7,5,8,2,7,5,8,2,1],…,null],
                 "hands":[[[1467,null,"r",…],[188,null,"r",…],[188,null,"r",…]],[[387,null,"b",…],[353,null,"b",…],[127,null,"b",…]]]},
                 "aggressive":false,"maxDepth":4,"maxTime":10,
                 "robust":{"aiPlayer":1,"opponent":0,"generated":[[1,0,0],[0,0,0]],"sampleCount":16,"npcIds":[],"handler":true,"fallbackCandidates":null}}
200 {"protocol":1,"status":"ok","move":{"slot":2,"row":2,"col":2,"cardId":127},
     "standard":{"move":{"slot":2,"row":1,"col":1,"cardId":127},"score":6.0600000000000005,"scoreBits":"40183D70A3D70A3E"},
     "depths":[{"depth":1,…,"nodes":13},{"depth":2,…},{"depth":3,…},{"depth":4,"move":{"slot":2,"row":1,"col":1,"cardId":127},"nodes":343,"nodesTotal":531,…}],
     "totals":{"nodes":531,"depthCompleted":4,"ttHits":48,"cutoffs":133,"rootMoves":12,"elapsed":0.0225},
     "robust":{"ran":true,"scenarios":16,"robustMove":{"slot":2,"row":2,"col":2,"cardId":127},"overridden":true,"exactNodes":20748,
               "candidates":[{"move":{…,"cardId":127},"finalScore":-2.325,"finalBits":"C00299999999999A","safetyRatio":0,…},…]},"timings":{…},"worker":0}
```

## 3. Python changes (the only ones allowed by the goal)

| File | Function | Change |
|---|---|---|
| `ai/csharp_engine.py` (new) | whole module | Engine client, standard library only (PyPy compatible): environment configuration, health check (protocol + card-database SHA-256), keep-alive connection per thread, circuit breaker, wire helpers (`card_array`, `state_payload`, `wire_max_depth`, `wire_max_time`). |
| `ai_server.py` | `_solve_minimax_python` (new) | The solver block of `ai_move` (Python search, robust endgame, override) moved here **unchanged** — proven by comparing the ASTs of the old inline block and the new function body (identical). |
| `ai_server.py` | `_solve_minimax_with_engine`, `_engine_outcome`, `_engine_move` (new) | Builds the request from `game_state` before any search, calls the engine, does the fallback-candidates round trip (random state saved and restored on failure), maps the returned slot to the Python card and verifies it (slot exists, card id equal, cell empty, playable under 秩序/混乱), builds the progress entries. Any exception becomes a fallback (`client_error`). |
| `ai_server.py` | `ai_move` | The minimax branch tries the engine when `TTC_CS_ENGINE` is not off, otherwise (or on any fallback) calls `_solve_minimax_python` with exactly the old arguments. `performance_stats` / `search_progress` come from the engine's statistics on the engine path (same keys); additive `engine` / `fallback_reason` fields, **absent when the engine is off**; `solver=monte_carlo` is untouched. |
| `ai_server.py` | `_record_request` (new) | Opt-in recorder `TTC_RECORD_REQUESTS=<dir>`: raw request JSON + Python random state (stores user requests; off by default). |
| `tools/parity/py_endgame_trace.py` | `load_override_block` | Finds the override block in `_solve_minimax_python` as well (a parity generator, not business code). |
| `tests/` (new) | `test_csharp_engine_client.py`, `test_engine_integration.py`, `test_engine_off_equivalence.py` | See sections 4 and 6. |

Nothing in `core/`, `ai/ai.py`, `ai/monte_carlo.py`, `app.py` or the analysis functions changed.

## 4. Fallback matrix and its tests

Every row is a pytest in `tests/test_csharp_engine_client.py` (a stub HTTP server plays the engine) unless noted; each fallback test asserts `engine == "python"`, the `fallback_reason`, and that the
response equals the engine-off response of the same request (normalised, see section 6).

| `fallback_reason` | Situation | Opens the circuit breaker | Test |
|---|---|---|---|
| `connection_failed` / `timeout` | nothing listens / the engine does not answer within `max_time` + margin (on Windows a refused connection can take longer than the connect timeout and then shows as `timeout`) | yes | `test_fallback_connection_refused_…`, `test_fallback_timeout_…`; real engine: `test_a_stopped_engine_falls_back_to_the_python_path` |
| `circuit_open` | within the back-off after such a failure | — | same tests; `test_the_breaker_closes_after_the_backoff` (engine used again afterwards) |
| `busy` | HTTP 503 | no | `test_fallback_busy`; engine side: `A_full_queue_answers_busy_and_the_engine_recovers` |
| `invalid_request` | HTTP 400 | no | `test_fallback_invalid_request`; engine side: 40 invalid payloads |
| `engine_error` | HTTP 500 | no | `test_fallback_engine_error` |
| `malformed_response` | not JSON / fields missing / wrong types | yes (JSON) | `test_fallback_malformed_json…`, `…_malformed_fields`, `…_illegal_move[bad_types]` |
| `http_error` | other status | no | `test_fallback_unknown_status` |
| `protocol_mismatch` | health reports another protocol | yes | `test_fallback_protocol_mismatch` |
| `card_db_mismatch` | card-database hashes differ | yes | `test_fallback_card_database_mismatch` |
| `health_failed` | health endpoint not 200/ok | yes | `test_fallback_health_failure` |
| `illegal_move` | slot out of range / card id differs / cell occupied or off the board / card not playable under 秩序·混乱 | no | `test_fallback_illegal_move[…]` (4 kinds), `test_fallback_a_card_that_the_order_rule_forbids` |
| `unsupported_parameter` | `max_depth` / `max_time` not a real number (string, bool, null): the Python path decides what to do with them | — | `test_unsupported_parameters_are_left_to_the_python_path` |
| `unexpected_status` | the second call (with fallback candidates) is not `ok` | no | `test_fallback_unexpected_status_…` (also checks the random generator is restored) |
| `client_error` | any exception in the client code | no | `test_fallback_client_error_never_fails_the_request` |

Also tested: the `needs_fallback_candidates` round trip consumes the random generator exactly like the Python path, and a failure after it restores the generator
(`test_needs_fallback_candidates_round_trip_…`, `test_a_failure_after_the_fallback_candidates_restores_the_random_state`); engine off never contacts the engine and adds no field;
the recorder is off by default and writes a replayable file when enabled.

## 5. End-to-end comparison (engine against the C# oracle and against Python; Q-5 (b))

`tools/phase7/gen_corpus.py` generates the corpus (360 seeded `/ai_move` requests: both owners, `currentPlayer` 0/1/2, every rule incl. 同类强化/同类弱化 (about 110 requests), 秩序/混乱,
选拔, 全明牌/三明牌, opponent unknown cards 0/1/2/3+, NPC id lists, duplicate card ids, aggressive and standard mode, 27 openings, mid-games, 222 robust-endgame candidates, 16 requests
whose NPC pool is blocked so that Python's legal enumeration is empty and the random fallback is needed, 3 full boards). `tools/phase7/e2e_reference.py` records, for each request (max_depth 4 and,
for positions with 5 or 6 empty cells, the full depth), **the payload the Python server really sends** (obtained by running `ai_move` with the engine call replaced by a recorder) and what Python
computes on copies of the pristine state: its standard search per depth (with its table; and with the table switched off, which is exact depth-limited minimax), and the robust table on the
request-order state. `EngineEndToEndTests` sends every payload to the engine and checks (404 records):

| What | Count |
|---|---|
| records (request × depth variant: max_depth 4, and the full depth for positions with 5-6 empty cells) | 404 from 360 requests |
| of which robust-endgame records / records on the fallback-candidates path (the engine asked for candidates before searching, then accepted them) / records with no move | 209 / 16 / 13 |
| completed search levels compared | 1,392 |
| engine root score **bit-identical to the oracle** | 1,392 of 1,392 |
| engine move **in the exact tie set** | 1,392 of 1,392 |
| exact tie-set size over all levels (1 / 2 / 3 / 4 / 6 moves) | 1,294 / 86 / 8 / 3 / 1 |
| Python move per level identical to the engine's | 1,381 |
| Python move different but in the exact tie set (Q-5 b) | 6 |
| Python move outside the tie set, **explained by D-006** (Python's own score differs from the exact value, Python without its table is in the tie set) | 5 |
| **unexplained** | **0** |
| Python's score equal to the exact value / different (D-006) | 1,384 / 8 |
| robust candidate tables **bit-identical to Python's** (move, `final_score`, `safety_ratio`, `robust_score`, `corner_risk`, order), records | 209 of 209 (1,274 candidates) |
| scenario counts equal (101 of them are the base state as the only scenario) | 209 of 209 |
| engine final move == override decision from Python's table and the engine's standard move: overridden / standard kept | 96 / 113 |

The invalid-request path (side to move without cards) did not occur in the corpus; it is covered by the engine tests. M4-3 is closed (section 1).

Methods: the oracle is ReferenceCompat with the table off; the exact tie set of a depth is the set of root moves whose full-window value has the same bits as the root score, computed in the test.
A Python move outside the tie set is "explained by D-006" only if Python's own score at that depth differs from the exact value **and** Python run without its table finds a move in the tie set.
The override check uses the engine's standard move with **Python's** table and robust move (`EndgameRobust.DecideOverride`, whose equality with the block extracted from `ai_move` by AST is
established by the Phase 5 override golden).

## 6. Engine-off equivalence

`tools/phase7/equivalence.py` captures `/ai_move` responses through Flask's test client (fixed `random.seed` per request, `max_depth` 4, `max_time` 10^6) from a checkout of commit
`03b4704` (the last commit before Phase 7, a separate git worktree) and from the new code with `TTC_CS_ENGINE=off`, and compares them. Result on the 360-request corpus: **360 of 360 responses identical**
(347 with a move, 13 without; status codes equal). Excluded fields, all of which measure time: `search_progress[*].time_elapsed / time_remaining / nodes_per_second`, `search_summary.total_time_seconds /
search_efficiency`, and the intermediate `search_progress` entries without a best move that minimax itself emits every `PROGRESS_INTERVAL` seconds (their number depends on the wall clock: the same request
gave 0, 1 or 2 of them from run to run in the *old* code as well). The comparison was run with `PYTHONHASHSEED=0` and without it (the first, unseeded run differed in exactly those intermediate entries
and in nothing else). `ai_move` with the engine on but unreachable (or any other fallback) returns the same response plus the `engine` / `fallback_reason` fields (every fallback test compares it with the
engine-off response). The regression sample `tests/test_engine_off_equivalence.py` (43 requests, expected responses stored in `tests/data/engine_off_expected.json`) runs in the normal pytest suite.
The solver block moved into `_solve_minimax_python` is AST-identical to the old inline block.

## 7. Concurrency determinism (xUnit, `EngineConcurrencyTests`)

* `Concurrent_responses_equal_a_sequential_run_on_a_fresh_engine_field_by_field`: 230 different requests (170 standard searches on golden positions at depth 4 and 60 robust-endgame requests at depth 3,
  both modes) are sent with 16 requests in flight to an engine with 4 workers, twice with different orders; every response equals, field by field except the timings and the worker id, the response of a
  **sequential run on a fresh single-worker engine**. 0 differences.
* `A_request_never_sees_what_an_earlier_request_left_on_the_same_worker`: the same 60 requests on one worker in two orders give equal answers, and a sample equals the answer of an engine that had never served
  anything. `A_full_queue_answers_busy_and_the_engine_recovers`, `The_worker_pool_is_bounded_and_creates_contexts_lazily`.
* Across concurrency levels and worker counts (`tools/phase7/determinism_check.py`, `docs/migration/reports/phase7/determinism_check.txt`): the engine replayed at concurrency 1, 4, 16 with 1, 8, 20 workers
  (openings, mid-games, robust endgames; 346 distinct requests whose search is not cut by the time limit) and the Flask server in front of it (94 requests without unknown cards): **36 cells, 0 differences**.
  The load test (section 8.2) repeats the within-cell check for every cell of the matrix (all identical).

## 8. Load test and latency

Machine: i7-13650HX (6 P-cores with SMT = logical CPUs 0–11, mask `0xFFF`; 8 E-cores = 12–19; 20 logical CPUs), 32 GB, Windows 11, .NET 10, CPython 3.11.4. **These numbers describe this machine only;
nothing here is an estimate for the Windows Server.** The procedure for measuring on the server is in `docs/DEPLOY_WINDOWS.md` section 8.

### 8.1 Single-request latency (pinned protocol)

Python solver (engine off) versus the engine path (Flask + engine), same requests, production settings (`max_depth` 10, `max_time` 10 s), both processes pinned to `0xFFF`, 3 independent runs
each started with fresh processes; cold = first request of its kind in a new process, warm = median of the 2nd and 3rd pass (`tools/phase7/latency.py`). "depth" is the search depth reached.

CPU set: 0xFFF, 3 independent runs. Seconds, min / median / max.

| request | tags | Python solver, warm | engine path, warm | Python, cold (median) | engine path, cold (median) | depth Python / engine |
|---|---|---|---|---|---|---|
| 22 | rule:同类强化, rule:同类弱化, k:0, unknown:0 | 10.0170 / 10.0185 / 10.0197 | 0.5621 / 0.5858 / 0.6345 | 10.026 | 0.797 | 5 / 9 |
| 23 | rule:同数, rule:加算, k:0, unknown:1 | 10.0274 / 10.0284 / 10.0314 | 2.4636 / 2.4858 / 2.7494 | 10.029 | 2.534 | 5 / 9 |
| 27 | rule:同类强化, k:0, unknown:0 | 10.0224 / 10.0232 / 10.0232 | 0.7918 / 0.8151 / 0.9165 | 10.025 | 0.829 | 5 / 9 |
| 7 | rule:同数, rule:同类强化, rule:秩序, k:2 | 3.7618 / 3.9301 / 4.0249 | 0.0081 / 0.0089 / 0.0094 | 3.958 | 0.011 | 10 / 7 |
| 14 | rule:none, k:3, unknown:0 | 3.7677 / 3.8164 / 4.0176 | 0.0043 / 0.0045 / 0.0061 | 3.687 | 0.005 | 10 / 6 |
| 5 | rule:同类强化, k:4, unknown:1 | 0.5306 / 0.5350 / 0.5386 | 0.0026 / 0.0035 / 0.0036 | 0.505 | 0.003 | 10 / 5 |
| 1 | rule:同数, rule:同类弱化, k:5, unknown:0 | 0.2242 / 0.2296 / 0.2311 | 0.0023 / 0.0023 / 0.0025 | 0.216 | 0.022 | 10 / 4 |
| 11 | rule:加算, rule:逆转, rule:同类弱化, k:5 | 0.2971 / 0.2996 / 0.3035 | 0.0025 / 0.0028 / 0.0032 | 0.283 | 0.003 | 10 / 4 |
| 2 | rule:同数, rule:加算, rule:同类弱化, rule:混乱 | 0.0407 / 0.0410 / 0.0411 | 0.0043 / 0.0044 / 0.0053 | 0.041 | 0.018 | 10 / 3 |
| 3 | rule:混乱, rule:选拔, k:6, unknown:1 | 0.0817 / 0.0854 / 0.0884 | 0.0052 / 0.0066 / 0.0091 | 0.079 | 0.006 | 10 / 3 |
| 4 | rule:同数, rule:加算, rule:同类强化, rule:混乱 | 0.0129 / 0.0151 / 0.0163 | 0.0039 / 0.0039 / 0.0042 | 0.014 | 0.005 | None / None |
| 10 | rule:加算, k:7, unknown:1 | 0.0193 / 0.0200 / 0.0203 | 0.0020 / 0.0028 / 0.0033 | 0.019 | 0.003 | 10 / 2 |


Reading it: on openings the Python search runs into its 10 s limit at depth 5, while the engine solves the whole game tree (depth 9) in 0.6–2.5 s; mid-game and endgame requests take 2–9 ms on the
engine path against 15 ms–4 s in Python. For a few cheap endgame requests the ratio is small because Flask's parsing and the analysis text dominate (the Python overhead is reported in 8.2).

### 8.2 Load test

`tools/phase7/run_load_matrix.py` starts the engine for each worker count with a CPU affinity (all CPUs, or `0xFFF`), then `tools/bench/load_test.py` replays the corpus at concurrency 1, 2, 4, 8, 12, 16
with production settings. Cells show requests per second / p95 latency in seconds. Openings: `max(12, 3 × concurrency)` requests per cell, mid-games 400, robust endgames 600.
The determinism check of the load test (normalised responses identical at every concurrency; searches that hit their time limit excluded) is reported per category in the raw files
(`docs/migration/reports/phase7/load_*.txt`).


**engine directly, opening requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=2 | c=4 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|
| cpu=0xFFF workers=1 | 0.9 / 2.364 | 0.9 / 5.684 | 1.1 / 7.465 | 1.3 / 10.241 | 1.0 / 15.223 | 1.0 / 21.636 |
| cpu=0xFFF workers=4 | 1.2 / 1.642 | 2.0 / 1.693 | 2.3 / 1.839 | 3.8 / 4.260 | 3.0 / 5.593 | 3.7 / 6.515 |
| cpu=0xFFF workers=8 | 1.1 / 1.698 | 1.9 / 1.833 | 2.2 / 2.092 | 4.3 / 3.817 | 3.9 / 5.900 | 4.8 / 6.788 |
| cpu=0xFFF workers=12 | 1.1 / 1.703 | 1.9 / 1.843 | 2.2 / 1.933 | 4.1 / 3.890 | 4.5 / 5.435 | 5.6 / 6.438 |
| cpu=0xFFF workers=20 | 1.1 / 1.705 | 1.8 / 1.857 | 2.2 / 1.941 | 3.9 / 4.337 | 4.4 / 5.371 | 5.5 / 6.522 |
| cpu=all (0xFFFFF) workers=1 | 0.5 / 3.407 | 0.5 / 10.013 | 0.5 / 14.096 | 0.6 / 19.323 | 0.5 / 25.237 | 0.6 / 37.149 |
| cpu=all (0xFFFFF) workers=4 | 0.5 / 4.207 | 1.0 / 4.073 | 1.3 / 3.319 | 2.2 / 7.120 | 1.7 / 10.839 | 1.7 / 14.680 |
| cpu=all (0xFFFFF) workers=8 | 0.7 / 2.781 | 1.2 / 2.928 | 1.3 / 3.137 | 2.4 / 7.890 | 2.3 / 10.157 | 3.1 / 10.011 |
| cpu=all (0xFFFFF) workers=12 | 0.7 / 2.828 | 1.1 / 3.030 | 1.2 / 3.229 | 2.3 / 7.697 | 2.7 / 10.002 | 3.2 / 10.741 |
| cpu=all (0xFFFFF) workers=20 | 0.6 / 4.473 | 1.0 / 3.762 | 1.2 / 3.763 | 2.4 / 7.407 | 2.7 / 9.272 | 3.4 / 10.002 |

**engine directly, mid requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=2 | c=4 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|
| cpu=0xFFF workers=1 | 22.7 / 0.223 | 24.6 / 0.289 | 25.1 / 0.606 | 23.2 / 0.845 | 26.4 / 0.959 | 30.4 / 0.854 |
| cpu=0xFFF workers=4 | 30.4 / 0.169 | 57.3 / 0.177 | 91.9 / 0.210 | 96.3 / 0.229 | 97.7 / 0.294 | 97.6 / 0.325 |
| cpu=0xFFF workers=8 | 28.2 / 0.174 | 52.6 / 0.190 | 98.1 / 0.203 | 141.0 / 0.243 | 127.5 / 0.318 | 127.2 / 0.313 |
| cpu=0xFFF workers=12 | 29.1 / 0.175 | 54.5 / 0.188 | 93.8 / 0.212 | 129.6 / 0.279 | 159.2 / 0.338 | 157.5 / 0.348 |
| cpu=0xFFF workers=20 | 27.7 / 0.183 | 54.5 / 0.187 | 100.3 / 0.194 | 140.5 / 0.250 | 159.8 / 0.337 | 160.2 / 0.419 |
| cpu=all (0xFFFFF) workers=1 | 13.7 / 0.367 | 14.4 / 0.522 | 14.7 / 1.020 | 14.3 / 1.367 | 11.9 / 2.116 | 12.0 / 2.461 |
| cpu=all (0xFFFFF) workers=4 | 17.2 / 0.296 | 32.4 / 0.320 | 59.8 / 0.320 | 60.7 / 0.366 | 57.8 / 0.495 | 45.6 / 0.735 |
| cpu=all (0xFFFFF) workers=8 | 17.1 / 0.290 | 31.7 / 0.329 | 61.1 / 0.324 | 84.1 / 0.430 | 83.4 / 0.456 | 85.5 / 0.469 |
| cpu=all (0xFFFFF) workers=12 | 16.4 / 0.301 | 31.4 / 0.311 | 60.2 / 0.324 | 84.0 / 0.418 | 96.6 / 0.559 | 95.9 / 0.555 |
| cpu=all (0xFFFFF) workers=20 | 20.1 / 0.257 | 41.5 / 0.245 | 71.8 / 0.273 | 81.8 / 0.417 | 96.3 / 0.548 | 112.8 / 0.596 |

**engine directly, robust requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=2 | c=4 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|
| cpu=0xFFF workers=1 | 1603.4 / 0.001 | 2246.7 / 0.002 | 2895.4 / 0.002 | 3711.4 / 0.003 | 3850.4 / 0.007 | 3704.5 / 0.009 |
| cpu=0xFFF workers=4 | 1523.7 / 0.001 | 3277.9 / 0.001 | 4180.2 / 0.002 | 4298.1 / 0.003 | 4807.1 / 0.004 | 4689.1 / 0.006 |
| cpu=0xFFF workers=8 | 1590.4 / 0.001 | 3366.7 / 0.001 | 4455.6 / 0.002 | 4713.9 / 0.003 | 4603.3 / 0.005 | 3907.8 / 0.010 |
| cpu=0xFFF workers=12 | 1505.7 / 0.002 | 2965.1 / 0.002 | 4801.5 / 0.002 | 5052.2 / 0.003 | 5041.2 / 0.004 | 4695.0 / 0.007 |
| cpu=0xFFF workers=20 | 1776.9 / 0.001 | 3270.1 / 0.001 | 3877.0 / 0.002 | 5417.8 / 0.003 | 4989.1 / 0.004 | 5528.4 / 0.005 |
| cpu=all (0xFFFFF) workers=1 | 556.9 / 0.005 | 895.0 / 0.006 | 886.0 / 0.009 | 1188.7 / 0.013 | 1138.2 / 0.017 | 946.5 / 0.042 |
| cpu=all (0xFFFFF) workers=4 | 698.0 / 0.004 | 1383.8 / 0.004 | 2057.8 / 0.004 | 2275.7 / 0.006 | 2448.1 / 0.008 | 2338.3 / 0.011 |
| cpu=all (0xFFFFF) workers=8 | 996.5 / 0.002 | 1860.7 / 0.002 | 2324.9 / 0.003 | 2496.4 / 0.005 | 2546.8 / 0.008 | 2408.5 / 0.011 |
| cpu=all (0xFFFFF) workers=12 | 858.5 / 0.003 | 1712.8 / 0.002 | 2373.9 / 0.003 | 2596.8 / 0.005 | 2549.4 / 0.007 | 2523.7 / 0.010 |
| cpu=all (0xFFFFF) workers=20 | 1033.2 / 0.002 | 2243.9 / 0.002 | 2726.5 / 0.003 | 3156.8 / 0.004 | 3178.8 / 0.006 | 3315.9 / 0.008 |

**Flask + engine, opening requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| cpu=0xFFF engine workers=20 + Flask | 0.9 / 2.361 fb=0 | 2.2 / 1.646 fb=0 | 4.0 / 4.545 fb=0 | 5.5 / 6.815 fb=0 |
| cpu=all (0xFFFFF) engine workers=20 + Flask | 0.6 / 4.989 fb=0 | 1.6 / 4.916 fb=0 | 2.8 / 6.359 fb=0 | 3.4 / 10.027 fb=0 |

**Flask + engine, mid requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| cpu=0xFFF engine workers=20 + Flask | 20.3 / 0.138 fb=0 | 72.8 / 0.180 fb=0 | 96.3 / 0.221 fb=0 | 111.7 / 0.322 fb=0 |
| cpu=all (0xFFFFF) engine workers=20 + Flask | 15.9 / 0.230 fb=0 | 47.8 / 0.281 fb=0 | 69.6 / 0.400 fb=0 | 75.9 / 0.523 fb=0 |

**Flask + engine, robust requests** (requests/s; p95 latency in s)

| configuration | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| cpu=0xFFF engine workers=20 + Flask | 56.6 / 0.032 fb=0 | 286.8 / 0.032 fb=0 | 440.5 / 0.032 fb=0 | 451.6 / 0.045 fb=0 |
| cpu=all (0xFFFFF) engine workers=20 + Flask | 51.7 / 0.040 fb=0 | 232.0 / 0.036 fb=0 | 238.3 / 0.052 fb=0 | 234.1 / 0.089 fb=0 |

**Engine process memory after the load**

| configuration | working set MiB | private MiB | peak working set MiB |
|---|---|---|---|
| cpu=0xFFF workers=1 | 169 | 133 | 170 |
| cpu=0xFFF workers=4 | 464 | 429 | 464 |
| cpu=0xFFF workers=8 | 842 | 808 | 848 |
| cpu=0xFFF workers=12 | 1231 | 1197 | 1234 |
| cpu=0xFFF workers=20 | 1621 | 1590 | 1621 |
| cpu=all (0xFFFFF) workers=1 | 164 | 130 | 168 |
| cpu=all (0xFFFFF) workers=4 | 461 | 428 | 461 |
| cpu=all (0xFFFFF) workers=8 | 845 | 813 | 846 |
| cpu=all (0xFFFFF) workers=12 | 1233 | 1201 | 1236 |
| cpu=all (0xFFFFF) workers=20 | 1622 | 1592 | 1622 |

Reading the numbers (this machine only):

* **Openings** (the whole 9-ply game tree; a request takes about 0.5–2.5 s on one worker, a few hit the 10 s limit): one stream gives 0.9–1.1 requests/s on the P-cores; with 12 workers and 16 concurrent
  requests 5.6 requests/s. Throughput stops growing near as many workers as the CPU set has logical CPUs (12 on `0xFFF`): 12 and 20 workers are equal. With **all 20 logical CPUs** the same engine is
  *slower* (3.4 requests/s at 20 workers) than pinned to the six P-cores, because on this hybrid CPU the scheduler places workers on the slower E-cores and the load generator competes. The server
  (E5-2698B v3) has no such split, so do not read this as advice for it.
* **Mid-games and robust endgames** take a median of 2–6 ms inside the engine (the slowest mid-games 0.2–0.4 s at p95): 160 requests/s (mid-game) and about 5,500 requests/s (robust) at 20 workers on the P-cores.
* **The whole Flask path** is limited by Python: the time spent in Python per request (parsing, unknown-card sampling, analysis text, win probability; latency minus the engine's own time) is a median of
  9–36 ms in most cells (p95 up to 47 ms; up to 63 ms median and 95 ms p95 when 16 requests queue on the GIL on all CPUs). Because of the GIL the Flask path tops out at about 450 requests/s (P-cores) or 235 requests/s (all CPUs) for the cheap
  robust requests while the engine alone does 3,300–5,500; for openings the engine dominates (5.5 requests/s with Flask in front, equal to the engine alone).
* **No fallback** occurred in any Flask cell (`fb=0`), and no cell had errors or `busy` answers.
* **Memory:** the engine process after the load: about 77 MiB per worker that has been used (the 96 MiB tables are touched on demand) plus about 90 MiB for the runtime and the HTTP stack: 164 MiB with 1 worker,
  461 MiB with 4, 845 MiB with 8, 1.2 GiB with 12, 1.6 GiB with 20 (working set; private bytes are about 30 MiB lower). An idle engine uses 50 MiB (tables are created on first use).

## 9. Changes in the shared C# code (allowed set only)

* `CardSnapshot.FromJson`: an owner other than `"r"`, `"b"`, `null` is rejected (L-9).
* `CompatSearcher.RunIterativeDeepening`: `maxDepth` clamped to Python's `min(max_depth, 100)` (L-1).
* `SearchLimits`: NaN = no limit, zero or less = already over, `DeadlineFrom` handles limits too large for the clock (L-2); both searchers use it. Positive limits behave as before.
* `SearchStats` + `FastSearcher.RunIterativeDeepening`: per completed level, the start/end clock and the running TT-hit and cutoff totals are stored (one clock read per level, for the engine's
  per-depth report). No change to the search itself; the existing 78 tests pass unchanged.

## 10. Known limits and remaining work (Phase 8 and later)

* The unknown-card sampling, request parsing, analysis text and Flask itself are still Python; the random fallback of the robust layer (`handler.generate_opponent_cards`) is called by the Python
  client and sent to the engine.
* The engine path never runs Python's search, so the `SEARCH_STATS` global is not updated and `performance_stats.search_depth` is the engine's (the engine stops once the game tree is searched, D-014).
* `branching_factor` in `search_progress` is the number of root moves (Python's value for the root); `nodes_per_second` uses the cumulative time of the level.
* A search that hits its time limit depends on the machine speed (as in Python); only fixed-depth results are comparable between runs.
* Flask's built-in server remains the HTTP front end (development server; see `docs/DEPLOY_WINDOWS.md` for a WSGI option). One connection per request thread is opened to the engine when the
  front end creates a new thread per connection.
* Not done by design: root parallelism, Lazy SMP, any performance work on the search, replacing Flask, porting `parse_hand`/sampling/analysis text/Monte Carlo (Phase 8).

---

## Notes added after the independent review (2026-10-10, Phase 7 hardening)

The text above is the Phase 7 delivery as it was; it is not rewritten. Two notes:

**Absolute latencies vary between sessions (review L7-2).** The reviewer re-ran the pinned measurements of section 8.1 and found every absolute time 35-45% lower than the
figures above, on **both** sides (Python and engine) and in the same proportion (request 7: about 440x in this report, about 470x in the review; Python's time-limited depth on
opening 27 rose from 5 to 6). It is a difference in the state of the machine between the two sessions, not an error in either measurement. The figures in this report are the
conservative ones. Absolute seconds, requests per second and memory figures of different sessions must not be compared with each other; **only ratios and comparisons made inside one
session (same machine state, same pinning, interleaved or back to back) are meaningful.** The same holds for every later measurement in `docs/migration/reports/PHASE7_HARDENING.md`.

**The automatic fallback described in sections 1, 3 and 4 was withdrawn** by the user's decision A-20 (`CSHARP_MIGRATION.md` section 8): while the engine is on it is the only
minimax solver, and every engine problem is answered with HTTP 503 and an `engine_error` code. The fallback matrix of section 4 is replaced by the error matrix in
`docs/migration/reports/PHASE7_HARDENING.md` and `docs/DEPLOY_WINDOWS.md` section 6; the circuit breaker no longer exists. The measurements of this report (latency, load, memory, the end-to-end
and equivalence comparisons) are unaffected: they never involved a fallback.
