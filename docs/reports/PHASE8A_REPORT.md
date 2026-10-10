# Phase 8A report — the C# server becomes the product backend

Date: 2026-10-11. Branch `csharp-solver-migration`, commits are local (nothing pushed). Authorisation: `.claude/phase8a_goal.md`. Status: **done, awaiting independent review** (no gate is set here).

## 1. What was done

| Step | Result | Commit |
|---|---|---|
| A | Reviewer files committed (hardening review, Phase 8 plan, mascot design, decisions A-21…A-24, Q-9, this phase's goal). The nine PNG originals in `assets/mascot/raw/` are **not** committed (the goal names only the README and `.gitkeep`; ~21 MB of art awaits the user's decision). | `8f78212` |
| B | RNG, request parsing, unknown-card sampling, native `/ai_move`, Monte Carlo retired. | `5afa01c`, `9b9fcba`, `7fc3273` |
| C | Per-level search callback; explanation facts v2 with Chinese templates; game logs; live events. | `4d6f2e3`, `968118b`, `833c8f1` |
| D | Priority scheduling with aging; API keys, limits, HTTPS, CORS, proxy, placeholder page; deployment guide; reports. | `8c97537`, `489edb6`, and the documentation commit |

### Project layout (`csharp/`)

* `TTC.Siren.Core`, `TTC.Siren.Search` — unchanged except one thing: `FastSearcher.RunIterativeDeepening` takes an optional `onLevel` callback, called once per completed depth (never per node).
* `TTC.Siren.Engine` — kept as a library and as the loopback host (`/v1/solve`) used by the tools and the Phase 7 tests. `SolveService` and `WorkerPool` live here and the server uses them **in process**; nothing is duplicated. `WorkerPool` got the cost-ordered queue with aging (O8-1); `SolveService` got optional progress hooks.
* `TTC.Siren.Compat` (new) — `PythonRandom` (CPython 3.11 `random`: MT19937, `getrandbits`, `_randbelow`, `randrange`/`randint`, `choice`, both `sample` algorithms, `shuffle`, `uniform`), the Python object model of the compatibility layer (`Legacy*`: request-time only, not in any search), request parser, `UnknownCardHandler`, `LegacyAnalysis` (the old response texts), `PyJson`/`Py` (Python's `round`, `repr(float)`, truthiness, `int()`).
* `TTC.Siren.Explain` (new) — `FlipAnalyzer`, the facts, `Explainer`, the template renderer and `templates/explain.zh-CN.json`.
* `TTC.Siren.Server` (new) — `ServerHost` (Kestrel), `AiMoveService`, game log (`GameLog/`), live events (`Live/`), keys/limits (`Security/`), proxy (`Proxy/`).
* `tools/phase8a/` — golden generators (RNG, parse, sampling, format, end-to-end), the request corpus generator, the HTTP latency and load tools, table renderer. Golden data is generated on demand by the tests and bound to the Python source by SHA-256 (`golden_sources8.py`, kept separate so that no older golden was invalidated); it is not committed.

## 2. Parity with Python (every comparison is exact; floats bit for bit)

| What | Compared | Result |
|---|---|---|
| CPython `random` | 600 seeded generators (seeds 0, 32/64-bit boundaries, > 64 bits, negative), > 10,000 operations in random mixes (`random`, `getrandbits`, `_randbelow`, `randrange`, `randint`, `choice`, `sample` with both algorithms, `shuffle`, `uniform`) | all equal |
| Request parsing | the 360 requests of the Phase 7 corpus + 2,400 new seeded requests (all rules and combinations, none, open modes, NPC pools of every shape, `inHand=false`, `canUse=false`, duplicate ids, `currentPlayer` 0/1/2/odd, boards with the all-zero rows, 76 requests Python refuses) | 2,684 parsed requests equal field for field (39,288 card fields: ids, types, owners, modifiers, base and display edges, flags, hand order), 76 refusals equal in status **and error text**; used ids, sample count and robust decision equal; the random generator stands at the same place afterwards in every case (same number of draws) |
| Unknown-card sampling | 5,400 seeded situations: every rule alone, pairs, triples, all nine, none (378 combinations), 1–5 unknown slots (1351/1371/1332/662/684), the opponent's and the player's own estimate, empty to nearly full boards, known 4/5-star cards, and the slot selection (random sample or highest threat: 2,143 cases) | 116,078 cards equal |
| Python's `round`, `repr(float)` | 6,000 values incl. exact ties | equal; `PyJson` reproduces Flask's `jsonify` form (sorted keys, `1.0`, `\uXXXX`) |
| End-to-end `/ai_move` | the Python server on its production path (C# engine on, same seed, same table size) against the C# server: the 360 Phase 7 requests + 600 fresh ones = 960 requests | 960 equal: 867 moves, 33 error responses (400: 6, 500: 14, 503: 13), 60 no-move/other 200 answers; 441 robust endgames (181 overridden), 85 with `show_search_progress`, 10 `solver=monte_carlo`; 44,569 values compared bit for bit |
| Over a real socket | a sample of the same requests through HTTP with the test seed header | equal |

Excluded from the end-to-end comparison (listed in `AiMoveParityTests.IsTiming`): `engine_info.timings`, `engine_info.worker`, and in `search_progress` the fields `time_elapsed`, `time_remaining`, `nodes_per_second` and in `search_summary` `total_time_seconds`, `search_efficiency` (they measure the clock or name a worker). The new fields exist only in the C# answer: `solver_used`, `solver_note` (only for `monte_carlo` requests), `explanation`, `game_id`.

**What the end-to-end comparison found** (this is why it exists): the first difference was a `type_modifier` in the description of the chosen card (`兽人+2` vs `+1`). Python plays the chosen card on a copy of the state before answering (`new_state.play_move(...)`, for its console), and because it passes the original card object, the type modifiers of that very card are recalculated. Reproduced as `PY-QUIRK` (D-023). All other differences of the porting process were found by the parse and sampling comparisons before they reached this stage.

### Non-exact items (all in `CSHARP_MIGRATION.md` §4, none unexplained)

* **D-020** a malformed JSON body: Python answers 500 with Flask's BadRequest text, C# 400 `Invalid input data`; any content type is accepted.
* **D-021** exotic JSON types in request fields: Python-specific exception texts are not all reproduced. Reproduced and tested exactly: missing keys, unknown card, index out of range, `KeyError: 0` of the draft rule. Integral floats and bools count as ints as in Python; ids beyond 2^63 are skipped.
* **D-022** production requests use a fresh 64-bit seed, not Python's one global stream; equality holds per seed.
* **D-023, D-024, D-025** Python quirks reproduced on purpose (the `play_move` on a copy, make/undo reordering of hands in the analysis texts, the handler bypass with no rules and the duplicate filter that compares 1000+id with id).
* Statistical tests were **not** needed: no construct depends on string hashing or set iteration order.

## 3. Monte Carlo (A-22)

`solver: "monte_carlo"` is accepted and answered by the minimax engine path (including the robust endgame, which Python never ran for it); the answer carries `solver_used: "minimax"` and a Chinese `solver_note`; `mc_simulations` is ignored. `ai/monte_carlo.py` is untouched in the repository. `API.md` is updated.

## 4. Explanation facts v2 (A-24)

Eight families, each verified by recomputation (`ExplanationFactsTests`, `FlipAnalyzerTests`; details in `docs/design/EXPLANATION_FACTS.md`):

| Family | Recomputed against | Result on the corpus |
|---|---|---|
| immediate effect: flips with causes, counts, modifier changes | the engine's `MakeMove` (flipped cells, owners, counts); the six causes via `FlipAnalyzer`, which equals the engine on 715,577 moves of 67,376 positions of the Python rules trace (194,691 with flips; Basic 97,557, Reverse 105,877, FallenAce 2,445, Same 2,258, Plus 13,927, Combo 5,305) | 583 decisions, 441 flips, 89 modifier changes |
| exposure and threats; probability that an unknown opponent card takes the placed card | every opponent card × every empty cell by `MakeMove`; every legal candidate card by `MakeMove` | 302 threats confirmed one by one, completeness checked (no other card captures); probabilities equal the brute force exactly |
| opponent reply and principal variation | legal replay with `MakeMove` | 1,766 plies replayed, 506 replies (336 from the search's own table, 170 from a bounded fresh search) |
| outcome | exact: a brute-force minimax (≤ 5 empty cells) and the exact solver (6); heuristic: the search score | 507 exact outcomes (win 218, draw 155, loss 134) |
| alternatives and why they are worse | fixed-depth searches (the best listed score equals the root value; ties are exact ties) | 531 decisions, 1,893 scores, 34 exact ties |
| robust endgame | the engine's candidate table: final score, robust score and safety ratio equal bit for bit; the override clause that fired | 414 decisions, 181 overrides |
| unknown cards, confidence | the request and the search statistics | all equal |
| text | every number in 583 brief and full texts (30,702 numbers) is a number of the facts; no win claim against the exact value; at most two sentences in the brief; 30 golden snapshots (`tests/data/explanation_snapshots_v2.json`) | |

The facts of a decision are reproducible: every search the explainer runs uses a node budget, never the clock.

### Examples (generated text; `explain=brief` gives the first block, `full` all)

**An exactly solved position** (corpus p7#53)

> **Brief:** 把第468号牌（上3右6下7左6）放在中心，当场翻转了1张对手的牌（基础比较）。已经算到终局：按双方手牌全部已知，双方最终牌数相同，结果是平局。

```text
这手把第468号牌（上3右6下7左6）放在中心。具体翻了：左边的第194号牌（左边6对5，数值更大）。牌数从我方5对对手5，变为我方6对对手4。

这手落在中心位置。放下后没有边朝向空格，下一手对手无法直接从相邻位置攻击这张牌。按现有的推测，下一手没有任何对手的牌能翻回这张牌。

搜索认为对手最好的应对是把第50号牌放在右上角，翻转1张牌。预计的后续走法：我方第468号牌放中心（翻1张），对手第50号牌放右上角（翻1张）。

已经算到终局：按双方手牌全部已知，双方最终牌数相同，结果是平局。

对比其他走法（同样搜索到第3层，当前选择评分-0.67）：
1. 第468号牌放右上角：评分-2.93，比当前选择低2.26；当场只翻0张，而当前选择当场翻1张
2. 第468号牌放左下角：评分-3.45，比当前选择低2.78；当场只翻0张，而当前选择当场翻1张

残局已完全算尽（对手手牌无需推测）：这手的最终牌差为0。常规搜索的首选与残局信息集分析一致。

对手没有需要推测的未知牌。

可信度：已搜到终局，深度3层，20个节点。
```

**A heuristic position with unknown cards and alternatives** (corpus p7#0)

> **Brief:** 第158号牌（上7右3下4左5）落在下边，当场没有翻牌，是在为后面的局面做准备。向后推演4层，评分-2.45，形势对我方不太有利。

```text
这手把第158号牌（上7右3下4左5）放在下边，没有翻牌，牌数仍是我方6张对对手4张。

这手落在边缘位置。朝向空格的边有：上边7、右边3、左边5，下一手对手可以从这些方向进攻。对手的第303号牌（推测牌）放在中心，可以靠基础比较翻回这张牌（该手一共翻1张）。（推测牌）第351号牌落在左下角时能以基础比较夺回这张牌，同时共翻1张。在469张符合条件的候选牌里，大约91%能让对手下一手夺回这张牌。

搜索认为对手最好的应对是把第351号牌（推测牌）放在右上角，翻转1张牌。预计的后续走法：我方第158号牌放下边（翻0张），对手第351号牌放右上角（翻1张），我方第336号牌放左边（翻0张），对手第303号牌放右边（翻0张）。

向后推演4层，评分-2.45，形势对我方不太有利。这是启发式评分，不是最终胜负：搜索深度第4层，评分-2.45（距终局还有8格）。

其他走法（统一按第4层评分，当前选择的评分为-2.45）：
1. 第158号牌放中心：评分-2.89，比当前选择低0.44；综合评分更低
2. 第336号牌放右上角：评分-2.89，比当前选择低0.44；综合评分更低
3. 第158号牌放左上角：评分-2.95，比当前选择低0.50；综合评分更低
4. 第158号牌放左下角：评分-3.05，比当前选择低0.60；综合评分更低

对手有3张未知牌，按规则驱动的对手行为模型（主要规则：选拔）做了推测，本次假设为：第303号牌（5星）、第351号牌（3星）、第120号牌（5星）。

可信度：搜索深度第4层（请求上限4层，还有8个空格），共1034个节点，结论是启发式的。
```


## 5. Game logs (A-21) — schema version 1

`docs/design/GAME_LOG_SCHEMA.md`. One JSONL file per game under `<GameLog.Directory>/<key id>/`, records `game_start`, `request` (raw request, seed, settings, state, result with per-depth statistics, explanation), `opponent_move` (inferred between requests, with flip causes recomputed by the rules), `game_end`. Game identity: the optional `gameId`, else inference (same key, rules and colour, same side to move, every old card in place with the same edges, ≤ 4 new cards, gap ≤ 15 min; configurable). Retention 30 days (configurable) with a cleanup job at start and daily. API: `GET /v1/games`, `GET /v1/games/{id}`, `/download`, `DELETE`; a key sees only its own games (404 for another's).

Tested (`GameLogTests`): **36 complete games**, the engine on both sides, requests built as the plugin builds them (12 rule sets, both colours, both first movers, with and without `gameId`): the log reconstructs the whole timeline exactly — 174 requests, 138 inferred opponent moves with 73 flips and their causes (basic 46, reverse 17, plus 4, combo 4, same 2) equal to the simulator's truth, 30 game ends with the right counts and winner; and **162 of 162 logged decisions replay identically** (move, standard move, score and move of every depth, robust result, all facts and both texts). Also identity (client id, inference, the 15-minute gap, other colour, restart, invalid ids), retention, deletion, isolation of keys, path-traversal ids refused.

Known limit: if the opponent places the ninth card and the plugin never asks again, the server never sees that card (no record, no game end).

## 6. Live events (C3)

`GET /v1/live[?game_id=]` (SSE, the key's own events only): `request.received`, `search.depth` (from the per-level callback), `robust.progress` (coarse: start and end of the robust layer, because `EndgameRobust` is Search code that was not touched), `decision`, `game.updated`, `game.ended`, plus `lagged` notices. Tested (`LiveEventTests`): order of the events of a request (`request.received` < `search.depth` 1…n < the request record < `decision`), ids strictly increasing, the last depth equal to the answer's depth, the decision equal to the answer, isolation between keys and by game filter, and back-pressure: a consumer that never reads loses its oldest events (20,000 published in 49–97 ms, 19,992 dropped, the newest 8 kept in order); through HTTP with a buffer of 4, 5,000 events arrived as 1,266–1,694 delivered + 3,306–3,734 reported missed in 573–1,583 `lagged` notices (it varies from run to run with timing), never out of order and always summing to 5,000.

## 7. Security (D)

* API keys: random 256-bit (`ttc_` + 43 base64url characters), stored as SHA-256 only, shown once by `TTC.Siren.Server keys add`; `list`, `revoke`, `set-limit`; changes are picked up within a second; `Authorization: Bearer` or `X-Api-Key`; 401 / 403 (revoked) / 429 with `Retry-After`.
* Limits per key: token bucket (rate per minute, burst) and maximum in-flight requests; open event streams per key.
* HTTPS: Kestrel with a PFX (password from an environment variable), HSTS, security headers on every answer, CORS only for exact configured origins (no wildcard; other origins' preflight answers 403), request-body limit 1 MiB, header limits and timeouts, loopback-only operator endpoints (refused when a proxy header is present), and the server **refuses to listen on a non-loopback address without HTTPS or an explicit `BehindTlsProxy`**.
* Proxy: `/api/recommend`, `/api/draft/analyze`, `/api/winrate` to the local Python app behind the same key; the caller's headers (above all the key) are not passed on; 502 when Python is down, 504 on timeout; other paths 404.
* No key in any log: a test captures every log message at Trace level while valid, invalid, revoked and malformed requests are made and asserts that neither a secret, nor a part of it, nor the text `Bearer` appears (83 messages), nor in any file of the game-log folder. **Found and fixed while writing this report:** the first version logged the first eight characters of a *rejected* key (to identify the attempt); a mistyped real key would have leaked part of itself, and the test had asserted that prefix. The log line now says only that an unknown key was rejected, with the client address; revoked keys are logged by their public id; the test asserts that not even a prefix of a presented key appears.
* `SecurityTests`: 13 tests (keys, CLI, headers, revocation without restart, loopback exemption off by default, rate and in-flight limits, stream limit, logs, HTTPS with a self-signed certificate, CORS, operator endpoints, proxy, page).

## 8. Scheduling (O8-1)

The wait queue is ordered by estimated cost (empty cells, robust endgame): cheap, medium, expensive; FIFO inside a class; a waiting request moves up one class per `AgingMs` (2 s), so an opening is overtaken for at most `2 × AgingMs`. The review's probe (`SchedulerTests`, one worker, twelve openings of 300 ms, then five cheap requests):

| mode | cheap requests answered after | openings answered after | longest opening wait | answered |
|---|---|---|---|---|
| first come first served (switch off) | 3,664 … 3,734 ms (queue wait 3,175 … 3,394 ms on average) | 311 … 3,684 ms | 3,021 … 3,222 ms | 17 / 17 |
| priority with aging | **303 … 626 ms** (queue wait 214 … 222 ms on average) | 303 … 3,669 ms | 3,153 … 3,163 ms | 17 / 17 |

No opening starves (all answered, the longest wait is no longer than under FIFO); the bound is also tested on the pool alone with a constant stream of cheap requests.

## 9. Latency and load (this machine only: i7-13650HX, 6 P-cores = CPU set `0xFFF`, 8 E-cores; the load generator runs unpinned)

### /ai_move latency over HTTP, seconds (min / median / max of 3 independent runs), CPU set 0xFFF, 4 search workers

| request (corpus #) | empty cells | depth | Python server + engine, warm | C# server, warm | cold median Python / C# |
|---|---|---|---|---|---|
| #22 | 9 | 9 | 0.3471 / 0.3546 / 0.3778 | 0.3523 / 0.3542 / 0.3600 | 0.563 / 0.660 |
| #23 | 9 | 9 | 2.1670 / 2.5679 / 3.1655 | 2.5365 / 2.5660 / 3.8389 | 1.683 / 2.231 |
| #27 | 9 | 9 | 0.5133 / 0.5200 / 0.5231 | 0.5178 / 0.5196 / 0.5212 | 0.528 / 0.544 |
| #7 | 7 | 7 | 0.0073 / 0.0170 / 0.0287 | 0.0120 / 0.0128 / 0.0246 | 0.026 / 0.030 |
| #14 | 6 | 6 | 0.0158 / 0.0233 / 0.0294 | 0.0135 / 0.0164 / 0.0238 | 0.029 / 0.025 |
| #5 | 5 | 5 | 0.0032 / 0.0032 / 0.0210 | 0.0015 / 0.0073 / 0.0134 | 0.006 / 0.014 |
| #1 | 4 | 4 | 0.0025 / 0.0088 / 0.0141 | 0.0067 / 0.0082 / 0.0086 | 0.014 / 0.024 |
| #11 | 4 | 4 | 0.0089 / 0.0099 / 0.0132 | 0.0013 / 0.0074 / 0.0131 | 0.003 / 0.003 |
| #2 | ? | 3 | 0.0045 / 0.0101 / 0.0109 | 0.0016 / 0.0018 / 0.0197 | 0.013 / 0.026 |
| #3 | 3 | 3 | 0.0112 / 0.0180 / 0.0256 | 0.0056 / 0.0103 / 0.0130 | 0.005 / 0.011 |
| #4 | ? | None | 0.0038 / 0.0088 / 0.0161 | 0.0110 / 0.0117 / 0.0121 | 0.004 / 0.025 |
| #10 | 2 | 2 | 0.0133 / 0.0147 / 0.0174 | 0.0013 / 0.0015 / 0.0073 | 0.022 / 0.004 |

### Load test: C# server (key, limits, game log on), CPU set 0xFFF, 12 search workers (requests per second; p50 / p95 / p99 latency in seconds)

| category | concurrency | requests/s | p50 | p95 | p99 | max | non-200 |
|---|---|---|---|---|---|---|---|
| opening | 1 | 1.2 | 0.531 | 1.507 | 4.597 | 4.597 | 0 |
| opening | 4 | 2.1 | 0.543 | 4.599 | 5.186 | 5.186 | 0 |
| opening | 8 | 4.3 | 0.585 | 3.751 | 5.551 | 5.551 | 0 |
| opening | 16 | 6.1 | 1.023 | 5.555 | 6.819 | 6.819 | 0 |
| mid | 1 | 26.0 | 0.010 | 0.174 | 0.400 | 0.503 | 0 |
| mid | 4 | 95.2 | 0.008 | 0.190 | 0.540 | 0.561 | 0 |
| mid | 8 | 128.7 | 0.012 | 0.264 | 0.607 | 0.780 | 0 |
| mid | 16 | 150.8 | 0.034 | 0.352 | 0.725 | 0.987 | 0 |
| robust | 1 | 148.7 | 0.007 | 0.010 | 0.012 | 0.025 | 0 |
| robust | 4 | 618.8 | 0.005 | 0.014 | 0.019 | 0.035 | 0 |
| robust | 8 | 825.3 | 0.008 | 0.019 | 0.024 | 0.042 | 0 |
| robust | 16 | 930.1 | 0.016 | 0.030 | 0.038 | 0.042 | 0 |

### Load test: Python Flask + C# engine process, CPU set 0xFFF, 12 search workers (requests per second; p50 / p95 / p99 latency in seconds)

| category | concurrency | requests/s | p50 | p95 | p99 | max | non-200 |
|---|---|---|---|---|---|---|---|
| opening | 1 | 0.9 | 0.596 | 2.119 | 5.083 | 5.083 | 0 |
| opening | 4 | 2.2 | 0.474 | 4.417 | 5.057 | 5.057 | 0 |
| opening | 8 | 4.2 | 0.606 | 4.349 | 5.758 | 5.758 | 0 |
| opening | 16 | 5.1 | 1.084 | 6.702 | 8.228 | 8.228 | 0 |
| mid | 1 | 21.7 | 0.017 | 0.199 | 0.508 | 0.897 | 0 |
| mid | 4 | 79.7 | 0.012 | 0.223 | 0.583 | 0.740 | 0 |
| mid | 8 | 114.5 | 0.018 | 0.288 | 0.642 | 0.885 | 0 |
| mid | 16 | 133.6 | 0.043 | 0.379 | 0.758 | 1.023 | 0 |
| robust | 1 | 90.7 | 0.012 | 0.026 | 0.027 | 0.028 | 0 |
| robust | 4 | 313.7 | 0.009 | 0.029 | 0.032 | 0.035 | 0 |
| robust | 8 | 424.3 | 0.018 | 0.030 | 0.035 | 0.045 | 0 |
| robust | 16 | 402.4 | 0.038 | 0.052 | 0.063 | 0.067 | 0 |

### Load test: C# server (key, limits, game log on), CPU set all, 20 search workers (requests per second; p50 / p95 / p99 latency in seconds)

| category | concurrency | requests/s | p50 | p95 | p99 | max | non-200 |
|---|---|---|---|---|---|---|---|
| opening | 1 | 0.9 | 0.599 | 2.875 | 4.817 | 4.817 | 0 |
| opening | 4 | 2.2 | 0.576 | 2.436 | 4.911 | 4.911 | 0 |
| opening | 8 | 4.2 | 0.610 | 3.940 | 5.762 | 5.762 | 0 |
| opening | 16 | 5.5 | 0.986 | 6.620 | 8.227 | 8.227 | 0 |
| mid | 1 | 25.9 | 0.010 | 0.176 | 0.385 | 0.502 | 0 |
| mid | 4 | 90.4 | 0.010 | 0.197 | 0.538 | 0.581 | 0 |
| mid | 8 | 110.1 | 0.012 | 0.304 | 0.682 | 0.944 | 0 |
| mid | 16 | 157.4 | 0.016 | 0.364 | 0.827 | 1.167 | 0 |
| robust | 1 | 143.7 | 0.007 | 0.011 | 0.014 | 0.046 | 0 |
| robust | 4 | 504.9 | 0.006 | 0.016 | 0.025 | 0.029 | 0 |
| robust | 8 | 663.5 | 0.010 | 0.026 | 0.033 | 0.041 | 0 |
| robust | 16 | 807.3 | 0.018 | 0.035 | 0.046 | 0.056 | 0 |



Reading: single-request latency is dominated by the search, so the native server and the Python-plus-engine stack are equal for openings and mid-games (the Python layer cost about 10–20 ms on small requests and is gone); what changes is throughput: the robust-endgame category, which was limited by Flask's GIL, runs at 825 req/s (8 clients) and 930 req/s (16 clients) against 424 and 402 on the Python stack, i.e. about 2×, with the key check, the limits, the explanation and the game log on. With all 20 logical CPUs (hybrid CPU, E-cores are slower) openings do not improve over the P-core set. These numbers are not a prediction for the 16-core / 32-thread Windows Server; `docs/DEPLOY_WINDOWS.md` §9 says how to measure there.

## 10. Build and tests

* `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`: 0 warnings, 0 errors.
* `dotnet test`: **228 passed**, 0 failed, 0 skipped (155 before this phase; 5 consecutive full runs green).
* `python -m pytest -q`: **152 passed**. (`tests/test_csharp_engine_pool.py::test_more_concurrent_requests_than_the_pool_size_...` is timing sensitive: it failed twice right after a `dotnet test` run on the same machine, with nothing changed in Python, and passes when run alone or again.)
* Two tests of this phase were flaky while being written and were fixed rather than loosened: the game-log test depended on the random production seeds (it now uses the test seed header, so the games are reproducible) and one scheduler test did not wait until two cheap requests were queued before releasing the worker.
* `Core` and `Search`: only the callback. The per-level callback is proved to leave scores, moves and every statistic unchanged on all reference scenarios in both modes, to fire once per completed level in order with the statistics of that level, and the steady-state search with a callback still allocates 0 bytes (`LevelCallbackTests`; the existing SIG, determinism and zero-allocation tests are unchanged and green).

## 11. Known limits

* The web UI (8B), deck optimisation and the port of the three `/api/*` endpoints (8C), coaching, reports and puzzles (8D) are **not started**.
* `robust.progress` has only two points (start and end); a finer progress needs a callback inside `EndgameRobust` (Search), which was not authorised.
* The ninth card of the opponent is invisible to the log (section 5). Games that the plugin never finishes simply stay open until the retention deletes them.
* Windows service installation, win-acme and NSSM steps in the deployment guide were written but not executed here; the load figures are from this machine only; PyPy and a real waitress were not run.
* The nine mascot PNGs are untracked pending the user's decision.
