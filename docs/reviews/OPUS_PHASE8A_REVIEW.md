# Opus independent review — Phase 8A (the C# server becomes the product backend)

Date: 2026-10-11. Reviewer: Opus. Independent review: every claim was re-run, or re-checked with the reviewer's own probes, against the Python reference.
Goal under review: `.claude/phase8a_goal.md`. Implementer documents:
- `docs/reports/PHASE8A_REPORT.md`
- `docs/design/EXPLANATION_FACTS.md`
- `docs/design/GAME_LOG_SCHEMA.md`
- `API.md`
- `docs/DEPLOY_WINDOWS.md`
- `CSHARP_MIGRATION.md`: §4 D-020 … D-025 and §7 "Phase 8A 交付记录"

## 1. Gate

**`PASS_WITH_NONBLOCKING_FINDINGS`**

- **Finding counts:** CRITICAL 0, HIGH 0, MEDIUM 2 (M8-1, M8-2), LOW 4 (L8-1 … L8-4), OPTIONAL 3 (O8-1 … O8-3).
- **Steps B, C and D: all PASS.**
  - Parity with Python holds on fresh seeds and in the reviewer's own end-to-end comparison.
  - The explanation facts agree with the Python rules and the Python exact solver.
  - Game logs reconstruct real simulated games and replay exactly.
  - Security behaves as documented.
  - The scheduling fix works.
- **8B may start once the user authorises it.** Recommendation: fix M8-1 and M8-2, and L8-2 (the UI will read the logs), as the first step of the 8B goal.
- **Cloud go-live: conditional.** M8-1 must be fixed, or worked around with verification (§8).
- No remediation goal is needed.

## 2. Repository and scope

- **Branch and remote:**
  - Branch `csharp-solver-migration`, HEAD `756eddd`, 22 commits ahead of `origin` (`45a9b28`).
  - **Nothing pushed**; `main` is `eb60a02`; no new PR; the repository is private (verified).
- **Step A:** `8f78212` contains the reviewer's files. `git diff 8f78212 HEAD` on `.claude/`, `docs/design/PHASE8_PLAN.md`, `docs/design/MASCOT.md`, `docs/reviews/` and `assets/` is empty, so they are unchanged.
- **Builds:** every code commit was built independently from `git archive` during the Phase 7 hardening review. For 8A, the HEAD build passes, and a full clean build in an empty copy passes (§3).
- **Not committed (correctly):**
  - build output, golden data, corpora, game logs, key stores, certificates (`.gitignore` covers `keys.json`, `gamelogs/`, `*.pfx`, `/publish/`);
  - the nine mascot PNGs, which are still untracked and await the user's decision. The report says so.
- **Python business code unchanged:** `git diff 8f78212 HEAD` on `core/`, `ai/`, `ai_server.py`, `app.py`, `serve_waitress.py` and the other apps is empty.
- **Core/Search changed in one place only:** the per-level callback in `FastSearcher.RunIterativeDeepening` (`4d6f2e3`). It is invoked once per completed depth and is null-checked. The reviewer compared `Bench` signatures of `45a9b28` (built separately) and HEAD: **search 22 / fast 40 SIG lines, exact values 3, endgame 72/72, make/undo checksums 6, all identical**. The zero-allocation tests pass.
- **8B/8C/8D not started.** There is a placeholder page only.

## 3. Correctness runs (reviewer)

| command | result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test` | **228 passed**, 0 failed, 0 skipped |
| `python -m pytest -q` | **152 passed** |
| Clean `git archive HEAD` copy with **no golden data**, all 8A seeds changed (`Seed` 31337211, `RngSeed` 31337212, `E2eSeed` 31337210), full `dotnet test` | **226 passed, 2 failed, 497 s**; every golden file was regenerated through the harness, with no hang. The 2 failures are explained (see below). |

**What the fresh-seed run reported:**
- parse parity: 2,689 parsed requests equal field for field, 71 refusals identical, 39,672 card fields;
- sampling parity: 5,400 cases, **117,292 assumed cards equal**, 384 rule combinations;
- `/ai_move` parity: **960 requests equal**, 44,037 values bit for bit;
- Phase 7 end-to-end: 1,418 levels, engine score equal to the oracle, 0 unexplained;
- all Phase 2–7 parity suites green.

**The two failures:**
1. **`SchedulerTests.An_opening_waits_at_most_twice_the_aging_period…` (M8-2).**
   - The test asserts wall-clock time: an aging period of 100 ms, a bound of 200 ms, and it measured 1,545 ms while the reviewer's other probes loaded the machine.
   - Its own output says the opening was overtaken by **0** cheap requests, so the ordering was correct.
   - Re-run alone on an idle machine: **5/5 green**.
2. **`ExplanationFactsTests.The_generated_text_matches_the_committed_snapshots`.**
   - The 30 snapshots are tied to the records of the default-seed corpus, so with other seeds other records are selected.
   - This is expected, not a defect. With the default seeds it is green in the main run.

## 4. Step B — port and parity: PASS

- **CPython RNG, reviewer's own probe:**
  - 306 seeds: 0, 2^32−1, 2^32, 2^64−1, 2^64, 2^128+1 and 300 others.
  - Each seed ran 40 mixed calls: `random`, `_randbelow`, `randint`, `getrandbits`, `sample`, `choice`, `shuffle`.
  - Result: **12,240 outputs identical** to CPython 3.11.4. The probe is a small C# console referencing `TTC.Siren.Compat`, run against the same script in Python.
- **Parsing and sampling:** exact on fresh seeds (§3).
  - D-020 … D-025 were checked one by one: each is either a deliberate, documented difference on malformed input (D-020, D-021) or a faithfully reproduced Python quirk (D-023 … D-025, with `PY-QUIRK` in the code).
  - No statistical fallback was needed. This is plausible: no sampling path iterates a hash-ordered set of strings.
- **End-to-end, reviewer's own probe:**
  - The published C# server over real HTTP, with an API key and the test seed header, was compared with the Python server in process (C# engine on, `random.seed(seed)`).
  - It covered 300 fresh requests (`gen_requests.py --seed 4242001`) plus 120 requests of the Phase 7 corpus.
  - **416/420 equal** at first. The 4 differences were all `solver: "monte_carlo"`, where Python runs its Monte Carlo solver and C# answers with the engine, as specified (A-22). With those 4 mapped to `minimax` on the Python side: **300/300 equal**. Total: **420/420**.
  - Excluded fields: timing only (`engine_info.timings/worker`, `search_progress` time fields, `search_summary` time fields) plus the additive fields `explanation`, `game_id`, `solver_used`, `solver_note`.
- **Monte Carlo retired:** the answer carries `solver_used: "minimax"` and a Chinese `solver_note`, and `API.md` is updated.
- **Test seed switch:** off by default and absent from `appsettings.json` and the deployment guide's table. **It is not guarded** against being switched on for a public listener (L8-1).

## 5. Step C — explanation facts, game logs, live events: PASS

**Explanation facts, reviewer's recomputation with the Python rules** (requests with fully known hands from both corpora, `explain=full`, `max_depth` 10):

| check | result |
|---|---|
| `immediate_effect.flips` against the cells Python's `make_move` flips | **125 / 125 equal** |
| `outcome` with `kind=exact`, `basis=known_hands`, against Python's `_solve_endgame_exact` after the move | **91 / 91 equal** |
| texts that claim a win, against the exact value | 46 win claims, **0 without an exact win** |

- The implementer's broader recomputation tests also pass on fresh seeds:
  - 543 decisions with alternatives;
  - 1,929 scores against fixed-depth root values;
  - 313 threats confirmed one by one;
  - capture probabilities against brute force;
  - every number in a text found among the facts.
- The templates are an embedded data file, and `Server:TemplatesDirectory` can replace them without a rebuild (needed for 8D).

**Game logs, reviewer's own simulation.**
- **Setup:** four complete games through the real server. Requests were built as the plugin builds them (D-011 edge naming, owner codes, `currentPlayer`), with the opponent playing random legal moves and the true state kept by the Python rules. The games covered:
  - rules: 全明牌 plus 同数+加算, 逆转, 同类强化, or none;
  - both colours, both first movers;
  - with and without `gameId`.
- **Requests and inference:** all requests were logged, and `seq` has no gaps.
  - Game identity is `client` (with `gameId`) or a newly started game, and following requests attach by inference.
  - **Every opponent move the server can see** (between two of our requests) was inferred with the right edges, cell and flipped cells, and marked consistent.
  - The opponent's move before our first request and after our last request is not visible, as documented.
- **Game end:** `game_end` was written with the right final counts and winner when our move filled the board.
- **Replay:** every logged decision was re-sent with its logged seed (18 requests). Move, all explanation facts and both texts were **identical (18/18)**; only measured `*_ms` values differ.
- **Isolation, deletion, inference limits:**
  - another key gets 404 for a game it does not own;
  - `DELETE`, then `GET`, gives 404;
  - two database cards with identical edges and type (ids 12 and 369) cannot be told apart from a plugin request. The log names the first one, exactly as Python's `find_card_id_by_stats` would (L8-4: document it).

**Live events (reviewer probe):**
- K1 received:
  - `request.received` 18;
  - `search.depth` 60;
  - `decision` 18;
  - `game.updated` 38;
  - `game.ended` 2;
  - `robust.progress` 16.
- The order inside each request is correct.
- The second key received only its own 18 requests, and no game id of the first key's server-generated games.
- With a **subscriber that never reads** connected for the whole run, request latency stayed at a median of 25 ms (max 125 ms).

## 6. Step D — cloud foundation: PASS, except the publish step (M8-1)

**Reviewer probe against the published server (`sec8.py`, `sec8b.py`):**

| check | result |
|---|---|
| key store | only SHA-256 hashes; `keys add/list/revoke` work; the key is shown once |
| no key / wrong key | 401 / 401 (loopback exemption off by default) |
| `X-Api-Key` and `Bearer` | both accepted |
| revoked while running | 403 after about 1 s, no restart |
| rate limit (120/min, burst 20) | 20 × 200 then 429 with `Retry-After` |
| HTTPS with a self-signed PFX (made in memory, no certificate store touched) | 200 over TLS; `Strict-Transport-Security: max-age=31536000; includeSubDomains` |
| security headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` |
| CORS | preflight from a foreign origin 403 and no `Access-Control-Allow-Origin`; from the configured origin 204 with the origin echoed |
| body limit 1 MiB | a valid 2 MiB JSON body is refused, but with **400**, not 413 (L8-3) |
| operator endpoint `/admin/status` | 200 from loopback; 403 with `X-Forwarded-For`, even with a key |
| proxy `/api/*` | to the real Python app (waitress): 200; without a key: 401; Python down: **502**; slow upstream with a 1 s timeout: **504** |
| non-loopback address without TLS | refuses to start with a clear message |
| key leakage | no key and no key prefix of 8 or more characters in the server output or in any game-log file |

**Scheduling (O8-1), reviewer probe:** one search worker, 12 concurrent openings at production settings, then 5 cheap endgame requests 3 s later.

| mode | cheap requests answered after | openings | longest opening |
|---|---|---|---|
| FIFO (`PriorityScheduling=false`) | 8.36 s each | 12/12 | 11.3 s |
| priority with aging (default) | **0.88 s** (the opening that was already running) | 12/12 | 11.3 s, no starvation |

**Load spot check** (`tools/phase8a/load_server.py --cpu auto --workers 12`):
- The first cell after a server start is cold: JIT and the first use of each worker's table. Mid c=4 measured 44.7 then 93.3; robust c=4 measured 334.5 then 600.3.
- Warm, the cells match the report within 3%:

  | cell | reviewer | report |
  |---|---|---|
  | mid c=4 | 93.3 | 95.2 |
  | mid c=8 | 119.8 / 132.9 | 128.7 |
  | robust c=4 | 600.3 | 618.8 |
  | robust c=8 | 808.5 / 836.1 | 825.3 |

- The report's matrix starts at c=1, which warms the server first.

**Deployment-guide walk-through:**
- runtime, configuration, key issuance, HTTPS with a PFX and the start-up behaviour work as written;
- the **`dotnet publish` step does not** (M8-1);
- service registration (`sc.exe`/NSSM) and win-acme were not executed: they need elevation and change system configuration.

## 7. Findings

### M8-1 (MEDIUM) — the documented publish command fails

| | |
|---|---|
| **File / project** | `csharp/src/TTC.Siren.Server/TTC.Siren.Server.csproj` (references `TTC.Siren.Engine`, a Web-SDK project with its own `appsettings.json`); `docs/DEPLOY_WINDOWS.md` §2 |
| **Observed** | `dotnet publish csharp\src\TTC.Siren.Server -c Release -o …` fails with **NETSDK1152**: two publish outputs named `appsettings.json` (Engine's and Server's). With `-p:ErrorOnDuplicatePublishOutputFiles=false` it builds, and this time the Server's file won, but which file wins is not guaranteed. Engine's file would listen on port 5300 with engine settings. |
| **Expected** | The guide's publish command produces a server folder with the server's `appsettings.json`. |
| **Impact** | The deployment cannot be followed as written; a workaround can silently ship the wrong configuration. No effect on correctness or tests. |
| **Fix** | Keep the Engine's `appsettings.json` out of referencing projects. Either mark it in `TTC.Siren.Engine.csproj` with `CopyToOutputDirectory/CopyToPublishDirectory` metadata that does not flow to referencing projects, or rename it, or move the shared code into a non-Web class library. Then re-run the guide's command. |
| **Regression test** | A test or script that runs the documented `dotnet publish` of the server and asserts the published `appsettings.json` is the server's (`"Server"` section, port 5000). |
| **Human decision** | No. |

### M8-2 (MEDIUM) — a scheduler test depends on wall-clock time

| | |
|---|---|
| **File / test** | `csharp/tests/TTC.Siren.Tests/SchedulerTests.cs`, `An_opening_waits_at_most_twice_the_aging_period_…` |
| **Observed** | It failed in the reviewer's clean run while the machine was loaded (1,545 ms against a 200 ms bound, with 0 overtakes). It passes 5/5 when idle. The report already mentions two flaky tests that were fixed, plus one timing-sensitive pytest. |
| **Impact** | A red build without a defect; it wears down trust in the suite (CLAUDE.md §7 expects a reliably green suite). |
| **Fix** | Assert the property itself: the number of cheap requests that overtook the opening, or the ordering. Use an injected `TimeProvider`/logical clock for aging instead of real time, or set a bound that is independent of machine speed. |
| **Regression test** | The same test run under artificial CPU load (for example a parallel busy loop) stays green. |
| **Human decision** | No. |

### LOW

- **L8-1 — the test seed switch has no guard.** `Server:TestSeedHeader` (environment variable `TTC_Server__TestSeedHeader=true`) is accepted for any listener. Impact is small, since it only fixes the sampling seed per request, but a test flag should not be possible in production.
  - **Fix:** refuse it in `ServerOptions.Validate` unless the host is loopback (or a second explicit flag is set), and log a warning when it is on. Add a test.
- **L8-2 — game logs and the games API write integers as floats.** For example `"max_depth": 4.0`, `"card_id": 357.0`, `"currentPlayer": 2.0`.
  - `request_raw` is therefore not "the plugin's request exactly as sent" (`GAME_LOG_SCHEMA.md`). Replay still works, because integral floats are read as ints (D-021).
  - The 8B UI and exported files would see floats.
  - **Fix:** write integers as integers, and store `request_raw` from the received text (the server already has it). Add a round-trip test.
- **L8-3 — a body over 1 MiB answers 400 `Invalid input data`, not 413.** The limit itself works.
  - **Fix:** map Kestrel's body-size `BadHttpRequestException` to 413, and document it in `API.md`.
- **L8-4 — inferred opponent cards with identical edges and type are ambiguous** (for example ids 12 and 369). The log names the first card, as Python would.
  - **Fix:** document it in `GAME_LOG_SCHEMA.md`, and let the UI show edges (and "ambiguous") instead of only an id.

### OPTIONAL

- **O8-1:** `robust.progress` has only start and end. A finer progress needs a hook in `EndgameRobust` (Search code); decide in 8B whether the live view needs it.
- **O8-2:** add a `Content-Security-Policy` once 8B serves the web UI from the same origin (the JSON API does not need it).
- **O8-3:** warm the server at start (one search per worker, and touching the tables) so the first requests after a restart are not 2× slower.

## 8. Cloud go-live checklist (Windows Server, E5-2698B v3)

1. **Fix M8-1**, or publish with the workaround and **check that `appsettings.json` in the output has the `Server` section**. Then copy the folder.
2. Install the ASP.NET Core Runtime 10, and copy `data\幻卡数据库.csv`. Set `CardDbPath`, `GameLog:Directory` and `Security:KeyStoreFile` **outside** the application folder, and back up the latter two.
3. **HTTPS:**
   - either Kestrel with a real certificate (`Security:Https:*`, password in `TTC_PFX_PASSWORD`, `HttpEnabled=false` for a public listener),
   - or a TLS reverse proxy with `BehindTlsProxy=true`, keeping the server on loopback and **not** publishing `/admin/*`.
   - Open only 443 in the firewall.
4. **Keys:** issue one key per user with `TTC.Siren.Server.exe keys add --name …` and hand it over securely. Set limits per key (`set-limit`) if the defaults (120/min, burst 20, 4 in flight) do not fit.
5. **Never set `TestSeedHeader`** (L8-1).
6. **Python app for `/api/*`:** waitress on loopback (`serve_waitress.py --host 127.0.0.1 --port 5001`), as a service. If it is down, the `/api/*` endpoints answer 502.
7. **Workers and warm-up:** start with `--workers 16`, run the load matrix on the server, and adjust. Expect about 1.6 GiB for 16 workers, and that the first minute after a restart is slower (O8-3).
8. **The game plugin** must be updated to send the server URL and the key, and ideally a `gameId`. Without a key every call answers 401.
9. **Rollback:** point the plugin back to the Python server (`app.py`/waitress, with `TTC_CS_ENGINE` set as wanted).

## 9. Test-coverage gaps

- No test publishes the server (M8-1).
- Timing-based assertions in scheduler tests (M8-2); the timing-sensitive pytest named in the report.
- The game-log games API is not tested for number types (L8-2).
- Service installation, win-acme and real certificates were not executed. PyPy is not covered (the C# server no longer needs Python on the `/ai_move` path, but `/api/*` still does).
- Snapshot texts depend on the default-seed corpus; a seed change selects other records (by design).

## 10. Next steps

1. **8B can be authorised.** Its goal should start with M8-1, M8-2 and L8-1 … L8-4. The UI needs L8-2 (clean numbers) and L8-4 (show edges for ambiguous cards) anyway.
2. **The mascot art is ready to use:**
   - 8 expressions at 1024×1536 RGBA with real transparency, plus the reference sheet;
   - semi-transparent pixels are only 1–5% (hair edges, bubbles, chiffon);
   - the framing is fuller than the spec (content reaches the top and side edges), so 8B should pad the art rather than crop it.
   - The user should decide whether the PNGs are committed (about 21 MB; the repository is private) or kept outside Git.
3. **Push:** the 22 local commits need the user's authorisation.

## 11. Commands run by the reviewer

```
git status -sb; git log 45a9b28..HEAD; git ls-remote origin; gh repo view --json isPrivate; gh pr list --state all; git show --stat <each commit>; git diff 8f78212 HEAD -- <Python files> <reviewer files>
dotnet build ... --no-incremental; dotnet test; python -m pytest -q                                  -> 0/0, 228/228, 152
git archive HEAD (scratch, no golden), seeds 31337210/11/12; dotnet test (detailed)                 -> 226 passed, 2 explained failures, 497 s
dotnet test --filter SchedulerTests (5x, idle)                                                       -> 5/5 green each
reviewer RNG probe (C# console on TTC.Siren.Compat vs CPython, 306 seeds x 40 ops)                    -> 12,240 identical
dotnet publish TTC.Siren.Server (documented)                                                        -> NETSDK1152 (M8-1); with ErrorOnDuplicatePublishOutputFiles=false -> builds
python tools/phase8a/gen_requests.py --seed 4242001 --count 300; e2e8.py / e2e8_mc.py (C# server over HTTP vs Python in process, engine on)   -> 420/420 (MC mapped)
explain8.py (Python make_move and _solve_endgame_exact vs explanation facts)                           -> flips 125/125, exact 91/91, win claims 46/46 backed
game8b.py (4 simulated games through the server: logs, inference, end, replay, isolation, delete, live, slow subscriber)
sec8.py / sec8b.py (keys, 401/403/429, HTTPS PFX, HSTS, headers, CORS, body limit, operator endpoints, proxy 200/401/502/504, non-loopback refusal, key leakage)
sched8.py fifo|priority (1 worker, 12 openings + 5 cheap)                                              -> 8.36 s -> 0.88 s, 12/12 openings
tools/phase8a/load_server.py --cpu auto --workers 12 (mid/robust, c=4,4,8)
Bench search|fast|exact|endgame|makeundo on 45a9b28 vs HEAD                                           -> identical signatures
mascot PNG check (size, alpha, semi-transparent share)
```
