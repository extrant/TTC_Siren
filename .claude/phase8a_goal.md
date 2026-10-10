# TTC_Siren — Goal: Phase 8A, the C# server becomes the product backend

Run with: `/goal @.claude/phase8a_goal.md` — **in a new session**. Every older goal file in `.claude/` is completed or retired. If a Stop hook asks for another goal's report, that goal is stale: ignore it and tell the user.

The user authorized Phase 8A on 2026-10-11. The decisions are recorded in `CSHARP_MIGRATION.md` §9 Q-9 and §8 A-21 … A-24; the design is in `docs/design/PHASE8_PLAN.md`. Do the work **in this order**, as separate logical commits:

- **Step A:** commit the reviewer's files.
- **Step B:** the port. `/ai_move` served natively, parity with the Python server.
- **Step C:** what is new. Explanation facts v2, game logs, the live event stream.
- **Step D:** the cloud foundation. API keys, rate limits, HTTPS, proxy, scheduling, deployment.

**Step B must be complete, with every parity test green, before Step C starts.**

Not authorized:
- the web UI (8B) beyond a placeholder page;
- deck optimisation (8C) and porting `/api/recommend`, `/api/draft/analyze`, `/api/winrate` (8C);
- coaching, reports, puzzles (8D);
- **improving** the unknown-card sampling (port it as it is);
- changes to rules, evaluation, ordering or search semantics;
- root parallelism;
- deleting Python code;
- any push.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely; its rules win if anything here seems to conflict.
2. Read `CSHARP_MIGRATION.md` completely, especially §0, §4 (D-001, D-003, D-004, D-010, D-011, D-015, D-019), §7 Phase 7 and the hardening record, §8 (A-7, A-20 … A-24), §9, §11 and §12.
3. Read `docs/design/PHASE8_PLAN.md`, `docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md` (§7 and §12), `docs/DEPLOY_WINDOWS.md` and `API.md` (the plugin contract).
4. Read the Python code you will port:
   - `ai_server.py`: `ai_move`, `parse_board`, `parse_owner`, `parse_rules_and_open_mode`, `parse_hand`, `find_card_id_by_stats`, `_select_unknown_cards_for_slots`, the unknown-candidate helpers, the current-player inference, `analyze_global_star_usage`, and every analysis/text function used by the response (`analyze_opponent_hand` … `format_card_display`, `generate_search_summary`).
   - `ai/unknown_card_handler.py` (all of it).
   - `app.py` (start-up and routes).
   - `ai/monte_carlo.py` (only to understand what is retired).
5. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`: expect 0 warnings, 0 errors.
   - `dotnet test`: expect 155/155.
   - `python -m pytest -q`: expect 152 passed.
   - `git status -sb`: expect 7 hardening commits ahead of `origin` (`45a9b28`), plus the reviewer's uncommitted files.
6. In one short message, state the steps, their acceptance criteria and your project layout, then begin.

## 2. Global rules

- **Git:** never touch `main`; no push, no force, no PR, no tag. One logical change per commit. Never commit build output, `publish/`, golden data, corpora, game logs, API key stores or certificates.
- **Python:**
  - Python business code does not change. It remains the reference and the golden generator.
  - New generators and tests under `tools/` and `tests/` are allowed.
  - The Python app keeps running for the three `/api/*` endpoints, which the C# server proxies.
- **C# Core/Search:**
  - Unchanged, except **one per-completed-level callback** in `FastSearcher.RunIterativeDeepening` for live progress. It is called once per finished depth, never per node.
  - Prove that results are unchanged (the existing SIG and determinism checks) and that zero-allocation tests still pass. A null callback must cost nothing.
- **Parity:**
  - Deterministic parts are compared **bit for bit / field for field** with golden data produced by the Python code.
  - Random parts use the CPython-compatible generator (A-23) and are compared exactly as well.
  - Anything that cannot be made exact must be explained in `CSHARP_MIGRATION.md` §4, covered by a statistical test, and reported. Never loosen a comparison silently.
- **Tests:** never skip or weaken tests. Random tests have fixed seeds that appear in failure messages.
- **Stop conditions:** stop on an unexplained difference with Python, record it in §9 BLOCKED, and report to the user. Also stop and report if the plugin contract (`API.md`) cannot be kept.

## 3. Step A — reviewer files (local commit only)

- [ ] One docs commit containing:
  - `docs/reviews/OPUS_PHASE7_HARDENING_REVIEW.md` (with §12);
  - the `CSHARP_MIGRATION.md` edits;
  - `.claude/phase7_hardening_goal.md` (banner);
  - `docs/design/PHASE8_PLAN.md`, `docs/design/MASCOT.md`;
  - `assets/mascot/README.md`, `assets/mascot/raw/.gitkeep`;
  - this goal file.
- [ ] Build, `dotnet test` and pytest green. **Do not push.**

## 4. Step B — port `/ai_move` with parity

### B1. Project layout

- A new public server project `csharp/src/TTC.Siren.Server` (ASP.NET Core). It hosts the engine **in process**: reuse the worker pool and the solve service. Move shared code into a small library if that is cleaner; do not duplicate it.
- `TTC.Siren.Engine` may stay as the internal `/v1/solve` host for tests and tooling, or be merged. Decide, document and keep its tests green.
- Update `CLAUDE.md` §10 with the new directory.

### B2. Request parsing — field-for-field parity

- Port the following:
  - `parse_board`, including the D-011 left/right naming;
  - `parse_owner`;
  - `parse_rules_and_open_mode` (substring matching, open modes);
  - `parse_hand` (database lookup by edges, `inHand`/`canUse`, used cards, id offset 1000 for assumed cards, the generated/prediction flags, NPC id pools);
  - current-player inference;
  - the 同类 recalculation at load;
  - the robust-endgame decision and sample count.
- **Golden:** a Python generator exports, for each request, the fully parsed state (board, both hands in order with ids, flags, modifiers, display values, current player, rules, open mode, robust inputs) **before any search**.
  - Cover the 360-request Phase 7 corpus plus at least 2,000 new seeded requests: all rules, open modes, NPC pools, `inHand=false`, `canUse=false`, duplicate ids, currentPlayer 0/1/2.
  - C# must match field for field.

### B3. Unknown-card sampling — exact with a CPython-compatible RNG (A-23)

- Implement `PythonRandom` in C#: MT19937 and `random.seed(int)` exactly as CPython 3.11 (init_by_array), plus `random()`, `getrandbits`, `_randbelow`, `randrange`/`randint`, `choice`, `sample` (both of CPython's algorithms), `shuffle`, `uniform`, and `choices`/`gauss` if used.
- First verify the raw generator: golden sequences from Python for many seeds and call mixes, compared exactly.
- Then port `ai/unknown_card_handler.py` and every helper `parse_hand` uses (all strategies, weighting, star quotas, draft constraints, NPC pools, the fallback `generate_opponent_cards`), **logic unchanged**, drawing from `PythonRandom`.
- **Golden:** for each request and a given seed, the assumed cards per slot, in order, exactly as Python produces after `random.seed(seed)`. Cover at least 5,000 seeded cases across all rule families and unknown counts 1 … 5.
- Iteration order of Python sets or dicts can matter. Reproduce it where it is deterministic. If a construct depends on string hashing, record it, add a statistical test and report it.
- **Production seeding:** every request gets a fresh 64-bit seed from a cryptographic RNG. The seed is written to the game log, so any decision can be replayed exactly.

### B4. Response compatibility — the plugin contract (`API.md`)

- Port every field `ai_move` returns today on the engine path, so that existing plugins keep working:
  - `card`, `card_id`, `pos`;
  - `opponent_hand_analysis`, `win_probability`, `recommendation`, `performance_stats` (same keys);
  - `search_progress`/`search_summary` when requested;
  - `engine`/`engine_info`;
  - the no-move answer and the error forms.
- **Port the analysis/text functions faithfully** (they are deterministic). They stay as the compatibility fields; the new text goes into the additive `explanation` field (Step C).
- **End-to-end parity reference:** the Python server with the engine **on** (today's production path), given the same seed. C# needs a test-only switch, enabled only by configuration and never in production, that seeds `PythonRandom` like `random.seed(seed)` before the request.
- On the 360-request corpus and on a fresh-seed corpus, C# responses must equal Python's field for field. The only exceptions are timing fields and new additive fields, and the exceptions must be listed.

### B5. Retire Monte Carlo (A-22)

- `solver: "monte_carlo"` is answered by the minimax engine path. The response gets `solver_used: "minimax"` and `solver_note` (Chinese text: Monte Carlo has been retired).
- `mc_simulations` is ignored.
- `ai/monte_carlo.py` stays in the repository untouched. Update `API.md`.

### Step B acceptance

- All of the parity above is green.
- The latency of the C# server vs the Python server (engine on), pinned protocol, is recorded.
- Commit the checkpoint, then start Step C.

## 5. Step C — explanation facts v2, game logs, live events

### C1. Explanation facts v2 (A-24)

- Add an additive `explanation` object to `/ai_move` (and to `/v1/` responses): `{ version, facts: [...], text: { brief, full } }`.
- Request parameter `explain`: `brief` | `full` (default `brief`; the plugin can ask for `full`).
- **Facts** are computed by the engine and checked by recomputation in tests. At least:
  1. **Immediate effect:**
     - cards flipped, each with its cause: basic / 同数 / 加算 / 连携 (one level, D-013) / 逆转 / 王牌杀手;
     - the card count before and after;
     - modifier changes from 同类强化/弱化.
  2. **Exposure:**
     - every exposed side of the placed card with its effective value;
     - which opponent cards (known, or assumed with probability) can capture it next turn, and with what;
     - the corner/edge safety used by the engine.
  3. **The opponent's strongest reply** (from the search line) and its effect.
  4. **Principal variation:**
     - the best line for the next plies, as far as it is reliable;
     - the expected outcome: an exact card difference when solved to the end (exact solver), otherwise the heuristic score and the depth.
  5. **Alternatives:**
     - the top 3–5 other moves with score, difference to the chosen move, and whether they are tied (Q-5 b);
     - for each, the decisive reason it is worse, as a fact (for example "exposes right 3 to their known 9").
  6. **Robust endgame (when active):**
     - number of scenarios, safety ratio, worst/average outcome;
     - whether the override happened and the rule that decided it.
  7. **Unknown cards:** how many, how they were assumed (strategy, NPC pool), the cards assumed in this decision.
  8. **Confidence:** solved to the end or not; depth reached; time used.
- **Text:**
  - Chinese templates with several phrasings per fact family. Selection is deterministic from the facts (no randomness), so tests can snapshot it.
  - Brief: one or two sentences. Full: structured paragraphs following the fact order.
  - **Text may only state what the facts contain.**
  - Templates are data files, so 8D can add Seren's voice without code changes.
- **Tests:** a recomputation test for each fact family; golden snapshots of the text; no fact contradicts the engine result (for example, no "winning" claim when the exact value is ≤ 0).

### C2. Game logs (A-21; schema `docs/design/GAME_LOG_SCHEMA.md`, versioned)

- **Storage:** one JSONL file per game under a data directory, per API key, with retention configurable (default 30 days) and a cleanup job.
- **Records** (every record has a sequence number, wall-clock UTC time and elapsed time):
  - **game start:** rules, open mode, owners, NPC, known decks;
  - **request:** full state (board with ids, base and display values, owners, modifiers; hands known / unknown / assumed), the RNG seed, solver settings, per-depth statistics, chosen move, explanation facts, timings;
  - **inferred opponent move:** the difference between consecutive requests, giving card, cell, owner, and flips with causes, recomputed by the rules;
  - **game end:** when the board is full, the final counts and winner.
- **Game identity:** optional `gameId` in the request (document it in `API.md` for future plugins). Otherwise infer it: same key, the new state is a consistent successor of the last one, gap < 15 min (configurable). Record how each game id was obtained.
- **Replay:** a test reconstructs every logged decision from its record (state + seed + settings) and gets the same move and the same facts.
- **API:**
  - `GET /v1/games` (list, paged);
  - `GET /v1/games/{id}` (JSON);
  - `GET /v1/games/{id}/download` (JSONL);
  - `DELETE /v1/games/{id}`.
  - Each key sees only its own games.
- **Tests:**
  - simulate complete games: both sides chosen by the engine, requests built as the plugin would send them;
  - the log must reconstruct the full timeline exactly, including inferred opponent moves and flip causes;
  - retention and deletion tests.

### C3. Live event stream (data source of 8B's real-time view and "live game follow")

- `GET /v1/live` (Server-Sent Events, authenticated by key) streams that key's events:
  - `request.received`;
  - `search.depth` (depth, score, best move, nodes, elapsed: from the per-level callback);
  - `robust.progress` (scenarios done / total, if cheap to report);
  - `decision` (move + brief explanation);
  - `game.updated` (log record);
  - `game.ended`.
- Events carry the game id. A slow consumer must never slow the search: use a bounded per-subscriber buffer, drop oldest, and send a `lagged` notice.
- **Tests:** ordering, isolation between keys, back-pressure.

## 6. Step D — cloud foundation, scheduling, deployment

- **API keys (A-21):**
  - random 256-bit keys, stored only as SHA-256 hashes, with name, created time, enabled flag and rate limit;
  - admin CLI on the server binary: `keys add --name`, `keys list`, `keys revoke`, `keys set-limit`;
  - accepted as `Authorization: Bearer <key>` or `X-Api-Key`;
  - **keys never appear in logs** (show a short prefix only);
  - a configurable loopback exemption for development, **off by default**.
- **Rate limits and concurrency:** per key (token bucket and maximum in-flight requests), configurable, with `429` plus `Retry-After`.
- **HTTPS:** Kestrel with a PFX certificate from configuration, or documented operation behind a reverse proxy. Also HSTS, request-body limits, security headers, a CORS allowlist (for the 8B origin), and admin endpoints bound to loopback only.
- **Proxy:** `/api/recommend`, `/api/draft/analyze`, `/api/winrate` are forwarded to the local Python app (waitress on loopback; configurable URL), authenticated by the same key, with a timeout and a clear 502/504 when Python is down.
- **Scheduling (review O8-1):**
  - queued requests are ordered by estimated cost: the number of empty cells, and whether the robust endgame is active, so cheap endgame/mid-game requests are not stuck behind openings;
  - FIFO within a class, with **aging**, so an opening is never overtaken indefinitely (bounded wait, configurable);
  - repeat the review's probe (1 worker, 12 openings + 5 cheap requests): the cheap requests are answered while the openings queue, and no opening starves.
- **Placeholder web page** at `/`: product name and server status only (8B builds the real UI). Static files are served from a configurable directory.
- **Deployment guide (`docs/DEPLOY_WINDOWS.md`, updated for the cloud topology):**
  - runtime;
  - publish;
  - a certificate (for example win-acme / Let's Encrypt, or the user's own);
  - firewall: open 443 only; engine and Python bound to loopback;
  - service registration for the C# server and the Python app (waitress);
  - key management; backup of the key store and logs; rollback (point the plugin back to the old Python server).
- **Docs:**
  - `API.md`: authentication, `gameId`, `explain`, `explanation`, `solver_note`, `/v1/*`, `/v1/live`, error codes 401/403/429/502/503/504.
  - `docs/reports/PHASE8A_REPORT.md`: what was ported, the parity counts, the latency/load tables, security notes, known limits.
  - `CSHARP_MIGRATION.md`:
    - §0 row 8A: DONE, awaiting review; **do not set a gate**;
    - §4: new D-ids for anything non-exact;
    - §7: delivery record;
    - §10: measurements;
    - §11 and §12: handoff.

## 7. Acceptance criteria (all required)

- [ ] Step A committed locally. Nothing pushed; `main` unchanged; no PR.
- [ ] Build 0 warnings / 0 errors; all C# tests and pytest pass. No test skipped or weakened.
- [ ] Parse parity: field for field on the 360-request corpus plus ≥ 2,000 new requests.
- [ ] RNG parity: exact raw sequences.
- [ ] Sampling parity: exact on ≥ 5,000 seeded cases. Any non-exact construct is documented, statistically tested and reported.
- [ ] `/ai_move` end-to-end parity with the Python server (engine on, same seed) on two corpora, with the excluded fields listed.
- [ ] Monte Carlo retired as specified.
- [ ] Core/Search unchanged except the per-level callback: SIG/determinism checks identical, zero-allocation tests green.
- [ ] Explanation facts:
  - every family recomputed in tests;
  - text only from facts;
  - brief and full text;
  - deterministic snapshots.
- [ ] Game logs:
  - simulated complete games reconstructed exactly;
  - replay reproduces every decision;
  - retention and deletion work;
  - keys are isolated.
- [ ] Live stream: ordering, isolation and back-pressure tested.
- [ ] Auth, rate limits, CORS, HTTPS configuration, the proxy and the admin CLI are tested. No key appears in any log.
- [ ] Scheduling probe passes; the load test and latency tables are recorded (pinned protocol for latency; CPU set recorded for load).
- [ ] Docs updated (`API.md`, `DEPLOY_WINDOWS.md`, `GAME_LOG_SCHEMA.md`, `PHASE8A_REPORT.md`, `CSHARP_MIGRATION.md`, `CLAUDE.md` §10).

## 8. Final report

Report briefly:
- the Step A hash;
- the commits per step;
- the project layout;
- the parity counts (parse, RNG, sampling, end-to-end) and any non-exact constructs;
- the Monte Carlo handling;
- the explanation fact families with examples (brief and full);
- the log schema version and the replay result;
- the live events;
- the security setup;
- the scheduling probe result;
- latency/load headline numbers with CPU sets;
- exact build, test and pytest counts, and `git status -sb`;
- confirmation that 8B/8C/8D have not been started and nothing was pushed.
