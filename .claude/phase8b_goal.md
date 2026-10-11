# TTC_Siren — Goal: Phase 8B (8A leftovers, Cloudflare readiness, access tiers, the web UI)

> **STATUS: COMPLETED AND REVIEWED (2026-10-11).** Gate `PASS_WITH_NONBLOCKING_FINDINGS` (`docs/reviews/OPUS_PHASE8B_REVIEW.md`). **Do not run this goal again.** If a session still has this goal active (for example a Stop hook asks for its final report), the goal is stale: ignore it and tell the user. The open findings (M8B-1 … M8B-3, L8B-1 … L8B-5, O8B-1 … O8B-3) go into the next authorized goal.

Run with: `/goal @.claude/phase8b_goal.md` — **in a new session**. Every other goal file in `.claude/` is completed or retired. If a Stop hook asks for another goal's report, that goal is stale: ignore it and tell the user.

The user authorized Phase 8B on 2026-10-11. The decisions are recorded in `CSHARP_MIGRATION.md` §9 Q-10 and §8 A-27 (access and capacity, Cloudflare) and A-29 (web UI); the design is in `docs/design/PHASE8_PLAN.md` §6. Do the work **in this order**, as separate logical commits:

- **Step A:** commit the reviewer's files and the mascot originals, then push.
- **Step B:** fix the 8A review findings and make the server ready for Cloudflare.
- **Step C:** access tiers and capacity classes (backend).
- **Step D:** the web UI.

**Steps B and C must be green before Step D starts.**

Not authorized:
- 8C (deck optimisation, porting `/api/*`);
- 8D (coaching, post-game report, puzzles, mascot line library);
- 8E (NPC library, NPC data export);
- user accounts or passwords (the user chose key-only);
- third-party auth or CAPTCHA code (Cloudflare is configured outside the code);
- changes to rule, evaluation, ordering or search semantics;
- deleting Python code;
- pushing Step B–D commits.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely; its rules win if anything here seems to conflict.
2. Read `CSHARP_MIGRATION.md`: §0, §7 Phase 8A, §8 A-21 … A-29, §9 Q-9 and Q-10, §11 and §12.
3. Read `docs/reviews/OPUS_PHASE8A_REVIEW.md` completely (findings §7, checklist §8), `docs/design/PHASE8_PLAN.md` (especially §2 8B and §6), `docs/design/MASCOT.md` (§2 and §4), `docs/design/EXPLANATION_FACTS.md`, `docs/design/GAME_LOG_SCHEMA.md`, `API.md`, `docs/DEPLOY_WINDOWS.md`.
4. Read the server code you will change: `csharp/src/TTC.Siren.Server/*` (`ServerHost`, `ServerOptions`, `Security/`, `GameLog/`, `Live/`), `csharp/src/TTC.Siren.Engine/WorkerPool.cs`, `csharp/src/TTC.Siren.Explain/` (templates and renderer).
5. Baseline, recorded in your first message:
   - `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`: expect 0 warnings, 0 errors.
   - `dotnet test`: expect 228/228.
   - `python -m pytest -q`: expect 152 passed.
   - `node --version`: Node 22 is installed.
   - `git status -sb`: expect 22 commits ahead of `origin` at `45a9b28`, the reviewer's uncommitted files, and 9 untracked PNGs in `assets/mascot/raw/`.
6. In one short message, state the steps, their acceptance criteria and your plan for the web project layout, then begin.

## 2. Global rules

- **Git:**
  - Never touch `main`. Never force-push, rewrite pushed history, tag, release or open a PR.
  - The user authorized **one push**: at the end of Step A, push all local commits up to and including Step A to `origin/csharp-solver-migration`, fast-forward, not squashed.
  - Steps B–D stay **local**.
  - One logical change per commit.
  - Never commit: `bin/`, `obj/`, `publish/`, `node_modules/`, web build output, golden data, corpora, game logs, key stores, session stores, certificates, Playwright reports/traces.
- **Python:** business code does not change.
- **C# Core/Search:** unchanged. Prove it with the `Bench` SIG comparison against the start of the phase, and keep the zero-allocation tests green.
- **Dependencies:**
  - Web: pin exact versions and commit the lockfile (`npm ci` must reproduce the build). Use only what the UI needs, and justify each package in `docs/design/WEB_UI.md`.
  - C#: only Microsoft packages.
- **Tests:** never skip or weaken tests. A test that depends on wall-clock time must be made deterministic, not loosened (see M8-2). Random tests have fixed seeds in their failure messages.
- **Security first:**
  - the web key never reaches browser storage, URLs or logs;
  - session ids are never logged;
  - CSP has no `unsafe-inline` for scripts.
- **Stop conditions:**
  - a parity or determinism test fails without an explanation → record it in `CSHARP_MIGRATION.md` §9 BLOCKED and report to the user;
  - a requirement here conflicts with the plugin contract in `API.md` → stop and report.

## 3. Step A — reviewer files, mascot art, push

- [ ] One docs commit containing:
  - `docs/reviews/OPUS_PHASE8A_REVIEW.md`;
  - the `CSHARP_MIGRATION.md` edits (8A gate, A-27 … A-29, Q-10, row 8);
  - `.claude/phase8a_goal.md` (status banner);
  - the additions to `docs/design/PHASE8_PLAN.md` (§6);
  - this goal file.
- [ ] One assets commit: the nine PNG originals in `assets/mascot/raw/` (`seren_ref_sheet.png`, `seren_e01` … `e08`), exactly as they are.
- [ ] Build, `dotnet test` and pytest green. Push to `origin/csharp-solver-migration`, fast-forward only. Confirm with `git ls-remote origin` that the remote branch equals your Step A head and that `main` is still `eb60a02`.

## 4. Step B — 8A findings and Cloudflare readiness

- [ ] **M8-1 (publish):**
  - Make `dotnet publish csharp\src\TTC.Siren.Server -c Release -o <dir>` work without flags, with the **server's** `appsettings.json` in the output. Remove the duplicate-file conflict at its source.
  - Add a test or script, run by the test suite or documented in the report, that publishes and asserts the output configuration.
  - Update `DEPLOY_WINDOWS.md` §2 if needed.
- [ ] **M8-2 (time-dependent test):**
  - Make the scheduler aging test deterministic: an injected `TimeProvider`/logical clock, or an assertion on ordering and overtake counts.
  - Do the same for any other wall-clock assertion you find: the report names `tests/test_csharp_engine_pool.py::test_more_concurrent_requests_than_the_pool_size_...`.
  - Prove stability by running the scheduler tests 20 times while a parallel CPU-burner keeps all cores busy.
- [ ] **L8-1 (test seed switch):** `TestSeedHeader` is refused at start-up unless the listener is loopback-only; when enabled, log a warning. Add a test.
- [ ] **L8-2 (log numbers):**
  - Integers are written as integers everywhere (game logs, `/v1/games*`).
  - `request_raw` is stored from the received text exactly as sent.
  - Bump the log schema version and keep reading version-1 files.
  - Round-trip test: the downloaded `request_raw` equals the posted bytes.
- [ ] **L8-3 (413):** a body over the limit answers **413** with the usual error form. Update `API.md`.
- [ ] **L8-4 (ambiguous cards):**
  - An inferred card whose edges and type match several database cards is marked `ambiguous: true`, with the candidate ids, in the log record and the API.
  - Document it in `GAME_LOG_SCHEMA.md`. The UI shows the edges.
- [ ] **O8-3 (warm-up):** optional warm-up at start (one search per worker, tables touched), configurable and on by default. Measure the first-minute latency after a restart before and after, with the pinned protocol, and keep it only if it helps.
- [ ] **Cloudflare readiness (A-27):**
  - **Real client IP:** taken from `CF-Connecting-IP` only when the direct peer is a trusted proxy. The trusted list is configurable: Cloudflare's published IPv4/IPv6 ranges as a data file, plus loopback for a local `cloudflared`.
  - **Other peers:** the peer address is used and the header is ignored. Test spoofing attempts.
  - **Operator endpoints** refuse any request that carries `CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For` or `Forwarded`, even from loopback. Test it.
  - **SSE `/v1/live`:** sends a comment heartbeat at a configurable interval (default 20 s, well below Cloudflare's idle timeout) and supports `Last-Event-ID` resumption from the subscriber buffer where possible.
  - **Caching:** dynamic endpoints answer `Cache-Control: no-store`; static assets get long-lived immutable caching with hashed names.
  - **Origin timeout:** a configurable `OriginTimeoutSeconds` (default 95). Validate at start that `max_time` cap + queue wait + margin stays below it, and cap per request accordingly.
  - **Docs:** a "Cloudflare" section in `DEPLOY_WINDOWS.md`:
    - DNS proxied;
    - SSL mode Full (strict) with an origin certificate, or a `cloudflared` tunnel;
    - cache bypass rules for `/ai_move`, `/v1/*`, `/api/*`;
    - optional WAF rate rules;
    - never expose `/admin/*`;
    - how to refresh the IP-range file.

## 5. Step C — access tiers and capacity classes (A-27)

- **Three classes:**
  - **plugin**: API key in `Authorization`/`X-Api-Key`;
  - **web**: a session cookie obtained with a key;
  - **anonymous**: neither.
- **Capacity in `WorkerPool`:**
  - class priority plugin > web > anonymous;
  - inside a class, the existing cost order with aging;
  - **a configurable reserved share for plugin** (plugin requests never wait behind web or anonymous work beyond the running searches);
  - **a configurable maximum share for anonymous** (default 25 % of the workers);
  - a global anonymous queue cap answering 503 `busy` with `Retry-After`.
  - Tests: under a saturating anonymous load, plugin latency stays at its unloaded value (± one search); anonymous never exceeds its share; nothing starves inside a class.
- **Anonymous limits:**
  - per real-IP token bucket and in-flight cap;
  - search caps (maximum depth and time) applied to every anonymous compute request;
  - all configurable, with tests.
- **8B exposes no anonymous compute endpoint**: anonymous gets read-only pages. 8D (coaching) will use this class. Test the class at pool and service level.
- **Web login with the key:**
  - `POST /v1/session` (key in the body) answers `Set-Cookie` with a random 256-bit session id. The cookie is `HttpOnly`, `Secure` (when HTTPS or `BehindTlsProxy`), `SameSite=Strict`, `Path=/`, with a configurable lifetime.
  - `DELETE /v1/session` logs out; `GET /v1/me` returns the key's id, name and limits, never the secret.
  - Sessions live server-side and die when the key is revoked or disabled. Session ids are never logged.
  - Login attempts are throttled per real IP, with an audit log entry: time, IP, success, key id on success.
- **CSRF:** cookie-authenticated state-changing requests (`DELETE /v1/games/{id}`, logout) require `SameSite=Strict` plus a matching `Origin` (or `Referer`) header. Test cross-origin attempts.
- **Endpoints and authentication:**
  - cookie auth is accepted on `/v1/games*`, `/v1/live`, `/v1/me` and the session endpoints;
  - `/ai_move` keeps requiring the key (plugin) and does not accept the cookie;
  - document every endpoint's accepted authentication in `API.md`.
- **Tests:** a security test suite for the tiers (cookie flags, expiry, revocation, CSRF, IP throttling with spoofed headers, no key or session id in any log).

## 6. Step D — the web UI (A-29; `docs/design/WEB_UI.md`)

### Stack and layout

- `web/`: Vite + React + TypeScript + three.js through react-three-fiber.
- Built files go into a directory the server serves (`StaticDirectory`, default set by the publish step).
- SPA routing falls back to `index.html`.
- `dotnet publish` of the server (or one documented script) builds the UI and includes it, so deployment stays one folder.

### Pages

1. **Landing (anonymous):**
   - product introduction;
   - the mascot;
   - a **demo replay**: a bundled sample game log, played on the 3D table with explanations;
   - login.
2. **Login:** enter the key once. Errors say only "login failed". Throttling is visible as "try again later".
3. **Live solve view (logged in), "real-time solve status" + "live game follow":**
   - attaches to `/v1/live` and follows the user's current game (by default the newest active one; a game picker is also available);
   - shows the board on the 3D table;
   - per-depth progress as it arrives: depth, score, best move, nodes, NPS, elapsed;
   - the principal variation drawn as ghost cards;
   - robust-endgame scenarios and safety ratio;
   - the assumed unknown cards;
   - the explanation (brief and full);
   - reconnects with `Last-Event-ID` and states clearly when the stream lagged.
4. **Game log viewer (logged in):**
   - list of games with paging;
   - one game as a **timeline**: every record with time, card, cell, owner, flips with their causes, modifier changes, solver data and explanation;
   - a scrubber and play/pause/step replay on the table;
   - download (JSONL) and delete (with confirmation);
   - ambiguous inferred cards shown by their edges.
5. **Settings:** language (中文 / English), 3D on/off, reduced motion, mascot on/off. Stored per browser; preferences only, never the key.

### 3D table and cards

- Deep-sea casino table, 3×3 glass board.
- **Original procedural card faces:** edge numbers (A for 10), star rating, an original type glyph, owner tint, base/display values (modifier badge for 同类强化/弱化). **No game artwork.**
- **3D flip animations with a distinct, recognisable effect per cause:** basic, 同数, 加算, 连携 (sequenced after the triggering flip), 逆转, 王牌杀手.
- Placement and flip order follow the facts of the explanation/log, never re-derived in the UI.
- **Fallbacks:**
  - without WebGL, or on narrow/touch screens, a 2D board with the same information and simple animations;
  - `prefers-reduced-motion` and the setting turn animations into instant state changes.
- **Performance:** measure and report frame rate on this machine's integrated and discrete GPU if available. Initial JS bundle budget: three.js lazily loaded for the table; record the gzipped sizes.

### Mascot dock (`docs/design/MASCOT.md` §4)

- **Build step:** WebP 512 px and 1024 px from `assets/mascot/raw/`.
- **Framing:** the art reaches the edges, so **pad, never crop**.
- **Behaviour:**
  - expressions mapped to events, with a crossfade of about 150 ms;
  - can be hidden;
  - a short neutral caption per event in both languages (the line library itself is 8D);
  - captions only state what the facts say.

### Languages

- **UI chrome:** every string in zh-CN and en, with a test that both key sets are identical.
- **Server explanations:**
  - add `explain.en.json` (same keys as `explain.zh-CN.json`) and a request parameter `lang` (`zh-CN` default, `en`) for `/ai_move`, `/v1/games*` replays and live events;
  - the facts stay language-neutral;
  - extend the explanation tests to English: every number from the facts, no win claim against the exact value, deterministic snapshots;
  - the UI requests the user's language.

### Security of the UI

- CSP:
  - `default-src 'self'`; `script-src 'self'` (no inline, no eval); `style-src 'self'` (or hashed);
  - `img-src 'self' data:`; `connect-src 'self'`;
  - `frame-ancestors 'none'`.
- No key in `localStorage`, `sessionStorage`, IndexedDB, URLs or logs (an e2e test inspects them after login).
- Every server text is rendered as text (no HTML injection).

### Accessibility

- Keyboard navigation for all controls.
- Labels for screen readers.
- Contrast WCAG AA in the dark theme.
- The table state is also available as a text list.

### Tests

- **Unit (Vitest):**
  - live-event reducer (ordering, lag, reconnect);
  - log parsing (schema v1 and v2, ambiguous cards);
  - i18n completeness;
  - settings.
- **End to end (Playwright), against a real `TTC.Siren.Server` started with a test key store, `TestSeedHeader` on loopback and a temporary game-log folder:**
  - landing and demo replay (anonymous);
  - login and logout (cookie flags, no key in storage);
  - a simulated plugin game posted to `/ai_move` appears live in the solve view (depth events, decision, explanation) and then in the log viewer, with the correct timeline;
  - replay controls;
  - download and delete;
  - the 2D fallback (WebGL disabled) and a narrow viewport;
  - language switch (UI and explanation);
  - reduced motion.
- **Screenshots** of each page in both languages, 3D and 2D, saved under `docs/reports/phase8b/` (small PNGs).

## 7. Documentation

- `docs/design/WEB_UI.md`: architecture, pages, state and events, the 3D table and flip effects, fallbacks, i18n, security (CSP, cookies), accessibility, every dependency and why.
- `docs/reports/PHASE8B_REPORT.md`:
  - each finding fixed, with commits and tests;
  - the Cloudflare readiness;
  - the access tiers with their measured behaviour under load (plugin latency with and without anonymous saturation);
  - UI screenshots;
  - bundle sizes and frame rates;
  - the warm-up measurement;
  - known limits.
- `API.md`: session endpoints, accepted authentication per endpoint, `lang`, 413, `ambiguous`, heartbeat and `Last-Event-ID`.
- `DEPLOY_WINDOWS.md`: Cloudflare, UI build and publish, sessions, the access-tier configuration.
- `CSHARP_MIGRATION.md`: §0 row 8 "8B DONE, awaiting independent review" (**do not set a gate**); §7 delivery record; §10 measurements; §11 and §12 handoff. Also `CLAUDE.md` §10 with the `web/` directory.

## 8. Acceptance criteria (all required)

- [ ] Step A committed and pushed (fast-forward); Steps B–D local; `main` unchanged; no PR.
- [ ] Build 0 warnings / 0 errors; all C# tests, pytest and web tests (unit and e2e) pass. No test skipped or weakened. The scheduler tests stay green 20/20 under CPU load.
- [ ] Core/Search unchanged (SIG comparison) and zero-allocation tests green.
- [ ] M8-1, M8-2, L8-1 … L8-4 fixed with tests; the documented publish works without flags.
- [ ] Cloudflare readiness:
  - real-IP logic with spoofing tests;
  - operator endpoints refuse proxy headers;
  - SSE heartbeat and resume;
  - `no-store` on dynamic endpoints;
  - the origin-timeout check.
- [ ] Access tiers:
  - plugin latency unaffected by anonymous saturation (measured);
  - anonymous share and caps enforced;
  - session cookie flags, expiry, revocation and CSRF tested;
  - no key or session id in any log or browser storage.
- [ ] Web UI:
  - every page works in zh-CN and en;
  - 3D and 2D;
  - live view and log viewer verified end to end with a simulated plugin game;
  - mascot reacts to events;
  - CSP has no unsafe directives for scripts;
  - accessibility basics present;
  - screenshots and measurements in the report.
- [ ] English explanation templates with the same guarantees as the Chinese ones.
- [ ] Docs updated as listed in §7.

## 9. Final report

Report briefly:
- the Step A hashes and push confirmation;
- the commits per step;
- each finding fixed;
- the Cloudflare readiness;
- the access-tier measurements;
- the UI pages with screenshot paths;
- the bundle sizes and frame rates;
- the test counts (C#, pytest, Vitest, Playwright);
- `git status -sb`;
- confirmation that 8C/8D/8E have not been started and that Steps B–D are not pushed.
