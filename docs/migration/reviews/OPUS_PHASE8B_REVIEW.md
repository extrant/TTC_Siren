# Opus independent review — Phase 8B (8A leftovers, Cloudflare readiness, access tiers, the web UI)

Date: 2026-10-11. Reviewer: Opus.

This is an independent review. Every claim was re-run, or re-checked with the reviewer's own probes, against a server that the reviewer published from a clean checkout. The reviewer also used a browser walk of their own.

Goal under review: `docs/migration/goals/phase8b_goal.md`.

Implementer documents:
- `docs/migration/reports/PHASE8B_REPORT.md`
- `docs/design/WEB_UI.md`
- `docs/design/GAME_LOG_SCHEMA.md` (schema 2)
- `API.md` (§5.3–§5.6, 413, `?lang`)
- `docs/DEPLOY_WINDOWS.md` (§2a, §4a, §5a)
- `CSHARP_MIGRATION.md` §7 "Phase 8B 交付记录"

## 1. Gate

**`PASS_WITH_NONBLOCKING_FINDINGS`**

| Severity | Count | Items |
|---|---|---|
| CRITICAL | 0 | |
| HIGH | 0 | |
| MEDIUM | 3 | M8B-1, M8B-2, M8B-3 |
| LOW | 5 | L8B-1 … L8B-5 |
| OPTIONAL | 3 | O8B-1 … O8B-3 |

**Every acceptance criterion of the goal (§8) is met:**
- Step A: pushed fast-forward, nothing squashed.
- Steps B–D: local only.
- Build: 0 warnings, 0 errors.
- Tests: 373 C#, 152 pytest, 82 Vitest and 19 Playwright, all green.
- Timing-dependent tests: 20/20 green under full CPU load.
- Core/Search: unchanged (the 62 `SIG` lines are byte-identical).
- 8A findings: M8-1, M8-2 and L8-1 … L8-4 are fixed and verified.
- Cloudflare readiness: verified.
- Access tiers: measured again by the reviewer, with the same result.
- Web UI: works in both languages, in 3D and 2D, and on a phone.
- UI security: the CSP is strict, the key never reaches browser storage, and server text is rendered as text.

**None of the findings blocks Phase 8C.** Two of them, M8B-1 and M8B-2, are security-relevant and should be fixed before the public launch (see §8).

## 2. Repository and scope

**Pushes and branches (`git ls-remote origin`):**
- `refs/heads/csharp-solver-migration` = `6d13de5`, the Step A head.
- `main` = `eb60a02`, unchanged.
- `refs/pull/1/head` = `d0ef721` is the old PR of 2026-10-07; no new PR was opened.
- The tag `1.0.0.0` is on `main` and is pre-existing.

**Step A:**
- `c85481d` contains exactly the reviewer's files: the 8A review, `CSHARP_MIGRATION.md` (the 8A gate lines, A-27 … A-29, Q-10, row 8), the `phase8a_goal.md` banner, `PHASE8_PLAN.md` §6 and the 8B goal.
- `6d13de5` contains the nine PNG originals, unmodified.
- The push was a fast-forward (`45a9b28..6d13de5`), not squashed. The reflog shows only commits and two amends that happened before the push.

**Steps B–D:**
- 19 local commits, `b98051c` … `1d7bdb0`, none pushed; `git status` is clean.
- **Every one of the 19 commits builds on its own** (reviewer, in a separate worktree): 0 warnings, 0 errors each.

**Nothing forbidden was committed:**
- No `node_modules`, `dist`, `wwwroot` or `bin/obj`.
- No key stores, session stores, game logs, certificates, or Playwright reports or traces.
- `.gitignore` covers all of them.
- What was committed: 30 small screenshots (2.8 MB in total) and the raw measurement files, as in 8A.

**Python business code is unchanged.** The only Python changes are:
- new tools under `tools/`;
- `tests/test_csharp_engine_pool.py`, whose burst test was made truly concurrent with a barrier. The assertion is stronger than before, not weaker.

**Core/Search unchanged:**
- `git diff 6d13de5 HEAD -- csharp/src/TTC.Siren.Core csharp/src/TTC.Siren.Search` is empty.
- The reviewer built `Bench` at `6d13de5` and at `HEAD` and ran `search 1` and `fast 1`. All 62 `SIG` lines are byte-identical: nodes, score bits, moves, and allocation 0.

**C# packages:** no new packages. The only one is `Microsoft.Extensions.Hosting.WindowsServices`, as before.

**Web dependencies:**
- Exact pins and a lockfile.
- Runtime dependencies: `react`, `react-dom`, `three` and `@react-three/fiber`.
- Build and test dependencies: `vite`, `typescript`, `vitest`, `jsdom`, `@playwright/test`, `sharp`, `@types/*` and `@vitejs/plugin-react`.
- `npm ci` reported 0 vulnerabilities.

**8C, 8D and 8E were not started.** There is no NPC data export, no deck work, and no coaching endpoint.

**Housekeeping:** an old worktree `C:\Users\Administrator\AppData\Local\Temp\wt_stepA` (detached at `03b4704`, from Phase 6) is still registered. It is harmless; remove it with `git worktree remove` when convenient.

## 3. Correctness runs (reviewer)

| Run | Result |
|---|---|
| `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` | 0 warnings, 0 errors |
| `dotnet test -c Release` | **373 / 373** passed (30 s) |
| `python -m pytest -q` | **152 passed** |
| `npm test` (Vitest) | **82 / 82** in 9 files |
| `npm run e2e` (Playwright, Edge, real server) | **19 / 19** passed |
| Each of the 19 commits, built on its own | 0 warnings, 0 errors every time |
| Bench SIG, `6d13de5` vs `HEAD` (`search 1`, `fast 1`) | 62 / 62 lines identical |
| `tools/phase8b/repeat_under_load.py --runs 20` with 20 burner processes; `SchedulerTests`, `LiveEventTests`, `AccessClassTests` and **`WebSessionTests`** = 63 tests (the implementer ran 45) | **20 / 20** runs green, 40–48 s each |
| Clean worktree at `HEAD`: `npm ci`, then the documented `dotnet publish csharp\src\TTC.Siren.Server -c Release -o publish\server` with no flags | Succeeds. The UI is built into `wwwroot/` (hashed assets); the server's own `appsettings.json` (127.0.0.1:5000) ships; no `engine.appsettings.json` in the output. The published exe starts, warms up (40 searches on 20 workers, 0.58 s) and answers `/health`. |

Bundle sizes from the reviewer's clean build, as printed by Vite:

| Part | Raw | gzip | Report | Difference |
|---|---|---|---|---|
| Main script | 343.5 kB | 105.1 kB | 104.9 KiB (first page) | within 1 % |
| `Board3D` chunk | 917.3 kB (= 895.8 KiB) | 243.8 kB | 235.3 KiB | 3.6 % (the gzip level differs between Vite and the size script) |

Both are well inside the 15 % tolerance.

The frame rate was not reproduced. The report measured it on a software renderer only (SwiftShader) and says so; see coverage gaps (§9).

## 4. Step B — 8A findings and Cloudflare readiness: PASS

**M8-1 — publish.** Verified as above, in a clean checkout, with no flags. The fix is at the source: the engine's file is now `engine.appsettings.json`, and a publish target drops the engine's host files. `PublishTests` covers it.

**M8-2 — timing-dependent tests.**
- The scheduler now takes a `TimeProvider`, and the tests use a `LogicalClock` with grant numbers. No wall-clock assertions remain in `SchedulerTests`.
- The Python pool test now waits on a barrier, so all 12 requests really are concurrent, and it asserts `accepted >= 12`.
- The reviewer ran it 20/20 green under full load, on a wider test set than the implementer used.

**L8-1 — test seed switch.** Probed with the published exe:
- On `0.0.0.0`, start-up is refused (exit 2: "allowed only when the server listens on a loopback address").
- With `TrustLoopback` or `BehindTlsProxy`, start-up is refused (exit 2).
- On loopback the server starts and logs the warning "TestSeedHeader is ON … must never be on in production".
- The header itself is honoured only from a loopback peer that carries no proxy headers.

**L8-2 — log numbers and the stored request.**
- A game was logged in two requests:
  - one sent with `json.dumps` default separators (with spaces);
  - one sent compact, with `max_depth: 3.0`.
- No integer field of the log outside `request_raw` is written as `x.0`.
- `request_raw` of the compact request equals the posted bytes exactly, including `3.0`.
- `request_raw` of the spaced request equals the posted text with the whitespace removed. This is documented (GAME_LOG_SCHEMA.md, API.md), but it falls short of the goal's wording "exactly as sent"; see **L8B-1**.
- The games API reports `schema: 2`.

**L8-3 — 413.** A 1 MiB + 12 B body to `/ai_move` gets **413** with `{"error":"Request body too large (limit 1048576 bytes)"}`.

**L8-4 — ambiguous cards.** Cards #12 and #369 both have edges 2/5/5/2 and no type. The reviewer played a game in which the opponent placed that card. The `opponent_move` record and `GET /v1/games/{id}` both carry `ambiguous: true`, `candidate_ids: [12, 369]` and `card_edges: [2,5,5,2]`.

**O8-3 — warm-up.** Present, on by default, measured in the report. The reviewer saw it run on the published exe.

**Cloudflare (reviewer's probes against the published exe):**

| Check | Result |
|---|---|
| No trusted proxy configured. Nine failed logins, each with a different `CF-Connecting-IP` and `X-Forwarded-For` | Throttled after 5 (`429`, `Retry-After: 10`). The audit records `127.0.0.1` every time: the spoofed headers are ignored. |
| `TrustLoopback=true` (tunnel), eight different `CF-Connecting-IP` values | Each address has its own bucket (8 × 401, no 429). The audit shows the header addresses. |
| Same configuration, but only `X-Forwarded-For`, an invalid IP, or `CF-Connecting-IP` sent twice | Not believed: the peer address is used. |
| `Cidrs=10.0.0.0/8` only, or `UseCloudflareRanges=true` (the shipped file), with a loopback peer sending `CF-Connecting-IP` | Ignored; throttled by the peer address. |
| Operator endpoints from loopback | No proxy header: 200. With any of `CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For`, `Forwarded`, `X-Real-IP`, `True-Client-IP` or `Via` (any capitalisation), on both `/admin/status` and `/admin/reload-keys`: **403**. |
| SSE heartbeat | `: heartbeat` arrived at 21.0 s and 41.0 s (default 20 s; the timer restarts after each event). `retry: 3000` is sent first. |
| SSE resume | Reconnecting with `Last-Event-ID: 23` replayed exactly ids 24–30, in order. An id beyond the newest one gives `resync` with `reason: stream_restarted`. See L8B-2 for the restart case. |
| Caching | `no-store` on `/health`, `/v1/*`, 401 answers and JSON 404s. `no-store, no-cache` on the stream. `no-cache` with the CSP on `index.html` and on SPA fallbacks. `public, max-age=31536000, immutable` on hashed `/assets/*`. |
| Origin timeout | `OriginTimeoutSeconds=30` with the defaults: start refused with the worst-case arithmetic (20 + 60 + 10 = 90 s). `MaxSearchSeconds=0` with an origin timeout: refused. The defaults (90 < 95) start. |
| Missing Cloudflare range file | Start is refused, but by an unhandled exception rather than a configuration error (L8B-3). |

## 5. Step C — access tiers and the web login: PASS, with M8B-1 and M8B-2

**Capacity, measured again by the reviewer** with `tools/phase8b/tier_load.py` (12 workers, P-cores, 1 run):

| Plugin latency, p50 / p95 / max (ms) | Quiet | Anonymous flood, with classes | Same flood, no classes |
|---|---|---|---|
| Mid-game | 10.5 / 14.0 / 108.3 | 10.4 / 13.4 / 15.7 | 7,285 / 7,816 / 7,816 |
| Robust endgame | 6.0 / 9.4 / 12.6 | 5.2 / 6.4 / 10.4 | 7,275 / 7,614 / 7,614 |
| Opening (1 s limit) | 1,003.9 / 1,008.4 / 1,008.4 | 1,004.2 / 1,007.5 / 1,007.5 | 8,749 / 8,782 / 8,782 |

**Under the anonymous flood, with classes:**
- at most 3 anonymous searches ran at once;
- at most 16 anonymous requests were queued;
- **0 plugin requests were queued**;
- the flood got 20,743 `busy` answers.

This matches the report: the plugin is unaffected, and anonymous work stays within its share.

**Code read.** `WorkerPool.Admissible` and `Dispatch`:
- plugin requests may use every free worker;
- web and anonymous work together never go past `workers − reserved`;
- anonymous work also stays under its own maximum;
- inside a class the O8-1 cost order with aging is unchanged.

There is no starvation inside a class: the aging rule is unchanged, and the scheduler tests stay green under load. Between classes the priority is strict, by design (A-27).

**Session cookie** (reviewer, plain-HTTP loopback):
- The cookie is `ttc_session=<43 chars>`, with `HttpOnly`, `SameSite=Strict`, `Path=/`, `Max-Age=43200` and `Expires`.
- `Secure` and the `__Host-` prefix are absent because there is no TLS. The code sets them whenever the request is HTTPS, `BehindTlsProxy` is set or `Https.Enabled` is on.
- The login body contains the key's id and name, never the secret.
- `/v1/me` through the cookie gives `class: web`, `via: session`, and neither the secret nor the session id.

**`/ai_move` with only the cookie gets 401.** The plugin endpoint is key-only, as decided.

**CSRF.**
- A cookie-authenticated `DELETE /v1/games/x` is refused (**403**) for 8 forged forms:
  - no Origin or Referer;
  - a foreign Origin;
  - `Origin: null`;
  - a foreign Referer;
  - the correct Origin but `Sec-Fetch-Site: cross-site`;
  - the correct Origin but `Sec-Fetch-Site: same-site`;
  - another port;
  - a host-prefix trick.
- The same request with the correct Origin passes the check (and gets 404, game not found).
- A cross-site logout and a cross-site login are both 403.

**Revocation.** After `keys revoke`, new requests on that key's session fail with **401 within 1.03 s**, as documented ("within a second"; the key file is checked at most once a second). **However, a `/v1/live` stream opened before logout or revocation stays open and keeps delivering events** (M8B-1).

**Logout.**
- `DELETE /v1/session` ends the session: `/v1/me` then gets 401, and the browser cookie is cleared (0 cookies left).
- The browser's storage stays empty.
- An already-open stream still keeps receiving events (M8B-1).

**Login throttling and audit.** Verified above. The audit lines hold time, IP, outcome and key id, and **no fragment of any presented key**.

**Logs.** The reviewer scanned everything the published servers wrote for every test key (full text and 16-character fragments) and for cookie names:
- server stdout and stderr at Information level;
- the four audit files;
- every game-log file.

No hits.

**Development switch behind a tunnel: M8B-2.**

## 6. Step D — the web UI: PASS, with M8B-3

The reviewer walked the UI with their own Playwright script (headless Edge, the published build, a real server). Results:

**Landing and demo:**
- The landing page works at 1366×768, at 1920×1080 and in the narrow layout.
- The demo replay plays anonymously.
- The mascot is padded, never cropped, as MASCOT.md requires.

**Login:**
- A wrong key shows only "登录失败".
- After login:
  - `localStorage`, `sessionStorage` and IndexedDB are all empty;
  - `document.cookie` is empty (the cookie is HttpOnly);
  - the URL is `/live`;
  - **the key, and every 16-character fragment of it, appears nowhere.**
- The key field is `type=password` with `autocomplete=off`, and is emptied the moment the request is sent.

**Live view.** A plugin game was posted in real time while the view was open:
- the connection badge reads "已连接";
- "follow newest game" works, and the game picker lists the game;
- the depth table fills in as depths complete (depth, score, best move, nodes, NPS, time);
- the decision, the assumed opponent cards, the explanation (brief, with "show details") and the solved/exact badges all appear;
- the principal variation is drawn as ghost cards (see L8B-5).

**Game logs:** list, replay with first/previous/play/next/last, speed, scrubber, the timeline, download and delete. These were exercised by the reviewer and by the e2e suite.

**Flip effects against the facts.** The reviewer recorded the `data-flip` cause that the flat table animates on each step, and compared it with the log:

| Game | Step | UI | Log |
|---|---|---|---|
| Demo | 2 | cell 8 `basic` | `basic` |
| Demo | 3 (opponent) | cell 8 `fallen_ace` | `fallen_ace` |
| Demo | 5 | cell 7 `same`, then cell 8 `combo` | same order |
| Demo | 6 | cell 5 `basic` | `basic` |
| Demo | 8 | cell 2 `plus`, then cell 5 `combo` | same order |
| Reviewer's 逆转 game | 2 | cells 1 and 3 `reverse` | `reverse` ×2 |

All six causes match cell for cell. The UI takes flips only from the facts and the `opponent_move` record; `buildTimeline` and `planFlips` apply no rules, and Combo waits for the flip that triggered it.

**Text is rendered as text.** The reviewer posted games whose rules and NPC-name fields held `<img src=x onerror=…><b>xss</b>`, and one whose `max_depth` held that string, so that it is echoed in the server's 400 error.
- No element was created, and the page title was untouched.
- The game list shows the markup as plain text.
- There is no `dangerouslySetInnerHTML`, `innerHTML` or `eval` in `src/`.

**CSP.**
- The header is `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; …; frame-ancestors 'none'`, with no `unsafe-*`.
- `index.html` has no inline script.
- No CSP violation was reported on any page the reviewer visited.
- The only console errors are the expected 401s of the anonymous `/v1/me` probe.

**Languages.** The UI switches between zh-CN and en.

**English explanations:**
- `explain.en.json` has the same 115 keys as zh-CN, and the placeholders match. English adds `card_cap`, a capitalised form, which the renderer provides.
- Outside placeholders there are no literal numbers.
- The words "win" and "wins" occur only in `outcome_exact_win` and in the rule descriptions of basic/Reverse flips ("the higher value wins").
- The Chinese snapshots are unchanged, so the Chinese output is byte-identical to 8A.
- `ExplanationEnglishTests` adds the same guarantees and English snapshots.

**Accessibility:**
- Tab order: skip link, then navigation, language switch, logout, back, download, delete, and the replay controls (`aria-label`), speed, the scrubber (labelled) and the timeline items.
- No control lacks an accessible name.
- The table state is also given as a text list.
- Contrast of the theme tokens: text 16.2:1 and 13.2:1, muted 8.7:1 and 7.2:1, accent 10.1:1 and 8.3:1, gold 11.0:1, danger 8.3:1 and 6.8:1, button ink 9.1:1. All pass WCAG AA (≥ 4.5).

**Narrow screens:** the 2D table is used, and there is no horizontal scroll (`scrollWidth = innerWidth = 390`).

**Reduced motion, no WebGL, and mascot off** are covered by e2e tests 13–19, which the reviewer ran.

**Mascot behaviour.** The expression and caption change with events: "已算到终局：结果是平局。" on an exact result, "等待下一次请求。" while idle, and a different pose while explaining. **But at common desktop widths the mascot covers content** (M8B-3).

## 7. Findings

### M8B-1 (MEDIUM) — an open `/v1/live` stream outlives the session or key that opened it

- **Location:** `csharp/src/TTC.Siren.Server/ServerHost.cs`, the `MapGet("/v1/live")` handler; `Live/LiveStream.cs`, `RunAsync`.
- **Problem:** authentication is checked once, when the stream opens. The loop then runs until the client disconnects. Logout, session expiry, eviction past `MaxPerKey`, and key revocation all leave an already-open stream running.
- **Evidence:** the reviewer logged in, opened `/v1/live` with the cookie, and logged out. `/v1/me` was then 401. Then a plugin request was sent with the same key, and the **logged-out stream received 8 events** (`request.received`, `search.depth` ×n, `decision`, `game.updated`). After `keys revoke`, new requests failed within 1.03 s, but the old stream was still open.
- **Expected:** API.md §5.6 and DEPLOY §4a say that a session "ends" on logout, expiry or revocation. A stream authorised by a session (or by a key) should end at that moment, or at the latest at the next heartbeat.
- **Impact:**
  - The live events of a key (boards, hands, explanations) keep flowing to a browser tab or a stolen cookie after logout or expiry, and the 12 h lifetime does not bound them.
  - After revocation no new events are produced, because the key can no longer post, so the leak is mainly about logout, expiry and eviction.
  - Requires a valid session at the time the stream opens.
- **Fix:**
  - Pass a re-validation callback into `LiveStream.RunAsync`: `sessions.Find(cookie)` for a web stream, `keys.GetById(id).Enabled` for a key stream.
  - Call it before writing each event and at each heartbeat; on failure, write `event: session_ended` and return.
  - Alternatively, have `SessionStore.Remove`, `RemoveAllOf` and expiry cancel a per-session token that the stream links to.
- **Regression tests:**
  - open a stream with a cookie, log out, post a plugin request: the stream must close (or receive `session_ended`) and receive no further `decision`;
  - the same with expiry (a `LogicalClock` moved past the lifetime);
  - the same with `keys revoke`, for both a key stream and a session stream.
- **Human decision:** none.

### M8B-2 (MEDIUM) — the development switch `AllowLoopbackWithoutKey` opens the server to the internet behind the documented tunnel

- **Location:**
  - `Security/SecurityOptions.cs:74` (`Validate` only checks that the host is loopback);
  - `Security/KeyAuthenticator.cs:42` (a loopback peer without a key is admitted as `local-dev`, whatever headers it carries).
- **Problem:** in the recommended `cloudflared` topology (DEPLOY §5a 2b), every internet request arrives from `127.0.0.1` and the host is loopback. If `AllowLoopbackWithoutKey` is also on, validation passes, and **any request that comes through the tunnel is authenticated with no key**.
- **Evidence:** the reviewer started the published exe with `AllowLoopbackWithoutKey=true` and `TrustLoopback=true`. It started (no configuration error). A request carrying `CF-Connecting-IP` and `CF-Ray` and **no key**:
  - got 200 on `GET /v1/games`;
  - got **200 on `POST /ai_move`** (a full search).
- **Expected:** the same protection that L8-1 gave the test seed switch.
  - Refuse `AllowLoopbackWithoutKey` at start when `BehindTlsProxy` or `TrustedProxies.AnyTrusted` is set.
  - Ignore it for any request that carries a proxy header (`ClientIpResolver.CarriesProxyHeaders`), the same check the operator endpoints already use.
- **Impact:** a development flag that is default-off and documented as "development only" becomes a complete authentication bypass in the production tunnel set-up. One forgotten setting exposes the plugin endpoint, the game logs and the solver to the internet. The keyless `local-dev` identity also bypasses the per-key limits.
- **Fix:** both checks above (start-up and per request). Add a sentence to DEPLOY §5a.
- **Regression tests:**
  - `ServerOptions.Validate` throws for `AllowLoopbackWithoutKey` with `TrustLoopback`, with `UseCloudflareRanges`, with `Cidrs` and with `BehindTlsProxy`;
  - a loopback request with `CF-Connecting-IP` and no key gets 401 even when the switch is on.
- **Human decision:** none.

### M8B-3 (MEDIUM, UX) — the mascot dock covers the core content at common desktop widths

- **Location:** `web/src/components/Mascot.tsx` and the dock CSS in `web/src/styles/app.css`.
- **Problem:** the dock is a fixed overlay at the bottom right, with no space reserved in the layout. Its caption bubble and figure overlap the explanation text, the result badges and the timeline:
  - at 1180×760 in the implementer's own screenshots (`live-zh-CN-3d.png`, `replay-our-move-en-3d.png`);
  - at **1366×768** in the reviewer's walk: the live explanation, the badges and the right edge of the timeline;
  - on a phone, where the caption bubble covers the bottom-right card of the board.
- **Expected:** with the mascot on (the default), the decision, the explanation, the badges and the board are never covered.
- **Impact:** the explanation is one of the product's main features. By default, at the most common laptop resolution, part of it is hidden. The report lists this as a known limit.
- **Fix:**
  - On wide screens, give the dock its own column or bottom gutter (for example `padding-right` on `main` equal to the dock width, from about 1200 px).
  - Below that, collapse the dock to a small avatar with the caption on demand.
  - On phones, render the caption below the board, not over it.
- **Regression test:** a Playwright check at 1180×760, 1366×768 and 390×844 that the bounding boxes of `[data-testid=mascot]` and of the explanation, decision, board and timeline do not intersect.
- **Human decision:** the user may prefer a particular layout (column versus collapsed avatar).

### LOW

- **L8B-1 — `request_raw` is the request with its whitespace removed** (`GameLog/GameLogService.cs:487`, `RawJson.Compact`).
  - Tokens, member order and number spellings are kept, and a compact body round-trips byte for byte.
  - A request sent with spaces (plain `json.dumps`) does not.
  - This deviates from the goal's "exactly as sent" and "equals the posted bytes". It is documented, and nothing that depends on the log is affected.
  - Options: keep it and amend the wording (recommended; replay needs only tokens), or store the original text as a JSON string beside it.
  - Human decision: which of the two.
- **L8B-2 — after a server restart, a stream can resume from a wrong position without being told.**
  - Event ids restart at 1 per key (`Live/LiveHub.cs:146` treats an id ≤ the new sequence as "resumed").
  - A browser that saw id 50 before a restart, and reconnects after the new run has produced ≥ 50 events of that key, gets a silent "resume". It misses the new run's first events, and no `resync` is sent.
  - Fix: prefix ids with a run epoch (`<run>-<seq>`), or send `resync` whenever the client's id belongs to an earlier run.
- **L8B-3 — a missing Cloudflare range file crashes the start with an unhandled exception** (`FileNotFoundException` in `ClientIpResolver`'s constructor, called from `ServerHost.Create`).
  - The exit code is 0xE0434352, and the text is a stack trace instead of the usual "configuration error: … (exit 2)".
  - The intended effect (no start) holds.
  - Fix: catch the exception in `Program` (or validate in `TrustedProxyOptions.Validate`) and report it like the other configuration errors.
- **L8B-4 — a broken command in `docs/DEPLOY_WINDOWS.md` §5a item 9** (lines 162–163).
  - `python tools\refresh_cloudflare_ips.py C:\ttc\server\cloudflare-ips.txt` was written with escapes: `\r` became a line break and `\t` a TAB.
  - The file now reads `python tools` / `efresh_cloudflare_ips.py C:<TAB>tc\server\…`.
  - Fix the text and check the file for other control characters (none were found in the other changed `.md` files).
- **L8B-5 — the principal-variation ghost cards are not explained.**
  - On the table (`Board2D.tsx:47`, `cardArt.ts:132`) they are drawn at 62 % opacity with a ply number in the type circle, which on the dark theme looks much like a real card.
  - The text list, correctly, says those cells are empty, and nothing on the page explains the difference. The reviewer at first read a ghost as a real card.
  - The ghost cells carry no `data`/ARIA marking.
  - Fix: a one-line legend ("faded cards = predicted continuation, number = ply"), a dashed outline on ghosts, and `aria-label` on ghost cells.

### OPTIONAL

- **O8B-1 — the wait queue is shared between classes.**
  - `QueueLimit` (256) counts every class. Only anonymous work has its own cap (16).
  - Today web work never queues (8B has no web compute), so nothing is affected.
  - Once 8C/8D add web compute, a flood of web work could fill the queue, and a plugin request that arrives while all workers are busy would get `busy`.
  - Before that: give web its own queue cap too, or let plugin requests ignore the global limit.
- **O8B-2 — `/health` is public and says a lot.**
  - It returns the full version string, including the commit SHA, the card-DB hash and per-class statistics.
  - The warm-up's 40 searches are counted as plugin `served`.
  - Consider a minimal public `/health` (`status` only) and keep the details on `/admin/status`; start the counters after the warm-up.
- **O8B-3 — the frame rate was measured with software WebGL only.** Measure once on the target machines (a real GPU laptop and a phone) before launch.

## 8. Go-live checklist (public web behind Cloudflare)

Carried over from the 8A review §8. 8A's M8-1 is closed by 8B.

| # | Item | Status |
|---|---|---|
| 1 | **Fix M8B-1** (streams end with their session or key) | open |
| 2 | **Fix M8B-2** (the development switch is refused behind proxies). Until then, make sure `Security:AllowLoopbackWithoutKey` is `false`; it is false by default. | open |
| 3 | Fix M8B-3 (the mascot covers content), or ship with the mascot off by default | recommended |
| 4 | Fix the DEPLOY §5a item 9 command (L8B-4) | recommended |
| 5 | Cloudflare dashboard: DNS proxied; Full (strict) with an Origin CA PFX, **or** a `cloudflared` tunnel with `TrustLoopback=true`; cache bypass for `/ai_move`, `/v1/`, `/api/`, `/health`, `/admin/`; a WAF rule blocking `/admin`; rate rules for `POST /v1/session` | to do (outside the code) |
| 6 | `OriginTimeoutSeconds`, `MaxSearchSeconds` and `QueueWaitMs` checked against the plan's timeout (free plan: 100 s) | the defaults pass |
| 7 | `TestSeedHeader` off (it is refused automatically behind a proxy) | verified |
| 8 | The worker count measured on the E5-2698B v3 (`tools/phase7/run_load_matrix.py`) and the access-class shares chosen for it | to do |
| 9 | Back up `keys.json`, `audit.jsonl` and the game-log directory; keep them out of the repository | the `.gitignore` is in place |

## 9. Test-coverage gaps

- **No test that an open stream ends on logout, expiry or revocation** (M8B-1).
- **No test of `AllowLoopbackWithoutKey` together with trusted proxies or proxy headers** (M8B-2).
- No layout test that the mascot dock never overlaps content (M8B-3).
- No test of SSE resume across a restart (L8B-2).
- No automated accessibility audit (for example axe) and no measured contrast inside the canvas card faces. The reviewer computed the CSS token contrast by hand.
- The frame rate was measured only with software WebGL. No other browser than Edge (Chromium) was run.
- The Cloudflare path was simulated with headers on loopback. A real Cloudflare or `cloudflared` hop was not run, and no Origin CA certificate was tested.
- The access-tier numbers are from one machine on loopback.

## 10. Next steps

- **Phase 8C may be authorized** (deck optimisation and porting `/api/*`; see `PHASE8_PLAN.md`).
- Recommendation: make the **first step of the 8C goal** the fix of M8B-1 and M8B-2 with their regression tests, plus L8B-3 and L8B-4. This mirrors how 8B started with the 8A findings.
- Also give web work its own queue cap there (O8B-1), because 8C introduces web compute.
- M8B-3 (the mascot layout) and L8B-5 (the ghost legend) can go in the same goal, or wait for 8D, which reworks the coaching UI anyway. This is the user's call.
- L8B-1 needs a one-line decision: keep the compacted `request_raw` and amend the wording (recommended), or store the verbatim text as well.
- No remediation goal was written: nothing blocks 8C.
- Steps B–D are still local. Pushing them needs the user's authorization.

## 11. Commands and probes run by the reviewer

- `git status -sb`, `git log`, `git reflog`, `git ls-remote origin`, `git show --stat` of each Step A commit, `git diff 6d13de5 HEAD` (name lists, Core/Search, Python, Engine, Server, Explain, web).
- `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`; `dotnet test -c Release --no-build`; `python -m pytest -q`; `npm test`; `npm run e2e`.
- Bench at `6d13de5` and at `HEAD` (worktree): `TTC.Siren.Bench search 1`, `fast 1`; the `SIG` lines diffed.
- A worktree per commit: `dotnet build -c Release` for each of the 19 commits.
- A clean worktree at `HEAD`: `npm ci`, the documented `dotnet publish … -o publish\server` (no flags), then the published exe started with a scratch key store (`keys add` ×5, `keys revoke` ×2).
- `python tools/phase8b/repeat_under_load.py --runs 20 -- dotnet test … --filter SchedulerTests|LiveEventTests|AccessClassTests|WebSessionTests`.
- `python tools/phase8b/tier_load.py --cpu auto --workers 12 --runs 1`.
- Reviewer probes (scratch, not committed):
  - headers, 413, admin and proxy headers, cookie flags, CSRF, revocation, throttling and spoofing, audit;
  - integers, `request_raw` and `ambiguous`;
  - SSE heartbeat, resume and resync;
  - trusted-proxy configurations and every start-up refusal, including the tunnel bypass;
  - the stream after logout or revoke;
  - a scan of every server log for keys and cookies.
- Reviewer UI walks (headless Edge, the published build, temporary scripts deleted afterwards):
  - landing, login, storage and cookies, live during a real-time plugin game, logs, replay, settings, 390 px, keyboard order, accessible names, logout, CSP and console;
  - flip cause per step against the log (demo and a 逆转 game);
  - markup injection through rules, NPC name and an error message.
- Contrast ratios of the CSS tokens (WCAG formula).
