# Phase 8B report — access classes, Cloudflare readiness, and the web UI

Date: 2026-10-11. Branch `csharp-solver-migration`. Authorisation: `.claude/phase8b_goal.md`. Status: **done, awaiting independent review** (no gate is set here). Step A is pushed; Steps B–D are local commits only.

## 1. What was done

| Step | Result | Commits |
|---|---|---|
| A | Reviewer files and the nine mascot originals committed and pushed fast-forward `45a9b28..6d13de5` to `origin/csharp-solver-migration` (`6d13de5`; `main` still `eb60a02`). | `c85481d` (docs, review, A-27…A-29, Q-10, plan §6, goal), `6d13de5` (mascot originals, unmodified) |
| B | Findings of the 8A review fixed; Cloudflare readiness; start-up warm-up. | see §2, §3 |
| C | Access classes with reserved capacity; web login; CSRF check; measured. | `6800e34`, `07899c2` |
| D | Web UI; English explanations; hosting and publish. | `6c73a60`, `e503794`, `a996286`, `e4ce52a`, `8ce05a9`, and the documentation commits |

## 2. Findings of the 8A review

| Item | Fix | Commit | Test |
|---|---|---|---|
| **M8-1** `dotnet publish` failed (NETSDK1152, two `appsettings.json`) | The engine's own file is `engine.appsettings.json`; a publish target drops the engine's host files from the server's output. The documented command works without flags. | `b98051c` | `PublishTests` (runs the published command; checks the shipped file, the GC settings, the web UI) |
| **M8-2** scheduler tests depended on wall-clock time | A logical clock and grant numbers replace sleeps; the live-hub timing assertion became a hang detector; the burst test is deterministic. Nothing was loosened. | `3dd6898` | `SchedulerTests`, `LiveEventTests`; 20 consecutive runs with every logical CPU kept busy (§4) |
| **L8-1** the test seed header could be on in production | Refused unless the listener is loopback and no proxy is configured; a warning is logged when on. | `84a8569` | `TestSeedSwitchTests` |
| **L8-2 / L8-4** floats in the log, `request_raw` re-serialised, ambiguous opponent cards | Log schema v2: integers stay integers, `request_raw` is the request text verbatim (`RawJson.Compact`), `ambiguous` + `candidate_ids` on inferred opponent cards. | `ccadb43` | `GameLogFormatTests` |
| **L8-3** an oversized body reset the connection | 413 in the usual error form; the body is drained within a budget so the client reads the answer. | `deb1606` | `BodyLimitTests` |
| **O8-3** first requests after start were slow | Optional warm-up on every worker before the listener opens (`Server:WarmUp`). Measured in §5. | `321bd5d` | `WarmUpTests` |

## 3. Cloudflare readiness (`6642204`)

The real client address is taken from `CF-Connecting-IP` / `X-Forwarded-For` **only** when the peer is a trusted proxy (the published Cloudflare ranges in `cloudflare-ips.txt`, refreshable with `tools/refresh_cloudflare_ips.py`, or configured addresses); a header from anybody else is ignored. The SSE stream sends a heartbeat comment every 20 s (below Cloudflare's 100 s idle timeout) and `retry: 3000`, numbers its events per key and resumes from `Last-Event-ID` (or `?last_event_id=`); a gap or a restart is announced with a `resync` event. API responses carry `Cache-Control: no-store`. The origin's own timeouts are validated at start against the proxy's (`OriginTimeoutSeconds` / `OriginMarginSeconds`). Tests: `CloudflareReadinessTests`. Operation: `docs/DEPLOY_WINDOWS.md` §5a.

## 4. Access classes and the web login (`6800e34`, `07899c2`)

Three classes — **plugin** (API key), **web** (session cookie), **anonymous** — served in that strict order; inside a class, cost-ordered with aging (O8-1 unchanged). 25 % of the workers are never given to anything but plugin work; anonymous work is capped at 25 % of the workers (at least one), has its own queue (16, 10 s) that answers `503 busy` + `Retry-After` when full, a lower search limit (depth 8, 3 s) and a per-real-address bucket. The web login exchanges the key (request body only, 4 KiB at most) for a 256-bit session id in an HttpOnly, SameSite=Strict cookie (`__Host-` and Secure behind TLS); sessions end on logout, expiry (12 h), revocation of the key, or when a key has more than 20. Logins are throttled per real address and audited without the key and without the session id. A cookie-authenticated request that changes something must pass the same-origin check (Origin, else Referer; `Sec-Fetch-Site`); nine forged forms are tested.

**Measured (`tools/phase8b/tier_load.py`, `TierLoad/`; 12 workers, P-core mask `0xFFF`, 3 independent runs, medians; raw data `docs/reports/phase8b/tiers.*`).** The plugin sends mid-game, robust-endgame and opening requests while an anonymous client floods the server (an open loop, ~21,000–22,000 requests per run):

| Plugin latency, ms (p50 / p95 / max) | quiet | anonymous flood, with classes | same flood, **without** classes (control) |
|---|---|---|---|
| mid-game | 14.1 / 19.5 / 155.5 | 13.6 / 17.2 / 34.7 | 8,425 / 8,949 / 8,949 |
| robust endgame | 6.9 / 12.0 / 17.2 | 7.2 / 9.5 / 20.0 | 8,599 / 9,016 / 9,016 |
| opening (≈1 s search) | 1,005.8 / 1,011.0 / 1,011.0 | 1,005.5 / 1,010.6 / 1,010.6 | 9,272 / 9,791 / 9,791 |

With classes the plugin is unaffected by the flood (the flood got 29–35 answers and ~21,000 `busy` answers; at most 3 anonymous searches ran at once and 0 plugin requests ever queued); without them every plugin request waits behind the flood for 8–9 s. Caveat: one machine, loopback, pure CPU contention.

## 5. Start-up warm-up (`321bd5d`; `tools/phase8b/warmup_latency.py`, 3 runs, medians)

| | ready after | opening: first request (median of first requests) | mid-game first | robust first |
|---|---|---|---|---|
| warm-up off | 0.6 s | 1.392 s | 0.030 s | 0.031 s |
| warm-up on | 1.1 s | 0.637 s | 0.018 s | 0.018 s |

The steady state (per-minute median) is the same with and without. Warm-up costs 0.5 s at start and takes the JIT cost off the first user.

## 6. The web UI (`6c73a60`, `e503794`, `a996286`, `e4ce52a`, `8ce05a9`)

Design: `docs/design/WEB_UI.md` (architecture, state and events, flip effects, fallbacks, i18n, security, accessibility, every dependency and why).

- **Pages** (screenshots in `docs/reports/phase8b/screenshots/`, English and Chinese, 3D table and flat table, and a phone): landing with the demo replay, login (logged out and logged in), live solve (waiting / in progress), game logs, replay (our move, the opponent's move), settings, not found, "login needed". Files: `landing-*`, `login-*`, `live-*`, `live-login-needed-*`, `games-*`, `replay-our-move-*`, `replay-opponent-move-*`, `settings-*`, `notfound-*`, `phone-replay-*`.
- **Flip effects**: one per cause — plain, Same, Plus, Combo, Reverse, Fallen Ace — from the facts the engine gives; a Combo waits for the flip that set it off; reduced motion shows the settled position at once.
- **English explanations** (`6c73a60`): same keys and guarantees as Chinese (every number from the facts, no win claim without an exact win, deterministic snapshots); both languages are in every log record and live event; `?lang=` chooses at read time without rewriting a number. `ExplanationEnglishTests`, `LanguageTests`.
- **Hosting** (`e503794`): single-page application under a strict CSP (`script-src 'self'`, no inline, no eval); the API paths never answer with the page. `WebHostingTests`. **Publish** (`a996286`): `dotnet publish` builds the UI into `wwwroot` (`-p:SkipWebUi=true` for the API alone); `PublishTests` checks both.
- **Sample game** (`e4ce52a`): played by the engine on both sides through the real server; the demo replay and the plugin fixture of the browser tests are made from it, and `PluginGameFixtureTests` fails if they stop being what the engine produces.

**Bundle sizes** (`docs/reports/phase8b/bundle_sizes.txt`): the first page is 345.9 KiB raw / **104.9 KiB gzip** (HTML, main script, CSS). The 3D table (three.js) is one lazy chunk, 895.8 KiB raw / **235.3 KiB gzip**, loaded only when a table is shown in 3D. Mascot WebP: 916 KiB raw, loaded as needed (the first expression at 512 px is ~80 KiB).

**Frame rate** (`e2e/reports/perf.spec.ts`, `ui_frame_rate.json`; Microsoft Edge headless, 1180×760, **WebGL in software (SwiftShader)**, an 8 s replay with animations at 1×): 3D table 91.3 fps (frame time p50 12.4 ms, p95 16.8 ms, one 300 ms frame at the start; the earlier run gave 94.3), flat table 177.3 fps (p50 4.2 ms, p95 8.4 ms; earlier 182.9). Headless rendering is not paced to a display, so these are throughput numbers, not a measure of smoothness on a real monitor; no real GPU was measured.

## 7. Tests

| Suite | Result |
|---|---|
| C# (`dotnet test csharp -c Release`) | **373 passed**, 0 failed, 0 skipped (228 at the end of 8A) |
| Python (`python -m pytest -q`) | **152 passed** |
| Vitest (`npm test`) | 82 tests in 9 files: i18n completeness (keys, placeholders, no untranslated copy), settings (defaults, damaged storage, only four fields written), router, live reducer (order, repeats, reconnect, lag, resync, records, which game), timeline (v1/v2 logs, ambiguous cards, inconsistent steps, damaged logs), mood (no win claim without an exact win), flip plan, API client (key only in the login body; errors), source rules (no markup insertion, no `eval`, storage only for the preferences, no key in a URL, same origin only) |
| Playwright (`npm run e2e`) | 19 tests against a real server: landing and demo (anonymous); login/logout (cookie flags, no key in storage, URL or any request, one message for a wrong key); text as text (a key named `<i>xss</i>`); a plugin game posted to `/ai_move` appears live (depths, decision, explanation, end) and then in the log viewer with a timeline equal to the log; replay controls; download; delete with confirmation; no-WebGL fallback; 3D and the setting; narrow viewport; language switch of the UI and of the explanation (same numbers); reduced motion; mascot off; CSP header and no CSP violation on the pages visited. The screenshots and the frame rate are made by `npm run report` (a separate configuration; nothing is skipped in `npm run e2e`). |

Mutation checks were made on the important rules (each broken once, the suite fails): the reducer's repeat check, the access-class reserve and anonymous cap, the CSRF check, the revocation check, the error detail of a refused login (and, for the UI's reducer, the repeat check of `Last-Event-ID` events).

## 8. Final verification

- `dotnet build csharp -c Release --no-incremental`: 0 warnings, 0 errors. `dotnet test`: 373/373. `python -m pytest -q`: 152 passed (run without `PYTHONUTF8`: with it, `tests/test_engine_overload.py` fails to decode the localised `netstat` output — a property of the test's environment, not of the code). `npm test`: 82/82. `npm run e2e`: 19/19.
- **M8-2 under load** (`tools/phase8b/repeat_under_load.py`: 20 pure-Python burner processes, one per logical CPU, at normal priority; `SchedulerTests`, `LiveEventTests`, `AccessClassTests` = 45 tests; `docs/reports/phase8b/m8-2_scheduler_under_load.txt`): **20 of 20 consecutive runs passed**, 13.8–15.3 s each.
- **Core and Search unchanged**: `git diff 6d13de5 HEAD -- csharp/src/TTC.Siren.Core csharp/src/TTC.Siren.Search` is empty (and `6d13de5` is the head of Step A, i.e. the start of this phase's code changes). The `Bench` signature comparison against the start of the phase: `Bench search 1` + `Bench fast 1` — all 62 `SIG` lines (nodes, score bits, move, allocation 0) are **byte-identical** to the run at `6d13de5`; the 72 positions of `Bench endgame` give the same `same` verdict and scenario counts as at the start. The zero-allocation tests are part of the 373.
- Python business code: not touched. `git diff 6d13de5 HEAD --stat -- '*.py'` lists the new tools under `tools/` and one test, `tests/test_csharp_engine_pool.py`, whose timing dependence was removed under M8-2 (the assertions were not loosened).

## 9. Known limits

- The mascot dock can cover text on a small window; it can be hidden.
- Frame rates were measured on a software renderer only.
- No automated accessibility audit and no contrast measurement.
- Windows service, win-acme and Cloudflare dashboard steps are written, not executed; the Cloudflare ranges file is a snapshot (refresh with the tool).
- Access-class numbers are for one machine on loopback.
- Playwright uses the Edge that ships with Windows; other browsers were not run.
- The log viewer lists the games of the logged-in key only; the live view shows one game at a time.
- 8C, 8D and 8E are not started; no NPC data was exported; no review gate is set.
