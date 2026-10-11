# The web UI (Phase 8B)

React + TypeScript + three.js (through react-three-fiber), built with Vite, served by `TTC.Siren.Server` from the same origin. Source in `web/`. The solver runs only on the server; the browser draws what the server says and decides nothing about the game.

Decisions: A-27 (access classes, web login), A-29 (UI language, original card art, English explanations). Server side of the UI: `API.md` §5.3–§5.5, `docs/DEPLOY_WINDOWS.md` §2a, §4a.

## 1. Principles

1. **The UI shows what the server recorded. It applies no rule.** The board of a step comes from the request; our move and its flips come from the explanation facts (`immediate_effect`); the opponent's move and its flips come from the `opponent_move` record. Where the record cannot explain a step the UI says so (`inconsistent`) instead of working it out. (`src/domain/timeline.ts`, tested against the demo game and against damaged logs.)
2. **Text from the server is text.** It is rendered by React as text nodes; there is no `dangerouslySetInnerHTML` anywhere (a test greps for it) and a key whose name is `<i>xss</i>` is shown literally (Playwright).
3. **No secret in the browser.** The API key is typed into an uncontrolled input, sent once in the body of `POST /v1/session`, and the field is emptied at once. The session is an HttpOnly cookie the page cannot read. Only four preferences are stored (`lang`, `use3d`, `motion`, `mascot`), in `localStorage`, and a test asserts that exactly those keys exist.
4. **Everything has a fallback**: no WebGL, a narrow or touch screen, a failed 3D chunk, no storage, reduced motion, a server that is away (the stream reconnects).

## 2. Pages

| Route | Needs login | What it shows |
|---|---|---|
| `/` | no | What the product is, the demo replay of a real recorded game (`#demo`), links. The mascot dock is hidden here (the page has its own picture of her). |
| `/login` | no | The key field. One message for every failure; throttling says how many seconds to wait. Logged in: who, and a logout button. |
| `/live` | yes | The live solve. The table, the search by depth (score, best move, nodes, nodes/s, time), the robust-endgame progress, the decision, the assumed opposing cards, and the explanation of the move. Follows the newest game in progress, or the game picked. Connection status, "events dropped" and "reading again" notices. |
| `/games` | yes | The game logs of the key: id, start, requests, state, rules; open, download (JSON Lines), delete (with a confirmation in the page). Paged. |
| `/games/:id` | yes | The replay of one game: table, controls (first, previous, play/pause, next, last, speed 0.5–4×, scrubber), the step's detail (what was placed, each flip with its reason, the solver data), the explanation of our moves, the timeline of all steps. |
| `/settings` | no | Language, 3D table, animations (follow system / off / always), mascot. |
| anything else | no | "Page not found" (the server answers the page with 200 for an unknown UI path, so a reload or a pasted link works; paths of the API are never the page). |

## 3. State and events

**Live view** (`src/state/live.ts`, a pure reducer; `src/api/useLive.ts` wires it to `EventSource`).

- On opening: the newest games are read from the log (`games_loaded`), then `GET /v1/live?lang=…[&last_event_id=n]` is opened. The browser reconnects by itself and sends `Last-Event-ID`; the server replays what was missed.
- Events: `request.received`, `search.depth`, `robust.progress`, `decision`, `game.updated` (a record of the log), `game.ended`, `lagged` (this client was too slow: N events dropped), `resync` (the server cannot resume: read the games again; `stream_restarted` also resets the numbering, because a restarted server numbers from 1).
- **Order and repeats**: an event whose id is at or below the newest applied is ignored (a repeat after a resume); records are keyed by `seq`, depths by `depth`, so a repeated record or depth changes nothing. A late lower id is dropped, never applied over newer state.
- A game read from the log after it finished has no events; its depths and decision are read from the request's own record (`withRecord`), without overwriting what events brought.
- **Which game**: with "follow" on, the newest game that has not ended (else the newest); picking a game turns follow off.

**Replay** (`src/domain/timeline.ts`). Steps: *position* (a request), *our move* (from the facts), *opponent move* (from the record), *end*. The flips of a step play in the order of the facts; a Combo waits for the flip that set it off (`flipOrder`). Old logs (schema v1) and logs with missing records are read without error. An opponent card known only by its four edges is shown with its candidate ids (`ambiguous`), never as a single card.

**Mood of the mascot** (`src/state/mood.ts`, `docs/design/MASCOT.md` §4): thinking while searching, confident or worried from the outcome, surprised after an opponent's Combo, happy/neutral/disappointed at the end. A caption states only what the facts say: "solved" only for an exact result, a win only for an exact win; a heuristic result never says "win".

## 4. The table: 3D, flat, and the flip effects

- `BoardView` chooses `3d` only when `use3d` is on **and** WebGL exists **and** the screen is wider than 720 px **and** the pointer is not coarse; otherwise `2d` (`data-table-mode` on the element). The 3D scene (`Board3D`, three.js, ~236 KiB gzip) is a **lazy chunk**: the first page does not load it. A scene that throws falls back to the flat table (error boundary) with the same information.
- Under both tables there is a **text list** of the nine cells (`StateList`), so the position is available without seeing the table (screen readers, copy/paste).
- **Card faces are original and procedural**: drawn on a canvas from the card's four edge values, star count and type (`cardArt.ts`, `CardCanvas`); no game art. The card database is read at build time only for edges, star and type (`scripts/build-cards.mjs`; names are not used).
- **One effect per cause of a flip** (`src/table/flipEffects.ts`); the same plan drives the 3D cards and the CSS classes of the flat table:

| Cause | Effect |
|---|---|
| plain comparison (`basic`) | one turn about the horizontal axis, short lift |
| `same` | three half-turns about the x axis, a teal ring spreading from the card |
| `plus` | three half-turns, high lift, the sum rises as a floating number |
| `combo` | three half-turns about a diagonal axis, an arc from the card that set it off; starts only after that flip ended |
| `reverse` | one turn about the other diagonal, a violet ring and an arrow |
| `fallen_ace` | highest lift, a red ring, and the table shakes when the card lands |
| cause unknown | shown as a plain flip |

- A move plays as: the card lands (420 ms), then the flips follow each other 150 ms apart. With "reduce motion" (system or setting) the settled position is shown at once and a replay steps without waiting.

## 5. Internationalisation

Two languages, `zh-CN` and `en`, chosen by the browser at first and then by the toggle (stored). `src/i18n/zh-CN.ts` is the master; `en.ts` is typed `Record<MessageKey, string>`, so a missing key fails the build. Vitest additionally asserts: identical key sets, no empty text, identical `{placeholders}` per key, no text copied untranslated.

The explanation of a move comes from the server in **both** languages (`explanation.texts`; the live `decision` event carries `briefs`). The page asks with `?lang=` and the server puts the chosen text in place; **no number is rewritten**. Changing the language while a replay is open keeps the replay at the same step. English templates carry the same guarantees as the Chinese ones (every number from the facts, no win claim without an exact win, deterministic snapshots; `ExplanationEnglishTests`).

## 6. Security

- **Content-Security-Policy** (set by the server on the page): `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'`. No `unsafe-inline`, no `unsafe-eval`. The build inlines nothing (`assetsInlineLimit: 0`), the page has no inline script (a publish test checks), and Playwright listens for `securitypolicyviolation` on every page it visits. The 3D scene draws on a canvas and loads its textures from canvases, so it needs nothing beyond `'self'`.
- **Cookie**: `ttc_session` (`__Host-ttc_session` over HTTPS): HttpOnly, SameSite=Strict, Path=/, Secure over TLS. A request authenticated by the cookie that changes something must also pass the same-origin check (Origin / Referer / Sec-Fetch-Site).
- **Same origin only**: `fetch` is called with relative URLs and `credentials: 'same-origin'`; the client adds no `Authorization` header and no key anywhere except the login body.
- **Logging**: the server never logs a key or a session id; the UI logs nothing.
- Static assets carry a content hash and are cached for a year (`immutable`); `index.html` is revalidated, so a new release is picked up.

## 7. Accessibility (what is done, and what is not measured)

Done: a "skip to the content" link; landmarks (`header`, `nav` with a label, `main`, `footer`, the mascot as a labelled `aside`); the current page is `aria-current`; every control has a text or an `aria-label`; the delete confirmation is an `alertdialog` with the focus on the primary button; connection status and the mascot caption are `role=status`; errors are `role=alert`; the table has a text equivalent; the language of the page is set on `<html lang>`; animations obey `prefers-reduced-motion`; the layout was checked at 375 px with no horizontal scroll.
Not done / not measured: no automated audit (axe or similar) was run, colour contrast was not measured, and keyboard operation of the 3D canvas is not provided (the replay controls and the text list are).

## 8. Build, test, run

```
cd web
npm ci                 # exact versions from package-lock.json
npm run build          # data (cards.json, padded WebP mascot) + typecheck (src and e2e) + vite build -> web/dist
npm test               # Vitest
npm run e2e            # Playwright against a real server (needs the server built in Release and web/dist)
npm run report         # the screenshots and the frame rate for docs/reports/phase8b (same setup; asserts nothing)
```

`dotnet publish csharp\src\TTC.Siren.Server -c Release -o <dir>` runs `npm run build` and ships `web/dist` as `<dir>\wwwroot` (`npm ci` first only if `web/node_modules` is missing); `-p:SkipWebUi=true` publishes the API alone. A plain `dotnet build` and `dotnet test` never run npm. The server finds `wwwroot` next to the program, or `Server:StaticDirectory`.

**Playwright** starts the real server (`e2e/global-setup.ts`): a temporary key store (keys created with the server's own `keys add`), `TestSeedHeader` on loopback, a temporary game-log folder, the built UI. Tests post the sample game to `/ai_move` with the plugin's header, as the plugin does. The browser is the Microsoft Edge that Windows ships (`channel: 'msedge'`; set `E2E_CHANNEL` for another), so no browser is downloaded; WebGL is software (SwiftShader). Login is throttled per address as in production, so one logged-in state is made once in the setup and the login page itself is tested separately.

**The sample game** (`web/src/demo/demo-game.json`, the landing page's demo; `web/e2e/fixtures/plugin-game.json`, the plugin's requests) is played by the engine on both sides through the real server: `PluginGameFixtureTests` replays the seed and fails if the committed files are not what the engine produces now (`TTC_UPDATE_FIXTURES=1` rewrites them and searches for a seed that shows every cause of a flip except Reverse, which cannot appear together with plain comparison under the Reverse rule).

## 9. Dependencies and why

All versions are exact; `package-lock.json` is committed. Nothing is loaded from a CDN at run time.

| Package | Kind | Why |
|---|---|---|
| `react`, `react-dom` | runtime | The component model for a UI with this much state (live events, replay, settings, session). |
| `three` | runtime (lazy chunk) | The 3D table and the card flips. |
| `@react-three/fiber` | runtime (lazy chunk) | Renders the three.js scene from React components, so the 3D table follows the same state as the flat one. |
| `vite`, `@vitejs/plugin-react` | build | Bundling, hashed file names, the dev server. |
| `typescript` | build | Types for the event and log shapes; the build fails on a type error. |
| `@types/react`, `@types/react-dom`, `@types/three`, `@types/node` | build (types only) | Types of the above; `@types/node` types the Playwright setup files. |
| `sharp` | build | Turns the mascot's PNG originals into padded WebP files (originals are never changed; padding, never cropping). Not shipped. |
| `vitest`, `jsdom` | test | Unit tests of the reducer, the log reading, i18n, settings, the client. |
| `@playwright/test` | test | The browser tests against the real server. |

Not used on purpose: a router library (the history API is ~40 lines, `components/router.tsx`), a state library (a reducer is enough), a CSS framework, a UI kit, an icon font.

## 10. Known limits

- The mascot dock is fixed at the bottom right and can cover text on a small window; it can be hidden (× on the dock, or in settings).
- The 3D table needs a reasonable GPU for the effects; a software renderer (as in the tests) is slower. Frame rates measured there are in `docs/reports/PHASE8B_REPORT.md`.
- The live view shows one game at a time.
- The log viewer lists the games of the logged-in key only.
- No automated accessibility audit; no contrast measurement.
