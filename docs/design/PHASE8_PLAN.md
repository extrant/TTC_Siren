# Phase 8 plan — from bridge to product (8A → 8B → 8C → 8D)

> **SUPERSEDED (2026-10-12, decision A-30).** The user stopped the product direction: the product server, the web UI, API keys, game logs and the live stream (8A, 8B) were removed, and 8C/8D/8E will not be done. The product is again the original Python server, with the C# engine solving the minimax of `/ai_move` (started by the server itself). This plan is kept as a record only; the removed code is in the git history (up to `f5cf200`).

Status: design agreed with the user on 2026-10-11 (`CSHARP_MIGRATION.md` §9 Q-9, §8 A-21 … A-24). Each sub-phase is authorized separately with its own goal file and gets its own independent review. **Only 8A is authorized now** (`.claude/phase8a_goal.md`).

## 0. Decisions (user, 2026-10-11)

| Topic | Decision |
|---|---|
| Source protection | Repository made **private** (done 2026-10-11). The product is not open source. |
| Delivery | **Cloud service + web front end.** The solver runs only on the user's server. Browsers open the web UI; the game plugin calls the cloud API. Nothing algorithmic is shipped to end users. |
| Authentication | **One API key per user**, issued and revoked by the operator, sent by the plugin and the web UI. Per-key rate limits. |
| Monte Carlo solver (`solver=monte_carlo`) | **Retired, not ported.** It is weaker than minimax + exact/robust endgame, and it was adapted from a third-party algorithm (LinkRoss). Requests that ask for it are answered by the engine, with a note in the response. |
| Request parsing and unknown-card sampling | **Ported faithfully first** (same logic, verified against Python). Improving the sampling comes later, as its own phase, with the port as the measurable baseline. |
| Analysis text | **Engine facts + templates.** The engine computes structured, verifiable "explanation facts". Chinese templates turn them into text at several levels of detail. No language model (none is needed, and it cannot invent anything). |
| Game logs | **Stored on the server per user**, replayable in the web UI, downloadable and deletable; retention configurable (default 30 days). |
| Web technology | **React + TypeScript + three.js (react-three-fiber), built with Vite**, served by the C# server. |
| Deck optimisation | **Both modes**: against a chosen NPC/rule set, and general strength for a rule set. |
| Game coaching | **The user places both sides' cards** (replaying a real game), with an optional switch for "engine plays the opponent". |
| Extra features accepted | Live game follow, automatic post-game report, endgame puzzles, mascot line library (text only). |
| Mascot | Original character **"Seren" (瑟琳), a calm, elegant siren croupier** (`docs/design/MASCOT.md`). |

## 1. Target architecture

```
browser (web UI) ─┐                                 ┌─ TTC.Siren.Server (C#, ASP.NET Core, one process)
game plugin ──────┼── HTTPS + API key ─────────────▶│   /ai_move (compatible), /v1/* (new API), /v1/live (SSE)
                  │                                 │   in-process engine: worker pool, Fast ID, exact/robust endgame
                  │                                 │   request parsing + unknown-card sampling (ported)
                  │                                 │   explanation facts + templates, game logs, auth, rate limits
                  │                                 │   static web UI (8B)
                  │                                 └─▶ loopback proxy to the Python app for /api/recommend,
                  │                                     /api/draft/analyze, /api/winrate (until 8C ports them)
```

- After 8A, Python is no longer on the `/ai_move` path. It keeps serving the three `/api/*` endpoints behind the C# server until 8C. It also stays in the repository as the parity reference and golden generator (CLAUDE.md §1.2).
- Windows Server (E5-2698B v3, 16 cores / 32 threads, 16 GB) with the .NET 10 ASP.NET Core runtime. HTTPS terminates in Kestrel (PFX certificate) or in a reverse proxy in front of it.

## 2. Sub-phases

### 8A — backend becomes the product server (authorized; `.claude/phase8a_goal.md`)

- Port of request parsing and unknown-card sampling, verified against Python. A CPython-compatible random generator makes the sampling reproducible bit for bit (A-23).
- `/ai_move` served natively in C#, response **compatible** with today's plugin. Old fields are ported faithfully; new fields are additive.
- Retire Monte Carlo.
- Priority scheduling for cheap requests (review O8-1).
- **Explanation facts v2** and templated text (`explanation` field, brief and full levels).
- **Game logs** with timeline, replay data and opponent moves inferred between requests.
- **Live event stream** (SSE) per API key: progress per completed search depth, decisions, game updates. This is the data source of "real-time solve status" and "live game follow".
- API keys, rate limits, HTTPS configuration, CORS, admin CLI; reverse proxy to Python for `/api/*`; deployment guide for the cloud server.

### 8B — web UI (to be authorized later)

- **Real-time solve status** as its own view:
  - board state;
  - depth-by-depth progress with score and best move per depth;
  - principal variation drawn on the board;
  - nodes, NPS and elapsed time;
  - robust-endgame scenarios and safety ratio;
  - which unknown cards were assumed.
- **Live game follow:** the view attaches to the user's current game (same API key) and updates on every request from the plugin.
- **3D table:**
  - deep-sea casino table, 3×3 glass board, cards with base/display values and owner colour;
  - **3D card flip** animations with a distinct effect per cause (basic, 同数, 加算, 连携, 逆转, 王牌杀手);
  - floating modifier badges for 同类强化/弱化.
- **Game log viewer:** timeline scrubber with every move (time, card, cell, owner, flips and causes, modifiers, solver data), replay controls and export.
- **Mascot dock:** Seren reacts to events, with expression switching and a speech bubble from the line library.
- Accessibility basics: keyboard control, reduced motion, a 2D fallback when WebGL is unavailable.

### 8C — deck optimisation (to be authorized later)

- **Inputs:**
  - the user's deck (5 cards) and optionally owned cards;
  - the rules;
  - either a target NPC (known pool) or "general" for the rule set;
  - the **change level**: at most 1, 2, 3, 4 or 5 swaps.
- **Output per level:**
  - the best deck reachable with at most that many swaps, within the star limits (★4+ ≤ 2, ★5 ≤ 1, draft quotas);
  - the expected win/draw/loss rates against the target, estimated with engine-played games;
  - per swap, **why** (which matchups or rule interactions it fixes, with numbers).
- **Ports** the existing `/api/recommend`, `/api/draft/analyze` and `/api/winrate` (faithfully, verified), then removes Python from the server entirely.

### 8D — coaching, post-game report, puzzles (to be authorized later)

- **Game coaching:**
  - the user drags cards for both sides, or switches on "engine plays the opponent";
  - every move is rated against the engine (best / good / inaccuracy / mistake / blunder by score loss, exact when the endgame is solved);
  - each rating comes with the better move and the explanation facts of both moves.
- **Automatic post-game report** from the game log: turning points (largest evaluation swings), the opponent's good moves and mistakes, and what the unknown cards turned out to be.
- **Endgame puzzles** taken from real games, where exactly one move wins. The exact solver guarantees uniqueness.
- **Mascot line library:** text lines by event and expression, in Seren's voice.

## 3. Game log (summary; full schema written in 8A as `docs/design/GAME_LOG_SCHEMA.md`)

- One file per game (JSON Lines), per API key, with a schema version.
- **Records:**
  - **game start:** rules, open mode, owners, NPC, known decks;
  - **request:** wall-clock UTC time, sequence number, full board (card ids, base/display values, owner, modifiers), both hands (known, unknown, assumed), the sampling RNG seed, solver settings and per-depth statistics, the chosen move, the explanation facts, timings;
  - **inferred opponent move:** between two requests, the card, cell, owner, flips and causes;
  - **game end:** final counts and winner.
- **Identity of a game:** an optional `gameId` in the request (future plugin versions). Otherwise the server infers it: same key, consistent successor state, time gap below a limit.
- Every record carries enough data to replay it deterministically: state, seed and settings.

## 4. Explanation facts v2 (summary; specified in the 8A goal)

Each fact is computed by the engine, is checkable by recomputation, and is rendered by templates at two levels (brief: one or two sentences; full: paragraphs). The fact families are:

- immediate effect;
- modifier changes;
- exposure and counter-threats;
- the principal variation and its outcome (exact or heuristic);
- alternatives and why they are worse;
- the robust endgame (scenarios, safety, override and why);
- unknown-card assumptions;
- rule-specific notes;
- confidence (solved to the end or depth reached).

## 5. Open items for later phases

- Improving unknown-card sampling, after 8A gives the measurable baseline.
- Plugin changes: the plugin is the user's separate project. It needs a server URL, an API key and, ideally, a `gameId`.
- Q-3 (only one combo level): decide before 8C/8D explanations call it "intended".

## 6. Decisions of 2026-10-11 (after the 8A review; `CSHARP_MIGRATION.md` §9 Q-10, §8 A-27 … A-29)

### 6.1 Access model and capacity (A-27)

The user's position: the web site is for the general public, so the concern is **abuse, queueing and throttling**. Strict authentication matters where the real compute goes: `/ai_move` called by the game plugin during real games. Those requests must get the **highest priority**.

| Who | How they are identified | What they may do | Priority / capacity |
|---|---|---|---|
| Game plugin | API key (8A mechanism; **no user accounts**) | `/ai_move`, own game logs, live stream | **highest**; a configurable share of the workers is reserved for it |
| Web user with a key | logs in **once with the key**; the server answers with an HttpOnly, Secure, SameSite=Strict session cookie; the key is never stored in the browser; logout, expiry, revocation with the key | the real-time solve view, live game follow, own game logs | high (below the plugin) |
| Anonymous visitor | none; limited by real client IP and browser session | **read-only** pages (landing, demo replay, later the NPC library); **game coaching** (8D: places cards, gets advice) within anonymous limits | lowest; at most a configurable share of workers (for example 25 %), capped depth and time, per-IP rate limits, a global anonymous queue cap ("try again later" when full) |

**Cloudflare in front** (the user's choice for DDoS and bot protection):
- The real client IP comes from `CF-Connecting-IP`. This header is trusted **only** from configured proxies (Cloudflare's published ranges, or loopback when `cloudflared` runs locally).
- SSE needs heartbeats well below Cloudflare's idle timeout.
- `/ai_move` and `/v1/*` must not be cached.
- Responses must finish within Cloudflare's origin timeout (100 s on the free plan), so `max_time` + queue wait + margin stays below it.
- The operator endpoints must refuse any request that carries Cloudflare or proxy headers. With a local `cloudflared`, requests arrive from loopback.

### 6.2 8B scope (authorized; `.claude/phase8b_goal.md`)

8B covers:
- the 8A leftovers (M8-1, M8-2, L8-1 … L8-4);
- Cloudflare readiness;
- the access tiers and capacity classes (backend);
- web login with a key;
- the web UI:
  - landing page;
  - real-time solve view with live game follow;
  - 3D table with 3D flips per cause (2D fallback);
  - game-log viewer with timeline and replay;
  - mascot dock;
  - **Chinese and English** (UI chrome, plus English explanation templates);
  - deep-sea dark theme, desktop first, simplified 2D on phones.

**Not in 8B:** coaching, post-game report and puzzles (8D); deck optimisation (8C); the NPC library (8E).

### 6.3 8E — NPC "full clear" library (A-28; planned, not authorized)

**Game data findings** (EXDViewer on the user's machine, game data version 2026.09.15).

Sheets:
- `TripleTriad`: per NPC, `TripleTriadCardFixed[5]`, `TripleTriadCardVariable[5]`, `TripleTriadRule[2]`, `UsesRegionalRules`, fee, rewards.
- `TripleTriadRule`: 15 rules.
- `TripleTriadResident`: card order.
- NPC names and places come through `ENpcBase` and `Level`, which link to `TripleTriad`.

Counts:
- **144 NPCs** with cards.
- Fixed cards per NPC: 1–5 (mostly 3). Variable cards: 0–5 (mostly 4–5). A typical NPC therefore has about 10 possible hands.
- Open rules: **9 NPCs** have 全明牌, **21** have 三明牌, **114 hide their hand**.
- NPC rules: 加算 46, 同数 31, 交换 24, 混乱 22, 三明牌 21, 天选 20, 王牌杀手 14, 同类强化 10, 全明牌 9, 同类弱化 7, 秩序 6, 逆转 6, 不胜不休 3, 随机 2.
- **Regional rules ("流行规则")**:
  - The game's guide text says they change **daily per region**, with at most two, and each region has its own tendencies.
  - They are server-side, **not in the client data**.
  - **46 NPCs** use them.

**Decisions:**
- **Data:** exported from the game sheets into `data/` (and re-exported after game updates). The library is not built from the plugin's `npcIds`.
- **Regional rules:** the plugin reports the day's regional rules. The server computes "NPC + today's rules" on demand and caches it. The other NPCs are precomputed offline.
- **Rules the engine does not support:** 交换, 天选, 不胜不休, 随机 (43 NPCs). These NPCs are marked **unsupported** for now; there is no new rule work.
- **What "full clear" means:**
  - The engine's own deterministic policy (which does not look at the NPC's hidden hand) must **win against every possible NPC hand and every NPC reply, for both first movers**. This is proven by exhaustive enumeration over the NPC's choices with our policy fixed.
  - With 全明牌 the exact game value is used directly.
  - When no deck achieves a full clear, report the guaranteed-win share, the not-lose share and the lowest-risk deck.
- **Scope of precomputation:**
  - offline, the best / full-clear decks from the whole card database, plus the first move for every opening;
  - a user's own deck is computed on demand and cached in a database;
  - later moves of a game are answered live by the engine (milliseconds), so no full move tree is stored.

### 6.4 Updated order

8B (UI + leftovers + access tiers) → 8C (deck optimisation, port of `/api/*`) → 8D (coaching, also for anonymous visitors within limits; post-game report; puzzles; mascot lines) → 8E (NPC full-clear library). Each step is authorized separately.
