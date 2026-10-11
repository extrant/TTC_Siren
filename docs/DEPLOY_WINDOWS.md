# Deploying the TTC Siren server on Windows Server (Phase 8A, cloud topology)

```
game plugin ─┐                          ┌─ TTC.Siren.Server  (C#, ASP.NET Core, one process, the engine runs inside it)
browser UI  ─┼── HTTPS :443 + API key ──┤     /ai_move  /v1/games  /v1/live  /health  /   (placeholder page or the web UI)
             │                          └─ loopback ──> Python app (waitress, 127.0.0.1:5001)   only /api/recommend, /api/draft/analyze, /api/winrate
```

Since Phase 8A the Python server is **no longer on the `/ai_move` path**. The C# server parses the request, samples the unknown cards (a port of the Python
logic, exact for the same seed), searches (in process: worker pool, Fast iterative deepening, exact and information-set endgame), writes the explanation and the game log,
and answers. The Python app stays only for three endpoints until Phase 8C ports them, and as the reference implementation of the tests. The separate
`TTC.Siren.Engine` host (`/v1/solve`, loopback) is still built for the test tools; the server does not use it.

Rollback at any time: point the plugin back at the old Python server (`python app.py` / `serve_waitress.py` with `TTC_CS_ENGINE=on` and the engine of Phase 7). Nothing in the
Python code changed.

## 1. What you need

| Item | Notes |
|---|---|
| Windows Server 2016 or newer, x64 | Tested on Windows 11 here. |
| **.NET 10 ASP.NET Core Runtime** (or the Hosting Bundle) | The server is published framework-dependent. `dotnet --list-runtimes` must show `Microsoft.AspNetCore.App 10.x`. |
| Python 3 with Flask (and `waitress`) | Only for the three `/api/*` endpoints: `pip install -r requirements.txt waitress`. |
| The card database | `data\幻卡数据库.csv`. The server and the Python app must load the **same file**. |
| A TLS certificate | A PFX file (your own, or issued with win-acme / Let's Encrypt, see section 5). |
| A DNS name pointing at the server | The certificate must be valid for it. |

## 2. Publish

```
dotnet publish csharp\src\TTC.Siren.Server -c Release -o publish\server
```

The command works as written, without further flags. The output holds one configuration file, the server's own `appsettings.json` (section `Server`, loopback port 5000); the engine library
is part of the server and its host files are not shipped. (Before Phase 8B the command failed with NETSDK1152 because the engine project also had an `appsettings.json`; the engine's own file is now
`engine.appsettings.json`, and `PublishTests` publishes and checks the result.)

Copy `publish\server` to the server, for example `C:\ttc\server`, and `data\幻卡数据库.csv` to `C:\ttc\data\`. Never copy golden data, corpora, game logs or key files from a development machine.

## 2a. The web UI

The publish builds the web UI (`web/`, React + TypeScript + three.js, Vite) and ships it as `publish\server\wwwroot`; the server finds it there by itself (or set `Server:StaticDirectory`). Nothing else is needed at run time: no Node on the server.

* **On the build machine:** Node 22 and npm. `npm ci` runs from `package-lock.json` the first time (only when `web
ode_modules` is missing), then `npm run build` (data, type check, bundle). `data\幻卡数据库.csv` and `assets\mascotaw\*.png` must be in the checkout (they are inputs of the build: card edges, and the mascot pictures that are padded and converted to WebP).
* **API only** (no Node on the build machine, or the UI is hosted elsewhere): `dotnet publish csharp\src\TTC.Siren.Server -c Release -o publish\server -p:SkipWebUi=true`. A plain `dotnet build` and the tests never run npm.
* The page is served with a strict Content-Security-Policy (`script-src 'self'`, no inline script, no `eval`); do **not** add a proxy rule that rewrites or removes it. Behind Cloudflare, cache `/assets/*` (hashed names, `immutable`) and do not cache `/` (the page revalidates, so a new release shows at once).
* The UI uses the session cookie of the web login (section 4a) and the same-origin API (`/v1/...`); no CORS setting is needed for it. Server-Sent Events (`/v1/live`) must not be buffered by a proxy (section 5a).
* First look after a deploy: open `https://<name>/`, "See the demo replay" must play; then log in with a key, `/live` must say "Connected".

## 3. Configuration

Settings come from `appsettings.json` (section `Server`), environment variables (`TTC_Server__Security__Https__Port=443`), and the command line (`--port`, `--workers`, ... ; the command line wins).

| Key (`Server:`) | Default | Meaning |
|---|---|---|
| `Host`, `Port` | `127.0.0.1`, `5000` | The plain HTTP listener. Behind a TLS proxy keep loopback. For a direct internet-facing server use `0.0.0.0` with `HttpEnabled=false` and HTTPS on. |
| `HttpEnabled` | `true` | `false` serves HTTPS only. |
| `Workers` | logical CPUs | Searches that run at the same time (section 9). |
| `QueueLimit`, `QueueWaitMs` | `256`, `60000` | Requests that may wait for a worker, and for how long. More, or too long, are answered 503 `busy`. |
| `PriorityScheduling`, `AgingMs` | `true`, `2000` | The wait queue is ordered by estimated cost: endgame requests overtake openings; a waiting request moves up one class every `AgingMs`, so an opening is overtaken for at most `2 * AgingMs`. |
| `TtEntries`, `MemoEntries` | `4194304`, `32768` | Per worker table sizes (96 MiB + 0.75 MiB). |
| `CardDbPath` | found above the application | Absolute path of the CSV. Set it explicitly. |
| `LiveBuffer` | `256` | Events a `/v1/live` client may fall behind before the oldest are dropped. |
| `Live:HeartbeatSeconds` | `20` | A comment line on every idle event stream this often. Must stay well below Cloudflare's 100 s idle timeout (the server accepts 0.05 to 60). |
| `Live:ReplayEvents`, `ReplayBytes`, `ReplayMinutes` | `256`, `2097152`, `15` | The most recent events of each key are kept so that a client that reconnects with `Last-Event-ID` gets what it missed (at most this many events and bytes; forgotten after this many idle minutes). `ReplayEvents=0` switches resuming off. |
| `MaxSearchSeconds` | `20` | The largest `max_time` a request may use (a larger value is lowered and the answer says so in `engine_info.limits`). The plugin's own default is 10. |
| `OriginTimeoutSeconds`, `OriginMarginSeconds` | `95`, `10` | How long the proxy in front waits (Cloudflare: 100 s on the free plan) and the time a request needs besides queueing and searching. At start the server checks that `MaxSearchSeconds + QueueWaitMs + OriginMarginSeconds` stays **below** `OriginTimeoutSeconds` and refuses to start otherwise, with the numbers. `OriginTimeoutSeconds=0`: no proxy with a time limit. |
| `WarmUp` | `true` | Before the first connection is accepted, run an opening and an endgame on every worker (JIT, first touch of each worker's tables). It takes a few seconds and creates all worker contexts at once (the memory of section 9 is used from the start). |
| `Access:PluginReservedShare` | `0.25` | The share of the workers that only the plugin (an API key) may use; web and anonymous work together never take them (rounded down; nothing is reserved with a single worker). A plugin request waits only for a search of its own class that is already running. |
| `Access:AnonymousMaxShare` | `0.25` | The most workers visitors who are not logged in may hold at once (rounded down, at least one). |
| `Access:AnonymousQueueLimit`, `AnonymousQueueWaitSeconds` | `16`, `10` | Anonymous requests waiting for a worker: more are answered `503 busy` with `Retry-After` at once; one that waits longer than this is answered busy. |
| `Access:AnonymousMaxDepth`, `AnonymousMaxSeconds` | `8`, `3` | The search an anonymous request gets at most (its `max_depth` / `max_time` are lowered). |
| `Access:AnonymousRatePerMinute`, `AnonymousBurst`, `AnonymousMaxInFlight` | `30`, `10`, `2` | Per REAL client address (after the trusted-proxy rule) for anonymous work. |
| `Security:Sessions:LifetimeHours`, `MaxPerKey` | `12`, `20` | How long a web session lasts, and how many one key may have (the oldest is ended). Sessions live in memory: a restart logs everybody out. |
| `Security:Sessions:LoginAttemptsPerMinute`, `LoginBurst` | `6`, `5` | Login attempts per real address: five at once, then six a minute (a failed and a successful attempt count the same). Over the limit: `429` with `Retry-After`. |
| `Security:Sessions:AuditLogFile` | `audit.jsonl` | One JSON line per login, logout and throttled attempt: time, real address, outcome, key id on success. Never a key, never a session id. Relative to the application folder; empty switches the file off. Back it up with the logs; it grows by a line per attempt. |
| `Security:TrustedProxies:UseCloudflareRanges` | `false` | Believe `CF-Connecting-IP` from the peers listed in `cloudflare-ips.txt` (next to the program; section 5a). |
| `Security:TrustedProxies:TrustLoopback` | `false` | Believe the header from this machine: a `cloudflared` tunnel or a reverse proxy that runs here. |
| `Security:TrustedProxies:Cidrs`, `CidrFile`, `ClientIpHeader` | none, `cloudflare-ips.txt`, `CF-Connecting-IP` | More trusted proxies (CIDRs or addresses), the range file, and the header that carries the client address. |
| `StaticDirectory` | none | A folder of static files served at `/` (the web UI of Phase 8B). Without it `/` shows the placeholder page. |
| `TemplatesDirectory` | none | A folder with another `explain.zh-CN.json` (wording of the explanations). |
| `GameLog:Directory` | `<app>\gamelogs` | Where game logs are written (one folder per key). Set it outside the application folder. |
| `GameLog:RetentionDays` | `30` | Logs not written for this long are deleted (at start and daily). |
| `GameLog:GapMinutes` | `15` | Largest gap between two requests of one game for game identity by inference. |
| `Security:RequireKey` | `true` | Every API call needs a key. `false` is for development only. |
| `Security:KeyStoreFile` | `<app>\keys.json` | The key hashes. Back it up. |
| `Security:DefaultRatePerMinute`, `DefaultBurst`, `DefaultMaxInFlight` | `120`, `20`, `4` | Limits of a key that has no limits of its own. |
| `Security:MaxLiveStreamsPerKey` | `4` | Open event streams per key. |
| `Security:CorsOrigins` | none | Exact origins that may call from a browser (the web UI), e.g. `["https://app.example.org"]`. No wildcard. |
| `Security:Https:Enabled`, `Port`, `PfxPath` | off, `443` | HTTPS in Kestrel. The PFX password comes from the environment variable `TTC_PFX_PASSWORD` (or `Security:Https:PfxPassword`; prefer the variable). |
| `Security:Https:Hsts` | `true` | `Strict-Transport-Security` on HTTPS answers. |
| `TestSeedHeader` | `false` | **Test switch, never in production.** The server refuses to start with it unless it listens on a loopback address without a proxy in front, and logs a warning when it is on. |
| `Security:BehindTlsProxy` | `false` | Set to `true` only when a reverse proxy terminates TLS in front of a non-loopback listener. Without HTTPS or this flag the server **refuses to start** on a non-loopback address (keys would travel in clear text). |
| `Security:AllowLoopbackWithoutKey` | `false` | Development only: callers on 127.0.0.1 need no key (they act as the key `local-dev`). Needs a loopback `Host`. |
| `Proxy:PythonUrl`, `TimeoutSeconds` | `http://127.0.0.1:5001`, `120` | The Python app behind `/api/*`. A Python app that is down answers 502, a slow one 504. |

Example `appsettings.json` for the internet-facing server:

```json
{ "Server": {
    "Host": "0.0.0.0", "HttpEnabled": false, "Workers": 16, "CardDbPath": "C:\\ttc\\data\\幻卡数据库.csv",
    "GameLog": { "Directory": "D:\\ttc-data\\gamelogs", "RetentionDays": 30 },
    "Security": { "KeyStoreFile": "D:\\ttc-data\\keys.json", "CorsOrigins": [ "https://app.example.org" ],
                  "Https": { "Enabled": true, "Port": 443, "PfxPath": "D:\\ttc-data\\server.pfx" } },
    "Proxy": { "PythonUrl": "http://127.0.0.1:5001" } } }
```

Run it once by hand: `C:\ttc\server\TTC.Siren.Server.exe`; it prints the listening addresses, the workers and the SHA-256 of the card database.

**Garbage collector.** The server runs with Server GC, non-concurrent (as the engine did; chosen by measurement in Phase 7 hardening). `DOTNET_gcServer=0` switches to workstation GC without a rebuild if memory matters more than throughput.

## 4. API keys

Keys are random 256-bit values; the server stores only their SHA-256. Manage them with the server binary (the running server notices a change within a second):

```
TTC.Siren.Server.exe keys add --name alice --rate 120 --burst 20 --inflight 4     prints the key ONCE
TTC.Siren.Server.exe keys list                                                      id, name, state, limits, hash prefix
TTC.Siren.Server.exe keys set-limit alice --rate 60 --inflight 2
TTC.Siren.Server.exe keys revoke alice                                              the key is answered 403 from now on
```

Hand the key to the user over a safe channel; it cannot be recovered (revoke and issue a new one). The plugin sends it as `Authorization: Bearer <key>` or `X-Api-Key: <key>`.
A key never appears in the server's logs (an unknown key is logged by its first characters only; a revoked key by its public id `k-xxxxxxxx`). The game logs of a key are stored under its public id.
Over the limit: HTTP 429 with `Retry-After`. `POST /admin/reload-keys` and `GET /admin/status` are served to callers on the same machine only (and never through a proxy: a request with `X-Forwarded-For` is refused).

## 5. HTTPS

**With a certificate you have (PFX):** set `Security:Https` as above and the password in the machine environment variable `TTC_PFX_PASSWORD`. Kestrel serves TLS itself; HSTS is sent.

**With Let's Encrypt (win-acme):** run `wacs.exe`, create a certificate for the DNS name with the "Manual input / export PFX" target, write the PFX to the folder named in `PfxPath`, and
let win-acme's scheduled task renew it. The server reads the certificate at start: restart the service after a renewal (win-acme can run `sc.exe stop TTCSirenServer & sc.exe start TTCSirenServer` as an installation script).

**Behind IIS / nginx / Caddy:** let the proxy terminate TLS and forward to `http://127.0.0.1:5000`; keep `Host` loopback, or set `Security:BehindTlsProxy=true` for another interface. Forward the `Authorization` header, allow response streaming for `/v1/live`
(disable response buffering, long read timeout), and do **not** publish `/admin/*`.

## 4a. Who may do what: access classes, the web login

Three classes of caller share the workers (decision A-27; details in API.md section 5.5):

* **plugin** -- an API key in `Authorization` / `X-Api-Key`. Highest priority; a share of the workers (`Access:PluginReservedShare`, 25 % by default) is reserved for it, so web and anonymous work can never delay a game. With 16 workers: 4 reserved.
* **web** -- a browser that logged in with the same key: `POST /v1/session` exchanges the key for an HttpOnly, SameSite=Strict (Secure behind TLS) cookie. The key is not stored in the browser, not logged and not put in the cookie; the server keeps only a hash of the session id. Logout, expiry (12 h), revocation of the key and a restart end the session. The cookie is accepted on the game logs, the stream and `/v1/me` only, never on `/ai_move`.
* **anonymous** -- nobody. Read-only pages today; game coaching (8D) later. At most 25 % of the workers (4 of 16) at a time, its own queue of 16, a smaller search (depth 8, 3 s), a rate limit per real client address.

Give out one key per user (`keys add`); the web login needs no further setup. Revoking the key (`keys revoke`) ends its web sessions within a second. Watch `GET /health` (`classes`: running, queued, served, busy answers per class) and `/admin/status` (the number of sessions); the login audit is `audit.jsonl`.

## 5a. Cloudflare in front (recommended)

Cloudflare absorbs floods, hides the origin address and terminates the browser's TLS. The server is built for it (decision A-27); what you configure:

1. **DNS:** the record of the site is *proxied* (orange cloud). Nothing else in the zone should point at the origin address.
2. **TLS between Cloudflare and the server.** Either
   * SSL/TLS mode **Full (strict)** with a Cloudflare *Origin CA* certificate: create it in the dashboard (SSL/TLS, Origin Server), export a PFX, and use it in `Security:Https` (section 5). Open inbound 443 only from Cloudflare's ranges (`netsh advfirewall firewall add rule name="TTC Siren HTTPS" dir=in action=allow protocol=TCP localport=443 remoteip=<the ranges of cloudflare-ips.txt>`); or
   * a **`cloudflared` tunnel** that runs on this machine and forwards to `http://127.0.0.1:5000`. Then no port is open at all, the server stays on loopback with HTTP, and `Security:TrustedProxies:TrustLoopback=true`.
3. **The real client address.** The server believes `CF-Connecting-IP` only when the TCP peer is a trusted proxy: `UseCloudflareRanges=true` (the ranges in `cloudflare-ips.txt`, next to `TTC.Siren.Server.exe`) for a direct connection, or `TrustLoopback=true` for a local tunnel. From every other peer the header is ignored, so a client cannot choose its own address. Rate limits, login throttling and the audit log use this address. `X-Forwarded-For` is never read.
4. **Cache.** The server sends `Cache-Control: no-store` on every dynamic answer and a year-long `immutable` policy on the hashed files under `/assets/`. Cloudflare does not cache dynamic paths by default; make it explicit with a **Cache Rule**: if the URI path starts with `/ai_move`, `/v1/`, `/api/`, `/health` or `/admin/` then *Bypass cache*. (`/admin/*` must not be published at all, see 6.)
5. **Timeouts.** Cloudflare gives up after 100 seconds (free plan; error 524). The server finishes inside `OriginTimeoutSeconds` (95): it lowers a request's `max_time` to `MaxSearchSeconds` (20) and refuses to start with settings whose worst case (`MaxSearchSeconds + QueueWaitMs + OriginMarginSeconds`) is not below the timeout. A paid plan with a longer timeout lets you raise these together.
6. **The event stream** (`/v1/live`, Server-Sent Events) passes through Cloudflare unbuffered. The server sends a heartbeat comment every 20 seconds (`Live:HeartbeatSeconds`), well below the 100 second idle limit, and a browser that loses the connection reconnects after 3 seconds with `Last-Event-ID` and receives what it missed (API.md section 5.3).
7. **WAF (optional, in the dashboard — the server has no CAPTCHA or bot code):** a rate-limit rule such as `POST /v1/session` 10 per minute per IP, and `/ai_move` a few hundred per minute per IP; *Bot Fight Mode*; a custom rule that blocks the URI path `/admin`.
8. **Never expose `/admin/*`.** The operator endpoints answer only a caller on this machine that no proxy handled: a request carrying any of `CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For`, `Forwarded`, `X-Real-IP`, `Via` and the other proxy headers is refused with 403 even when it arrives from loopback (that is what a tunnel looks like).
9. **Refreshing the ranges.** Cloudflare publishes them at <https://www.cloudflare.com/ips-v4> and <https://www.cloudflare.com/ips-v6> and announces changes in advance. A few times a year: `python tools
efresh_cloudflare_ips.py C:	tc\server\cloudflare-ips.txt`, then restart the service (the file is read at start; a missing or empty file stops the start). A tunnel needs no ranges.

## 6. Services

The C# server runs as a Windows service directly (`UseWindowsService`). In an elevated prompt:

```
sc.exe create TTCSirenServer binPath= "C:\ttc\server\TTC.Siren.Server.exe" start= auto DisplayName= "TTC Siren server"
sc.exe failure TTCSirenServer reset= 86400 actions= restart/5000/restart/5000/restart/30000
sc.exe start TTCSirenServer
```

(a space after each `=`). Set the machine environment variables (`TTC_PFX_PASSWORD`, `TTC_Server__...`) before starting. The service account needs read access to the application folder and the PFX, and write access to the game-log folder and the key file.

The Python app is a script, not a service: run it with waitress through a service wrapper such as NSSM:

```
nssm install TTCSirenPython "C:\Python311\python.exe" "C:\ttc\python\serve_waitress.py --host 127.0.0.1 --port 5001 --threads 8 --no-console"
nssm set TTCSirenPython AppDirectory C:\ttc\python
nssm start TTCSirenPython
```

The Python app listens on loopback only. Start order does not matter: until the Python app is up, the three `/api/*` endpoints answer 502 and `/ai_move` is unaffected.

## 7. Firewall

Open **inbound TCP 443 only**. The C# server's plain port (5000) and the Python app (5001) must not be reachable from outside; with `HttpEnabled=false` the first does not even exist.

```
netsh advfirewall firewall add rule name="TTC Siren HTTPS" dir=in action=allow protocol=TCP localport=443
```

Check from another machine that 5000 and 5001 are closed, and that `http://<server>/` does not answer.

## 8. Health, logs, backup, rollback

* **Health:** `GET /health` (no key): status, version, card-database SHA-256, workers, idle workers, queue length, `busyAnswers`. Alert when `queued` stays above 0 with no idle worker.
* **Logs:** the console/Windows event log at warning level (rejected keys, damaged log files). Request logging is off; nothing identifies a user except the key id.
* **Back up** `keys.json` and the game-log folder (the logs are users' data: protect them like a database; deleting a user's games is `DELETE /v1/games/{id}` with their key, or deleting the files).
* **Roll back:** stop the C# server and point the plugin at the old Python server. Game logs and keys are only read by the C# server.

## 9. Workers, memory, scheduling

Every worker owns a transposition table (96 MiB by default) and a small memo, created when the worker is first needed. Working sets measured on the development machine: 1 worker 164 MiB, 4 workers 461 MiB, 8 workers 845 MiB, 12 workers 1.2 GiB, 20 workers 1.6 GiB; plan on workers × 97 MiB + 100 MiB as the upper bound (16 workers ≈ 1.7 GiB; 32 ≈ 3.2 GiB).
The search is CPU bound and each worker runs one single-threaded search: throughput grows with the number of workers up to about the number of physical cores (expect the knee near 16 on the 16-core E5-2698B v3).
Measure on the server with `python tools\phase8a\load_server.py --target server --cpu all --workers N` for several N (it creates its own key and key file), and keep the p95 latency of openings below `max_time` plus a few seconds.
The scheduling probe of the review is reproduced by `SchedulerTests` (`dotnet test --filter SchedulerTests`); see `docs/reports/PHASE8A_REPORT.md` for the figures.

## 10. Troubleshooting

| Symptom | Likely cause |
|---|---|
| The server does not start: "Refusing to listen on … without HTTPS" | Intended. Enable `Security:Https`, or set `Security:BehindTlsProxy=true` behind a TLS proxy, or use a loopback `Host`. |
| 401 `API key required` / `invalid API key` | No key, or a mistyped one. Header: `Authorization: Bearer ttc_…` or `X-Api-Key`. |
| 403 `API key revoked` | The key was revoked (`keys list` shows it). Issue a new one. |
| 429 | The key's rate or in-flight limit; honour `Retry-After`; raise it with `keys set-limit`. |
| 503 `busy` (`queue_full` / `queue_timeout`) | All workers busy and the queue full or the wait expired. Raise `Workers` (if CPUs allow), `QueueLimit` / `QueueWaitMs`, or reduce the traffic. |
| 503 `invalid_request` | The engine rejected the position (for example the side to move has no cards). |
| 502 on `/api/*` | The Python app is down or `Proxy:PythonUrl` is wrong. |
| 504 on `/api/*` | The Python app took longer than `Proxy:TimeoutSeconds`. |
| `/v1/live` stops without a message | A proxy buffers or times out the stream; disable buffering and raise the read timeout. |
| No explanation in the answer, `explanation_error` instead | The facts could not be computed; the move is unaffected. Please keep the request and its `game_id`; the log has everything to replay it. |
