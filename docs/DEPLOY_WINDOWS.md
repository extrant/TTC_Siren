# Running TTC_Siren with the C# engine on Windows

**Decision A-30 (2026-10-12):** the product is the original Python server (`app.py` / `serve_waitress.py` / the packaged `TTC_Siren.exe` or the PyPy build),
on `http://127.0.0.1:5000`, with every endpoint and every behaviour as before (no API key; `/api/draft/analyze`, `/api/recommend`, `/api/winrate` unchanged).
The only difference is the **minimax search of `/ai_move`, which the C# engine runs** (about 100 times faster than the Python search). The server **starts the engine by
itself** when it is not running (section 0) and stops it again when it exits. The separate product server, web UI and API keys of Phase 8 were removed.

The Python `ai_server` (Flask) parses requests, samples unknown cards and writes the analysis text. For the **minimax** solver it sends the
fully determined position to a local C# engine over HTTP (`127.0.0.1` only) and uses the engine's answer. **Since the Phase 7 hardening (decision A-20)
the engine is required for the minimax solver: there is no automatic fallback to the Python solver.** When the engine is missing, busy for too long, slow,
speaks another protocol, loaded another card database, or returns a move that is not legal, `/ai_move` answers with an HTTP error (503 plus an
`engine_error` code, section 6) and the game client may retry. The only way to use the Python solver is the operator's manual switch `TTC_CS_ENGINE=off`
(section 7). `solver=monte_carlo` never uses the engine.

```
client -> Flask ai_server (Python) --HTTP 127.0.0.1:5300--> TTC.Siren.Engine (one process, a pool of worker threads)
                         \-- engine problem --> HTTP 503 {"engine_error": ...}   (no Python search)
                         \-- TTC_CS_ENGINE=off (set by hand) --> the original Python search
```

**A machine without the engine** (a development box, a server where it is not installed yet) must set `TTC_CS_ENGINE=off`, otherwise every minimax request there
answers 503 `connection_failed`.

## 0. The short way: one click, or the packaged folder

| What | How |
|---|---|
| Run from the repository | Double-click **`run_local.bat`** (`run_local.bat pypy` for PyPy from `PYPY\pypy3.exe`). The first run builds the engine (`scriptsuild_engine.bat`, needs the .NET 10 SDK); then it runs `python app.py`, which starts the engine. `scriptsun_pypy.bat`, `python app.py` and `serve_waitress.py` start the engine the same way once it is built. |
| Build the engine only | `scriptsuild_engine.bat` -> `dist\engine\TTC.Siren.Engine.exe`: one self-contained file (about 50 MB); the machine that runs it needs **no .NET installed**. |
| Build the smallest package to hand out | `scriptsuild_release.bat` -> `distelease\` (`TTC_Siren.exe` + `engine\TTC.Siren.Engine.exe` + `使用说明.txt`) and `dist\TTC_Siren_release.zip`. The recipient unzips it and double-clicks `TTC_Siren.exe`; no Python, no .NET. |
| Build everything to distribute | `build_all.bat` -> `dist\engine\` (C# engine), `dist\TTC_Siren.exe` (CPython), `dist\TTC_Siren_PYPY\` (PyPy). Zip `dist\`; on the target machine double-click `TTC_Siren.exe` (or `TTC_Siren_PYPY.exe`) as before. |

**How the server starts the engine** (`ai/engine_launcher.py`, called by `app.py` and `serve_waitress.py` before they listen; the start-up banner says what happened):

1. `TTC_CS_ENGINE=off` -> nothing is started; the Python solver answers (as before Phase 7).
2. An engine already answers at `TTC_CS_ENGINE_URL` (default `http://127.0.0.1:5300`) -> it is used (for example the Windows service of section 4).
3. Otherwise, unless `TTC_CS_ENGINE_AUTOSTART=off` or the URL is not loopback, `TTC.Siren.Engine.exe` is looked for: `TTC_CS_ENGINE_EXE`, then `engine\`, `..\engine\` and
   `dist\engine\` next to the packaged exe and in the repository. It is started on the port of the URL with **the card database this server loads** (`--card-db`), its output
   goes to `engine.log` next to it, and it is tied to the server process (a Windows job object): it ends when the server ends, also when the console window is closed.
4. Not found -> the banner says `未找到（先运行 scriptsuild_engine.bat）` and minimax requests of `/ai_move` answer 503 `connection_failed` until an engine runs (A-20: no silent fallback).

## 1. What you need

| Item | Notes |
|---|---|
| Windows Server 2016 or newer, x64 | Tested only on Windows 11 here; the engine uses nothing Windows specific except the optional service hosting. |
| .NET | **Nothing** for the self-contained build of `scriptsuild_engine.bat`. Only the framework-dependent publish of section 2 needs the "ASP.NET Core Runtime 10.x" (check with `dotnet --list-runtimes`). Building needs the .NET 10 SDK. |
| The Python environment of the server | CPython or PyPy; the engine client uses the standard library only. |
| The card database | The engine and the Python server must load **the same file** `data\幻卡数据库.csv`; the Python side compares SHA-256 hashes at start-up and refuses the engine if they differ. |

## 2. Build and publish (on a development machine with the .NET 10 SDK)

`scriptsuild_engine.bat` (self-contained, one file, recommended), or framework-dependent:

```
dotnet publish csharp\src\TTC.Siren.Engine -c Release -o publish\engine
```

Copy the folder (`dist\engine` or `publish\engine`) to the server, for example to `C:\ttc\engine`. It contains `TTC.Siren.Engine.exe`, `engine.appsettings.json`
and (framework-dependent only) `TTC.Siren.Engine.dll` and the runtime configuration. Copy `data\幻卡数据库.csv` next to your Python server as before, for example `C:\ttc\data\`.

## 3. Configuration

Settings come from `engine.appsettings.json` (and an optional `appsettings.json`, section `Engine`) next to the executable, and from the command line (the command line wins).

| Command line | `engine.appsettings.json` key | Default | Meaning |
|---|---|---|---|
| `--host` | `Engine:Host` | `127.0.0.1` | Listen address. **Must be a loopback address.** The engine refuses to start on another address unless `--allow-remote` is given. It has no authentication; never expose it. |
| `--port` | `Engine:Port` | `5300` | TCP port. |
| `--workers` | `Engine:Workers` | number of logical CPUs | Searches that run at the same time (see section 8). |
| `--queue` | `Engine:QueueLimit` | `256` | Requests that may wait for a worker. More are answered `busy` (HTTP 503) at once, which the Python server passes on as 503 `busy`. |
| `--queue-wait-ms` | `Engine:QueueWaitMs` | `60000` | How long a queued request waits for a worker before it is answered `busy`. `/v1/health` reports it as `queueWaitMs`; the Python client's read timeout is `max_time` + this + a margin, so a queued request is always answered by the engine (a result or `busy`) and never by a client timeout. |
| `--card-db` | `Engine:CardDbPath` | found above the application folder | Absolute path of `幻卡数据库.csv`. Set it explicitly on a server. |
| `--tt-entries` | `Engine:TtEntries` | `4194304` | Entries (24 bytes each) of every worker's transposition table: 4 Mi entries = 96 MiB per worker. Power of two. |
| `--memo-entries` | `Engine:MemoEntries` | `32768` | Entries of every worker's exact-endgame memo (0.75 MiB). |

Example `engine.appsettings.json` next to the executable:

```json
{ "Engine": { "Port": 5300, "Workers": 16, "QueueLimit": 256, "QueueWaitMs": 60000,
              "CardDbPath": "C:\\ttc\\data\\幻卡数据库.csv" } }
```

**Why a deep queue and a long wait.** The Python solver is no longer a way out, and it was about 100 times slower per search anyway, so a request is better
off waiting for a free worker. A request that waits is dropped without a search if its HTTP connection has been closed meanwhile (the client gave up);
a search that is already running is never interrupted. Size the wait to the longest delay your game clients accept: a request that has waited `queue-wait-ms`
gets 503 `busy`, and a full queue answers 503 `busy` at once.

**Garbage collector.** The engine runs with **Server GC, non-concurrent**, written explicitly in the project file and visible in the published `TTC.Siren.Engine.runtimeconfig.json`
(`"System.GC.Server": true`, `"System.GC.Concurrent": false`); `/v1/health` reports `serverGc`. It was chosen by measurement (`docs/migration/reports/phase7/gc/gc_table.txt`,
`docs/migration/reports/PHASE7_HARDENING.md` section 7): equal to workstation GC for openings and mid-games, faster for the very short robust-endgame requests on the P-cores, same memory.
Server GC keeps one heap per logical CPU it may use; on the 32-thread server that is 32 small heaps, which was not measured there. If memory matters more than throughput,
`DOTNET_gcServer=0` in the engine's environment switches to workstation GC without a rebuild.

Run it once by hand to see that it works:

```
C:\ttc\engine\TTC.Siren.Engine.exe --workers 16 --card-db C:\ttc\data\幻卡数据库.csv
```

It prints the listening address, the number of workers and the SHA-256 of the card database.

## 4. Run it as a Windows service

The executable supports the Windows service host (`UseWindowsService`), so `sc.exe` can run it directly. In an elevated prompt:

```
sc.exe create TTCSirenEngine binPath= "C:\ttc\engine\TTC.Siren.Engine.exe --workers 16 --card-db C:\ttc\data\幻卡数据库.csv" start= auto DisplayName= "TTC Siren engine"
sc.exe description TTCSirenEngine "C# solver for the TTC Siren ai_server (127.0.0.1 only)"
sc.exe failure TTCSirenEngine reset= 86400 actions= restart/5000/restart/5000/restart/30000
sc.exe start TTCSirenEngine
```

(There must be a space after each `=`.) Use a service account that can read the folder; the engine needs no network permissions and no write access.
A Python server started afterwards finds the service running and starts nothing (or set `TTC_CS_ENGINE_AUTOSTART=off`). To change the settings: `sc.exe stop`, edit `engine.appsettings.json` (or `sc.exe config TTCSirenEngine binPath= ...`), `sc.exe start`.
To remove it: `sc.exe stop TTCSirenEngine` and `sc.exe delete TTCSirenEngine`.

## 5. Firewall

Nothing to open. The engine listens on `127.0.0.1` only, so Windows Firewall needs no inbound rule and the port is not reachable from the network.
Do **not** use `--allow-remote` unless you add your own authentication and network controls in front of it.

## 6. Python server settings

All optional, read from environment variables of the Python process (set them as machine/service variables, then restart the Python server):

| Variable | Default | Meaning |
|---|---|---|
| `TTC_CS_ENGINE` | `on` | `off` (also `0`, `false`, `no`) is the **manual** switch: the minimax solver then runs in Python exactly as before Phase 7 and the engine is never contacted. It is never set automatically. |
| `TTC_CS_ENGINE_URL` | `http://127.0.0.1:5300` | Where the engine listens. |
| `TTC_CS_ENGINE_AUTOSTART` | `on` | `off` (also `0`, `false`, `no`): the server never starts the engine itself (use it with the service of section 4). |
| `TTC_CS_ENGINE_EXE` | unset | Full path of `TTC.Siren.Engine.exe` to start, before the places of section 0 are searched. |
| `TTC_CS_ENGINE_CONNECT_TIMEOUT` | `0.5` | Seconds to wait for the connection. On Windows a refused connection to a closed port can take longer than this; it is then reported as `connection_failed`. |
| `TTC_CS_ENGINE_READ_MARGIN` | `5` | Seconds added to `max_time` + the engine's queue wait (from `/v1/health`) for the read timeout of a solve. With the defaults a request waits at most 10 + 60 + 5 = 75 s. |
| `TTC_CS_ENGINE_POOL_SIZE` | `32` | Idle keep-alive connections the Python process keeps to the engine (one process-wide pool, reused by every request thread). More connections may be open during a burst; the surplus is closed afterwards. |
| `TTC_RECORD_REQUESTS` | unset | If set to a folder, every `/ai_move` request is stored there with the state of Python's random generator (for replay). **This stores users' requests**; use it only where that is acceptable. |

(`TTC_CS_ENGINE_BACKOFF` of Phase 7 no longer exists: there is no circuit breaker. A failure never disables the engine; the next request tries again and the health is
verified again after any connection-level failure.)

**What the game client receives when the engine cannot answer** (the Python solver is not run):

```
HTTP 503   {"error": "C# engine: busy (queue_full)", "engine": "csharp", "engine_error": "busy"}
```

| `engine_error` | HTTP | Meaning |
|---|---|---|
| `connection_failed` | 503 | The engine is not running or not reachable (also a connect timeout, a reset connection). |
| `timeout` | 503 | No answer within `max_time` + queue wait + margin. |
| `busy` | 503 | The engine's queue is full (`queue_full`) or the request waited `queue-wait-ms` (`queue_timeout`). |
| `engine_error` | 503 | The engine reported an internal error (its HTTP 500). |
| `invalid_request` | 503 | The engine rejected the position (its HTTP 400). |
| `malformed_response`, `http_error`, `unexpected_status` | 503 | The answer was not what the protocol says. |
| `protocol_mismatch`, `card_db_mismatch`, `health_failed` | 503 | The engine is another version, loaded another card database, or its health check failed. Not cached: a corrected engine is accepted by the next request. |
| `illegal_move` | 503 | The engine's move is not legal for the position (should not happen; keep the request, see `TTC_RECORD_REQUESTS`). |
| `unsupported_parameter` | 400 | `max_depth` / `max_time` of the request is not a number (the Python solver would have raised a TypeError). A bool counts as the integer it is in Python. |
| `client_error` | 500 | A defect in the Python client code itself. |

A successful engine answer carries `"engine": "csharp"` and `engine_info`; with `TTC_CS_ENGINE=off` neither field is present.

**Monitoring.** Every failure is logged by the logger `ttc.engine` at **error** level (`C# engine failed (<code>): <detail>`); alert on these lines. The engine's
`GET /v1/health` shows the live state: `idleWorkers`, `queued`, `queueLimit`, `queueWaitMs`, `busyAnswers` (requests answered `busy`), `abandonedSkipped`
(queued requests dropped because the client had gone), `served`, `workersCreated`, `tableMemoryBytes`, `serverGc`. A growing `queued` with no idle workers
means the engine needs more workers or the traffic is too high.

**Front end.** `python app.py` still starts Flask's development server (`app.run(threaded=True)`, one thread per request) and works with the connection
pool. The recommended production front end is the optional launcher `serve_waitress.py`, which serves the same application with waitress and a fixed thread pool:

```
pip install waitress
python serve_waitress.py --host 127.0.0.1 --port 5000 --threads 16        (add --no-console when running as a service)
```

It performs the same initialisation as `app.py` and does not change `app.py`. `--threads` is the number of requests one Python process works on at once.

## 7. Health check, start order, switching off, rollback

* **Health check:** `Invoke-RestMethod http://127.0.0.1:5300/v1/health` (or `curl`). Expected: `"status":"ok"`, `"protocol":1`, the card-database `cardDbSha256`,
  `workers`, `idleWorkers`, `queued`, `busyAnswers`. The hash must equal the SHA-256 of the CSV the Python server loads
  (`Get-FileHash C:\ttc\data\幻卡数据库.csv -Algorithm SHA256`).
* **Start order:** the Python server starts the engine itself (section 0). With a service (section 4): engine first, then the Python server; a Python server that
  starts first would start its own engine on the same port. Set `TTC_CS_ENGINE_AUTOSTART=off` for the Python server when the engine runs as a service.
* **Switch the engine off** (manual, by the operator): set `TTC_CS_ENGINE=off` for the Python server and restart it; the Python solver then answers minimax
  requests (about 100 times slower). Use it on any machine without an engine and as an emergency measure; it is never switched on or off automatically.
* **Roll back:** `TTC_CS_ENGINE=off` restores the behaviour of the code before Phase 7 exactly (verified on the whole request corpus); no data is written by the engine.

## 8. Choosing the number of workers, memory

Every worker owns a transposition table (96 MiB by default) and a small memo (0.75 MiB), created when the worker is first needed (an idle engine uses about 50 MiB). The pages of a table are
touched as searches fill it, so the working set grows towards 96.75 MiB per worker. Measured on the development machine after the full load matrix:

| Workers | Working set measured (MiB) | Upper bound: workers × 96.75 MiB + 100 MiB |
|---|---|---|
| 1 | 164 | 0.2 GiB |
| 4 | 461 | 0.5 GiB |
| 8 | 845 | 0.9 GiB |
| 12 | 1,233 | 1.3 GiB |
| 20 | 1,622 | 2.0 GiB |

For the server (not measured there): 16 workers need at most about 1.7 GiB and 32 workers at most about 3.2 GiB; plan on the upper bound. With `--tt-entries 1048576` (24 MiB) the tables take a quarter of that.
16 GB is ample either way.

**How to pick the number on the server itself** (E5-2698B v3: 16 cores / 32 threads, 16 GB). The search is CPU bound and each worker runs one single-threaded search, so
throughput grows with the number of workers up to about the number of physical cores and then flattens; more workers than that only raise latency.

1. Install the runtime, publish the engine (sections 1 and 2) and make sure Python 3 is available.
2. From the repository on the server, generate the request payloads once: `python tools\phase7\gen_corpus.py --out tools\phase7\corpus\corpus_s20261010_n360.json` and
   `python tools\phase7\e2e_reference.py --corpus tools\phase7\corpus\corpus_s20261010_n360.json --out tools\parity\golden\e2e_reference_g1_s20261010_n360.jsonl.gz`.
3. Run the matrix for the worker counts you want to compare, with all CPUs:
   `python tools\phase7\run_load_matrix.py --cpu all --workers 4,8,16,24,32 --concurrency 1,4,8,16,24,32 --out C:\temp\load_server.jsonl`
   (add `--flask` to also measure the whole Flask path). It starts the engine for every worker count, replays openings, mid-games and robust endgames with the production
   settings (`max_depth` 10, `max_time` 10 s) and prints one JSON line per cell: requests per second and latency p50/p95/p99/max.
4. Choose the smallest worker count after which requests per second no longer grow noticeably (expect the knee near 16, and the 32 hardware threads to add little).
   Keep the p95 latency below `max_time` plus a few seconds: when the machine is oversubscribed, searches run into their time limit and play weaker.
5. Set `--workers` accordingly (section 4). Do **not** keep the queue short in the hope that Python takes the overflow: there is no fallback, and Python would be about
   100 times slower anyway. Keep the defaults (`--queue 256 --queue-wait-ms 60000`) or size the wait to what your game clients accept; a request that waits is answered
   by the engine, and only a full queue or an expired wait gives 503 `busy`.

The numbers measured on the development machine (i7-13650HX, 6 P-cores + 8 E-cores) are **not** predictions for the server.

## 9. Troubleshooting

| Symptom (HTTP 503 unless noted) | Likely cause |
|---|---|
| `"engine_error": "card_db_mismatch"` | The engine loaded another CSV. Pass the same absolute path with `--card-db`. |
| `"engine_error": "protocol_mismatch"` | The engine and the Python files are from different versions; redeploy both. |
| `"engine_error": "busy"` | The queue is full (`queue_full`) or the wait ran out (`queue_timeout`). Raise `--workers` (if CPUs allow) or `--queue` / `--queue-wait-ms`, or reduce the traffic. |
| `"engine_error": "connection_failed"` | The engine is not running or not answering; check the service and the health URL. On a machine without an engine set `TTC_CS_ENGINE=off`. |
| `"engine_error": "timeout"` | The engine did not answer within `max_time` + queue wait + margin; look at `queued` / `idleWorkers` in the health. |
| `"engine_error": "illegal_move"` | The engine returned a move that Python's state does not allow (should not happen; please keep the recorded request, see `TTC_RECORD_REQUESTS`). |
| `"engine_error": "unsupported_parameter"` (HTTP 400) | `max_depth` / `max_time` in the request is not a number. |
| The engine does not start: "Refusing to listen on non-loopback address" | Intended. Use `127.0.0.1`. |
