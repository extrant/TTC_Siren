# Deploying the C# engine on Windows Server (Phase 7)

The Python `ai_server` (Flask) parses requests, samples unknown cards and writes the analysis text. For the **minimax** solver it sends the
fully determined position to a local C# engine over HTTP (`127.0.0.1` only) and uses the engine's answer. If the engine is missing, busy,
slow, speaks another protocol, loaded another card database, or returns a move that is not legal, the server **automatically falls back to
the unchanged Python solver**. `solver=monte_carlo` never uses the engine.

```
client -> Flask ai_server (Python) --HTTP 127.0.0.1:5300--> TTC.Siren.Engine (one process, a pool of worker threads)
                         \-- on any problem --> the original Python search (fallback)
```

## 1. What you need

| Item | Notes |
|---|---|
| Windows Server 2016 or newer, x64 | Tested only on Windows 11 here; the engine uses nothing Windows specific except the optional service hosting. |
| **.NET 10 ASP.NET Core Runtime** | Download "ASP.NET Core Runtime 10.x" (or the "Hosting Bundle", which contains it) from dot.net and install it. The engine is published **framework-dependent**: no self-contained copy of .NET. Check with `dotnet --list-runtimes` (needs `Microsoft.AspNetCore.App 10.x`). |
| The Python environment of the server | CPython or PyPy; the engine client uses the standard library only. |
| The card database | The engine and the Python server must load **the same file** `data\幻卡数据库.csv`; the Python side compares SHA-256 hashes at start-up and refuses the engine if they differ. |

## 2. Build and publish (on a development machine with the .NET 10 SDK)

```
dotnet publish csharp\src\TTC.Siren.Engine -c Release -o publish\engine
```

Copy the folder `publish\engine` to the server, for example to `C:\ttc\engine`. It contains `TTC.Siren.Engine.exe`, `TTC.Siren.Engine.dll`,
`appsettings.json` and the runtime configuration. Copy `data\幻卡数据库.csv` next to your Python server as before, for example `C:\ttc\data\`.

## 3. Configuration

Settings come from `appsettings.json` (section `Engine`) and from the command line (the command line wins).

| Command line | `appsettings.json` key | Default | Meaning |
|---|---|---|---|
| `--host` | `Engine:Host` | `127.0.0.1` | Listen address. **Must be a loopback address.** The engine refuses to start on another address unless `--allow-remote` is given. It has no authentication; never expose it. |
| `--port` | `Engine:Port` | `5300` | TCP port. |
| `--workers` | `Engine:Workers` | number of logical CPUs | Searches that run at the same time (see section 8). |
| `--queue` | `Engine:QueueLimit` | `64` | Requests that may wait for a worker. More are answered `busy` (HTTP 503) at once, and the Python server falls back to its own solver for them. |
| `--queue-wait-ms` | `Engine:QueueWaitMs` | `30000` | How long a queued request waits for a worker before it is answered `busy`. |
| `--card-db` | `Engine:CardDbPath` | found above the application folder | Absolute path of `幻卡数据库.csv`. Set it explicitly on a server. |
| `--tt-entries` | `Engine:TtEntries` | `4194304` | Entries (24 bytes each) of every worker's transposition table: 4 Mi entries = 96 MiB per worker. Power of two. |
| `--memo-entries` | `Engine:MemoEntries` | `32768` | Entries of every worker's exact-endgame memo (0.75 MiB). |

Example `appsettings.json` next to the executable:

```json
{ "Engine": { "Port": 5300, "Workers": 16, "QueueLimit": 64, "QueueWaitMs": 30000,
              "CardDbPath": "C:\\ttc\\data\\幻卡数据库.csv" } }
```

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
To change the settings: `sc.exe stop`, edit `appsettings.json` (or `sc.exe config TTCSirenEngine binPath= ...`), `sc.exe start`.
To remove it: `sc.exe stop TTCSirenEngine` and `sc.exe delete TTCSirenEngine`.

## 5. Firewall

Nothing to open. The engine listens on `127.0.0.1` only, so Windows Firewall needs no inbound rule and the port is not reachable from the network.
Do **not** use `--allow-remote` unless you add your own authentication and network controls in front of it.

## 6. Python server settings

All optional, read from environment variables of the Python process (set them as machine/service variables, then restart the Python server):

| Variable | Default | Meaning |
|---|---|---|
| `TTC_CS_ENGINE` | `on` | `off` (also `0`, `false`, `no`) disables the engine completely: the server then behaves exactly as before Phase 7 and does not contact it. |
| `TTC_CS_ENGINE_URL` | `http://127.0.0.1:5300` | Where the engine listens. |
| `TTC_CS_ENGINE_CONNECT_TIMEOUT` | `0.5` | Seconds to wait for the connection. On Windows a refused connection to a closed port can take longer than this; it is then reported as a timeout. |
| `TTC_CS_ENGINE_READ_MARGIN` | `5` | Seconds added to the request's `max_time` for the read timeout. |
| `TTC_CS_ENGINE_BACKOFF` | `30` | Seconds the engine is skipped after a connection-level failure, so that a dead engine costs one timeout, not one per request. |
| `TTC_RECORD_REQUESTS` | unset | If set to a folder, every `/ai_move` request is stored there with the state of Python's random generator (for replay). **This stores users' requests**; use it only where that is acceptable. |

Every response that went through the engine path carries `"engine": "csharp"`; a response where the engine was tried and the Python solver answered
carries `"engine": "python"` and `"fallback_reason"` (`busy`, `timeout`, `connection_failed`, `circuit_open`, `protocol_mismatch`,
`card_db_mismatch`, `illegal_move`, ...). With `TTC_CS_ENGINE=off` neither field is present. Fallbacks are also logged by the logger `ttc.engine`.

The server's own start-up is unchanged (`app.run(threaded=True)`). That is Flask's development server; for production you may want a WSGI server
for Windows, for example `pip install waitress` and `waitress-serve --listen=127.0.0.1:5000 --threads=16 ai_server:app`. Phase 7 does not change how the
Python server is started.

## 7. Health check, start order, switching off, rollback

* **Health check:** `Invoke-RestMethod http://127.0.0.1:5300/v1/health` (or `curl`). Expected: `"status":"ok"`, `"protocol":1`, the card-database `cardDbSha256`,
  `workers`, `idleWorkers`, `queued`, `busyAnswers`. The hash must equal the SHA-256 of the CSV the Python server loads
  (`Get-FileHash C:\ttc\data\幻卡数据库.csv -Algorithm SHA256`).
* **Start order:** engine first, then the Python server. If the Python server starts first, it simply falls back until the engine answers (after at most
  the back-off time).
* **Switch the engine off** (without stopping anything else): set `TTC_CS_ENGINE=off` for the Python server and restart it; optionally `sc.exe stop TTCSirenEngine`.
* **Roll back:** the Python code path is the old one. Setting `TTC_CS_ENGINE=off` is a complete rollback of behaviour; no data is written by the engine.

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
5. Set `--workers` accordingly (section 4) and keep `--queue` small: a request that waits longer than it would take Python to answer is better served by the fallback.

The numbers measured on the development machine (i7-13650HX, 6 P-cores + 8 E-cores) are **not** predictions for the server.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Responses show `"engine": "python"`, `"fallback_reason": "card_db_mismatch"` | The engine loaded another CSV. Pass the same absolute path with `--card-db`. |
| `"fallback_reason": "protocol_mismatch"` | The engine and the Python files are from different versions; redeploy both. |
| `"fallback_reason": "busy"` | More concurrent requests than workers + queue. Raise `--workers` (if CPUs allow) or `--queue`. |
| `"fallback_reason": "connection_failed"` or `"timeout"` then `"circuit_open"` | The engine is not running or not answering; check the service and the health URL. The Python server retries after the back-off. |
| `"fallback_reason": "illegal_move"` | The engine returned a move that Python's state does not allow (should not happen; the response is the Python answer; please keep the recorded request, see `TTC_RECORD_REQUESTS`). |
| The engine does not start: "Refusing to listen on non-loopback address" | Intended. Use `127.0.0.1`. |
