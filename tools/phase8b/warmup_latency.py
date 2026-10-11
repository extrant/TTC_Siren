# -*- coding: utf-8 -*-
"""
Review O8-3: does the start-up warm-up make the first minute after a restart faster?

  python tools/phase8b/warmup_latency.py --cpu auto --workers 12 --runs 3 --out docs/reports/phase8b/warmup_latency.jsonl

For each run and each setting (warm-up off / on, alternating, a fresh server process every time, the process pinned to the P-cores like the other
measurements) the script starts the server, waits until it answers, and then drives it for ``--seconds`` with the same workload: four client threads that
cycle through opening, mid-game and robust-endgame requests of the Phase 7 corpus at production settings (max_depth 10, max_time 10 s).
Reported per category: the median latency of the FIRST requests (the first ``--first`` of the category) and the median of the whole minute, and the time
the server needed to become ready (the warm-up runs before the first connection is accepted, so that is its price).
The numbers describe THIS machine with THIS CPU set; they are not an estimate for another machine.
"""
import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase8a'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))
sys.path.insert(0, ROOT)
import load_server as ls  # noqa: E402
import pin  # noqa: E402


def drive(url, key, workload, seconds, threads):
    """workload: list of (category, request). Returns the list of (seconds since start, category, latency, status)."""
    parsed = urllib.parse.urlsplit(url)
    results = []
    lock = threading.Lock()
    counter = [0]
    t_start = time.perf_counter()

    def worker():
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
        while time.perf_counter() - t_start < seconds:
            with lock:
                i = counter[0]
                counter[0] += 1
            category, request = workload[i % len(workload)]
            body = json.dumps(request, ensure_ascii=False).encode('utf-8')
            t0 = time.perf_counter()
            try:
                conn.request('POST', '/ai_move', body, {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
                response = conn.getresponse()
                response.read()
                status = response.status
            except Exception:  # noqa: BLE001
                status = 0
                conn.close()
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
            took = time.perf_counter() - t0
            with lock:
                results.append((t0 - t_start, category, took, status))
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sorted(results)


def one_run(setting, args, mask, workload):
    tmp = tempfile.mkdtemp(prefix='ttc-warm-')
    store = os.path.join(tmp, 'keys.json')
    out = subprocess.run(['dotnet', ls.SERVER_DLL, 'keys', 'add', '--name', 'load', '--rate', '1000000', '--burst', '100000', '--inflight', '1000', '--store', store],
                         capture_output=True, text=True, cwd=ROOT, encoding='utf-8').stdout
    key = next(l.strip() for l in out.splitlines() if l.startswith('ttc_'))
    env = dict(os.environ, TTC_Server__Security__KeyStoreFile=store, TTC_Server__GameLog__Directory=os.path.join(tmp, 'logs'),
               TTC_Server__WarmUp='true' if setting == 'on' else 'false')
    t0 = time.perf_counter()
    server = pin.start_pinned(['dotnet', ls.SERVER_DLL, '--port', str(args.port), '--workers', str(args.workers), '--card-db', ls.DB], mask, cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = 'http://127.0.0.1:%d' % args.port
        ls.wait_ok(url, '/health')
        ready = time.perf_counter() - t0
        results = drive(url, key, workload, args.seconds, args.threads)
    finally:
        server.kill()
        server.wait()
    row = {'setting': setting, 'ready_seconds': ready, 'requests': len(results), 'errors': sum(1 for r in results if r[3] != 200)}
    for category in ('opening', 'mid', 'robust'):
        lat = [r[2] for r in results if r[1] == category and r[3] == 200]
        row[category] = {'n': len(lat), 'first_median': statistics.median(lat[:args.first]) if lat else None,
                         'first_max': max(lat[:args.first]) if lat else None, 'median': statistics.median(lat) if lat else None}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cpu', default='auto')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--first', type=int, default=6, help='how many of the first requests of a category count as "the first requests"')
    ap.add_argument('--port', type=int, default=5397)
    ap.add_argument('--out')
    args = ap.parse_args()
    mask = None if args.cpu == 'all' else pin.parse_mask(args.cpu)

    workload = []
    pools = {c: ls.corpus_requests(c) for c in ('opening', 'mid', 'robust')}
    n = max(len(v) for v in pools.values())
    for i in range(n):
        for c in ('mid', 'robust', 'opening', 'robust'):
            workload.append((c, pools[c][i % len(pools[c])]))
    rows = []
    for run in range(args.runs):
        for setting in (('off', 'on') if run % 2 == 0 else ('on', 'off')):
            row = one_run(setting, args, mask, workload)
            row['run'] = run
            rows.append(row)
            print('run %d warm-up %-3s ready %5.1fs  %s' % (run, setting, row['ready_seconds'],
                  '  '.join('%s first %.3f/%.3f med %.3f (n=%d)' % (c, row[c]['first_median'] or 0, row[c]['first_max'] or 0, row[c]['median'] or 0, row[c]['n']) for c in ('opening', 'mid', 'robust'))),
                  flush=True)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(row, cpu=pin.describe(mask) if mask is not None else 'all', workers=args.workers), ensure_ascii=False) + '\n')
    print()
    print('%-9s %-4s %12s %14s %14s %14s' % ('category', 'warm', 'ready s', 'first med s', 'first max s', 'minute med s'))
    for c in ('opening', 'mid', 'robust'):
        for setting in ('off', 'on'):
            sel = [r for r in rows if r['setting'] == setting and r[c]['median'] is not None]
            if not sel:
                continue
            print('%-9s %-4s %12.1f %14.3f %14.3f %14.3f' % (c, setting, statistics.median(r['ready_seconds'] for r in sel), statistics.median(r[c]['first_median'] for r in sel),
                                                           statistics.median(r[c]['first_max'] for r in sel), statistics.median(r[c]['median'] for r in sel)))


if __name__ == '__main__':
    main()
