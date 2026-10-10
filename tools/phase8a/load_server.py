# -*- coding: utf-8 -*-
"""
Load test of the native C# server (Phase 8A) over real HTTP, with an API key, and of the Python stack for comparison.

  python tools/phase8a/load_server.py --target server --cpu 0xFFF --workers 12 --out docs/reports/phase8a/load_p_cores.jsonl
  python tools/phase8a/load_server.py --target flask  --cpu 0xFFF --workers 12 --out docs/reports/phase8a/load_p_cores.jsonl

``--target server`` starts ``TTC.Siren.Server`` with the given CPU set (the process sees only those CPUs: ``Environment.ProcessorCount`` follows
the mask), creates an API key with ``keys add`` (so authentication and the per-key limits are part of what is measured; the limits are set high so
that they do not interfere), and posts requests of the Phase 7 corpus with production settings (max_depth 10, max_time 10 s) from a load generator
that runs unpinned. ``--target flask`` starts the C# engine and the Flask ``ai_server`` the way Phase 7 did, with the same CPU set, for the
comparison. Per category (opening / mid / robust) and concurrency level it reports throughput, latency p50/p95/p99/max and the number of non-200 answers.

The numbers say how THIS machine behaves with THIS CPU set; they are not an estimate for another machine.
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
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))
sys.path.insert(0, ROOT)
import pin  # noqa: E402
import gen_corpus  # noqa: E402

SERVER_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Server', 'bin', 'Release', 'net10.0', 'TTC.Siren.Server.dll')
ENGINE_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Engine', 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')
DB = os.path.join(ROOT, 'data', '幻卡数据库.csv')


def category_of(tags):
    t = set(tags)
    if 'opening' in t:
        return 'opening'
    if 'robust_candidate' in t:
        return 'robust'
    if any(x in t for x in ('k:1', 'k:2', 'k:3', 'k:4')):
        return 'mid'
    return 'other'


def corpus_requests(category):
    out = []
    for e in gen_corpus.generate(20261010, 360):
        cat = category_of(e['tags'])
        if cat != category or 'full_board' in e['tags']:
            continue
        req = dict(e['request'])
        req['max_depth'] = 10
        req['max_time'] = 10
        out.append(req)
    return out


def wait_ok(url, path, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + path, timeout=2):
                return
        except urllib.error.HTTPError:
            return  # the server answered (for example 404 from Flask at /)
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit('did not start: ' + url)


def run_level(url, key, requests, concurrency, total):
    parsed = urllib.parse.urlsplit(url)
    latencies = []
    errors = [0]
    lock = threading.Lock()
    counter = [0]

    def worker():
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
        while True:
            with lock:
                i = counter[0]
                counter[0] += 1
            if i >= total:
                break
            body = json.dumps(requests[i % len(requests)], ensure_ascii=False).encode('utf-8')
            headers = {'Content-Type': 'application/json'}
            if key:
                headers['Authorization'] = 'Bearer ' + key
            t0 = time.perf_counter()
            try:
                conn.request('POST', '/ai_move', body, headers)
                response = conn.getresponse()
                response.read()
                status = response.status
            except Exception:  # noqa: BLE001
                status = 0
                conn.close()
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
            took = time.perf_counter() - t0
            with lock:
                latencies.append(took)
                if status != 200:
                    errors[0] += 1
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    latencies.sort()

    def pct(p):
        return latencies[min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))]
    return {'requests': len(latencies), 'seconds': wall, 'rps': len(latencies) / wall, 'p50': pct(0.5), 'p95': pct(0.95), 'p99': pct(0.99),
            'max': latencies[-1], 'errors': errors[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=['server', 'flask'], required=True)
    ap.add_argument('--cpu', default='auto', help="'all', 'auto' (P-cores) or a hex mask")
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--concurrency', default='1,4,8,16')
    ap.add_argument('--categories', default='opening,mid,robust')
    ap.add_argument('--out')
    ap.add_argument('--port', type=int, default=5395)
    args = ap.parse_args()

    mask = None if args.cpu == 'all' else pin.parse_mask(args.cpu)
    tmp = tempfile.mkdtemp(prefix='ttc-load-')
    processes = []
    key = None
    try:
        if args.target == 'server':
            store = os.path.join(tmp, 'keys.json')
            out = subprocess.run(['dotnet', SERVER_DLL, 'keys', 'add', '--name', 'load', '--rate', '1000000', '--burst', '100000', '--inflight', '1000', '--store', store],
                                 capture_output=True, text=True, cwd=ROOT, encoding='utf-8').stdout
            key = next(l.strip() for l in out.splitlines() if l.startswith('ttc_'))
            env = dict(os.environ, TTC_Server__Security__KeyStoreFile=store, TTC_Server__GameLog__Directory=os.path.join(tmp, 'logs'))
            server = pin.start_pinned(['dotnet', SERVER_DLL, '--port', str(args.port), '--workers', str(args.workers), '--card-db', DB], mask, cwd=ROOT, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(server)
            url = 'http://127.0.0.1:%d' % args.port
            wait_ok(url, '/health')
        else:
            engine = pin.start_pinned(['dotnet', ENGINE_DLL, '--port', str(args.port + 1), '--workers', str(args.workers), '--card-db', DB], mask, cwd=ROOT,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(engine)
            wait_ok('http://127.0.0.1:%d' % (args.port + 1), '/v1/health')
            env = dict(os.environ, TTC_CS_ENGINE='on', TTC_CS_ENGINE_URL='http://127.0.0.1:%d' % (args.port + 1))
            flask = pin.start_pinned([sys.executable, os.path.join(ROOT, 'tools', 'phase7', 'serve_flask.py'), '--port', str(args.port)], mask, cwd=ROOT, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(flask)
            url = 'http://127.0.0.1:%d' % args.port
            time.sleep(2.0)
            wait_ok(url, '/')  # Flask has no / route: a 404 still means it is up (urllib raises, caught below)
        label = '%s cpu=%s workers=%d' % (args.target, pin.describe(mask) if mask is not None else 'all', args.workers)
        rows = []
        print('# ' + label, flush=True)
        print('%-8s %-5s %9s %9s %9s %9s %9s %6s' % ('category', 'conc', 'req/s', 'p50 s', 'p95 s', 'p99 s', 'max s', 'errors'), flush=True)
        for category in args.categories.split(','):
            reqs = corpus_requests(category)
            for concurrency in [int(c) for c in args.concurrency.split(',')]:
                total = {'opening': max(12, 3 * concurrency), 'mid': 300, 'robust': 400}[category]
                r = run_level(url, key, reqs, concurrency, total)
                row = dict(label=label, target=args.target, cpu=pin.describe(mask) if mask is not None else 'all', workers=args.workers, category=category,
                           concurrency=concurrency, **r)
                rows.append(row)
                print('%-8s %-5d %9.1f %9.3f %9.3f %9.3f %9.3f %6d' % (category, concurrency, r['rps'], r['p50'], r['p95'], r['p99'], r['max'], r['errors']), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'a', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    finally:
        for p in processes:
            p.kill()
            p.wait()


if __name__ == '__main__':
    main()
