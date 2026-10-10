# -*- coding: utf-8 -*-
"""
End-to-end latency over real HTTP, Python server on its production path (Flask + C# engine process) vs the native C# server (Phase 8A).

  python tools/phase8a/latency_http.py --affinity auto --runs 3 --out docs/reports/phase8a/latency_p_cores.json

Protocol (CLAUDE.md section 6, goal "pinned protocol"): every independent run starts FRESH server processes pinned to the same CPU set (the
mask is applied before the process executes anything, tools/bench/pin.py): first the Python stack (the engine process, 4 workers, plus
``tools/phase7/serve_flask.py``), then the C# server (4 workers). The same 12 requests of the Phase 7 corpus (3 openings, 3 mid-games, 6
robust-endgame requests; production settings max_depth 10, max_time 10 s) are posted three times in the same order by this driver, which is not pinned
(it only waits for answers). Pass 1 is COLD (first request of its kind in a new process), passes 2 and 3 are WARM. The tables report min / median /
max over the independent runs (cold: pass 1; warm: median of passes 2-3 of each run). The moves are compared as well: they agree when the
unknown-card sampling gives the same cards, which the servers do not guarantee here (the Python server's generator is not seeded per request),
so the comparison of moves is informative only; the response times are what this measures.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))
sys.path.insert(0, ROOT)
import pin  # noqa: E402
import latency as L7  # noqa: E402  (select_requests)
import gen_corpus  # noqa: E402

ENGINE_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Engine', 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')
SERVER_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Server', 'bin', 'Release', 'net10.0', 'TTC.Siren.Server.dll')
DB = os.path.join(ROOT, 'data', '幻卡数据库.csv')


def wait_ok(url, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit('server did not start: ' + url)


def post(url, body):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(url + '/ai_move', data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode('utf-8'))
        status = response.status
    return time.perf_counter() - t0, status, payload


def one_pass(url, chosen, pass_no):
    rows = []
    for e in chosen:
        request = dict(e['request'])
        request['max_depth'] = 10
        request['max_time'] = 10
        seconds, status, body = post(url, request)
        stats = body.get('performance_stats') or {}
        rows.append({'pass': pass_no, 'i': e['i'], 'seconds': seconds, 'status': status, 'card_id': body.get('card_id'), 'pos': body.get('pos'),
                     'depth': stats.get('search_depth'), 'nodes': stats.get('nodes_searched'),
                     'engine_ms': ((body.get('engine_info') or {}).get('timings') or {}).get('totalMs')})
    return rows


def measure(url, chosen):
    rows = []
    for pass_no in (1, 2, 3):
        rows.extend(one_pass(url, chosen, pass_no))
    return rows


def spread(values):
    return min(values), statistics.median(values), max(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--affinity', default='auto')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--engine-port', type=int, default=5398)
    ap.add_argument('--flask-port', type=int, default=5397)
    ap.add_argument('--server-port', type=int, default=5396)
    ap.add_argument('--out')
    args = ap.parse_args()

    mask = None if args.affinity == 'all' else pin.parse_mask(args.affinity)
    chosen = L7.select_requests(gen_corpus.generate(20261010, 360))
    engine_url = 'http://127.0.0.1:%d' % args.engine_port
    flask_url = 'http://127.0.0.1:%d' % args.flask_port
    server_url = 'http://127.0.0.1:%d' % args.server_port
    runs = []
    for run in range(args.runs):
        print('# run %d/%d' % (run + 1, args.runs), flush=True)
        # --- Python stack
        engine = pin.start_pinned(['dotnet', ENGINE_DLL, '--port', str(args.engine_port), '--workers', str(args.workers), '--card-db', DB], mask, cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env = dict(os.environ, TTC_CS_ENGINE='on', TTC_CS_ENGINE_URL=engine_url)
        flask = pin.start_pinned([sys.executable, os.path.join(ROOT, 'tools', 'phase7', 'serve_flask.py'), '--port', str(args.flask_port)], mask, cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        try:
            wait_ok(engine_url + '/v1/health')
            time.sleep(1.0)
            python_rows = measure(flask_url, chosen)
        finally:
            flask.kill(); flask.wait(); engine.kill(); engine.wait()
        # --- C# server
        server = pin.start_pinned(['dotnet', SERVER_DLL, '--port', str(args.server_port), '--workers', str(args.workers), '--card-db', DB], mask, cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ok(server_url + '/health')
            csharp_rows = measure(server_url, chosen)
        finally:
            server.kill(); server.wait()
        runs.append({'python': python_rows, 'csharp': csharp_rows})

    summary = {'affinity': pin.describe(mask) if mask is not None else 'all CPUs', 'runs': args.runs, 'workers': args.workers, 'requests': []}
    for idx, req in enumerate(chosen):
        row = {'i': req['i'], 'tags': [t for t in req['tags'] if t.startswith(('k:', 'unknown', 'rule:'))][:4]}
        for mode in ('python', 'csharp'):
            cold = [r['seconds'] for run in runs for r in run[mode] if r['i'] == req['i'] and r['pass'] == 1]
            warm = [statistics.median([r['seconds'] for r in run[mode] if r['i'] == req['i'] and r['pass'] in (2, 3)]) for run in runs]
            row[mode] = {'cold': spread(cold), 'warm': spread(warm)}
        row['depth'] = {m: runs[0][m][len(chosen) * 2 + idx].get('depth') for m in ('python', 'csharp')}
        summary['requests'].append(row)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'raw': runs}, f, ensure_ascii=False)

    print('\n# /ai_move over HTTP, seconds, min / median / max over %d runs, CPU set: %s, %d search workers' % (args.runs, summary['affinity'], args.workers))
    print('%-5s %-28s | %-28s | %-28s | %s' % ('i', 'tags', 'Python+engine warm', 'C# server warm', 'cold median Python / C#'))
    for r in summary['requests']:
        print('%-5d %-28s | %-28s | %-28s | %.3f / %.3f  depth %s/%s' % (
            r['i'], ','.join(r['tags'])[:28], '%.4f / %.4f / %.4f' % r['python']['warm'], '%.4f / %.4f / %.4f' % r['csharp']['warm'],
            r['python']['cold'][1], r['csharp']['cold'][1], r['depth']['python'], r['depth']['csharp']))


if __name__ == '__main__':
    main()
