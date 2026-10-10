# -*- coding: utf-8 -*-
"""
Single-request latency, Python solver vs C# engine, same requests, production settings (max_depth 10, max_time 10 s). Phase 7.

  python tools/phase7/latency.py --affinity auto --runs 3 --out docs/reports/phase7/latency.json

Protocol (goal 4.6, CLAUDE.md section 6): every independent run starts FRESH processes pinned to the same CPU set (the mask is checked and
printed): the Python solver (``TTC_CS_ENGINE=off``) in one process, and the engine path in a second process together with a freshly started
engine. Each process serves the same request list three times in the same order: pass 1 is COLD (first request of its kind in a new process:
JIT, first-use allocations, CSV loading), passes 2 and 3 are WARM. Requests go through ``ai_server.ai_move`` in process (Flask's test client) with
``random.seed(entry seed)``, so unknown-card sampling gives the same position in both modes. The reported figures are min / median / max over the
independent runs (cold: pass 1; warm: median of passes 2-3 of each run).

Requests: 3 openings, 3 mid-games, 6 robust-endgame requests from tools/phase7/corpus (tags in the output).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
sys.path.insert(0, HERE)

ENGINE_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Engine', 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')


def select_requests(entries):
    """A fixed, tagged selection: 3 openings, 3 mid-games, 6 robust-endgame requests."""
    def first(n, pred, taken):
        out = []
        for e in entries:
            if e['i'] in taken or 'full_board' in e['tags']:
                continue
            if pred(set(e['tags'])):
                out.append(e)
                taken.add(e['i'])
                if len(out) == n:
                    break
        return out
    taken = set()
    chosen = first(3, lambda t: 'opening' in t, taken)
    chosen += first(1, lambda t: 'k:2' in t and 'robust_candidate' not in t, taken)
    chosen += first(1, lambda t: 'k:3' in t, taken)
    chosen += first(1, lambda t: 'k:4' in t, taken)
    chosen += first(2, lambda t: 'robust_candidate' in t and 'k:5' in t and 'unknown:0' in t, taken)
    chosen += first(2, lambda t: 'robust_candidate' in t and 'k:6' in t and ('unknown:1' in t or 'unknown:2' in t), taken)
    chosen += first(2, lambda t: 'robust_candidate' in t and 'k:7' in t, taken)
    return chosen


def worker(args):
    """Runs inside a pinned process: three passes over the request list, prints one JSON line per request and pass."""
    import random
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    os.environ['TTC_CS_ENGINE'] = 'on' if args.mode == 'engine' else 'off'
    if args.mode == 'engine':
        os.environ['TTC_CS_ENGINE_URL'] = args.engine_url
    import ai_server
    import gen_corpus

    class NullUi:
        def __getattr__(self, name):
            return lambda *a, **k: None
    ai_server.console_ui = NullUi()
    try:
        import pin
        affinity = pin.describe(pin.current_affinity())
    except Exception:  # noqa: BLE001
        affinity = '?'
    entries = gen_corpus.generate(20261010, 360)
    chosen = select_requests(entries)
    client = ai_server.app.test_client()
    print(json.dumps({'header': True, 'mode': args.mode, 'affinity': affinity, 'requests': [{'i': e['i'], 'tags': e['tags']} for e in chosen]}), flush=True)
    for pass_no in (1, 2, 3):
        for e in chosen:
            request = dict(e['request'])
            request['max_depth'] = 10
            request['max_time'] = 10
            random.seed(e['seed'])
            t0 = time.perf_counter()
            response = client.post('/ai_move', json=request)
            took = time.perf_counter() - t0
            body = response.get_json() or {}
            row = {'pass': pass_no, 'i': e['i'], 'seconds': took, 'status': response.status_code, 'engine': body.get('engine', 'python-only'),
                   'card_id': body.get('card_id'), 'pos': body.get('pos'),
                   'depth': (body.get('performance_stats') or {}).get('search_depth'),
                   'nodes': (body.get('performance_stats') or {}).get('nodes_searched')}
            timings = (body.get('engine_info') or {}).get('timings') or {}
            if timings:
                row['engine_ms'] = timings.get('totalMs')
            print(json.dumps(row), flush=True)


def wait_health(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + '/v1/health', timeout=2):
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit('engine did not start')


def run_worker_pinned(mode, mask, engine_url):
    import pin
    cmd = [sys.executable, os.path.abspath(__file__), 'worker', '--mode', mode, '--engine-url', engine_url]
    p = pin.run_pinned(cmd, mask, cwd=ROOT, timeout=3600)
    if p.returncode != 0:
        raise SystemExit('worker failed:\n' + p.stdout[-2000:] + p.stderr[-2000:])
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]
    header = next(l for l in lines if l.get('header'))
    if mask is not None and ('0x%X' % mask) not in header['affinity']:
        raise SystemExit('worker ran with %s, requested 0x%X' % (header['affinity'], mask))
    return header, [l for l in lines if not l.get('header')]


def orchestrate(args):
    import pin
    mask = None if args.affinity == 'all' else pin.parse_mask(args.affinity)
    port = args.port
    url = 'http://127.0.0.1:%d' % port
    runs = []
    header = None
    for run in range(args.runs):
        print('# run %d/%d' % (run + 1, args.runs), flush=True)
        h_py, rows_py = run_worker_pinned('python', mask, url)
        engine = pin.start_pinned(['dotnet', ENGINE_DLL, '--port', str(port), '--workers', '4'], mask, cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health(url)
            h_en, rows_en = run_worker_pinned('engine', mask, url)
        finally:
            engine.kill()
            engine.wait()
        header = h_py
        runs.append({'python': rows_py, 'engine': rows_en})
    summary = summarise(header, runs)
    summary['affinity'] = pin.describe(mask) if mask is not None else 'all CPUs'
    summary['runs'] = args.runs
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'raw': runs}, f, ensure_ascii=False)
    print_table(summary)


def spread(values):
    return min(values), statistics.median(values), max(values)


def summarise(header, runs):
    reqs = header['requests']
    out = {'requests': []}
    for idx, req in enumerate(reqs):
        row = {'i': req['i'], 'tags': [t for t in req['tags'] if t.startswith(('k:', 'unknown', 'rule:'))][:4]}
        for mode in ('python', 'engine'):
            cold = [r['seconds'] for run in runs for r in run[mode] if r['i'] == req['i'] and r['pass'] == 1]
            warm = [statistics.median([r['seconds'] for r in run[mode] if r['i'] == req['i'] and r['pass'] in (2, 3)]) for run in runs]
            row[mode] = {'cold': spread(cold), 'warm': spread(warm)}
        row['engine_answers'] = all(r['engine'] == 'csharp' for run in runs for r in run['engine'] if r['i'] == req['i'])
        row['same_move'] = all((a['card_id'], a['pos']) == (b['card_id'], b['pos'])
                               for run in runs for a in run['python'] for b in run['engine'] if a['i'] == b['i'] == req['i'] and a['pass'] == b['pass'] == 3)
        row['depth'] = {m: runs[0][m][len(reqs) * 2 + idx].get('depth') for m in ('python', 'engine')}
        out['requests'].append(row)
    return out


def print_table(summary):
    def fmt(t):
        return '%.4f / %.4f / %.4f' % t
    print('\n# single-request latency, s, min / median / max over %d runs, CPU set: %s' % (summary['runs'], summary['affinity']))
    print('%-5s %-28s | %-33s | %-33s | %s' % ('i', 'tags', 'Python warm', 'engine path (Flask + engine) warm', 'cold: Python / engine (median)'))
    for r in summary['requests']:
        print('%-5d %-28s | %-33s | %-33s | %.3f / %.3f   depth %s/%s%s' % (
            r['i'], ','.join(r['tags'])[:28], fmt(r['python']['warm']), fmt(r['engine']['warm']), r['python']['cold'][1], r['engine']['cold'][1],
            r['depth']['python'], r['depth']['engine'], '' if r['engine_answers'] else '  (engine did not answer!)'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    w = sub.add_parser('worker')
    w.add_argument('--mode', choices=['python', 'engine'], required=True)
    w.add_argument('--engine-url', default='http://127.0.0.1:5399')
    ap.add_argument('--affinity', default='auto')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--port', type=int, default=5399)
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.cmd == 'worker':
        worker(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()
