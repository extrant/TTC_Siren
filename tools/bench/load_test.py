# -*- coding: utf-8 -*-
"""
Load test for the C# engine and the full Flask server (Phase 7). Standard library only.

  # (a) the engine directly: the payloads the Python server would send (tools/phase7/e2e_reference.py output)
  python tools/bench/load_test.py --target engine --url http://127.0.0.1:5300 --concurrency 1,2,4,8 --category robust --requests 24
  # (b) the whole Flask ai_server with the engine switched on (start it with tools/phase7/serve_flask.py)
  python tools/bench/load_test.py --target flask --url http://127.0.0.1:5001 --concurrency 1,4,8 --category all --requests 24

Requests are replayed with production settings by default (max_depth 10, max_time 10 s). ``--category``: opening (empty board), mid
(1-4 cards on the board), robust (the robust endgame is active: 5+ cards, at most two unknown opponent cards), all.
Per concurrency level it reports throughput (requests/s), latency p50/p95/p99/max, the number of non-engine answers (fallbacks) and, for the
Flask target, the Python-side time per request (latency minus the engine's own total time). With ``--check`` it also verifies that the
answers do not depend on the concurrency (the engine is deterministic: the normalised response must be identical at every level;
for the Flask target only requests without unknown cards are compared, because the unknown-card sampling uses Python's shared
random generator).

The numbers say how THIS machine behaves with THIS CPU set; they are not an estimate for another machine.
"""
import argparse
import gzip
import http.client
import json
import os
import statistics
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_CORPUS = os.path.join(ROOT, 'tools', 'phase7', 'corpus', 'corpus_s20261010_n360.json')
DEFAULT_REFERENCE = os.path.join(ROOT, 'tools', 'parity', 'golden', 'e2e_reference_g1_s20261010_n360.jsonl.gz')


def category_of(tags):
    tagset = set(tags)
    if 'opening' in tagset:
        return 'opening'
    if 'robust_candidate' in tagset:
        return 'robust'
    if any(t in tagset for t in ('k:1', 'k:2', 'k:3', 'k:4')):
        return 'mid'
    return 'other'


def load_flask_requests(corpus_path, category, max_depth, max_time):
    with open(corpus_path, encoding='utf-8') as f:
        entries = json.load(f)['entries']
    out = []
    for e in entries:
        cat = category_of(e['tags'])
        if category != 'all' and cat != category:
            continue
        if cat == 'other' or 'full_board' in e['tags']:
            continue
        req = dict(e['request'])
        req['max_depth'] = max_depth
        req['max_time'] = max_time
        out.append({'i': e['i'], 'tags': e['tags'], 'body': req, 'deterministic': not any(t in e['tags'] for t in ('unknown:1', 'unknown:2', 'unknown:3+'))})
    return out


def load_engine_requests(reference_path, category, max_depth, max_time):
    out = []
    seen = set()
    with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            rec = json.loads(line)
            if rec['i'] in seen:
                continue
            seen.add(rec['i'])
            cat = category_of(rec['tags'])
            if category != 'all' and cat != category:
                continue
            if cat == 'other' or 'full_board' in rec['tags']:
                continue
            payload = rec['payload_with_candidates'] if rec.get('expects_fallback') and rec.get('payload_with_candidates') else rec['payload']
            payload = dict(payload)
            payload['maxDepth'] = max_depth
            payload['maxTime'] = max_time
            out.append({'i': rec['i'], 'tags': rec['tags'], 'body': payload, 'deterministic': True})
    return out


def normalise_engine(body):
    body = json.loads(json.dumps(body))
    body.pop('timings', None)
    body.pop('worker', None)
    for d in body.get('depths', []) or []:
        d.pop('elapsed', None)
    if isinstance(body.get('totals'), dict):
        body['totals'].pop('elapsed', None)
    if isinstance(body.get('robust'), dict):
        body['robust'].pop('elapsed', None)
    return json.dumps(body, sort_keys=True)


def digest_flask(body):
    keep = {k: body.get(k) for k in ('card', 'card_id', 'pos', 'move', 'engine')}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


def run_level(target, url, requests, concurrency, total, timeout):
    parsed = urllib.parse.urlsplit(url)
    path = '/v1/solve' if target == 'engine' else '/ai_move'
    work = [requests[k % len(requests)] for k in range(total)]
    lock = threading.Lock()
    cursor = [0]
    results = []

    def worker():
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        while True:
            with lock:
                if cursor[0] >= len(work):
                    break
                k = cursor[0]
                cursor[0] += 1
            item = work[k]
            data = json.dumps(item['body'], ensure_ascii=False).encode('utf-8')
            t0 = time.perf_counter()
            status = 0
            body = None
            try:
                conn.request('POST', path, body=data, headers={'Content-Type': 'application/json'})
                response = conn.getresponse()
                raw = response.read()
                status = response.status
                body = json.loads(raw.decode('utf-8'))
            except Exception as e:  # noqa: BLE001
                body = {'error': repr(e)}
                conn.close()
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
            latency = time.perf_counter() - t0
            with lock:
                results.append({'i': item['i'], 'latency': latency, 'status': status, 'body': body, 'deterministic': item['deterministic']})
        conn.close()

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    return results, wall


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, max(0, int(round(p * (len(sorted_values) - 1)))))
    return sorted_values[k]


def summarise(target, results, wall, concurrency):
    lat = sorted(r['latency'] for r in results)
    ok = [r for r in results if r['status'] == 200 and isinstance(r['body'], dict)]
    row = {
        'concurrency': concurrency, 'requests': len(results), 'wall_s': round(wall, 3),
        'throughput_rps': round(len(results) / wall, 3) if wall > 0 else 0,
        'p50_s': round(percentile(lat, 0.50), 4), 'p95_s': round(percentile(lat, 0.95), 4),
        'p99_s': round(percentile(lat, 0.99), 4), 'max_s': round(lat[-1], 4) if lat else 0,
        'errors': len(results) - len(ok),
    }
    if target == 'engine':
        row['busy'] = sum(1 for r in results if r['status'] == 503)
        eng = [r['body'].get('timings', {}).get('totalMs', 0.0) / 1000.0 for r in ok]
        row['engine_p50_s'] = round(statistics.median(eng), 4) if eng else 0
    else:
        csharp = [r for r in ok if r['body'].get('engine') == 'csharp']
        row['engine_answers'] = len(csharp)
        row['fallbacks'] = sum(1 for r in ok if r['body'].get('engine') == 'python')
        reasons = {}
        for r in ok:
            if r['body'].get('engine') == 'python':
                reasons[r['body'].get('fallback_reason')] = reasons.get(r['body'].get('fallback_reason'), 0) + 1
        row['fallback_reasons'] = reasons
        overhead = []
        for r in csharp:
            engine_ms = (r['body'].get('engine_info') or {}).get('timings', {}) or {}
            overhead.append(r['latency'] - engine_ms.get('totalMs', 0.0) / 1000.0)
        row['python_overhead_p50_s'] = round(statistics.median(overhead), 4) if overhead else 0
        row['python_overhead_p95_s'] = round(percentile(sorted(overhead), 0.95), 4) if overhead else 0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=['engine', 'flask'], required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--concurrency', default='1,2,4,8')
    ap.add_argument('--category', default='all', choices=['opening', 'mid', 'robust', 'all'])
    ap.add_argument('--requests', type=int, default=0, help='requests per concurrency level (default: max(8, 3 x concurrency))')
    ap.add_argument('--max-depth', type=int, default=10)
    ap.add_argument('--max-time', type=float, default=10.0)
    ap.add_argument('--corpus', default=DEFAULT_CORPUS)
    ap.add_argument('--reference', default=DEFAULT_REFERENCE)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--check', action='store_true', help='verify that the answers do not depend on the concurrency')
    ap.add_argument('--label', default='')
    ap.add_argument('--json-out')
    args = ap.parse_args()

    if args.target == 'engine':
        requests = load_engine_requests(args.reference, args.category, args.max_depth, args.max_time)
    else:
        requests = load_flask_requests(args.corpus, args.category, args.max_depth, args.max_time)
    if not requests:
        raise SystemExit('no requests for category %s' % args.category)

    levels = [int(x) for x in args.concurrency.split(',')]
    print('# load test target=%s url=%s category=%s distinct requests=%d max_depth=%d max_time=%g %s' %
          (args.target, args.url, args.category, len(requests), args.max_depth, args.max_time, args.label), flush=True)
    rows = []
    reference_digests = {}
    changed = []
    for c in levels:
        total = args.requests or max(8, 3 * c)
        results, wall = run_level(args.target, args.url, requests, c, total, args.timeout)
        row = summarise(args.target, results, wall, c)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.check:
            for r in results:
                if r['status'] != 200 or not isinstance(r['body'], dict) or not r['deterministic']:
                    continue
                d = normalise_engine(r['body']) if args.target == 'engine' else digest_flask(r['body'])
                if r['i'] in reference_digests and reference_digests[r['i']] != d:
                    changed.append((r['i'], c))
                reference_digests.setdefault(r['i'], d)
    if args.check:
        print('# determinism check over %d distinct requests: %s' % (len(reference_digests), 'IDENTICAL at every concurrency' if not changed
                                                                      else 'CHANGED for %r' % changed[:10]))
    if args.json_out:
        with open(args.json_out, 'a', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(dict(row, target=args.target, category=args.category, label=args.label), ensure_ascii=False) + '\n')
    return 1 if changed else 0


if __name__ == '__main__':
    sys.exit(main())
