# -*- coding: utf-8 -*-
"""
Markdown tables from the raw measurement files of Phase 8A (docs/reports/phase8a): the HTTP latency comparison and the load tests.

  python tools/phase8a/render_tables.py
"""
import json
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
REPORTS = os.path.join(HERE, '..', '..', 'docs', 'reports', 'phase8a')


def latency():
    path = os.path.join(REPORTS, 'latency_p_cores.json')
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        data = json.load(f)['summary']
    print('### /ai_move latency over HTTP, seconds (min / median / max of %d independent runs), CPU set %s, %d search workers\n' % (data['runs'], data['affinity'], data['workers']))
    print('| request (corpus #) | empty cells | depth | Python server + engine, warm | C# server, warm | cold median Python / C# |')
    print('|---|---|---|---|---|---|')
    for r in data['requests']:
        k = next((t[2:] for t in r['tags'] if t.startswith('k:')), '?')
        empties = 9 - int(k) if k.isdigit() else '?'
        print('| #%d | %s | %s | %.4f / %.4f / %.4f | %.4f / %.4f / %.4f | %.3f / %.3f |' % (
            r['i'], empties, r['depth']['csharp'], *r['python']['warm'], *r['csharp']['warm'], r['python']['cold'][1], r['csharp']['cold'][1]))
    print()


def load():
    rows = []
    for name in ('load_p_cores.jsonl', 'load_all_cpus.jsonl'):
        path = os.path.join(REPORTS, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                rows.extend(json.loads(l) for l in f if l.strip())
    labels = []
    for r in rows:
        key = (r['target'], r['cpu'], r['workers'])
        if key not in labels:
            labels.append(key)
    for target, cpu, workers in labels:
        print('### Load test: %s, CPU set %s, %d search workers (requests per second; p50 / p95 / p99 latency in seconds)\n' % (
            'C# server (key, limits, game log on)' if target == 'server' else 'Python Flask + C# engine process', cpu, workers))
        print('| category | concurrency | requests/s | p50 | p95 | p99 | max | non-200 |')
        print('|---|---|---|---|---|---|---|---|')
        for r in rows:
            if (r['target'], r['cpu'], r['workers']) == (target, cpu, workers):
                print('| %s | %d | %.1f | %.3f | %.3f | %.3f | %.3f | %d |' % (r['category'], r['concurrency'], r['rps'], r['p50'], r['p95'], r['p99'], r['max'], r['errors']))
        print()


if __name__ == '__main__':
    latency()
    load()
