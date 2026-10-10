# -*- coding: utf-8 -*-
"""
Markdown tables for docs/reports/PHASE7_INTEGRATION.md from the measurement files (so no number is typed by hand).

  python tools/phase7/render_tables.py latency docs/reports/phase7/latency_p_cores.json
  python tools/phase7/render_tables.py load docs/reports/phase7/load_all_cpus.jsonl
"""
import collections
import json
import sys


def latency(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    s = data['summary']
    print('CPU set: %s, %d independent runs. Seconds, min / median / max.\n' % (s['affinity'], s['runs']))
    print('| request | tags | Python solver, warm | engine path, warm | Python, cold (median) | engine path, cold (median) | depth Python / engine |')
    print('|---|---|---|---|---|---|---|')

    def fmt(t, digits=4):
        return ' / '.join(('%.' + str(digits) + 'f') % x for x in t)
    for r in s['requests']:
        tags = ', '.join(r['tags'])
        print('| %d | %s | %s | %s | %.3f | %.3f | %s / %s |' % (
            r['i'], tags, fmt(r['python']['warm']), fmt(r['engine']['warm']), r['python']['cold'][1], r['engine']['cold'][1],
            r['depth']['python'], r['depth']['engine']))


def load(*paths):
    rows = []
    memory = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                (memory if row.get('memory') else rows).append(row)
    by = collections.defaultdict(dict)
    for r in rows:
        key = (r['target'], r['category'], r['label'])
        by[key][r['concurrency']] = r
    labels = collections.OrderedDict()
    for (target, category, label), cells in by.items():
        labels.setdefault((target, category), []).append((label, cells))
    for (target, category), items in labels.items():
        concs = sorted({c for _, cells in items for c in cells})
        print('\n**%s, %s requests** (requests/s; p95 latency in s)\n' % ('engine directly' if target == 'engine' else 'Flask + engine', category))
        print('| %s | %s |' % ('configuration', ' | '.join('c=%d' % c for c in concs)))
        print('|---|%s|' % '|'.join('---' for _ in concs))
        for label, cells in items:
            parts = []
            for c in concs:
                r = cells.get(c)
                if r is None:
                    parts.append('')
                else:
                    extra = ''
                    if r.get('busy'):
                        extra += ' busy=%d' % r['busy']
                    if r.get('errors'):
                        extra += ' err=%d' % r['errors']
                    if target == 'flask':
                        extra += ' fb=%d' % r.get('fallbacks', 0)
                    parts.append('%.1f / %.3f%s' % (r['throughput_rps'], r['p95_s'], extra))
            print('| %s | %s |' % (label, ' | '.join(parts)))
    if memory:
        print('\n**Engine process memory after the load**\n')
        print('| configuration | working set MiB | private MiB | peak working set MiB |')
        print('|---|---|---|---|')
        for m in memory:
            print('| %s | %.0f | %.0f | %.0f |' % (m['label'], m['working_set_bytes'] / 2**20, m['private_bytes'] / 2**20, m['peak_working_set_bytes'] / 2**20))


if __name__ == '__main__':
    {'latency': latency, 'load': load}[sys.argv[1]](*sys.argv[2:])
