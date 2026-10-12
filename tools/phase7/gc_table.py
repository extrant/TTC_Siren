# -*- coding: utf-8 -*-
"""
Table of the Server GC / workstation GC comparison written by tools/phase7/gc_compare.py (review L7-1).

  python tools/phase7/gc_table.py docs/migration/reports/phase7/gc > docs/migration/reports/phase7/gc/gc_table.txt

Per CPU set, worker count, category and concurrency: throughput (requests/s) and p95 latency (s) as median [min-max] over the runs for each
GC mode, the workstation/server ratio of the medians, and the engine's working set after the load. Cells where a run reported errors or
busy answers are marked, because their numbers are not comparable. Only comparisons made inside this one session are meaningful (L7-2).
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


def fmt(values, digits):
    values = sorted(values)
    return '%.*f [%.*f-%.*f]' % (digits, statistics.median(values), digits, values[0], digits, values[-1])


def main():
    directory = sys.argv[1]
    cells = defaultdict(lambda: defaultdict(list))      # (cpu, workers, category, conc) -> gc -> [row]
    memory = defaultdict(lambda: defaultdict(list))     # (cpu, workers) -> gc -> [MiB]
    runs = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(directory, 'gc_*_run*.jsonl'))):
        m = re.match(r'gc_(server|workstation)_(\w+)_run(\d+)\.jsonl', os.path.basename(path))
        gc, cpu, run = m.group(1), m.group(2), int(m.group(3))
        runs[(cpu, gc)].add(run)
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            workers = int(re.search(r'workers=(\d+)', r['label']).group(1))
            mask = re.search(r'cpu=(\S+(?: \(0x[0-9A-F]+\))?)', r['label']).group(1)
            if r.get('memory'):
                memory[(cpu, mask, workers)][gc].append(r['working_set_bytes'] / 2 ** 20)
            else:
                cells[(cpu, mask, workers, r['category'], r['concurrency'])][gc].append(r)
    print('runs per (cpu set, GC):', {'%s/%s' % k: sorted(v) for k, v in sorted(runs.items())})
    for key in sorted(cells, key=lambda k: (k[0], k[2], ['opening', 'mid', 'robust'].index(k[3]), k[4])):
        cpu, mask, workers, category, conc = key
        by = cells[key]
        if len(by) < 2:
            continue
        s, w = by['server'], by['workstation']
        bad = any(r['errors'] or r.get('busy') for r in s + w)
        tp_s, tp_w = statistics.median(r['throughput_rps'] for r in s), statistics.median(r['throughput_rps'] for r in w)
        print('%-4s %-22s w=%-2d %-7s c=%-2d | throughput server %-26s workstation %-26s ws/server %.3f | p95 server %-22s workstation %-22s%s' % (
            cpu, mask, workers, category, conc, fmt([r['throughput_rps'] for r in s], 1), fmt([r['throughput_rps'] for r in w], 1),
            tp_w / tp_s, fmt([r['p95_s'] for r in s], 4), fmt([r['p95_s'] for r in w], 4),
            '   (errors/busy in some run)' if bad else ''))
    print()
    for key in sorted(memory):
        cpu, mask, workers = key
        by = memory[key]
        print('memory %-4s %-22s workers=%-2d working set MiB: server %s  workstation %s' % (
            cpu, mask, workers, fmt(by['server'], 0), fmt(by['workstation'], 0)))


if __name__ == '__main__':
    main()
