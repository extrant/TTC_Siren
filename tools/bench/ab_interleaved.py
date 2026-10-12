# -*- coding: utf-8 -*-
"""
Interleaved A/B of several builds of TTC.Siren.Bench.exe, pinned to one CPU set (Phase 6).

The builds are run ALTERNATELY, sample by sample (A, B, C, A, B, C, ...), not in blocks. Cold-JIT timings are bimodal and depend on
tier-up timing, and the machine drifts over minutes, so a block A/B (all runs of A, then all runs of B) can show a 'regression' that
is only the order: a block comparison of the 22-item cold suite showed +25% for a change that is -2% when the builds alternate
(docs/migration/reports/PHASE6_OPTIMIZATION.md section 5).

  python tools/bench/ab_interleaved.py --mode search --cold --samples 25 --affinity auto name=path/to/Bench.exe name2=path2 ...
Modes: search (22 fixed-depth items), fast (Fast column), exact (mid3_ascension), endgame (72 positions). Prints min / p25 / median / p75 / max.
A build folder must sit inside a repository tree (the harness locates data/ above its own folder): use a git worktree.
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import pin  # noqa: E402

ARGS = {'search': ['search', '5'], 'fast': ['fast', '3'], 'exact': ['exact', '5'],
        'endgame': ['endgame', os.path.join(ROOT, 'tools', 'bench', 'endgame_bench_positions.json'), '5']}
KEY = {'search': 'total_median_s', 'fast': 'fast_median_s', 'exact': 'median_s', 'endgame': 'cs_median_s'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=sorted(ARGS))
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--samples', type=int, default=15)
    ap.add_argument('--affinity', required=True)
    ap.add_argument('builds', nargs='+', help='name=path/to/TTC.Siren.Bench.exe')
    a = ap.parse_args()
    mask = pin.parse_mask(a.affinity)
    exes = dict(b.split('=', 1) for b in a.builds)
    res = {k: [] for k in exes}
    for _ in range(a.samples):
        for name, exe in exes.items():
            p = pin.run_pinned([os.path.abspath(exe)] + (['--cold'] if a.cold else []) + ARGS[a.mode], mask, cwd=ROOT, timeout=1800)
            for line in p.stdout.splitlines():
                if line.startswith('RESULT ' + a.mode):
                    kv = dict(t.split('=', 1) for t in line.split()[2:] if '=' in t)
                    if a.mode == 'exact' and kv.get('item') != 'mid3_ascension':
                        continue
                    res[name].append(float(kv[KEY[a.mode]]))
    print('# %s %s, %d alternating samples, affinity %s' % (a.mode, 'cold' if a.cold else 'warm', a.samples, pin.describe(mask)))
    for name, v in res.items():
        v = sorted(v)
        print('%-18s n=%d min %.5f p25 %.5f median %.5f p75 %.5f max %.5f s' % (name, len(v), v[0], v[len(v) // 4], statistics.median(v), v[3 * len(v) // 4], v[-1]))


if __name__ == '__main__':
    main()
