# -*- coding: utf-8 -*-
"""
Phase 8B: plugin latency with and without a flood of anonymous work (decision A-27), on this machine with a pinned CPU set.

  python tools/phase8b/tier_load.py --cpu auto --workers 12 --runs 3 --out docs/reports/phase8b/tiers.json

Builds tools/phase8b/TierLoad (Release) and runs it pinned (default: the P-cores, like the other measurements), several times; prints the table of every run and
writes the raw JSON of each run next to --out. See tools/phase8b/TierLoad/Program.cs for what the three phases are.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
import pin  # noqa: E402

PROJECT = os.path.join(HERE, 'TierLoad', 'TierLoad.csproj')
DLL = os.path.join(HERE, 'TierLoad', 'bin', 'Release', 'net10.0', 'TierLoad.dll')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cpu', default='auto')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--flood', type=int, default=48)
    ap.add_argument('--plugin-rounds', type=int, default=3)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    subprocess.run(['dotnet', 'build', PROJECT, '-c', 'Release', '--nologo', '-v', 'q'], check=True, cwd=ROOT)
    mask = None if args.cpu == 'all' else pin.parse_mask(args.cpu)
    runs = []
    for run in range(args.runs):
        out = os.path.splitext(args.out)[0] + '.run%d.json' % run
        cmd = ['dotnet', DLL, '--workers', str(args.workers), '--flood', str(args.flood), '--plugin-rounds', str(args.plugin_rounds), '--out', out]
        print('=== run %d, cpu %s' % (run, pin.describe(mask) if mask is not None else 'all'), flush=True)
        if mask is None:
            subprocess.run(cmd, check=True, cwd=ROOT)
        else:
            proc = pin.start_pinned(cmd, mask, cwd=ROOT)
            if proc.wait() != 0:
                raise SystemExit('TierLoad failed')
        with open(out, encoding='utf-8') as f:
            runs.append(json.load(f))

    def med(values):
        return statistics.median(values)

    def stats(run, phase, kind):
        v = sorted(run[phase]['latenciesMs'][kind])
        if not v:
            return None
        return med(v), v[int(round(0.95 * (len(v) - 1)))], v[-1]
    print()
    print('Medians over %d runs (ms): p50 / p95 / max of the plugin requests, per kind' % len(runs))
    print('%-40s %-8s %10s %10s %10s' % ('phase', 'kind', 'p50', 'p95', 'max'))
    titles = {'A': 'A  quiet', 'B': 'B  anonymous flood, access classes', 'C': 'C  control: same flood, no classes'}
    for phase in 'ABC':
        for kind in ('mid', 'robust', 'opening'):
            rows = [x for x in (stats(r, phase, kind) for r in runs) if x]
            if not rows:
                continue
            print('%-40s %-8s %10.1f %10.1f %10.1f' % (titles[phase], kind, med([r[0] for r in rows]), med([r[1] for r in rows]), med([r[2] for r in rows])))
    for phase in 'BC':
        fl = [r[phase]['flood'] for r in runs]
        print('%s flood: answered %s busy %s, max anonymous running %s, max anonymous queued %s, max plugin queued %s' % (
            phase, [f['answered'] for f in fl], [f['busy'] for f in fl], [f['maxRunningAnonymous'] for f in fl], [f['maxQueuedAnonymous'] for f in fl], [f['maxQueuedPlugin'] for f in fl]))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'cpu': pin.describe(mask) if mask is not None else 'all', 'workers': args.workers, 'flood': args.flood, 'runs': runs}, f, indent=1)


if __name__ == '__main__':
    main()
