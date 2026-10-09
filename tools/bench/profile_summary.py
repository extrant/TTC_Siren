# -*- coding: utf-8 -*-
"""
Summarises a dotnet-trace Speedscope file (Phase 6 profiling).

  python tools/bench/profile_summary.py trace.speedscope.json [--top 25] [--filter TTC.Siren]

Takes the sampled thread with the most samples (the benchmark thread), counts for every frame the samples that contain it
(inclusive) and the samples where it is the leaf (exclusive), and prints both as a percentage of that thread's samples.
Methods marked AggressiveInlining are attributed to their callers by the JIT, so they do not appear on their own.

Collect a trace on the P-cores (both the tool and its child inherit the mask):
  python tools/bench/profile_summary.py --collect fast --out profile_fast
which runs: dotnet-trace collect --profile dotnet-sampled-thread-time,dotnet-common --format Speedscope -- Bench.exe <mode> ...
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import pin  # noqa: E402

BENCH_EXE = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Bench', 'bin', 'Release', 'net10.0', 'TTC.Siren.Bench.exe')
PSEUDO_FRAMES = {'CPU_TIME', 'UNMANAGED_CODE_TIME', 'BLOCKED_TIME'}
WORKLOADS = {
    'fast': ['loop', 'fast', '12'],
    'compat': ['loop', 'compat', '12'],
    'exact': ['loop', 'exact', '12'],
}


def summarise(path, top, name_filter):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    frames = [fr['name'] for fr in data['shared']['frames']]
    profiles = data['profiles']
    inclusive = collections.Counter()
    exclusive = collections.Counter()
    if profiles[0].get('type') == 'evented':
        # dotnet-trace writes open/close events with a timestamp; attribute the time between two events to the stack
        best = max(profiles, key=lambda p: len(p['events']))
        stack = []
        prev = best['startValue']
        for ev in best['events']:
            dt = ev['at'] - prev
            if dt > 0 and stack:
                # the last frame is a pseudo frame (CPU_TIME / UNMANAGED_CODE_TIME): the method is the frame below it
                real = [fr for fr in stack if frames[fr] not in PSEUDO_FRAMES]
                if real:
                    exclusive[frames[real[-1]]] += dt
                for fr in set(real):
                    inclusive[frames[fr]] += dt
            prev = ev['at']
            if ev['type'] == 'O':
                stack.append(ev['frame'])
            else:
                # close: pop down to (and including) that frame
                while stack and stack.pop() != ev['frame']:
                    pass
        total = sum(exclusive.values())
    else:
        best = max(profiles, key=lambda p: len(p.get('samples', [])))
        samples = best['samples']
        total = len(samples)
        for stack in samples:
            if not stack:
                continue
            exclusive[frames[stack[-1]]] += 1
            for name in {frames[i] for i in stack}:
                inclusive[name] += 1
    print('# %s: thread %r, %.0f %s attributed' % (os.path.basename(path), best.get('name'), total, best.get('unit', 'samples')))

    def show(title, counter):
        print('\n## %s' % title)
        shown = 0
        for name, n in counter.most_common():
            if name_filter and name_filter not in name:
                continue
            print('%6.1f%%  %s' % (100.0 * n / total, name))
            shown += 1
            if shown >= top:
                break

    show('inclusive (samples containing the method)', inclusive)
    show('exclusive (leaf)', exclusive)


def collect(workload, out):
    mask = pin.performance_core_mask()
    cmd = ['dotnet-trace', 'collect', '--profile', 'dotnet-sampled-thread-time,dotnet-common', '--format', 'Speedscope',
           '-o', out, '--', BENCH_EXE] + WORKLOADS[workload]
    print('running pinned to 0x%X: %s' % (mask, ' '.join(cmd)), flush=True)
    p = pin.run_pinned(cmd, mask, cwd=ROOT, timeout=3600)
    sys.stdout.write(p.stdout[-1500:])
    sys.stderr.write(p.stderr[-1500:])
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trace', nargs='?')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--filter', default='TTC.Siren')
    ap.add_argument('--collect', choices=sorted(WORKLOADS))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.collect:
        out = args.out or ('profile_' + args.collect)
        collect(args.collect, out)
        matches = sorted(glob.glob(out + '*.speedscope.json'), key=os.path.getmtime)
        if not matches:
            raise SystemExit('no speedscope file written next to ' + out)
        args.trace = matches[-1]
    if not args.trace:
        ap.error('give a trace file or --collect <fast|compat|exact>')
    summarise(args.trace, args.top, args.filter)


if __name__ == '__main__':
    main()
