# -*- coding: utf-8 -*-
"""
C#-only pinned benchmark runner for Phase 6 (CLAUDE.md section 6: Baseline -> Modify -> Build -> Test -> Benchmark -> Compare).

Runs the built TTC.Siren.Bench.exe once per process run, N process runs per mode, every run pinned to the same CPU set
(tools/bench/pin.py; the P-core set by default, review A-7), and aggregates min / median / max over the process runs.
Unpinned numbers are not evidence on this hybrid CPU, so the mask is mandatory (--affinity auto|<hex>).

Per mode it records
  timings   the RESULT lines of the harness (search total, fast on/off/fast totals, exact per item, endgame, make/undo per scenario)
  signature the deterministic SIG/RESULT fields (node counts, score bits, moves, values, checksums, allocation) with no timing in
            them. The signature must be identical in every process run (checked) and, with --compare, identical to the baseline:
            that is the "node counts, scores and moves unchanged" invariant of a pure performance change.

Usage
  python tools/bench/run_cs_pinned.py --modes search,fast,exact,endgame,makeundo --runs 3 --affinity auto [--cold]
         [--save out.json] [--compare baseline.json] [--label text]
Build first: dotnet build csharp/TTC.Siren.sln -c Release
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import pin  # noqa: E402

BENCH_EXE = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Bench', 'bin', 'Release', 'net10.0',
                         'TTC.Siren.Bench.exe' if os.name == 'nt' else 'TTC.Siren.Bench')
EXE = [BENCH_EXE]  # --exe overrides it (A/B comparison of two builds)
POSITIONS = os.path.join(ROOT, 'tools', 'bench', 'endgame_bench_positions.json')
BENCH_ARGS = {'search': ['search', '5'], 'fast': ['fast', '3'], 'exact': ['exact', '5'],
              'endgame': ['endgame', POSITIONS, '5'], 'makeundo': ['makeundo', '2000000', '5']}


def kv(line, skip):
    return dict(tok.split('=', 1) for tok in line.split()[skip:] if '=' in tok)


def run_once(mode, mask, cold):
    cmd = [EXE[0]] + BENCH_ARGS[mode] + (['--cold'] if cold else [])
    p = pin.run_pinned(cmd, mask, cwd=ROOT, timeout=3600)
    if p.returncode != 0:
        raise SystemExit('Bench %s failed (%d)\n%s\n%s' % (mode, p.returncode, p.stdout[-1500:], p.stderr[-1500:]))
    affinity = None
    for line in p.stdout.splitlines()[:4]:
        if 'affinity=' in line:
            affinity = int(line.split('affinity=')[1].split()[0], 16)
    if mask is not None and affinity != mask:
        raise SystemExit('Bench %s ran with affinity %s, requested 0x%X' % (mode, hex(affinity) if affinity else None, mask))
    timings = {}
    signature = []
    for line in p.stdout.splitlines():
        if line.startswith('SIG '):
            signature.append(line)
        elif line.startswith('RESULT '):
            parts = line.split()
            fields = kv(line, 2)
            kind = parts[1]
            if kind == 'search':
                timings['total'] = float(fields['total_median_s'])
                signature.append('SIG search-total nodes=%s' % fields['nodes'])
            elif kind == 'fast':
                for k in ('on', 'off', 'fast'):
                    timings[k] = float(fields[k + '_median_s'])
                signature.append('SIG fast-total nodes=%s/%s/%s identical=%s different=%s' % (
                    fields['nodes_on'], fields['nodes_off'], fields['nodes_fast'], fields['identical'], fields['different']))
            elif kind == 'exact':
                timings[fields['item']] = float(fields['median_s'])
                signature.append('SIG exact item=%s value=%s make_moves=%s' % (fields['item'], fields['value'], fields['make_moves']))
            elif kind == 'endgame':
                timings['total'] = float(fields['cs_median_s'])
                signature.append('SIG endgame same=%s different=%s' % (fields['same'], fields['different']))
            elif kind == 'makeundo':
                timings[fields['item']] = float(fields['median_s'])
                signature.append('SIG makeundo item=%s checksum=%s alloc=%s' % (fields['item'], fields['checksum'], fields['alloc']))
    if not timings:
        raise SystemExit('Bench %s printed no RESULT lines:\n%s' % (mode, p.stdout[-1500:]))
    return affinity, timings, signature


def spread(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modes', default='search,fast,exact,endgame,makeundo')
    ap.add_argument('--runs', type=int, default=3, help='independent process runs per mode (at least 3)')
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--affinity', required=True, help="hex CPU-set mask or 'auto' (the detected P-core set)")
    ap.add_argument('--label', default='')
    ap.add_argument('--exe', help='another build of TTC.Siren.Bench.exe (copy of a bin/Release/net10.0 folder), e.g. the baseline for an A/B run')
    ap.add_argument('--save')
    ap.add_argument('--compare')
    args = ap.parse_args()
    if args.runs < 3:
        ap.error('use at least 3 process runs (CLAUDE.md section 6)')
    if args.exe:
        EXE[0] = os.path.abspath(args.exe)
    if not os.path.exists(EXE[0]):
        ap.error('build first: dotnet build csharp/TTC.Siren.sln -c Release (missing %s)' % EXE[0])
    mask = pin.parse_mask(args.affinity)
    tag = 'cold' if args.cold else 'warm'
    print('# C# pinned benchmark [%s] label=%r runs=%d affinity=%s exe=%s' % (tag, args.label, args.runs, pin.describe(mask), EXE[0]), flush=True)

    result = {'label': args.label, 'mode': tag, 'affinity': '0x%X' % mask, 'runs': args.runs, 'modes': {}}
    for mode in args.modes.split(','):
        per_run = []
        sigs = []
        for r in range(args.runs):
            affinity, timings, signature = run_once(mode, mask, args.cold)
            per_run.append(timings)
            sigs.append(signature)
        for k in range(1, len(sigs)):
            if sigs[k] != sigs[0]:
                diff = [(a, b) for a, b in zip(sigs[0], sigs[k]) if a != b][:3]
                raise SystemExit('Bench %s: the deterministic output differs between process runs: %s' % (mode, diff))
        keys = list(per_run[0])
        agg = {k: spread([t[k] for t in per_run]) for k in keys}
        digest = hashlib.sha256('\n'.join(sigs[0]).encode()).hexdigest()[:16]
        result['modes'][mode] = {'timings': agg, 'signature_sha256_16': digest, 'signature': sigs[0]}
        shown = ('total', 'on', 'off', 'fast') if mode in ('search', 'fast', 'endgame') else tuple(keys)
        for k in keys:
            if k in shown or mode in ('exact', 'makeundo'):
                a = agg[k]
                print('%-9s %-16s min %.6f  median %.6f  max %.6f s   (spread %.1f%%)' % (
                    mode, k, a['min'], a['median'], a['max'], 100.0 * (a['max'] - a['min']) / a['median']), flush=True)
        print('%-9s signature %s (%d lines, identical in all %d runs)' % (mode, digest, len(sigs[0]), args.runs), flush=True)

    if args.compare:
        with open(args.compare, encoding='utf-8') as f:
            base = json.load(f)
        if base['mode'] != result['mode'] or base['affinity'] != result['affinity']:
            raise SystemExit('baseline was measured as %s/%s, this run is %s/%s' % (base['mode'], base['affinity'], result['mode'], result['affinity']))
        print('\n# compared with baseline %r' % base['label'])
        for mode, cur in result['modes'].items():
            if mode not in base['modes']:
                continue
            b = base['modes'][mode]
            same = b['signature'] == cur['signature']
            print('%-9s deterministic output %s' % (mode, 'IDENTICAL' if same else 'CHANGED'))
            if not same:
                changed = [(x, y) for x, y in zip(b['signature'], cur['signature']) if x != y][:5]
                for x, y in changed:
                    print('    base:', x, '\n    now :', y)
            for k, a in cur['timings'].items():
                if k in b['timings']:
                    old = b['timings'][k]['median']
                    print('%-9s %-16s median %.6f -> %.6f s  (%+.1f%%)' % (mode, k, old, a['median'], 100.0 * (a['median'] - old) / old))
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print('saved', args.save)


if __name__ == '__main__':
    main()
