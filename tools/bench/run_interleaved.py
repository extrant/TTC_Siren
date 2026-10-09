# -*- coding: utf-8 -*-
"""
Interleaved Python/C# benchmark runner (review A-2, A-3, L5-3).

Python and C# are measured alternately, round after round (Python suite 1, C# suite 1, Python suite 2, ...), so both
sides see the same machine load, instead of in separate batches. For every suite and round it prints the Python total,
the C# total and their ratio; at the end it prints min / median / max over the rounds for both sides and for the ratio.
Nothing else may run on the machine while this script runs (the figures are wall-clock times).

Suites
  fixed    the 22 fixed-depth items of CSHARP_MIGRATION.md section 3.4:  py_bench.py fixed   vs  Bench search
  exact    _solve_endgame_exact on end5_same_plus / mid4_rev_ace / mid3_ascension:
           py_endgame_bench.py --exact   vs  Bench exact
  endgame  the robust endgame pipeline on 72 deterministic positions (tools/bench/py_endgame_bench.py positions):
           py_endgame_bench.py   vs  Bench endgame   (C# also checks that it selects the same moves as Python)

The C# side runs in warm mode by default (the whole item set is warmed for 1.5 s before any timed run); --cold passes
--cold to the C# harness (one warm-up run per item: the Phase 5 report's configuration).
Build first:  dotnet build csharp/TTC.Siren.sln -c Release

CPU placement (review A-7). On a hybrid CPU Windows places a long single-thread run on a P-core or an E-core
non-deterministically; the same Python solve took 10.1-18.0 s unpinned and 23.4 s on an E-core, and .NET is 1.4-1.5x slower
on E-cores, so an unpinned interleaved run can report a ratio that is off by 1.5-2x. Use --affinity to run BOTH sides on the
same CPU set:
  --affinity auto    the P-core set, detected from the OS (GetLogicalProcessorInformationEx efficiency classes)
  --affinity FFF     an explicit hex mask (FFF = the 6 P-cores = logical processors 0-11 on the i7-13650HX of this repo;
                     FF000 = its 8 E-cores). The mask is a CPU SET: never pin .NET to one logical CPU (its JIT/GC
                     threads share it and the result is distorted by 2-4x).
The mask is applied to each child process before it runs its first instruction (tools/bench/pin.py), the C# harness is
started as the built TTC.Siren.Bench.exe (no `dotnet run` wrapper), and every child prints the affinity mask it actually
runs with; this script checks it. Without --affinity the run is unpinned: on a hybrid CPU that is NOT reproducible and a
warning is printed.

Usage:
  python tools/bench/run_interleaved.py --rounds 3 --affinity FFF [--suites fixed,exact,endgame] [--cold]
                                        [--cs-runs 5] [--py-runs 1] [--per-cell 6]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import pin  # noqa: E402  (tools/bench/pin.py: affinity helpers)

# The built harness, started directly (a `dotnet run` wrapper would spawn the real process itself).
BENCH_EXE = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Bench', 'bin', 'Release', 'net10.0',
                         'TTC.Siren.Bench.exe' if os.name == 'nt' else 'TTC.Siren.Bench')
BENCH = [BENCH_EXE]

MASK = None          # requested CPU set (int) or None = unpinned
SEEN_MASKS = set()   # masks the children actually reported


def run(cmd, timeout=3600):
    p = pin.run_pinned(cmd, MASK, cwd=ROOT, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit('command failed (%d): %s\n%s\n%s' % (p.returncode, ' '.join(cmd), p.stdout[-2000:], p.stderr[-2000:]))
    check_affinity(p.stdout, cmd)
    return p.stdout


def check_affinity(output, cmd):
    """Every child prints `affinity=0x...` in its first lines: it must be the requested mask."""
    for line in output.splitlines()[:4]:
        if 'affinity=' in line:
            text = line.split('affinity=')[1].split()[0]
            if text.startswith('0x'):
                actual = int(text, 16)
                SEEN_MASKS.add(actual)
                if MASK is not None and actual != MASK:
                    raise SystemExit('child ran with affinity 0x%X, requested 0x%X: %s' % (actual, MASK, ' '.join(cmd)))
            return
    raise SystemExit('child did not report its affinity: ' + ' '.join(cmd))


def results(output, name):
    """Parses the machine-readable `RESULT <name> k=v k=v ...` lines of the C# harness."""
    out = []
    for line in output.splitlines():
        if line.startswith('RESULT ' + name + ' '):
            out.append(dict(tok.split('=', 1) for tok in line.split()[2:] if '=' in tok))
    return out


def json_lines(output):
    out = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def python_fixed(args):
    recs = json_lines(run([sys.executable, 'tools/bench/py_bench.py', '.', 'fixed', str(args.py_runs)]))
    summary = next(r for r in recs if r.get('summary') == 'fixed')
    return {'total': summary['sec_median'], 'nodes': summary['nodes']}


def csharp_fixed(args):
    r = results(run(BENCH + ['search', str(args.cs_runs)] + (['--cold'] if args.cold else [])), 'search')[0]
    return {'total': float(r['total_median_s']), 'nodes': int(r['nodes']), 'min': float(r['total_min_s']), 'max': float(r['total_max_s'])}


def python_exact(args):
    recs = json_lines(run([sys.executable, 'tools/bench/py_endgame_bench.py', '--exact', '--runs', str(args.py_runs)]))
    return {r['scenario']: {'total': r['median_s'], 'make_moves': r['make_moves'], 'value': r['value']} for r in recs if 'scenario' in r}


def csharp_exact(args):
    rs = results(run(BENCH + ['exact', str(args.cs_runs)] + (['--cold'] if args.cold else [])), 'exact')
    return {r['item']: {'total': float(r['median_s']), 'make_moves': int(r['make_moves']), 'value': int(r['value'])} for r in rs}


def python_endgame(args, positions):
    out = run([sys.executable, 'tools/bench/py_endgame_bench.py', '--out', positions, '--per-cell', str(args.per_cell),
               '--runs', str(args.py_runs)], timeout=7200)
    summary = next(r for r in json_lines(out) if r.get('summary') == 'endgame')
    return {'total': summary['total_median_s'], 'positions': summary['positions']}


def csharp_endgame(args, positions):
    r = results(run(BENCH + ['endgame', positions, str(args.cs_runs)] + (['--cold'] if args.cold else [])), 'endgame')[0]
    return {'total': float(r['cs_median_s']), 'min': float(r['cs_min_s']), 'max': float(r['cs_max_s']),
            'same': int(r['same']), 'different': int(r['different'])}


def spread(values):
    return min(values), statistics.median(values), max(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--suites', default='fixed,exact,endgame')
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--cs-runs', type=int, default=5)
    ap.add_argument('--py-runs', type=int, default=1)
    ap.add_argument('--per-cell', type=int, default=6)
    ap.add_argument('--affinity', default=None,
                    help="hex CPU-set mask for BOTH sides, or 'auto' for the detected P-core set (see the module docstring)")
    args = ap.parse_args()
    if args.rounds < 3:
        ap.error('use at least 3 rounds (CLAUDE.md section 6)')
    global MASK
    if args.affinity is not None:
        MASK = pin.parse_mask(args.affinity)
        if MASK & ~(pin.current_affinity() or MASK):
            ap.error('mask 0x%X is not inside this process CPU set' % MASK)
    if not os.path.exists(BENCH_EXE):
        ap.error('build first: dotnet build csharp/TTC.Siren.sln -c Release (missing %s)' % BENCH_EXE)
    suites = args.suites.split(',')
    tmp = tempfile.mkdtemp(prefix='ttc_interleaved_')
    mode = 'cold' if args.cold else 'warm'
    tag = '[affinity %s]' % pin.describe(MASK)
    print('# interleaved Python (CPython %s) / C# [%s], rounds=%d, python runs per round=%d, C# runs per round=%d, CPU set: %s' %
          ('.'.join(map(str, sys.version_info[:3])), mode, args.rounds, args.py_runs, args.cs_runs,
           'unpinned' if MASK is None else pin.describe(MASK) + ' for both sides'), flush=True)
    classes = pin.core_classes()
    if classes:
        print('# CPU classes: ' + ', '.join('class %d = 0x%X' % (c, classes[c]) for c in sorted(classes, reverse=True)) +
              ('  (hybrid CPU: higher class = P-cores)' if len(classes) > 1 else ''), flush=True)
    if MASK is None and pin.is_hybrid():
        print('# WARNING: no --affinity on a hybrid CPU. Windows may place the Python and C# runs on different core classes '
              '(P vs E: 1.4-2x apart), so these ratios are NOT reproducible. Re-run with --affinity auto (P-cores = 0x%X).' %
              pin.performance_core_mask(), flush=True)

    py_tot = {s: [] for s in suites}
    cs_tot = {s: [] for s in suites}
    exact_py = {}
    exact_cs = {}
    sanity = []
    for rnd in range(1, args.rounds + 1):
        if 'fixed' in suites:
            p = python_fixed(args)
            c = csharp_fixed(args)
            assert p['nodes'] == c['nodes'], 'node counts differ: %s vs %s' % (p['nodes'], c['nodes'])
            py_tot['fixed'].append(p['total']); cs_tot['fixed'].append(c['total'])
            print('round %d  fixed    Python %.4f s   C# %.5f s (min %.5f max %.5f)   ratio %.1fx   nodes %d   %s' %
                  (rnd, p['total'], c['total'], c['min'], c['max'], p['total'] / c['total'], p['nodes'], tag), flush=True)
        if 'exact' in suites:
            p = python_exact(args)
            c = csharp_exact(args)
            line = []
            for name in p:
                assert p[name]['value'] == c[name]['value'], 'exact values differ for ' + name
                exact_py.setdefault(name, []).append(p[name]['total'])
                exact_cs.setdefault(name, []).append(c[name]['total'])
                line.append('%s %.4f/%.5f s = %.0fx' % (name, p[name]['total'], c[name]['total'], p[name]['total'] / c[name]['total']))
                sanity.append((name, p[name]['make_moves'], c[name]['make_moves']))
            py_tot['exact'].append(sum(v['total'] for v in p.values())); cs_tot['exact'].append(sum(v['total'] for v in c.values()))
            print('round %d  exact    ' % rnd + '   '.join(line) + '   ' + tag, flush=True)
        if 'endgame' in suites:
            positions = os.path.join(tmp, 'endgame_round%d.json' % rnd)
            p = python_endgame(args, positions)
            c = csharp_endgame(args, positions)
            assert c['different'] == 0, 'C# selected a different move than Python in %d positions' % c['different']
            py_tot['endgame'].append(p['total']); cs_tot['endgame'].append(c['total'])
            print('round %d  endgame  Python %.4f s   C# %.5f s (min %.5f max %.5f)   ratio %.1fx   %d/%d selections identical   %s' %
                  (rnd, p['total'], c['total'], c['min'], c['max'], p['total'] / c['total'], c['same'], c['same'] + c['different'], tag), flush=True)

    print('\n# summary %s (child processes reported affinity: %s)' % (tag, ', '.join(pin.describe(m) for m in sorted(SEEN_MASKS))))
    print('%-10s %-34s %-34s %-24s' % ('suite', 'Python s  min / median / max', 'C# s  min / median / max', 'ratio  min / median / max'))
    for s in suites:
        pmin, pmed, pmax = spread(py_tot[s])
        cmin, cmed, cmax = spread(cs_tot[s])
        ratios = [a / b for a, b in zip(py_tot[s], cs_tot[s])]
        rmin, rmed, rmax = spread(ratios)
        print('%-10s %10.4f /%10.4f /%10.4f  %10.5f /%10.5f /%10.5f   %6.1f /%6.1f /%6.1f x' %
              (s, pmin, pmed, pmax, cmin, cmed, cmax, rmin, rmed, rmax))
    if exact_py:
        print('\nexact solver per position (Python s / C# s per round; ratio)')
        for name in exact_py:
            pmin, pmed, pmax = spread(exact_py[name])
            cmin, cmed, cmax = spread(exact_cs[name])
            ratios = [a / b for a, b in zip(exact_py[name], exact_cs[name])]
            rmin, rmed, rmax = spread(ratios)
            makes = next((x for x in sanity if x[0] == name))
            print('  %-16s Python %.5f/%.5f/%.5f   C# %.6f/%.6f/%.6f   ratio %.0f/%.0f/%.0f x   make_move Python %d, C# %d (work ratio %.3f)' %
                  (name, pmin, pmed, pmax, cmin, cmed, cmax, rmin, rmed, rmax, makes[1], makes[2], makes[1] / makes[2]))
    print('\nratio = Python / C# per round, CPU set: %s (both sides). All ratios are versus CPython %s; PyPy was not measured.' %
          (pin.describe(MASK), '.'.join(map(str, sys.version_info[:3]))))


if __name__ == '__main__':
    main()
