# -*- coding: utf-8 -*-
"""
Server GC against workstation GC for the engine (review L7-1, CLAUDE.md section 6: Baseline -> Modify -> ... -> Compare).

  python tools/phase7/gc_compare.py --runs 3 --out docs/migration/reports/phase7/gc

The same build is started with ``DOTNET_gcServer=1`` and ``=0`` (the environment variable overrides ``System.GC.Server`` of the
runtimeconfig.json, so only the GC differs). For every repetition, CPU set and GC mode ``run_load_matrix.py`` runs the engine load matrix
(openings, mid-games, robust endgames; workers 12 and 16; concurrency 1/4/8/16) and appends one JSON line per cell plus the engine's
memory. The order of the two GC modes alternates from repetition to repetition so that a drifting machine does not favour one of them.
CPU sets: the P-cores (``auto``, 0xFFF on the development machine) and all CPUs. The engine's /v1/health ``serverGc`` is checked first.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import engine_process  # noqa: E402


def verify_mode(gc):
    env = dict(os.environ)
    env['DOTNET_gcServer'] = gc
    port = engine_process.free_port()
    proc = subprocess.Popen(['dotnet', engine_process.ENGINE_DLL, '--port', str(port), '--workers', '1'], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/v1/health' % port, timeout=2) as r:
                    return json.loads(r.read().decode('utf-8'))['serverGc']
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        raise SystemExit('engine did not start')
    finally:
        proc.kill()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', default='12,16')
    ap.add_argument('--concurrency', default='1,4,8,16')
    ap.add_argument('--cpus', default='auto,all')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    modes = {'server': '1', 'workstation': '0'}
    for name, value in modes.items():
        actual = verify_mode(value)
        assert actual == (value == '1'), '%s: DOTNET_gcServer=%s gave serverGc=%s' % (name, value, actual)
        print('# DOTNET_gcServer=%s -> serverGc=%s (%s)' % (value, actual, name), flush=True)
    for rep in range(1, args.runs + 1):
        order = list(modes) if rep % 2 == 1 else list(reversed(list(modes)))
        for cpu in args.cpus.split(','):
            for name in order:
                out = os.path.join(args.out, 'gc_%s_%s_run%d.jsonl' % (name, cpu, rep))
                if os.path.exists(out):
                    os.remove(out)
                env = dict(os.environ)
                env['DOTNET_gcServer'] = modes[name]
                print('# run %d cpu=%s gc=%s -> %s' % (rep, cpu, name, out), flush=True)
                subprocess.run([sys.executable, os.path.join(HERE, 'run_load_matrix.py'), '--cpu', cpu, '--workers', args.workers,
                                '--concurrency', args.concurrency, '--out', out, '--port', '5397'],
                               env=env, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
