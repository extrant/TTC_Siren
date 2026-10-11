#!/usr/bin/env python
"""Runs a command several times while every logical CPU is kept busy by a pure-Python busy loop.

Review M8-2: a test that depends on how fast the machine is shows up when the machine is loaded. This is how the scheduler tests are
shown to be stable: the same command, N times in a row, with one burner process per logical CPU (the burners run at normal priority, so
the tests really compete for the cores).

    python tools/phase8b/repeat_under_load.py --runs 20 -- dotnet test csharp/TTC.Siren.sln -c Release --no-build --filter FullyQualifiedName~SchedulerTests

Prints one line per run and a summary; the exit code is 0 only if every run passed.
"""
import argparse
import multiprocessing
import os
import subprocess
import sys
import time


def burn(stop_file):
    x = 0
    while not os.path.exists(stop_file):
        for _ in range(200000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--burners', type=int, default=0, help='number of burner processes (default: the number of logical CPUs)')
    ap.add_argument('--log', default='', help='write the output of failing runs to this file')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command:
        ap.error('give the command after --')
    burners = args.burners or os.cpu_count() or 4
    stop_file = os.path.join(os.environ.get('TEMP', '.'), 'ttc_burner_stop_%d' % os.getpid())
    if os.path.exists(stop_file):
        os.remove(stop_file)
    procs = [multiprocessing.Process(target=burn, args=(stop_file,)) for _ in range(burners)]
    for p in procs:
        p.start()
    passed = 0
    failures = []
    try:
        time.sleep(1.0)  # let the burners take the cores
        for i in range(1, args.runs + 1):
            t0 = time.time()
            env = dict(os.environ, DOTNET_CLI_UI_LANGUAGE='en', DOTNET_NOLOGO='1', PYTHONIOENCODING='utf-8')
            r = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
            ok = r.returncode == 0
            passed += ok
            tail = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ['']
            print('run %2d/%d: %s in %.1f s  %s' % (i, args.runs, 'PASS' if ok else 'FAIL', time.time() - t0, tail[0].strip()), flush=True)
            if not ok:
                failures.append((i, r.stdout))
    finally:
        open(stop_file, 'w').close()
        for p in procs:
            p.join(10)
            if p.is_alive():
                p.terminate()
        try:
            os.remove(stop_file)
        except OSError:
            pass
    print('SUMMARY: %d/%d runs passed with %d burner processes' % (passed, args.runs, burners))
    if failures and args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            for i, out in failures:
                f.write('===== run %d =====\n%s\n' % (i, out))
    return 0 if passed == args.runs else 1


if __name__ == '__main__':
    sys.exit(main())
