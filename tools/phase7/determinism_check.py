# -*- coding: utf-8 -*-
"""
Do the answers depend on the concurrency or on the number of engine workers? (Phase 7, load-test requirement)

  python tools/phase7/determinism_check.py --out docs/reports/phase7/determinism_check.txt

For worker counts 1, 8 and 20 it starts the engine, replays the corpus at concurrency 1, 4 and 16 for openings, mid-games and robust
endgames, and compares every answer with the first answer given to the same request in ANY earlier cell (tools/bench/load_test.py --digests):
the normalised engine response must be identical across concurrency levels and worker counts. Then the same for the Flask server in front of an
engine (requests without unknown cards: the unknown-card sampling uses Python's shared random generator). Searches that run into their
time limit are excluded (they depend on the machine speed, in Python as well).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
import pin  # noqa: E402
import run_load_matrix as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=5391)
    ap.add_argument('--workers', default='1,8,20')
    ap.add_argument('--concurrency', default='1,4,16')
    ap.add_argument('--opening-requests', type=int, default=16)
    args = ap.parse_args()
    url = 'http://127.0.0.1:%d' % args.port
    digests_engine = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'determinism_digests_engine.json')
    digests_flask = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'determinism_digests_flask.json')
    for f in (digests_engine, digests_flask):
        if os.path.exists(f):
            os.remove(f)
    lines = []

    def run(cmd):
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        for line in p.stdout.splitlines():
            if line.startswith('#'):
                print(line, flush=True)
                lines.append(line)

    for workers in [int(w) for w in args.workers.split(',')]:
        engine = pin.start_pinned(['dotnet', M.ENGINE_DLL, '--port', str(args.port), '--workers', str(workers), '--queue', '512',
                                   '--queue-wait-ms', '600000'], None, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            M.wait_health(url)
            for category in ('opening', 'mid', 'robust'):
                for conc in args.concurrency.split(','):
                    n = args.opening_requests if category == 'opening' else 300
                    run([sys.executable, M.LOAD_TEST, '--target', 'engine', '--url', url, '--category', category, '--concurrency', conc,
                         '--requests', str(n), '--check', '--digests', digests_engine, '--label', 'determinism workers=%d' % workers])
            if workers == int(args.workers.split(',')[1]):
                flask_port = args.port + 1
                env = dict(os.environ, TTC_CS_ENGINE='on', TTC_CS_ENGINE_URL=url)
                flask = pin.start_pinned([sys.executable, os.path.join(HERE, 'serve_flask.py'), '--port', str(flask_port)], None, cwd=ROOT,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                try:
                    M.wait_http('http://127.0.0.1:%d/' % flask_port)
                    for category in ('opening', 'mid', 'robust'):
                        for conc in args.concurrency.split(','):
                            n = args.opening_requests if category == 'opening' else 120
                            run([sys.executable, M.LOAD_TEST, '--target', 'flask', '--url', 'http://127.0.0.1:%d' % flask_port,
                                 '--category', category, '--concurrency', conc, '--requests', str(n), '--check', '--digests', digests_flask,
                                 '--label', 'determinism Flask'])
                finally:
                    flask.kill()
        finally:
            engine.kill()
            engine.wait()
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    bad = [l for l in lines if 'determinism check' in l and 'IDENTICAL' not in l]
    print('RESULT: %d cells, %d with differences' % (sum(1 for l in lines if 'determinism check' in l), len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
