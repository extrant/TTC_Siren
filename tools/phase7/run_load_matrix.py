# -*- coding: utf-8 -*-
"""
The Phase 7 load-test matrix on one machine: engine workers x concurrency x category x CPU set.

  python tools/phase7/run_load_matrix.py --cpu all    --out docs/reports/phase7/load_all.jsonl
  python tools/phase7/run_load_matrix.py --cpu 0xFFF  --out docs/reports/phase7/load_p.jsonl      # P-cores of the i7-13650HX
  python tools/phase7/run_load_matrix.py --cpu auto   ...                                          # the detected P-core set

For every engine worker count it starts the published/built engine (``dotnet <dll>``) with that CPU affinity (the engine sees only those
CPUs: Environment.ProcessorCount follows the mask), waits for ``/v1/health`` and runs tools/bench/load_test.py against it for every
category and concurrency level. ``--flask`` additionally starts the Flask server (same affinity) in front of one engine and measures
the full path. The load generator itself runs unpinned.

On the Windows Server: run it with ``--cpu all`` (and, to see how the NUMA/hyper-thread layout matters, with a mask of your choice),
with ``--workers 4,8,16,24,32`` and the engine published for that machine; see docs/DEPLOY_WINDOWS.md. Numbers measured here say nothing
about that server.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'bench'))
import pin  # noqa: E402

ENGINE_DLL = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Engine', 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')
LOAD_TEST = os.path.join(ROOT, 'tools', 'bench', 'load_test.py')


def wait_health(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + '/v1/health', timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit('engine did not become healthy: ' + url)


def wait_http(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except urllib.error.HTTPError:
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit('server did not answer: ' + url)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cpu', default='all', help="'all', 'auto' (P-cores) or a hex mask such as 0xFFF")
    ap.add_argument('--workers', default='1,4,8,12,20')
    ap.add_argument('--concurrency', default='1,2,4,8,12,16')
    ap.add_argument('--categories', default='opening,mid,robust')
    ap.add_argument('--requests', type=int, default=0)
    ap.add_argument('--port', type=int, default=5391)
    ap.add_argument('--flask', action='store_true', help='also measure the full Flask path (one engine with the largest worker count)')
    ap.add_argument('--flask-concurrency', default='1,4,8,16')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tt-entries', type=int, default=0, help='engine table size per worker (default: the engine default, 4 Mi entries)')
    args = ap.parse_args()

    mask = None if args.cpu == 'all' else pin.parse_mask(args.cpu)
    label = 'cpu=%s' % ('all (0x%X)' % (pin.current_affinity() or 0) if mask is None else pin.describe(mask))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    url = 'http://127.0.0.1:%d' % args.port

    for workers in [int(w) for w in args.workers.split(',')]:
        cmd = ['dotnet', ENGINE_DLL, '--port', str(args.port), '--workers', str(workers), '--queue', '512', '--queue-wait-ms', '600000']
        if args.tt_entries:
            cmd += ['--tt-entries', str(args.tt_entries)]
        engine = pin.start_pinned(cmd, mask, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health(url)
            for category in args.categories.split(','):
                subprocess.run([sys.executable, LOAD_TEST, '--target', 'engine', '--url', url, '--category', category,
                                '--concurrency', args.concurrency, '--requests', str(args.requests), '--check',
                                '--label', '%s workers=%d' % (label, workers), '--json-out', args.out], check=False)
            if args.flask and workers == int(args.workers.split(',')[-1]):
                flask_port = args.port + 1
                env = dict(os.environ, TTC_CS_ENGINE='on', TTC_CS_ENGINE_URL=url)
                flask = pin.start_pinned([sys.executable, os.path.join(HERE, 'serve_flask.py'), '--port', str(flask_port)], mask, cwd=ROOT,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                try:
                    wait_http('http://127.0.0.1:%d/' % flask_port)
                    for category in args.categories.split(','):
                        subprocess.run([sys.executable, LOAD_TEST, '--target', 'flask', '--url', 'http://127.0.0.1:%d' % flask_port,
                                        '--category', category, '--concurrency', args.flask_concurrency, '--requests', str(args.requests),
                                        '--check', '--label', '%s engine workers=%d + Flask' % (label, workers), '--json-out', args.out],
                                       check=False)
                finally:
                    flask.kill()
        finally:
            engine.kill()
            engine.wait()


if __name__ == '__main__':
    main()
