# -*- coding: utf-8 -*-
"""
Python reference for the end-to-end comparison of the native C# ``/ai_move`` (Phase 8A, goal section B4): the COMPLETE responses of the
Python server on its production path (the C# engine on), for the Phase 7 corpus and a corpus of fresh seeded requests.

  python tools/phase8a/py_e2e_trace.py --out tools/parity/golden/p8_e2e_g1_s20261011_n600.jsonl.gz --seed 20261011 --count 600

The real ``ai_server.ai_move`` is driven through Flask's test client with ``random.seed(seed)`` before every request; the C# engine runs as
its own process (tests/engine_process.py, 2 workers, 262144-entry tables: the C# test uses the same table size, because the node counts
of a fixed-depth search depend on it). Nothing is re-implemented: what is recorded is what the Python server answers. Search settings
are deterministic (``max_time`` = 10^6, ``max_depth`` 4, and the full depth for positions with at most 6 empty cells). A record holds the
request as the C# server will receive it, the seed, the HTTP status and the parsed JSON response.

The Monte Carlo solver is retired in the C# server (A-22). A request with ``solver = monte_carlo`` is sent to Python as a minimax request
(Python would otherwise run ai/monte_carlo.py), while the record keeps the original request, which C# answers on the minimax path.
"""
import argparse
import contextlib
import copy
import gzip
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
os.chdir(ROOT)
os.environ['TTC_CS_ENGINE'] = 'on'

import golden_sources8 as GS  # noqa: E402
import gen_corpus  # noqa: E402
import gen_requests  # noqa: E402
import engine_process  # noqa: E402

GENERATOR_VERSION = 'p8_e2e/1'
P7_SEED = 20261010
P7_COUNT = 360
TT_ENTRIES = 262144


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


def settle_depth(request, rng):
    """Deterministic search settings: 4 plies, or the whole game when at most 6 cells are empty."""
    req = copy.deepcopy(request)
    if isinstance(req, dict) and isinstance(req.get('max_depth', 4), int) and not isinstance(req.get('max_depth', 4), bool):
        board = req.get('board')
        empties = 9 - len(board) if isinstance(board, list) else 9
        req['max_depth'] = empties if 1 <= empties <= 6 else 4
        req['max_time'] = 1000000
    return req


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=600)
    args = ap.parse_args()

    import ai_server
    ai_server.console_ui = _NullUi()

    entries = [(e, 'p7') for e in gen_corpus.generate(P7_SEED, P7_COUNT)] + [(e, 'p8') for e in gen_requests.generate(args.seed, args.count)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    sink = io.StringIO()
    rng = random.Random(args.seed)
    ok = 0
    with engine_process.engine_process(workers=2, tt_entries=TT_ENTRIES) as eng, gzip.open(tmp, 'wt', encoding='utf-8', newline='\n') as out:
        os.environ['TTC_CS_ENGINE_URL'] = eng['url']
        from ai import csharp_engine
        csharp_engine.reset()
        header = dict({'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count, 'p7_seed': P7_SEED, 'p7_count': P7_COUNT,
                       'tt_entries': TT_ENTRIES}, **GS.header_fields('e2e'))
        out.write(json.dumps({'header': header}, ensure_ascii=False, separators=(',', ':')) + '\n')
        client = ai_server.app.test_client()
        for entry, src in entries:
            request = settle_depth(entry['request'], rng)
            reference = copy.deepcopy(request)
            if isinstance(reference, dict) and reference.get('solver') == 'monte_carlo':
                reference['solver'] = 'minimax'
            random.seed(entry['seed'])
            with contextlib.redirect_stdout(sink):
                response = client.post('/ai_move', json=reference)
            body = response.get_json(silent=True)
            record = {'i': entry['i'], 'src': src, 'seed': entry['seed'], 'tags': entry['tags'], 'request': request,
                      'status': response.status_code, 'response': body}
            ok += 1 if response.status_code == 200 else 0
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, args.out)
    print('e2e trace: %d records (%d answered 200) -> %s' % (len(entries), ok, args.out))


if __name__ == '__main__':
    main()
