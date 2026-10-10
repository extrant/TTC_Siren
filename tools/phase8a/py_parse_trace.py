# -*- coding: utf-8 -*-
"""
Python reference for the Phase 8A request-parsing and unknown-card-sampling parity tests.

  python tools/phase8a/py_parse_trace.py --out tools/parity/golden/p8_parse_g1_s20261011_n2400.jsonl.gz --seed 20261011 --count 2400

For every request of the Phase 7 corpus (360, tools/phase7/gen_corpus.py) and of the Phase 8A corpus (``--count`` requests,
tools/phase8a/gen_requests.py) the REAL ``ai_server.ai_move`` is driven through Flask's test client after ``random.seed(seed)``. The
engine call is replaced by a recorder that captures the fully parsed state at the moment the Python server would hand it to the C#
engine, i.e. after ``parse_board`` / ``parse_hand`` (with the unknown-card sampling) / current-player inference / the 同类 recalculation
and before any search. Nothing is re-implemented here: what is recorded is what ``ai_move`` computed.

A record is either ``{"i", "src", "seed", "ok": true, "state", "flags", "rules", ...}`` or, when Python refuses the request,
``{"i", "src", "seed", "ok": false, "status", "error"}``. ``state`` is the engine payload form (the 13-value card arrays of
``ai.csharp_engine.card_array``) and ``flags`` holds, per card, ``[_is_generated, _is_prediction]``. ``rng_next`` is the next value of
the random generator after the parse (``float.hex``), which proves that the C# port consumed exactly the same number of draws.
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
os.chdir(ROOT)
os.environ.setdefault('TTC_CS_ENGINE', 'on')

import golden_sources8 as GS  # noqa: E402
import gen_corpus  # noqa: E402
import gen_requests  # noqa: E402

GENERATOR_VERSION = 'p8_parse/1'
P7_SEED = 20261010
P7_COUNT = 360


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


_S = None


def _init():
    global _S
    import ai_server
    ai_server.console_ui = _NullUi()
    _S = ai_server


def card_rec(c):
    arr = _S.csharp_engine.card_array(c)
    return arr + [1 if getattr(c, '_is_generated', False) else 0, 1 if getattr(c, '_is_prediction', False) else 0]


def capture_for(entry, src):
    S = _S
    request = copy.deepcopy(entry['request'])
    # The Monte Carlo solver is retired (A-22): the C# server answers solver=monte_carlo on the minimax path, so the reference parses it
    # as a minimax request (Python itself would skip the unknown-card sampling and call ai/monte_carlo.py).
    retired_mc = request.get('solver') == 'monte_carlo'
    if retired_mc:
        request['solver'] = 'minimax'
    captured = {}
    original = S._solve_minimax_with_engine

    def recorder(game_state, data, aggressive_mode, use_endgame_robust, opp_unknown_count, my_owner, opp_hand, used_cards,
                 rules, board, opp_owner, npc_ids, show_search_progress):
        captured.update({
            'state': {
                'cur': int(game_state.current_player_idx),
                'board': [None if game_state.board.get_card(r, c) is None else card_rec(game_state.board.get_card(r, c))
                          for r in range(3) for c in range(3)],
                'hands': [[card_rec(card) for card in player.hand] for player in game_state.players],
            },
            'rules': list(rules), 'my_owner': my_owner, 'opp_owner': opp_owner, 'aggressive': bool(aggressive_mode),
            'npc_ids': [int(i) for i in npc_ids], 'used_cards': sorted(int(i) for i in used_cards),
            'opp_unknown_count': int(opp_unknown_count), 'use_endgame_robust': bool(use_endgame_robust),
            'show_progress': bool(show_search_progress),
            'rng_next': random.random().hex(),
            'generated_flags': [[1 if S._is_generated_unknown_card(c) else 0 for c in p.hand] for p in game_state.players],
        })
        raise S.csharp_engine.EngineError('e2e_capture')

    S._solve_minimax_with_engine = recorder
    sink = io.StringIO()
    try:
        random.seed(entry['seed'])
        with contextlib.redirect_stdout(sink):
            response = S.app.test_client().post('/ai_move', json=request)
    finally:
        S._solve_minimax_with_engine = original
    record = {'i': entry['i'], 'src': src, 'seed': entry['seed'], 'tags': entry['tags'], 'retired_mc': retired_mc, 'request': entry['request']}
    if captured:
        rules_value = request.get('rules', '')
        rules, open_mode = S.parse_rules_and_open_mode(rules_value)
        record.update({'ok': True, 'open_mode': open_mode, 'current_player': request.get('currentPlayer', 0)
                       if isinstance(request.get('currentPlayer', 0), (int, float, str, type(None))) else None})
        record.update(captured)
    else:
        body = response.get_json(silent=True) or {}
        record.update({'ok': False, 'status': response.status_code, 'error': body.get('error')})
    return record


def phase7_entries():
    return gen_corpus.generate(P7_SEED, P7_COUNT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=2400)
    args = ap.parse_args()
    _init()
    entries = [(e, 'p7') for e in phase7_entries()] + [(e, 'p8') for e in gen_requests.generate(args.seed, args.count)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8', newline='\n') as out:
        header = dict({'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count, 'p7_seed': P7_SEED, 'p7_count': P7_COUNT},
                      **GS.header_fields('parse'))
        out.write(json.dumps({'header': header}, ensure_ascii=False, separators=(',', ':')) + '\n')
        ok = 0
        for entry, src in entries:
            record = capture_for(entry, src)
            ok += 1 if record['ok'] else 0
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, args.out)
    print('parse trace: %d records (%d parsed, %d refused) -> %s' % (len(entries), ok, len(entries) - ok, args.out))


if __name__ == '__main__':
    main()
