# -*- coding: utf-8 -*-
"""
Python side of the end-to-end comparison (Phase 7, goal section 4.4): for every corpus request, what Python computes and what the
Python server would send to the C# engine.

  python tools/phase7/e2e_reference.py --corpus tools/phase7/corpus/corpus_s20261010_n360.json \
         --out tools/parity/golden/e2e_reference_g1_s20261010_n360.jsonl.gz [--workers 6] [--limit N]

The C# test ``EngineEndToEndTests`` reads the output, sends the recorded engine payloads to the engine and compares. Nothing in
production code is changed for this: ``ai_server.ai_move`` is driven through Flask's test client, and two hooks capture what is
needed. (1) ``csharp_engine.solve`` is replaced by a recorder, so ``ai_server._solve_minimax_with_engine`` builds the payload from the
pristine ``game_state`` exactly as in production (including the second payload with the random fallback candidates when Python's
legal enumeration is empty), then falls back. (2) At that moment, and before any search, the references are computed on COPIES of
the pristine state:
  * the standard search of the Python server (``find_best_move_parallel``, with its transposition table): per completed depth the best
    score (float.hex) and move (hand slot in the request's order, row, col, card id);
  * the same iterative deepening with the transposition table switched off (no stores), which is exact depth-limited minimax: this is the
    explanation for a Python move outside the exact tie set (D-006: Python's table has no bound types);
  * when the request uses the robust endgame: ``_build_endgame_scenarios`` + ``select_endgame_robust_move`` on the request-order
    state (the D-015 decision), with the very fallback candidates the client sent: the candidate table (move, final_score, safety_ratio,
    robust_score, corner_risk as float.hex) in the returned order, the scenario count and Python's robust move.
Search settings are deterministic: ``max_time`` = 10^6, ``max_depth`` = 4, and the full depth (= number of empty cells) for positions with
5 or 6 empty cells.
"""
import argparse
import copy
import gzip
import json
import multiprocessing
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'parity'))
os.chdir(ROOT)
os.environ.setdefault('TTC_CS_ENGINE', 'on')

import golden_sources as GS  # noqa: E402

GENERATOR_VERSION = 'phase7_e2e_reference/1'

_ai_server = None
_A = None


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _NoStoreDict(dict):
    """A transposition table that never remembers anything: exact depth-limited minimax."""

    def __setitem__(self, key, value):
        pass


def _init():
    global _ai_server, _A
    import ai_server
    import ai.ai as A
    ai_server.console_ui = _NullUi()
    ai_server.ensure_handler_initialized()
    _ai_server = ai_server
    _A = A


def hexf(x):
    return float(x).hex()


def card_id_of(card):
    return -1 if card.card_id is None else int(card.card_id)


def levels_from_stats(slot_by_card, stats):
    out = []
    for i, (score, move) in enumerate(zip(stats.best_score_history, stats.best_move_history)):
        entry = {'depth': i + 1, 'score': hexf(score), 'move': None}
        if move is not None:
            card, (row, col) = move
            entry['move'] = [slot_by_card[id(card)], row, col, card_id_of(card)]
        out.append(entry)
    return out


def run_python_search(game_state, max_depth, aggressive, max_time, with_table):
    """Iterative deepening on a COPY of the pristine state; moves are reported as slots of the request-order hand."""
    A = _A
    state = game_state.copy()
    slot_by_card = {id(c): k for k, c in enumerate(state.current_player.hand)}
    try:
        if with_table:
            A.find_best_move_parallel(state, max_depth=max_depth, verbose=False, max_time=max_time, aggressive=aggressive)
        else:
            A.TRANSPOSITION_TABLE = _NoStoreDict()
            A.iterative_deepening_search(state, max_time=max_time, verbose=False, max_depth=min(max_depth, 100),
                                         progress_callback=None, aggressive=aggressive)
        return levels_from_stats(slot_by_card, A.SEARCH_STATS)
    finally:
        A.TRANSPOSITION_TABLE = {}


def robust_reference(game_state, used_cards, rules, opp_owner, opp_idx, sample_count, npc_ids, aggressive, candidates):
    S = _ai_server
    base = game_state.copy()
    slot_by_card = {id(c): k for k, c in enumerate(base.current_player.hand)}
    handler = S.get_unknown_card_handler()
    original = handler.generate_opponent_cards
    if candidates is not None:
        handler.generate_opponent_cards = lambda *a, **k: candidates  # the cards the client's single call produced
    try:
        scenarios = S._build_endgame_scenarios(base, base.players[opp_idx].hand, used_cards, rules, base.board, opp_owner,
                                               opp_idx, sample_count, candidate_card_ids=npc_ids)
    finally:
        handler.generate_opponent_cards = original
    move, table = S.select_endgame_robust_move(base, scenarios, base.current_player_idx, aggressive=aggressive)

    def encode(m):
        card, (row, col) = m
        return [slot_by_card[id(card)], row, col, card_id_of(card)]

    return {
        'scenarios': len(scenarios),
        'same_object': bool(len(scenarios) == 1 and scenarios[0] is base),
        'robust_move': None if move is None else encode(move),
        'table': [{'move': encode(item['move']), 'final': hexf(item['final_score']), 'safety': hexf(item['safety_ratio']),
                   'robust': hexf(item['robust_score']), 'corner': hexf(item['corner_risk'])} for item in table],
    }


def reference_for(entry):
    S = _ai_server
    request = entry['request']
    occupancy = len(request['board'])
    empties = 9 - occupancy
    depths = [4]
    if 4 < empties <= 6:
        depths.append(empties)
    records = []
    for max_depth in depths:
        req = copy.deepcopy(request)
        req['max_depth'] = max_depth
        req['max_time'] = 1000000
        captured = {}

        def fake_solve(payload, max_time, _captured=captured):
            _captured.setdefault('payloads', []).append(copy.deepcopy(payload))
            robust = payload.get('robust')
            if robust is not None and robust['fallbackCandidates'] is None and _captured.get('expects_fallback'):
                return {'protocol': 1, 'status': 'needs_fallback_candidates', 'unknownCount': 1}
            raise S.csharp_engine.EngineFallback('e2e_capture')

        original = S._solve_minimax_with_engine

        def wrapper(game_state, data, aggressive_mode, use_endgame_robust, opp_unknown_count, my_owner, opp_hand, used_cards,
                    rules, board, opp_owner, npc_ids, show_search_progress, _captured=captured, _original=original):
            handler = S.get_unknown_card_handler()
            unknown = S._get_unknown_hand_indices(opp_hand)
            _captured['expects_fallback'] = bool(
                use_endgame_robust and unknown and handler and
                not S._build_legal_unknown_candidates(game_state, opp_hand, board, opp_owner, npc_ids))
            # the real client code builds the payload(s); the handler's random cards are recorded for the references
            recorded = {}
            real_generate = handler.generate_opponent_cards if handler else None

            def recording_generate(*a, **k):
                result = real_generate(*a, **k)
                recorded['cards'] = result
                return result

            if handler:
                handler.generate_opponent_cards = recording_generate
            saved_solve = S.csharp_engine.solve
            S.csharp_engine.solve = fake_solve
            try:
                _original(game_state, data, aggressive_mode, use_endgame_robust, opp_unknown_count, my_owner, opp_hand,
                          used_cards, rules, board, opp_owner, npc_ids, show_search_progress)
            finally:
                S.csharp_engine.solve = saved_solve
                if handler:
                    handler.generate_opponent_cards = real_generate

            # references on copies of the pristine state
            max_depth_py = data['max_depth']
            _captured['py'] = run_python_search(game_state, max_depth_py, aggressive_mode, data['max_time'], True)
            _captured['py_no_table'] = run_python_search(game_state, max_depth_py, aggressive_mode, data['max_time'], False)
            if use_endgame_robust:
                opp_idx = 0 if my_owner == 'blue' else 1
                sample = 1 if opp_unknown_count == 0 else min(24, max(16, opp_unknown_count * 12))
                _captured['robust'] = robust_reference(game_state, used_cards, rules, opp_owner, opp_idx, sample, npc_ids,
                                                       aggressive_mode, recorded.get('cards'))
            return None, 'e2e_capture'

        S._solve_minimax_with_engine = wrapper
        saved_python = S._solve_minimax_python
        S._solve_minimax_python = lambda *a, **k: None
        try:
            random.seed(entry['seed'])
            response = S.app.test_client().post('/ai_move', json=req)
        finally:
            S._solve_minimax_with_engine = original
            S._solve_minimax_python = saved_python
        if 'payloads' not in captured or 'py' not in captured:
            records.append({'i': entry['i'], 'max_depth': max_depth, 'error': 'not captured (HTTP %d)' % response.status_code})
            continue
        payloads = captured['payloads']
        records.append({
            'i': entry['i'], 'tags': entry['tags'], 'seed': entry['seed'], 'max_depth': max_depth, 'empties': empties,
            'aggressive': bool(request.get('aggressive', False)),
            'payload': payloads[0],
            'payload_with_candidates': payloads[1] if len(payloads) > 1 else None,
            'expects_fallback': bool(captured.get('expects_fallback')),
            'py': captured['py'],
            'py_no_table': captured['py_no_table'],
            'robust': captured.get('robust'),
        })
    return records


def header(count, corpus_path):
    return {'header': dict({'generator': GENERATOR_VERSION, 'count': count, 'corpus': os.path.basename(corpus_path)},
                           **GS.header_fields('e2e'))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 3))
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    with open(args.corpus, encoding='utf-8') as f:
        entries = json.load(f)['entries']
    if args.limit:
        entries = entries[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    n = 0
    with gzip.open(tmp, 'wt', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(header(len(entries), args.corpus), ensure_ascii=False, separators=(',', ':')) + '\n')
        with multiprocessing.Pool(args.workers, initializer=_init) as pool:
            for records in pool.imap(reference_for, entries, chunksize=1):
                for rec in records:
                    out.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
                    n += 1
                if n % 50 == 0:
                    print('  %d records' % n, flush=True)
    os.replace(tmp, args.out)
    print('e2e reference: %d records -> %s' % (n, args.out))


if __name__ == '__main__':
    main()
