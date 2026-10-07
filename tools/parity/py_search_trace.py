# -*- coding: utf-8 -*-
"""
Golden generator for the Phase 3 C# parity tests (evaluation, move ordering, history, minimax, iterative deepening,
transposition table). Read-only use of the Python reference (ai/ai.py, core/*): nothing is modified.

Kinds of output (one gzip JSON-lines file per kind; first line is a {"header": ...} record):

  --kind eval    evaluate_state / evaluate_move bit patterns (float.hex()) on random positions
  --kind search  fixed-depth minimax (fresh TT and history per run): score, best move, node/cutoff/TT counters,
                 final history table. Includes the 22 benchmark items of CSHARP_MIGRATION.md section 3.4.
  --kind id      iterative_deepening_search with unlimited time (per-depth score/move/nodes) and with a node budget
                 (ai.is_time_up replaced by a deterministic nodes_searched > N predicate)
  --kind tt      get_state_hash equivalence: sha1 of the Python key string per state, grouped in families that contain
                 deliberate collisions (hand ids concatenated without separator, D-005)

Position encoding is shared with tools/parity/py_rules_trace.py (same compact card arrays).

Usage:
  python tools/parity/py_search_trace.py --kind eval   --out <file.jsonl.gz> --seed 20261008 --count 100000
  python tools/parity/py_search_trace.py --kind search --out <file.jsonl.gz> --seed 20261008 --count 600
  python tools/parity/py_search_trace.py --kind id     --out <file.jsonl.gz> --seed 20261008 --count 200
  python tools/parity/py_search_trace.py --kind tt     --out <file.jsonl.gz> --seed 20261008 --count 3000
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)  # ai.get_card_star_map() reads data/ relative to the working directory (D-009)

import ai.ai as A  # noqa: E402
from core.board import Board  # noqa: E402
from core.card import Card  # noqa: E402
from core.game_state import GameState  # noqa: E402
from core.player import Player  # noqa: E402
import py_rules_trace as R  # noqa: E402

GENERATOR_VERSION = 'py_search_trace/1'
INF = float('inf')


def hexf(x):
    return float(x).hex()


def hand_slot(state, card):
    return next(i for i, h in enumerate(state.current_player.hand) if h is card)


def move_code(state, move):
    card, (r, c) = move
    return hand_slot(state, card) * 9 + r * 3 + c


def fmt_move(move):
    if move is None:
        return None
    card, (r, c) = move
    return [card.card_id, r, c]


def export_history(history):
    return sorted(([k[0], k[1], k[2], v] for k, v in history.items()),
                  key=lambda e: (-1 if e[0] is None else e[0], e[1], e[2]))


def clone(state):
    return state.copy()  # deepcopy: keeps hand order, owners, modifiers and display edges


def occupancy(state):
    return sum(1 for r in range(3) for c in range(3) if state.board.grid[r][c] is not None)


def make_position(rng, db, index, empties_bias):
    """Random position (shared builder). empties_bias='late' favours boards with <= 5 empty cells."""
    rules = R.rules_for_index(rng, index)
    for _ in range(100):
        state = R.build_position(rng, db, rules)
        occ = occupancy(state)
        if empties_bias == 'late' and occ < 4:
            continue
        if state.current_player.hand:
            return rules, state
    return rules, state


# --------------------------------------------------------------------------- eval

def random_history(rng, state):
    if rng.random() < 0.4:
        return {}
    ids = [c.card_id for p in state.players for c in p.hand]
    hist = {}
    for _ in range(rng.randint(1, 12)):
        cid = rng.choice(ids) if ids and rng.random() < 0.9 else rng.choice([None, 7, 1007, 5001])
        key = (cid, rng.randint(0, 2), rng.randint(0, 2))
        value = rng.choice([2, 4, 8, 16, 64, 1024, 5000, 9999, 10000, 65536, 524288, 1000000, 30, 170])
        hist[key] = value
    return hist


def eval_record(rng, db, index):
    rules, state = make_position(rng, db, index, 'any')
    hist = random_history(rng, state)
    moves = state.get_available_moves()
    sel = list(range(len(moves)))
    if len(sel) > 10:
        sel = sorted(rng.sample(sel, 10))
    rec = {
        'i': index, 'rules': rules, 'state': R.export_state(state),
        'hist': export_history(hist),
        'es': [[hexf(A.evaluate_state(state, ai, False)), hexf(A.evaluate_state(state, ai, True))] for ai in (0, 1)],
        'em': [[move_code(state, moves[j]),
                hexf(A.evaluate_move(moves[j], state, hist, False)),
                hexf(A.evaluate_move(moves[j], state, hist, True))] for j in sel],
    }
    return rec


# --------------------------------------------------------------------------- search

def run_fixed_depth(base_state, depth, aggressive, budget=None):
    state = clone(base_state)
    A.TRANSPOSITION_TABLE = {}
    A.SEARCH_STATS.reset()
    A.START_TIME = time.time()
    A.TIME_LIMIT = 1e9
    history = {}
    orig = A.is_time_up
    if budget is not None:
        A.is_time_up = lambda: A.SEARCH_STATS.nodes_searched > budget
    try:
        res = A.minimax(state, depth, -INF, INF, True, state.current_player_idx, False, True, history,
                        aggressive=aggressive)
    finally:
        A.is_time_up = orig
    s = A.SEARCH_STATS
    return {
        'd': depth, 'budget': budget, 'score': hexf(res.eval_score), 'move': fmt_move(res.best_move),
        'nodes': s.nodes_searched, 'cut': s.alpha_beta_cutoffs, 'tt': s.tt_hits, 'ttc': s.tt_cutoffs,
        'me': s.move_evaluations, 'hist': export_history(history),
    }


def search_depths(base_state, aggressive, node_cap):
    out = []
    # Depth 1..5 on every position. Searching deeper than the number of empty cells is legal (the search stops at the
    # end of the game) and exercises the terminal / no-move paths; only the node cap stops expensive open positions.
    for d in range(1, 6):
        r = run_fixed_depth(base_state, d, aggressive)
        out.append(r)
        if r['nodes'] > node_cap:
            break
    return out


def search_record(rng, db, index):
    rules, state = make_position(rng, db, index, 'late' if rng.random() < 0.85 else 'any')
    aggressive = index % 4 == 3
    return {'i': index, 'rules': rules, 'aggr': aggressive, 'state': R.export_state(state),
            'runs': search_depths(state, aggressive, 5000)}


def bench_positions():
    """The five benchmark positions of CSHARP_MIGRATION.md section 3.3 (identical to tools/bench/py_bench.py)."""
    rows = {r['id']: r for r in R.load_db_rows()}

    def card(cid, owner):
        r = rows[cid]
        c = Card(r['up'], r['right'], r['down'], r['left'], None, cid, r['type'])
        c = c.copy()  # Card.copy() refreshes the display edges like py_bench does
        c.owner = owner
        return c

    def state(board_items, red_ids, blue_ids, cur, rules):
        b = Board()
        for (r, c), cid, owner in board_items:
            b.place_card(r, c, card(cid, owner))
        players = [Player('red', [card(i, 'red') for i in red_ids]), Player('blue', [card(i, 'blue') for i in blue_ids])]
        s = GameState(b, players, current_player_idx=cur, rules=list(rules))
        if '同类强化' in rules or '同类弱化' in rules:
            s.recalculate_type_modifiers()
        return s

    red5, blue5 = [56, 64, 27, 46, 7], [59, 51, 35, 48, 8]
    return {
        'open_none': (state([], red5, blue5, 0, []), [1, 2, 3, 4]),
        'open_same_plus': (state([], red5, blue5, 0, ['同数', '加算']), [1, 2, 3, 4]),
        'mid3_ascension': (state([((1, 1), 40, 'red'), ((0, 0), 41, 'blue'), ((2, 2), 52, 'red')],
                                 [42, 43, 31, 15], [53, 54, 47, 14], 1, ['同类强化']), [1, 2, 3, 4, 5]),
        'mid4_rev_ace': (state([((0, 1), 1, 'red'), ((1, 0), 2, 'blue'), ((1, 2), 3, 'red'), ((2, 1), 4, 'blue')],
                               [56, 64, 27], [59, 51, 35], 0, ['逆转', '王牌杀手']), [1, 2, 3, 4, 5]),
        'end5_same_plus': (state([((0, 0), 56, 'red'), ((0, 1), 59, 'blue'), ((1, 1), 64, 'red'),
                                  ((2, 0), 51, 'blue'), ((2, 2), 27, 'red')],
                                 [46, 7], [35, 48], 1, ['同数', '加算']), [1, 2, 3, 4]),
    }


def bench_records():
    recs = []
    for name, (state, depths) in bench_positions().items():
        rec = {'i': -1, 'name': name, 'rules': list(state.rules), 'aggr': False, 'state': R.export_state(state),
               'runs': [run_fixed_depth(state, d, False) for d in depths]}
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- iterative deepening

def run_id(base_state, max_depth, aggressive, budget=None):
    state = clone(base_state)
    A.TRANSPOSITION_TABLE = {}
    orig = A.is_time_up
    if budget is not None:
        A.is_time_up = lambda: A.SEARCH_STATS.nodes_searched > budget
    try:
        move, _path = A.iterative_deepening_search(state, max_time=1e9, max_depth=max_depth, aggressive=aggressive)
    finally:
        A.is_time_up = orig
    s = A.SEARCH_STATS
    return {
        'maxd': max_depth, 'budget': budget, 'move': fmt_move(move),
        'nodes': s.nodes_searched, 'cut': s.alpha_beta_cutoffs, 'tt': s.tt_hits, 'ttc': s.tt_cutoffs,
        'me': s.move_evaluations, 'done': s.depth_completed,
        'npd': list(s.nodes_per_depth), 'scores': [hexf(x) for x in s.best_score_history],
        'moves': [fmt_move(m) for m in s.best_move_history],
    }


def id_record(rng, db, index):
    rules, state = make_position(rng, db, index, 'late')
    aggressive = index % 5 == 4
    empties = 9 - occupancy(state)
    max_depth = min(8, empties + 2) if empties <= 5 else 3
    runs = [run_id(state, max_depth, aggressive)]
    # A few node budgets: exercises time-up control flow (partial results, discarded iterations, TT writes on timeout).
    total = runs[0]['nodes']
    for _ in range(3):
        runs.append(run_id(state, max_depth, aggressive, budget=rng.randint(1, max(2, total))))
    # direct minimax with a budget (time-up inside one search)
    depth = min(4, max(1, empties))
    direct = run_fixed_depth(state, depth, aggressive, budget=rng.randint(1, 400))
    return {'i': index, 'rules': rules, 'aggr': aggressive, 'state': R.export_state(state),
            'runs': runs, 'direct': direct}


def bench_id_records():
    recs = []
    for name, (state, _depths) in bench_positions().items():
        if name in ('open_none', 'open_same_plus'):
            continue  # unlimited-time ID to depth 10 from an empty board is infeasible (max_time=10 s in production)
        recs.append({'i': -1, 'name': name, 'rules': list(state.rules), 'aggr': False, 'state': R.export_state(state),
                     'runs': [run_id(state, 10, False)], 'direct': None})
    return recs


# --------------------------------------------------------------------------- transposition key equivalence

def key_digest(state):
    return hashlib.sha1(A.get_state_hash(state).encode('utf-8')).hexdigest()


def tt_family(rng, db, index):
    """A base position plus variants; some variants intentionally collide in Python's key (D-005) or differ only in
    fields that the key ignores (can_use, card_type, hand modifiers, display edges)."""
    rules, base = make_position(rng, db, index, 'any')
    members = [base]

    def variant(fn):
        s = clone(base)
        fn(s)
        members.append(s)

    hands = [p.hand for p in base.players]
    all_hand_cards = [c for h in hands for c in h]
    variant(lambda s: None)  # identical copy
    if all_hand_cards:
        variant(lambda s: setattr(next(c for p in s.players for c in p.hand), 'can_use',
                                  not next(c for p in s.players for c in p.hand).can_use))
        variant(lambda s: setattr(next(c for p in s.players for c in p.hand), 'type_modifier', 2))
        variant(lambda s: setattr(next(c for p in s.players for c in p.hand), 'up', 9))  # display edge only
    variant(lambda s: setattr(s, 'current_player_idx', 1 - s.current_player_idx))
    for r in range(3):
        for c in range(3):
            if base.board.grid[r][c] is not None:
                def flip(s, r=r, c=c):
                    card = s.board.grid[r][c]
                    card.owner = 'blue' if card.owner == 'red' else 'red'
                variant(flip)
                variant(lambda s, r=r, c=c: setattr(s.board.grid[r][c], 'type_modifier', 1))
                break
        else:
            continue
        break
    # hand id concatenation collisions: [1, 23] vs [12, 3] (and across the red/blue boundary)
    pair = [(1, 23), (12, 3)]
    for ids in pair:
        def setids(s, ids=ids):
            s.players[0].hand = [Card(5, 5, 5, 5, 'red', ids[0]), Card(5, 5, 5, 5, 'red', ids[1])]
            s.players[1].hand = []
        variant(setids)

    def split_a(s):
        s.players[0].hand = [Card(5, 5, 5, 5, 'red', 1)]
        s.players[1].hand = [Card(5, 5, 5, 5, 'blue', 23)]

    def split_b(s):
        s.players[0].hand = [Card(5, 5, 5, 5, 'red', 12)]
        s.players[1].hand = [Card(5, 5, 5, 5, 'blue', 3)]
    variant(split_a)
    variant(split_b)

    def none_ids(s):
        s.players[0].hand = [Card(0, 0, 0, 0, 'red', None)]
        s.players[1].hand = []
    variant(none_ids)
    return {'i': index, 'rules': rules,
            'members': [{'state': R.export_state(s), 'key': key_digest(s)} for s in members]}


# --------------------------------------------------------------------------- driver

def generate(kind, out_path, seed, count):
    db = R.load_db_rows()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    header = {'header': {'generator': GENERATOR_VERSION, 'kind': kind, 'seed': seed, 'count': count,
                         'python': platform.python_version()}}
    t0 = time.time()
    with gzip.open(out_path, 'wt', encoding='utf-8', compresslevel=6, newline='\n') as f:
        def emit(obj):
            f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')

        emit(header)
        if kind == 'search':
            for rec in bench_records():
                emit(rec)
        if kind == 'id':
            for rec in bench_id_records():
                emit(rec)
        for index in range(count):
            rng = random.Random(seed * 1_000_003 + index)
            if kind == 'eval':
                emit(eval_record(rng, db, index))
            elif kind == 'search':
                emit(search_record(rng, db, index))
            elif kind == 'id':
                emit(id_record(rng, db, index))
            elif kind == 'tt':
                emit(tt_family(rng, db, index))
            else:
                raise SystemExit('unknown kind ' + kind)
            if (index + 1) % max(1, count // 10) == 0:
                print('  %s: %d / %d (%.0fs)' % (kind, index + 1, count, time.time() - t0), flush=True)
    print('%s: %d records -> %s' % (kind, count, out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', required=True, choices=['eval', 'search', 'id', 'tt'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--count', type=int, default=100)
    args = ap.parse_args()
    generate(args.kind, args.out, args.seed, args.count)


if __name__ == '__main__':
    main()
