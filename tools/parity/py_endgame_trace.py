# -*- coding: utf-8 -*-
"""
Golden generator for the Phase 5 C# parity tests (exact endgame, information-set robust endgame, deterministic
unknown-card scenario construction, endgame override). Read-only use of the Python reference (ai_server.py, ai/ai.py,
core/*): nothing is modified, the functions are only imported and called.

Kinds (one gzip JSON-lines file per kind; the first line is a {"header": ...} record that binds the file to the
reference source, see golden_sources.py):

  --kind scenario  _build_legal_unknown_candidates / _score_unknown_candidate / _build_endgame_scenarios on random
                   endgame positions: legal candidate ids (database order), the threat score of candidates, and the
                   complete scenario states (order matters). Positions whose legal candidate list is empty are not
                   recorded: Python would then sample at random (handler.generate_opponent_cards), which is outside
                   the deterministic scope (CSHARP_MIGRATION.md section 6, A-7).
  --kind unique    _unique_unknown_candidates on lists with deliberate duplicates.
  --kind corner    _calculate_corner_safety_risk (bit patterns) for many (card, cell) pairs on random positions.
  --kind reply     _best_immediate_reply_score (bit patterns) on random positions, both AI sides.
  --kind exact     _solve_endgame_exact: the value with a fresh cache, the values after every first move with ONE cache
                   shared across the moves (how _evaluate_endgame_move_robustly uses it) and with fresh caches.
  --kind robust    select_endgame_robust_move on Python-built scenarios, both modes: for every move final / safety /
                   robust / corner (bit patterns) in the returned list order, and the selected move.
  --kind override  the endgame override block of ai_move, EXTRACTED FROM THE SOURCE BY AST and executed on synthetic
                   candidate tables (the block is inline in the Flask handler and cannot be imported).

Usage:
  python tools/parity/py_endgame_trace.py --kind scenario --out <file.jsonl.gz> --seed 20261009 --count 300
"""
import argparse
import ast
import gzip
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
os.chdir(ROOT)  # ai_server.get_card_db() reads data/幻卡数据库.csv relative to the working directory (D-009)

import golden_sources as GS  # noqa: E402
import py_rules_trace as R  # noqa: E402
import ai_server as S  # noqa: E402
from core.board import Board  # noqa: E402
from core.card import Card  # noqa: E402
from core.game_state import GameState  # noqa: E402
from core.player import Player  # noqa: E402

GENERATOR_VERSION = 'py_endgame_trace/1'


class _NullUi:
    """Stands in for the rich console panel (module attribute of ai_server; no source file is touched)."""

    def __getattr__(self, name):
        return lambda *a, **k: None


S.console_ui = _NullUi()
S.ensure_handler_initialized()


def hexf(x):
    return float(x).hex()


def num(x):
    """Exact numbers: ints stay ints, floats become float.hex() strings."""
    return x if isinstance(x, int) else hexf(x)


COLOR = ['red', 'blue']


# --------------------------------------------------------------------------- positions

def build_endgame_position(rng, db, rules, occupancy, unknown):
    """A realistic endgame position: `occupancy` filled cells, hand sizes from the number of cards played, the side
    to move derived from who started (with some off-nominal shapes), `unknown` generated placeholder cards in the
    opponent's hand (the opponent is the side that is not to move)."""
    typed_bias = rng.random() < 0.5
    factory = R.CardFactory(rng, db, rules, typed_bias)
    board = Board()
    for cell in rng.sample(range(9), occupancy):
        owner = None if rng.random() < 0.02 else rng.choice(COLOR)
        board.place_card(cell // 3, cell % 3, factory.make(owner, rng.random() < 0.85, set()))

    first = rng.randint(0, 1)
    played = [(occupancy + 1) // 2, occupancy // 2]  # cards placed by [first, second]
    sizes = [max(0, 5 - played[0]), max(0, 5 - played[1])]
    sizes = [sizes[0], sizes[1]] if first == 0 else [sizes[1], sizes[0]]
    if rng.random() < 0.2:  # off-nominal hand sizes
        sizes = [max(1, min(5, s + rng.choice([-1, 0, 0, 1]))) for s in sizes]
    cur = first if occupancy % 2 == 0 else 1 - first

    players = []
    for pidx in range(2):
        ids = set()
        hand = []
        for _ in range(sizes[pidx]):
            owner = COLOR[pidx] if rng.random() < 0.93 else rng.choice(['red', 'blue', None])
            card = factory.make(owner, rng.random() < 0.85, ids)
            ids.add(card.card_id)
            hand.append(card)
        players.append(Player(COLOR[pidx], hand))

    state = GameState(board, players, current_player_idx=cur, rules=list(rules))
    if ('同类强化' in rules or '同类弱化' in rules) and rng.random() < 0.85:
        state.recalculate_type_modifiers()

    # predicted placeholders in the opponent's hand, as parse_hand produces them
    opp = 1 - cur
    hand = players[opp].hand
    slots = rng.sample(range(len(hand)), min(unknown, len(hand)))
    for slot in slots:
        old = hand[slot]
        r = rng.choice(db)
        used = {c.card_id for c in hand}
        if rng.random() < 0.7:
            card = Card(r['up'], r['right'], r['down'], r['left'], old.owner, 1000 + r['id'], r['type'], old.can_use)
            card._is_prediction = True
        else:
            card = Card(r['up'], r['right'], r['down'], r['left'], old.owner, r['id'], r['type'], old.can_use)
            card._is_generated = True
        if card.card_id in used:
            continue
        hand[slot] = card
    return state


def occupancy_of(state):
    return S._board_occupancy(state.board)


def export_gen(state):
    """Python's _is_generated_unknown_card(card) for every hand slot (the C# state has no such attribute)."""
    return [[1 if S._is_generated_unknown_card(c) else 0 for c in p.hand] for p in state.players]


def move_code(state, move):
    hand = state.current_player.hand
    card, (r, c) = move
    slot = next(i for i, h in enumerate(hand) if h is card)
    return slot * 9 + r * 3 + c


def pick_rules(rng, index):
    return R.rules_for_index(rng, index)


def pick_occupancy(rng, lo=0, hi=8):
    weights = {o: 1 for o in range(lo, hi + 1)}
    for o in (5, 6, 7):
        if o in weights:
            weights[o] = 3
    keys = sorted(weights)
    return rng.choices(keys, [weights[k] for k in keys])[0]


# --------------------------------------------------------------------------- scenario

def pick_npc(rng, db):
    if rng.random() < 0.7:
        return None
    k = rng.choice([4, 6, 10, 20, 40, 120])
    ids = rng.sample([r['id'] for r in db], k)
    return ids


def route_sample_count(unknown):
    return 1 if unknown == 0 else min(24, max(16, unknown * 12))


def scenario_record(rng, db, index):
    for _ in range(200):
        rules = pick_rules(rng, index)
        occupancy = pick_occupancy(rng, 1, 8)
        unknown = rng.choice([0, 1, 1, 2, 2, 3])
        state = build_endgame_position(rng, db, rules, occupancy, unknown)
        opp = 1 - state.current_player_idx
        opp_hand = state.players[opp].hand
        npc = pick_npc(rng, db)
        legal = S._build_legal_unknown_candidates(state, opp_hand, state.board, COLOR[opp], npc)
        n_unknown = len(S._get_unknown_hand_indices(opp_hand))
        # More than three unknown slots is far outside production (the route needs <= 2) and makes Python permute
        # tens of millions of assignments; the random card factory can create such hands through synthetic ids >= 1000.
        if n_unknown <= 3 and (n_unknown == 0 or legal):
            break
    else:
        raise RuntimeError('no position with legal candidates found')

    sample_count = route_sample_count(n_unknown) if rng.random() < 0.6 else rng.choice([1, 2, 3, 5, 16, 24, 50])
    rec = {'i': index, 'rules': rules, 'state': R.export_state(state), 'gen': export_gen(state), 'opp': opp,
           'npc': npc, 'sample': sample_count, 'unknown': n_unknown,
           'legal': [c.card_id for c in legal]}
    if legal and rng.random() < 0.35:
        rec['scores'] = [[c.card_id, hexf(S._score_unknown_candidate(c, state.board, rules or [], COLOR[opp]))] for c in legal]
    scenarios = S._build_endgame_scenarios(state, opp_hand, set(), rules, state.board, COLOR[opp], opp, sample_count,
                                           candidate_card_ids=npc)
    rec['same_object'] = bool(len(scenarios) == 1 and scenarios[0] is state)
    rec['scen'] = [R.export_state(s) for s in scenarios]
    return rec


# --------------------------------------------------------------------------- unique

def unique_record(rng, db, index):
    pool = [rng.choice(db) for _ in range(rng.randint(2, 14))]
    cards = []
    for r in pool:
        for _ in range(rng.choice([1, 1, 1, 2, 3])):
            cid = r['id'] if rng.random() < 0.9 else rng.choice([None, 7, 1000 + r['id']])
            typ = r['type'] if rng.random() < 0.9 else rng.choice([None, '兽人', '拂晓'])
            edges = [r['up'], r['right'], r['down'], r['left']]
            if rng.random() < 0.1:
                edges[rng.randint(0, 3)] = rng.randint(0, 10)
            card = Card(edges[0], edges[1], edges[2], edges[3], 'blue', cid, typ, True)
            cards.append(card)
    rng.shuffle(cards)
    out = S._unique_unknown_candidates(cards)

    def arr(c):
        return [c.card_id, c.card_type, c.base_up, c.base_right, c.base_down, c.base_left]

    return {'i': index, 'in': [arr(c) for c in cards], 'out': [arr(c) for c in out]}


# --------------------------------------------------------------------------- corner

def corner_record(rng, db, index):
    rules = pick_rules(rng, index)
    state = build_endgame_position(rng, db, rules, pick_occupancy(rng, 0, 8), 0)
    cells = [(r, c) for r in range(3) for c in range(3) if state.board.is_empty(r, c)]
    hand = state.current_player.hand
    checks = []
    for _ in range(rng.randint(1, 8)):
        if not hand or not cells:
            break
        slot = rng.randrange(len(hand))
        r, c = rng.choice(cells)
        checks.append([slot, r * 3 + c, hexf(S._calculate_corner_safety_risk(hand[slot], r, c, state.board, state.rules))])
    return {'i': index, 'rules': rules, 'state': R.export_state(state), 'checks': checks}


# --------------------------------------------------------------------------- reply

def reply_record(rng, db, index):
    rules = pick_rules(rng, index)
    state = build_endgame_position(rng, db, rules, pick_occupancy(rng, 2, 8), 0)
    out = []
    for ai in (0, 1):
        probe = state.copy()
        out.append(hexf(S._best_immediate_reply_score(probe, ai)))
    return {'i': index, 'rules': rules, 'state': R.export_state(state), 'reply': out}


# --------------------------------------------------------------------------- exact

def exact_record(rng, db, index):
    rules = pick_rules(rng, index)
    occupancy = rng.choice([5, 6, 6, 7, 7, 8, 8, 9])  # occupancy 4 (five empty cells) costs seconds per record in Python
    state = build_endgame_position(rng, db, rules, occupancy, 0)
    rec = {'i': index, 'rules': rules, 'state': R.export_state(state), 'ai': []}
    for ai in (0, 1):
        value = S._solve_endgame_exact(state.copy(), ai, {})
        item = {'ai': ai, 'value': value, 'moves': []}
        shared = {}
        for card, (r, c) in state.get_available_moves():
            slot = next(i for i, h in enumerate(state.current_player.hand) if h is card)
            after = state.copy()
            scenario_card = next((x for x in after.current_player.hand if x.card_id == card.card_id), None)
            if scenario_card is None or after.make_move(r, c, scenario_card) is None:
                continue
            fresh = S._solve_endgame_exact(after.copy(), ai, {})
            with_shared = S._solve_endgame_exact(after, ai, shared)
            item['moves'].append([slot * 9 + r * 3 + c, fresh, with_shared])
        item['shared_cache_entries'] = len(shared)
        rec['ai'].append(item)
    return rec


# --------------------------------------------------------------------------- robust

def robust_record(rng, db, index):
    for _ in range(300):
        rules = pick_rules(rng, index)
        occupancy = pick_occupancy(rng, 1, 8)
        unknown = rng.choice([0, 0, 1, 1, 2])
        state = build_endgame_position(rng, db, rules, occupancy, unknown)
        opp = 1 - state.current_player_idx
        opp_hand = state.players[opp].hand
        n_unknown = len(S._get_unknown_hand_indices(opp_hand))
        npc = pick_npc(rng, db)
        legal = S._build_legal_unknown_candidates(state, opp_hand, state.board, COLOR[opp], npc)
        if n_unknown > 3 or (n_unknown and not legal):
            continue
        break
    else:
        raise RuntimeError('no position with legal candidates found')

    sample_count = route_sample_count(n_unknown)
    if occupancy == 5 and n_unknown:
        # Four empty cells: every (move, scenario) pair is an exact solve. Python needs seconds per solve for the
        # larger hands, so these positions use a small scenario count (the builder takes any count).
        sample_count = rng.choice([2, 3, 4])
    scenarios = S._build_endgame_scenarios(state, opp_hand, set(), rules, state.board, COLOR[opp], opp, sample_count,
                                           candidate_card_ids=npc)
    ai = state.current_player_idx
    rec = {'i': index, 'rules': rules, 'state': R.export_state(state), 'gen': export_gen(state), 'opp': opp,
           'npc': npc, 'sample': sample_count, 'unknown': n_unknown, 'occupancy': occupancy_of(state),
           'use_robust': bool(S._should_use_endgame_robust_mode(state.board, n_unknown)),
           'scen': [R.export_state(s) for s in scenarios], 'modes': []}
    for aggressive in (False, True):
        move, scored = S.select_endgame_robust_move(state, scenarios, ai, aggressive=aggressive)
        rec['modes'].append({
            'aggr': aggressive,
            'best': None if move is None else move_code(state, move),
            'scored': [[move_code(state, it['move']), hexf(it['final_score']), hexf(it['safety_ratio']),
                        hexf(it['robust_score']), hexf(it['corner_risk'])] for it in scored],
        })
    return rec


# --------------------------------------------------------------------------- override (AST-extracted)

def load_override_block():
    """The `if move is not None and robust_move is not None:` block of ai_server.ai_move, compiled from the source."""
    path = os.path.join(ROOT, 'ai_server.py')
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'ai_move')
    found = [n for n in ast.walk(fn)
             if isinstance(n, ast.If) and ast.unparse(n.test) == 'move is not None and robust_move is not None']
    if len(found) != 1:
        raise SystemExit('override block not found exactly once in ai_move (found %d)' % len(found))
    module = ast.Module(body=[found[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, 'ai_server.py:ai_move:override', 'exec')


_OVERRIDE = None


def run_override(move, robust_move, items, aggressive):
    global _OVERRIDE
    if _OVERRIDE is None:
        _OVERRIDE = load_override_block()
    env = {
        '_move_key': S._move_key,
        'console_ui': _NullUi(),
        'aggressive_mode': aggressive,
        'scenario_states': [None],
        'move': move,
        'robust_move': robust_move,
        'robust_lookup': {S._move_key(it['move']): it for it in items},
    }
    exec(_OVERRIDE, env)
    return env['move']


BOUNDARY_DELTAS = [0.0, 0.05, 0.0500000001, 0.04999999, -0.05, 1.0, -1.0, 0.999999, -0.999999, 1.000001, -1.000001,
                   2.5, 2.4999999, 2.5000001, -2.5, 0.3, -0.3, 7.0, -7.0]
SAFETY_VALUES = [0.0, 0.25, 0.4, 0.4999999, 0.5, 0.5000001, 0.6, 0.7, 0.7499999, 0.75, 0.8, 0.9, 0.9999999, 1.0]


def override_record(rng, db, index):
    n = rng.randint(2, 7)
    cells = rng.sample(range(9), n)
    ids = rng.sample([r['id'] for r in db], n)
    moves = []
    for cid, cell in zip(ids, cells):
        card = Card(1, 1, 1, 1, 'red', rng.choice([cid, None]) if rng.random() < 0.05 else cid, None, True)
        moves.append((card, (cell // 3, cell % 3)))
    base = rng.choice([-8.0, -2.0, -0.5, 0.0, 0.25, 1.5, 3.0, 12.25])
    items = []
    for m in moves:
        final = base + rng.choice(BOUNDARY_DELTAS)
        if rng.random() < 0.04:
            final = float('-inf')
        items.append({'move': m, 'final_score': final, 'safety_ratio': rng.choice(SAFETY_VALUES),
                      'robust_score': final, 'corner_risk': 0.0})
    standard = rng.choice(moves + [None])
    robust = rng.choice(moves + [None])
    if rng.random() < 0.1:  # a move that is not in the table at all
        ghost = Card(1, 1, 1, 1, 'red', 999, None, True)
        if rng.random() < 0.5:
            standard = (ghost, (rng.randint(0, 2), rng.randint(0, 2)))
        else:
            robust = (ghost, (rng.randint(0, 2), rng.randint(0, 2)))
    # make the standard move's score close to the robust move's more often (that is where the thresholds are)
    if standard is not None and robust is not None and rng.random() < 0.7:
        std_item = next((it for it in items if it['move'] is standard), None)
        rob_item = next((it for it in items if it['move'] is robust), None)
        if std_item is not None and rob_item is not None and rob_item['final_score'] != float('-inf'):
            std_item['final_score'] = rob_item['final_score'] - rng.choice(BOUNDARY_DELTAS)
    aggressive = rng.random() < 0.5

    def key(m):
        return None if m is None else [m[0].card_id, m[1][0], m[1][1]]

    chosen = run_override(standard, robust, items, aggressive)
    return {
        'i': index, 'aggr': aggressive, 'standard': key(standard), 'robust': key(robust),
        'items': [[it['move'][0].card_id, it['move'][1][0], it['move'][1][1], hexf(it['final_score']),
                   hexf(it['safety_ratio'])] for it in items],
        'chosen': key(chosen),
    }


# --------------------------------------------------------------------------- driver

BUILDERS = {
    'scenario': scenario_record,
    'unique': unique_record,
    'corner': corner_record,
    'reply': reply_record,
    'exact': exact_record,
    'robust': robust_record,
    'override': override_record,
}


def generate(kind, out_path, seed, count):
    db = R.load_db_rows()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    header = {'header': dict({'generator': GENERATOR_VERSION, 'kind': kind, 'seed': seed, 'count': count,
                              'python': platform.python_version()}, **GS.header_fields('endgame'))}
    build = BUILDERS[kind]
    t0 = time.time()
    with gzip.open(out_path, 'wt', encoding='utf-8', compresslevel=6, newline='\n') as f:
        f.write(json.dumps(header, ensure_ascii=False, separators=(',', ':')) + '\n')
        for index in range(count):
            rng = random.Random(seed * 1_000_003 + index)
            f.write(json.dumps(build(rng, db, index), ensure_ascii=False, separators=(',', ':')) + '\n')
            if (index + 1) % max(1, count // 10) == 0:
                print('  %s: %d / %d (%.0fs)' % (kind, index + 1, count, time.time() - t0), flush=True)
    print('%s: %d records -> %s' % (kind, count, out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', required=True, choices=sorted(BUILDERS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261009)
    ap.add_argument('--count', type=int, default=100)
    args = ap.parse_args()
    generate(args.kind, args.out, args.seed, args.count)


if __name__ == '__main__':
    main()
