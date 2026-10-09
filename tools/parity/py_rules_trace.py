# -*- coding: utf-8 -*-
"""
Golden trace generator for the C# rules parity tests (CSHARP_MIGRATION.md, Phase 2).

Read-only use of the Python reference implementation: this script only *imports*
core/ (Card, Board, Player, GameState) and csv_compat; it never modifies them.

For every sequence it builds a random position (random cards, owners, can_use, rules,
modifiers, side to move), then performs a random interleaving of make_move / undo_move.
After the initial position and after every step it exports the complete state:

  state = {"cur": 0|1,
           "board": [card|null] * 9              # row-major
           "hands": [[card...], [card...]]}      # ordered hands
  card  = [id|null, type|null, "r"|"b"|null, type_modifier,
           base_up, base_right, base_down, base_left,
           disp_up, disp_right, disp_down, disp_left, can_use(0|1)]

plus the list of available moves (slot * 9 + cell, in Python generation order) and
[red_count, blue_count, winner] where winner is 0 (red), 1 (blue) or -1.

Usage:
  python tools/parity/py_rules_trace.py --out tools/parity/golden/rules_trace.jsonl.gz --seed 20261007 --count 20000
  python tools/parity/py_rules_trace.py --carddb-out tools/parity/golden/carddb.json
"""
import argparse
import gzip
import json
import os
import platform
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_compat as pd  # noqa: E402
import golden_sources as GS  # noqa: E402
from core.board import Board  # noqa: E402
from core.card import Card  # noqa: E402
from core.game_state import GameState  # noqa: E402
from core.player import Player  # noqa: E402

GENERATOR_VERSION = 'py_rules_trace/3'  # 2: D-002 fixed (undo restores type_modifier by card identity); 3: header binds the source (M4-2)
RULE_NAMES = ['同数', '加算', '逆转', '王牌杀手', '同类强化', '同类弱化', '秩序', '混乱']
TYPE_NAMES = ['兽人', '拂晓', '帝国', '蛮神']
OWNER_CODE = {'red': 'r', 'blue': 'b', None: None}


# --------------------------------------------------------------------------- card database

def load_db_rows():
    path = os.path.join(ROOT, 'data', '幻卡数据库.csv')
    rows = []
    for _, row in pd.read_csv(path).iterrows():
        card_type = row['TripleTriadCardType']
        rows.append({
            'id': int(row['序号']),
            'up': int(row['Top']),
            'right': int(row['Right']),
            'down': int(row['Bottom']),
            'left': int(row['Left']),
            'star': int(row['星级']),
            'type': card_type if pd.notna(card_type) and card_type != '' else None,
        })
    return rows


def dump_carddb(out_path):
    rows = load_db_rows()
    lookup = {}
    for r in rows:  # same rule as ai_server.get_card_lookup: first match wins
        key = (r['up'], r['right'], r['down'], r['left'])
        if key not in lookup:
            lookup[key] = r['id']
    doc = {
        'header': dict(GS.header_fields('carddb'), generator=GENERATOR_VERSION, kind='carddb'),
        'cards': [[r['id'], r['up'], r['right'], r['down'], r['left'], r['star'], r['type']] for r in rows],
        'lookup': [[k[0], k[1], k[2], k[3], v] for k, v in lookup.items()],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, separators=(',', ':'))
    print('carddb: %d cards, %d distinct edge keys -> %s' % (len(rows), len(lookup), out_path))


# --------------------------------------------------------------------------- export

def card_arr(c):
    return [c.card_id, c.card_type, OWNER_CODE[c.owner], c.type_modifier,
            c.base_up, c.base_right, c.base_down, c.base_left,
            c.up, c.right, c.down, c.left, 1 if c.can_use else 0]


def export_state(s):
    return {
        'cur': s.current_player_idx,
        'board': [None if s.board.grid[r][c] is None else card_arr(s.board.grid[r][c])
                  for r in range(3) for c in range(3)],
        'hands': [[card_arr(c) for c in p.hand] for p in s.players],
    }


def export_moves(s):
    hand = s.current_player.hand
    out = []
    for card, (r, c) in s.get_available_moves():
        slot = next(i for i, h in enumerate(hand) if h is card)
        out.append(slot * 9 + r * 3 + c)
    return out


def export_counts(s):
    red, blue = s.count_cards()
    winner = s.get_winner()
    w = -1 if winner is None else (0 if winner == s.players[0].name else 1)
    return [red, blue, w]


# --------------------------------------------------------------------------- random positions

class CardFactory:
    def __init__(self, rng, db, rules, typed_bias):
        self.rng = rng
        self.db = db
        self.typed_db = [r for r in db if r['type']]
        self.rules = rules
        self.typed_bias = typed_bias
        self.synthetic_counter = 0

    def _edge_value(self):
        # Small alphabet so that same / plus / ace-killer situations occur often.
        return self.rng.choice([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 10])

    def make(self, owner, can_use, forbidden_ids):
        """Create one card whose id is not in forbidden_ids (None counts as an id)."""
        rng = self.rng
        for _ in range(50):
            roll = rng.random()
            if roll < 0.52:
                pool = self.typed_db if (self.typed_bias and rng.random() < 0.7) else self.db
                r = rng.choice(pool)
                card = Card(r['up'], r['right'], r['down'], r['left'], owner, r['id'], r['type'], can_use)
            elif roll < 0.82:
                self.synthetic_counter += 1
                t = None
                if rng.random() < (0.8 if self.typed_bias else 0.3):
                    t = rng.choice(TYPE_NAMES)
                card = Card(self._edge_value(), self._edge_value(), self._edge_value(), self._edge_value(),
                            owner, 5000 + self.synthetic_counter, t, can_use)
            elif roll < 0.88:
                card = Card(0, 0, 0, 0, owner, None, None, can_use)  # unknown placeholder
            else:
                r = rng.choice(self.db)
                card = Card(r['up'], r['right'], r['down'], r['left'], owner, 1000 + r['id'], r['type'], can_use)
            if card.card_id not in forbidden_ids:
                return card
        raise RuntimeError('could not draw a card with a fresh id')


def build_position(rng, db, rules):
    typed_bias = rng.random() < 0.5
    factory = CardFactory(rng, db, rules, typed_bias)

    board = Board()
    k = rng.choice([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8])
    cells = rng.sample(range(9), k)
    for cell in cells:
        owner = None if rng.random() < 0.02 else rng.choice(['red', 'blue'])
        board.place_card(cell // 3, cell % 3, factory.make(owner, rng.random() < 0.85, set()))

    players = []
    for pidx, name in enumerate(['red', 'blue']):
        size = rng.choice([0, 1, 2, 3, 3, 4, 4, 5, 5, 5]) if rng.random() < 0.08 else rng.randint(1, 5)
        ids = set()
        hand = []
        for _ in range(size):
            if rng.random() < 0.9:
                owner = name
            else:
                owner = rng.choice(['red', 'blue', None])
            card = factory.make(owner, rng.random() < 0.85, ids)
            ids.add(card.card_id)
            hand.append(card)
        players.append(Player(name, hand))

    state = GameState(board, players, current_player_idx=rng.randint(0, 1), rules=list(rules))

    all_cards = [state.board.grid[r][c] for r in range(3) for c in range(3) if state.board.grid[r][c]]
    all_cards += [c for p in state.players for c in p.hand]
    has_type_rule = '同类强化' in rules or '同类弱化' in rules
    if has_type_rule and rng.random() < 0.7:
        state.recalculate_type_modifiers()
    if rng.random() < 0.12:
        for c in all_cards:
            if rng.random() < 0.5:
                c.apply_type_modifier(rng.randint(-3, 3))
    if rng.random() < 0.05:
        for c in all_cards:
            if rng.random() < 0.3:
                c.type_modifier = rng.randint(-2, 2)  # leaves .up/.right/... stale on purpose
    return state


# --------------------------------------------------------------------------- sequences

def rules_for_index(rng, index):
    mask = index % 256  # every subset of the 8 settlement rules appears equally often
    rules = [RULE_NAMES[b] for b in range(8) if mask >> b & 1]
    if rng.random() < 0.25:
        rules.append('选拔')  # no effect on resolution; exercised for rule parsing parity
    return rules


def generate_sequence(seed, index, db):
    rng = random.Random(seed * 1_000_003 + index)
    rules = rules_for_index(rng, index)
    state = build_position(rng, db, rules)

    seq = {
        'i': index,
        'rules': rules,
        'init': export_state(state),
        'init_moves': export_moves(state),
        'init_cw': export_counts(state),
        'steps': [],
    }
    stack = []
    for _ in range(rng.randint(3, 16)):
        moves = state.get_available_moves()
        can_make = bool(moves) and len(stack) < 16
        can_undo = bool(stack)
        if not can_make and not can_undo:
            break
        do_undo = can_undo and (not can_make or rng.random() < 0.35)
        if do_undo:
            record = stack.pop()
            if rng.random() < 0.5:
                assert state.undo_move(record)
            else:
                assert state.undo_move()
            step = {'op': 'undo'}
        else:
            card, (r, c) = rng.choice(moves)
            slot = next(i for i, h in enumerate(state.current_player.hand) if h is card)
            record = state.make_move(r, c, card)
            assert record is not None
            stack.append(record)
            step = {'op': 'make', 'slot': slot, 'cell': r * 3 + c,
                    'flips': sorted(fr * 3 + fc for fr, fc, _ in record.flipped_cards)}
        step['after'] = export_state(state)
        step['moves'] = export_moves(state)
        step['cw'] = export_counts(state)
        seq['steps'].append(step)
    return seq


def generate(out_path, seed, count):
    db = load_db_rows()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    header = {'header': dict({'generator': GENERATOR_VERSION, 'seed': seed, 'count': count,
                              'python': platform.python_version()}, **GS.header_fields('rules'))}
    total_steps = 0
    with gzip.open(out_path, 'wt', encoding='utf-8', compresslevel=6, newline='\n') as f:
        f.write(json.dumps(header, ensure_ascii=False, separators=(',', ':')) + '\n')
        for index in range(count):
            seq = generate_sequence(seed, index, db)
            total_steps += len(seq['steps'])
            f.write(json.dumps(seq, ensure_ascii=False, separators=(',', ':')) + '\n')
            if (index + 1) % 5000 == 0:
                print('  %d / %d sequences' % (index + 1, count), flush=True)
    print('rules trace: %d sequences, %d steps -> %s' % (count, total_steps, out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', help='output .jsonl.gz for the rules trace')
    ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--count', type=int, default=20000)
    ap.add_argument('--carddb-out', help='also/only dump the card database reference (json)')
    args = ap.parse_args()
    if not args.out and not args.carddb_out:
        ap.error('nothing to do: pass --out and/or --carddb-out')
    if args.carddb_out:
        dump_carddb(args.carddb_out)
    if args.out:
        generate(args.out, args.seed, args.count)


if __name__ == '__main__':
    main()
