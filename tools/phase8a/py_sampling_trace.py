# -*- coding: utf-8 -*-
"""
Python reference for the unknown-card sampler of Phase 8A (decision A-23): the handler functions called directly, in thousands of
seeded situations.

  python tools/phase8a/py_sampling_trace.py --out tools/parity/golden/p8_sampling_g1_s20261011_n5400.jsonl.gz --seed 20261011 --count 5400

Each case sets up a board and a known hand from database rows, applies ``random.seed(seed)``, and calls
``UnknownCardHandler.generate_opponent_cards`` (the opponent's estimate) or ``generate_unknown_cards`` (the player's own) with 1..5 unknown
slots under one rule family or a combination (none, each of the nine rules alone, pairs, triples, everything), and then
``ai_server._select_unknown_cards_for_slots`` on the result (random sample, or the highest-threat selection late in the game). The
records hold the input and, for every produced card, ``[id, type, up, right, down, left, can_use, generated, prediction]``, then the
selection, and the next value of the random generator (``float.hex``), which proves the number of draws.
"""
import argparse
import contextlib
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
sys.path.insert(0, os.path.join(ROOT, 'tools', 'parity'))
os.chdir(ROOT)

import golden_sources8 as GS  # noqa: E402
import py_rules_trace as R  # noqa: E402

GENERATOR_VERSION = 'p8_sampling/1'
RULE_NAMES = ['同数', '加算', '逆转', '王牌杀手', '同类强化', '同类弱化', '秩序', '混乱', '选拔']


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


def rule_sets(rng, count):
    """Rule combinations in a fixed rotation (every family alone, pairs, triples, all, none) plus random subsets."""
    fixed = [[]] + [[r] for r in RULE_NAMES] + [['同数', '加算'], ['逆转', '王牌杀手'], ['同类强化', '同类弱化'], ['秩序', '选拔'],
                                                 ['加算', '同类弱化', '王牌杀手'], ['同数', '逆转', '选拔'], list(RULE_NAMES)]
    out = []
    for i in range(count):
        if i % 2 == 0:
            out.append(list(fixed[(i // 2) % len(fixed)]))
        else:
            out.append([r for r in RULE_NAMES if r in rng.sample(RULE_NAMES, rng.choice([1, 2, 3, 4]))])
    return out


def card_rec(c):
    return [c.card_id, c.card_type, c.up, c.right, c.down, c.left, 1 if c.can_use else 0,
            1 if getattr(c, '_is_generated', False) else 0, 1 if getattr(c, '_is_prediction', False) else 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=5400)
    args = ap.parse_args()

    import ai_server
    from core.board import Board
    from core.card import Card
    ai_server.console_ui = _NullUi()
    ai_server.ensure_handler_initialized()
    handler = ai_server.get_unknown_card_handler()

    db = R.load_db_rows()
    first = {}
    for r in db:
        first.setdefault((r['up'], r['right'], r['down'], r['left']), r['id'])
    pool = [r for r in db if (r['up'], r['right'], r['down'], r['left']) != (0, 0, 0, 0) and first[(r['up'], r['right'], r['down'], r['left'])] == r['id']]
    high = [r for r in pool if r['star'] >= 4]
    by_id = {r['id']: r for r in db}

    meta = random.Random(args.seed)
    rule_list = rule_sets(meta, args.count)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    sink = io.StringIO()
    with gzip.open(tmp, 'wt', encoding='utf-8', newline='\n') as out:
        header = dict({'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count}, **GS.header_fields('sampling'))
        out.write(json.dumps({'header': header}, ensure_ascii=False, separators=(',', ':')) + '\n')
        for i in range(args.count):
            rules = rule_list[i]
            opponent = meta.random() < 0.7
            owner = meta.choice(['red', 'blue'])
            count = meta.choice([1, 1, 2, 2, 3, 3, 4, 5])
            occupied = meta.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
            used_rows = set()
            board_cells = []
            board = Board()
            for cell in meta.sample(range(9), occupied):
                row = meta.choice([r for r in pool if r['id'] not in used_rows])
                used_rows.add(row['id'])
                o = meta.choice(['red', 'blue'])
                board.place_card(cell // 3, cell % 3, Card(row['up'], row['right'], row['down'], row['left'], o, row['id'], row['type']))
                board_cells.append([cell, row['id'], o])
            known = []
            for _ in range(meta.choice([0, 0, 1, 2, 3, 4])):
                src = high if meta.random() < 0.25 else pool
                cand = [r for r in src if r['id'] not in used_rows]
                row = meta.choice(cand)
                used_rows.add(row['id'])
                known.append(Card(row['up'], row['right'], row['down'], row['left'], owner, row['id'], row['type']))
            extra_used = sorted(used_rows | set(meta.sample(range(1, 476), meta.choice([0, 0, 5, 40, 200]))))
            seed = meta.randrange(1, 10 ** 9)
            select_slots = meta.randrange(1, count + 1)

            random.seed(seed)
            with contextlib.redirect_stdout(sink):
                if opponent:
                    cards = handler.generate_opponent_cards(count=count, rules=rules, used_cards=set(extra_used), board_state=board,
                                                            known_hand=known, owner=owner, can_use=True)
                else:
                    cards = handler.generate_unknown_cards(count=count, rules=rules, used_cards=set(extra_used), board_state=board,
                                                           known_hand=known, owner=owner, can_use=True)
                selected = ai_server._select_unknown_cards_for_slots(cards, select_slots, board, rules, owner, opponent)
            record = {
                'i': i, 'seed': seed, 'opponent': opponent, 'owner': owner, 'rules': rules, 'count': count, 'occupied': occupied,
                'board': board_cells, 'known': [c.card_id for c in known], 'used': extra_used, 'select_slots': select_slots,
                'cards': [card_rec(c) for c in cards], 'selected': [card_rec(c) for c in selected], 'rng_next': random.random().hex(),
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, args.out)
    print('sampling trace: %d cases -> %s' % (args.count, args.out))


if __name__ == '__main__':
    main()
