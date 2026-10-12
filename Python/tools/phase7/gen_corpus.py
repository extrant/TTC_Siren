# -*- coding: utf-8 -*-
"""
Deterministic corpus of ``POST /ai_move`` requests for Phase 7 (engine-off equivalence, end-to-end comparison, load test).

  python tools/phase7/gen_corpus.py --out tools/phase7/corpus/corpus_s20261010_n360.json --seed 20261010 --count 360

Every entry is ``{"i", "tags", "seed", "request"}``. ``request`` is exactly the JSON body the Unity/Dalamud client sends
(``board`` items with ``pos``/``numU``/``numL``/``numD``/``numR``/``owner``; hand items with ``numU``..``numR``/``inHand``/``canUse``;
``rules`` as one string; ``currentPlayer``; ``npcIds``; ...); ``seed`` is the value to pass to ``random.seed`` before the request so
that the unknown-card sampling is reproducible. The numbers follow ``ai_server.parse_board`` / ``parse_hand``: a card is looked up by
(numU, numL, numD, numR) = (Top, Right, Bottom, Left) of the database, which is why ``numL`` carries the database's Right edge.

Coverage (counted by --report): both owners; currentPlayer 0 (inferred), 1 and 2; every rule incl. 同类强化/同类弱化, 秩序/混乱, 选拔,
全明牌/三明牌; opponent unknown cards 0, 1, 2 and 3+; NPC id lists; duplicate card ids inside one hand; aggressive and standard mode;
openings (empty board, both sides to move) and mid-games; requests that trigger the robust endgame (occupancy >= 5 and at most two
unknown opponent cards), among them requests whose NPC pool is blocked so that Python's legal enumeration is empty and the random
fallback (handler.generate_opponent_cards) is needed.
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'parity'))
sys.path.insert(0, ROOT)
import py_rules_trace as R  # noqa: E402  (only for load_db_rows)

RULE_NAMES = ['同数', '加算', '逆转', '王牌杀手', '同类强化', '同类弱化', '秩序', '混乱', '选拔']
GENERATOR_VERSION = 'phase7_corpus/1'


def hand_item(row, can_use=True, in_hand=True):
    # parse_hand: up, right, down, left = numU, numL, numD, numR  (D-011: the left/right swap of the protocol)
    return {'numU': row['up'], 'numL': row['right'], 'numD': row['down'], 'numR': row['left'],
            'inHand': in_hand, 'canUse': can_use}


def unknown_item(can_use=True):
    return {'numU': 0, 'numL': 0, 'numD': 0, 'numR': 0, 'inHand': True, 'canUse': can_use}


def board_item(row, cell, owner):
    return {'pos': [cell // 3, cell % 3], 'numU': row['up'], 'numL': row['right'], 'numD': row['down'], 'numR': row['left'],
            'owner': owner}


def make_entry(rng, db, index, spec):
    tags = []
    k = spec['occupancy']
    pool = [r for r in db if (r['up'], r['right'], r['down'], r['left']) != (0, 0, 0, 0)]
    first_lookup = {}
    for r in pool:  # the id parse_board / parse_hand will recognise: first row with these four edges
        first_lookup.setdefault((r['up'], r['right'], r['down'], r['left']), r['id'])
    pool = [r for r in pool if first_lookup[(r['up'], r['right'], r['down'], r['left'])] == r['id']]

    rules = []
    for name in spec.get('rules', []):
        rules.append(name)
    rules_str = ','.join(rules)
    open_mode = spec.get('open_mode')
    if open_mode:
        rules_str += ',' + open_mode
        tags.append('open:' + open_mode)
    for name in rules:
        tags.append('rule:' + name)
    if not rules:
        tags.append('rule:none')

    first = rng.choice([1, 2])        # 1 = blue moved first, 2 = red moved first (client numbering)
    second = 3 - first
    played = {first: (k + 1) // 2, second: k // 2}
    cur = first if k % 2 == 0 else second
    sizes = {1: 5 - played[1], 2: 5 - played[2]}

    my_color = cur if rng.random() < spec.get('my_turn', 0.92) else 3 - cur
    opp_color = 3 - my_color

    used = set()
    cells = rng.sample(range(9), k)
    board = []
    board_ids = []
    for cell in cells:
        row = rng.choice([r for r in pool if r['id'] not in used])
        used.add(row['id'])
        board_ids.append(row['id'])
        board.append(board_item(row, cell, rng.choice([1, 2])))

    def draw_known(n):
        out = []
        for _ in range(n):
            row = rng.choice([r for r in pool if r['id'] not in used])
            used.add(row['id'])
            out.append(row)
        return out

    my_size, opp_size = sizes[my_color], sizes[opp_color]
    typed_bias = spec.get('typed_bias', False)

    def draw_known_biased(n):
        if not typed_bias:
            return draw_known(n)
        out = []
        for _ in range(n):
            typed = [r for r in pool if r['id'] not in used and r['type']]
            row = rng.choice(typed if typed and rng.random() < 0.7 else [r for r in pool if r['id'] not in used])
            used.add(row['id'])
            out.append(row)
        return out

    n_unknown = min(spec.get('unknown', 0), opp_size)
    my_unknown = 1 if (spec.get('my_unknown', False) and my_size >= 1) else 0
    my_known = draw_known_biased(my_size - my_unknown)
    opp_known = draw_known_biased(opp_size - n_unknown)

    order_rules = 'order' in spec.get('flags', []) or '秩序' in rules or '混乱' in rules

    def make_hand(known, unknown_n):
        items = [hand_item(r, can_use=(not order_rules) or rng.random() > 0.3) for r in known]
        items += [unknown_item(can_use=(not order_rules) or rng.random() > 0.3) for _ in range(unknown_n)]
        rng.shuffle(items)
        return items

    my_hand = make_hand(my_known, my_unknown)
    opp_hand = make_hand(opp_known, n_unknown)

    if spec.get('dup'):
        target = rng.choice([my_hand, opp_hand])
        known_items = [it for it in target if it['numU'] or it['numL'] or it['numD'] or it['numR']]
        if known_items and len(target) < 5:
            target.insert(rng.randint(0, len(target)), dict(rng.choice(known_items)))
        elif len(known_items) >= 2:
            src = rng.choice(known_items)
            for j, it in enumerate(target):
                if it is not src and (it['numU'] or it['numL'] or it['numD'] or it['numR']):
                    target[j] = dict(src)
                    break
        tags.append('dup_ids')

    # used cards listed with inHand = False (skipped by parse_hand but remembered as used)
    if rng.random() < 0.2:
        for r in draw_known(rng.randint(1, 2)):
            my_hand.append(hand_item(r, in_hand=False)) if rng.random() < 0.5 else opp_hand.append(hand_item(r, in_hand=False))
        tags.append('inhand_false')

    request = {
        'board': board,
        'myHand': my_hand,
        'oppHand': opp_hand,
        'myOwner': my_color,
        'rules': rules_str,
        'aggressive': bool(spec.get('aggressive', False)),
        'max_depth': spec.get('max_depth', 4),
        'max_time': 1000000,
    }
    cp = spec.get('current_player', 'explicit')
    if cp == 'explicit':
        request['currentPlayer'] = cur
        tags.append('cp:%d' % cur)
    elif cp == 'inferred':
        request['currentPlayer'] = 0
        tags.append('cp:0')
    else:
        request['currentPlayer'] = cp
        tags.append('cp:%d' % cp)

    if spec.get('npc') == 'pool':
        ids = rng.sample([r['id'] for r in pool if r['id'] not in used], rng.choice([8, 20, 60]))
        request['npcIds'] = ids
        request['npcName'] = 'npc-%d' % index
        tags.append('npc')
    elif spec.get('npc') == 'blocked':
        # every id of the pool is already on the board or in a hand: Python's legal enumeration is empty -> random fallback
        blocked = sorted(set(board_ids) | {r['id'] for r in opp_known} | {r['id'] for r in my_known})
        request['npcIds'] = blocked[:40]
        request['npcName'] = 'blocked-%d' % index
        tags.append('npc_blocked')
    if spec.get('progress'):
        request['show_search_progress'] = True
        tags.append('progress')

    tags.append('k:%d' % k)
    tags.append('mine:%s' % ('blue' if my_color == 1 else 'red'))
    tags.append('unknown:%s' % (str(n_unknown) if n_unknown < 3 else '3+'))
    if request['aggressive']:
        tags.append('aggressive')
    if k == 0:
        tags.append('opening')
    if k >= 5 and n_unknown <= 2:
        tags.append('robust_candidate')
    if spec.get('flags'):
        tags.extend(spec['flags'])
    return {'i': index, 'tags': tags, 'seed': rng.randrange(1, 10 ** 9), 'request': request}


def build_specs(rng, count):
    """A mix with enough requests in every category; the first entries are the mandatory ones, the rest is filled randomly."""
    specs = []

    def rules_subset(force=None, n=None):
        n = rng.choice([0, 1, 1, 2, 2, 3, 4]) if n is None else n
        picked = set(rng.sample(RULE_NAMES, n))
        if force:
            picked.add(force)
        return [r for r in RULE_NAMES if r in picked]

    def add(**kw):
        specs.append(kw)

    # openings: empty board, both sides to move first
    for i in range(24):
        add(occupancy=0, unknown=rng.choice([0, 0, 1, 2]), rules=rules_subset(), aggressive=i % 4 == 3,
            my_turn=0.95, max_depth=rng.choice([3, 4]), dup=(i % 12 == 5))
    # type rules (>= 30): early and late
    for i in range(48):
        force = rng.choice(['同类强化', '同类弱化'])
        add(occupancy=rng.choice([1, 2, 3, 4, 5, 5, 6, 6, 7]), unknown=rng.choice([0, 0, 1, 2]), rules=rules_subset(force),
            aggressive=i % 5 == 4, typed_bias=True, dup=(i % 16 == 3))
    # mid-games
    for i in range(70):
        add(occupancy=rng.choice([1, 2, 3, 4]), unknown=rng.choice([0, 0, 1, 1, 2, 3, 4]), rules=rules_subset(),
            aggressive=i % 4 == 2, current_player=rng.choice(['explicit', 'explicit', 'inferred']), dup=(i % 14 == 7),
            progress=(i % 17 == 0))
    # robust endgame candidates: occupancy 5..8, unknown 0..2
    for i in range(150):
        add(occupancy=rng.choice([5, 5, 6, 6, 7, 7, 8]), unknown=rng.choice([0, 0, 1, 1, 2]), rules=rules_subset(),
            aggressive=i % 3 == 1, npc=rng.choice([None, None, None, 'pool']), dup=(i % 10 == 4),
            current_player=rng.choice(['explicit', 'explicit', 'explicit', 'inferred']), my_unknown=(i % 23 == 0),
            progress=(i % 19 == 0))
    # the random fallback path of the robust layer: the NPC pool is blocked
    for i in range(16):
        add(occupancy=rng.choice([5, 6, 7]), unknown=rng.choice([1, 1, 2]), rules=rules_subset(), npc='blocked',
            aggressive=i % 3 == 0, flags=['fallback_path'])
    # special rules
    for i in range(14):
        add(occupancy=rng.choice([0, 2, 4, 5, 6]), unknown=rng.choice([0, 1]), rules=rules_subset(n=rng.choice([1, 2])),
            open_mode=rng.choice(['全明牌', '三明牌']))
    for i in range(10):
        add(occupancy=rng.choice([1, 3, 5, 6]), unknown=rng.choice([0, 1]), rules=rules_subset(force=rng.choice(['秩序', '混乱'])))
    for i in range(6):
        add(occupancy=rng.choice([2, 5, 6]), unknown=rng.choice([0, 1]), rules=rules_subset(force='选拔'))
    # full board and nearly full board: no move, one move
    for i in range(3):
        add(occupancy=9, unknown=0, rules=rules_subset(n=1), flags=['full_board'])
    for i in range(3):
        add(occupancy=8, unknown=0, rules=rules_subset(n=1))
    rng.shuffle(specs)
    return specs[:count] if len(specs) >= count else specs + [dict(occupancy=rng.choice(range(0, 9)), rules=rules_subset())
                                                              for _ in range(count - len(specs))]


def generate(seed, count):
    rng = random.Random(seed)
    db = R.load_db_rows()
    specs = build_specs(rng, count)
    entries = []
    for i, spec in enumerate(specs):
        entries.append(make_entry(rng, db, i, spec))
    return entries


def report(entries):
    c = collections.Counter()
    for e in entries:
        for t in e['tags']:
            c[t] += 1
        req = e['request']
        c['myOwner=%d' % req['myOwner']] += 1
    robust = [e for e in entries if 'robust_candidate' in e['tags']]
    print('requests: %d' % len(entries))
    print('robust candidates (occupancy >= 5, unknown <= 2): %d, fallback-path requests: %d' % (len(robust), c['fallback_path']))
    for key in sorted(c):
        print('  %-20s %d' % (key, c[key]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out')
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--count', type=int, default=360)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    entries = generate(args.seed, args.count)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'header': {'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count}, 'entries': entries},
                      f, ensure_ascii=False, separators=(',', ':'))
        print('wrote %d requests -> %s' % (len(entries), args.out))
    if args.report or not args.out:
        report(entries)


if __name__ == '__main__':
    main()
