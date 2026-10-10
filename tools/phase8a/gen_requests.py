# -*- coding: utf-8 -*-
"""
Seeded ``POST /ai_move`` requests for the Phase 8A parity tests (request parsing, unknown-card sampling, end-to-end responses).

  python tools/phase8a/gen_requests.py --out tools/phase8a/corpus/requests_s20261011_n2400.json --seed 20261011 --count 2400

The Phase 7 corpus (tools/phase7/gen_corpus.py, 360 requests) stays the base of the end-to-end comparison; this generator adds many more
requests with a different mix, aimed at the parser and at the sampler: every rule and rule combination (also none: the handler-bypass
path of ``parse_hand``), all open modes, opponent unknown cards 0..5 and own unknown cards, NPC id pools (lists, comma strings with
junk, pools smaller than the number of unknown slots, pools that are completely blocked), known opponent hands with 4/5-star cards
(the deck-limit logic), ``inHand = false`` entries, ``canUse = false``, duplicate ids, ``currentPlayer`` 0/1/2 and odd values, boards
with the all-zero database rows (D-012), and a few requests that Python refuses (missing keys, cards that are not in the database).

Every entry is ``{"i", "tags", "seed", "request"}``; ``seed`` is the value for ``random.seed`` before the request.
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
GENERATOR_VERSION = 'p8_requests/1'


def hand_item(row, can_use=True, in_hand=True):
    # parse_hand: up, right, down, left = numU, numL, numD, numR  (D-011: the left/right swap of the protocol)
    return {'numU': row['up'], 'numL': row['right'], 'numD': row['down'], 'numR': row['left'], 'inHand': in_hand, 'canUse': can_use}


def unknown_item(can_use=True):
    return {'numU': 0, 'numL': 0, 'numD': 0, 'numR': 0, 'inHand': True, 'canUse': can_use}


def board_item(row, cell, owner):
    return {'pos': [cell // 3, cell % 3], 'numU': row['up'], 'numL': row['right'], 'numD': row['down'], 'numR': row['left'], 'owner': owner}


def make_request(rng, db, index):
    tags = []
    pool_all = [r for r in db if (r['up'], r['right'], r['down'], r['left']) != (0, 0, 0, 0)]
    first_lookup = {}
    for r in db:
        first_lookup.setdefault((r['up'], r['right'], r['down'], r['left']), r['id'])
    pool = [r for r in pool_all if first_lookup[(r['up'], r['right'], r['down'], r['left'])] == r['id']]
    high = [r for r in pool if r['star'] >= 4]
    typed = [r for r in pool if r['type']]

    # --- rules
    mode = rng.random()
    if mode < 0.14:
        rules = []
    elif mode < 0.55:
        rules = [rng.choice(RULE_NAMES)]
    else:
        rules = rng.sample(RULE_NAMES, rng.choice([2, 2, 3, 3, 4, 5]))
    rules = [r for r in RULE_NAMES if r in rules]
    rules_str = ','.join(rules)
    open_mode = rng.choice([None, None, None, '全明牌', '三明牌'])
    if open_mode:
        rules_str += (',' if rules_str else '') + open_mode
        tags.append('open:' + open_mode)
    for name in rules:
        tags.append('rule:' + name)
    if not rules:
        tags.append('rule:none')
    if rng.random() < 0.1:   # the rule text is not always a tidy comma list
        rules_str = rng.choice([' ' + rules_str, rules_str.replace(',', '、'), '规则：' + rules_str])
        tags.append('rules_text_odd')

    # --- position
    k = rng.choice([0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9])
    first = rng.choice([1, 2])
    second = 3 - first
    played = {first: (k + 1) // 2, second: k // 2}
    cur = first if k % 2 == 0 else second
    sizes = {1: max(0, 5 - played[1]), 2: max(0, 5 - played[2])}
    my_color = cur if rng.random() < 0.85 else 3 - cur
    opp_color = 3 - my_color

    used = set()
    cells = rng.sample(range(9), k)
    board = []
    board_ids = []
    for cell in cells:
        if rng.random() < 0.015:   # an all-zero database row on the board (D-012): looked up as id 474
            row = db[473]
            board.append(board_item(row, cell, rng.choice([1, 2])))
            board_ids.append(row['id'])
            tags.append('zero_card_on_board')
            continue
        src = typed if rng.random() < 0.35 else pool
        row = rng.choice([r for r in src if r['id'] not in used])
        used.add(row['id'])
        board_ids.append(row['id'])
        board.append(board_item(row, cell, rng.choice([1, 2])))

    def draw(n, high_n=0):
        out = []
        for _ in range(high_n):
            cand = [r for r in high if r['id'] not in used]
            if cand:
                row = rng.choice(cand)
                used.add(row['id'])
                out.append(row)
        for _ in range(n - len(out)):
            src = typed if rng.random() < 0.3 else pool
            row = rng.choice([r for r in src if r['id'] not in used])
            used.add(row['id'])
            out.append(row)
        return out

    my_size, opp_size = sizes[my_color], sizes[opp_color]
    n_unknown = min(opp_size, rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 5]))
    my_unknown = min(my_size, rng.choice([0, 0, 0, 0, 1, 1, 2]))
    high_n = rng.choice([0, 0, 1, 2, 3]) if opp_size - n_unknown > 0 else 0
    opp_known = draw(opp_size - n_unknown, min(high_n, opp_size - n_unknown))
    my_known = draw(my_size - my_unknown)

    def can_use():
        return rng.random() > (0.3 if ('秩序' in rules or '混乱' in rules) else 0.08)

    def make_hand(known, unknown_n):
        items = [hand_item(r, can_use()) for r in known] + [unknown_item(can_use()) for _ in range(unknown_n)]
        rng.shuffle(items)
        return items

    my_hand = make_hand(my_known, my_unknown)
    opp_hand = make_hand(opp_known, n_unknown)

    if rng.random() < 0.1:   # duplicate ids inside one hand
        target = rng.choice([my_hand, opp_hand])
        known_items = [it for it in target if it['numU'] or it['numL'] or it['numD'] or it['numR']]
        if known_items:
            target.insert(rng.randint(0, len(target)), dict(rng.choice(known_items)))
            tags.append('dup_ids')

    if rng.random() < 0.25:   # used cards listed with inHand = false
        for r in draw(rng.randint(1, 3)):
            (my_hand if rng.random() < 0.5 else opp_hand).append(hand_item(r, in_hand=False))
        if rng.random() < 0.3:
            (my_hand if rng.random() < 0.5 else opp_hand).append(dict(unknown_item(), inHand=False))
        tags.append('inhand_false')

    request = {
        'board': board, 'myHand': my_hand, 'oppHand': opp_hand, 'myOwner': my_color, 'rules': rules_str,
        'aggressive': rng.random() < 0.3, 'max_depth': 4, 'max_time': 1000000,
    }
    cp = rng.choice(['explicit', 'explicit', 'explicit', 'zero', 'zero', 'odd'])
    if cp == 'explicit':
        request['currentPlayer'] = cur
    elif cp == 'zero':
        request['currentPlayer'] = 0
    else:
        request['currentPlayer'] = rng.choice([3, -1, 1.0, 2.0, True, '1', None, 7])
        if rng.random() < 0.3:
            del request['currentPlayer']
    tags.append('cp:%s' % (request.get('currentPlayer', 'absent'),))

    # --- NPC pool in many shapes
    npc = rng.choice([None, None, None, 'list', 'list', 'small', 'blocked', 'string', 'messy'])
    free_ids = [r['id'] for r in pool if r['id'] not in used]
    if npc == 'list':
        request['npcIds'] = rng.sample(free_ids, rng.choice([8, 20, 60, 200]))
    elif npc == 'small':
        request['npcIds'] = rng.sample(free_ids, rng.choice([1, 2, 3]))
    elif npc == 'blocked':
        request['npcIds'] = sorted(used)[:40]
    elif npc == 'string':
        request['npcIds'] = ','.join(str(i) for i in rng.sample(free_ids, rng.choice([5, 12, 40]))).replace(',', rng.choice([',', ';', ', ']))
    elif npc == 'messy':
        ids = rng.sample(free_ids, 15)
        request['npcIds'] = ids + [ids[0], 0, -4, 'abc', None, str(ids[1]), ' %d ' % ids[2], 12.7, True, '1_0', 0.0] + (list(used)[:3])
        rng.shuffle(request['npcIds'])
    if npc:
        tags.append('npc:' + npc)
        if rng.random() < 0.7:
            request['npcName'] = 'npc-%d' % index
    if rng.random() < 0.15:
        request['show_search_progress'] = rng.choice([True, True, 1, 'yes', False])
    if rng.random() < 0.05:
        request['solver'] = rng.choice(['monte_carlo', 'minimax', 'something-else'])
        tags.append('solver:' + request['solver'])
    tags.extend(['k:%d' % k, 'mine:%s' % ('blue' if my_color == 1 else 'red'), 'unknown:%d' % n_unknown, 'my_unknown:%d' % my_unknown])
    if high_n:
        tags.append('high_star_known')
    return tags, request


def broken_request(rng, base):
    """A request Python refuses: the error forms."""
    kind = rng.choice(['no_board', 'no_hand', 'no_owner', 'empty', 'missing_numU', 'board_unknown_card', 'hand_unknown_card',
                       'missing_pos', 'bad_pos', 'missing_owner_field', 'bad_depth'])
    req = json.loads(json.dumps(base))
    if kind == 'no_board':
        del req['board']
    elif kind == 'no_hand':
        del req['oppHand']
    elif kind == 'no_owner':
        del req['myOwner']
    elif kind == 'empty':
        req = {}
    elif kind == 'missing_numU':
        target = rng.choice([req['myHand'], req['oppHand']] + ([req['board']] if req['board'] else []))
        if target:
            del rng.choice(target)['numU']
    elif kind == 'board_unknown_card':
        if req['board']:
            req['board'][0]['numU'] = 11
    elif kind == 'hand_unknown_card':
        known = [it for it in req['myHand'] + req['oppHand'] if it['numU']]
        if known:
            known[0]['numU'] = 11
    elif kind == 'missing_pos':
        if req['board']:
            del req['board'][0]['pos']
    elif kind == 'bad_pos':
        if req['board']:
            req['board'][0]['pos'] = [rng.choice([3, 5, -4]), 1]
    elif kind == 'missing_owner_field':
        if req['board']:
            del req['board'][0]['owner']
    elif kind == 'bad_depth':
        req['max_depth'] = rng.choice(['deep', None, [3]])
    return kind, req


def generate(seed, count):
    rng = random.Random(seed)
    db = R.load_db_rows()
    entries = []
    for i in range(count):
        tags, request = make_request(rng, db, i)
        if i % 40 == 39:   # one in forty is broken on purpose
            kind, request = broken_request(rng, request)
            tags = ['broken:' + kind]
        entries.append({'i': i, 'tags': tags, 'seed': rng.randrange(1, 10 ** 9), 'request': request})
    return entries


def report(entries):
    c = collections.Counter()
    for e in entries:
        for t in e['tags']:
            c[t.split(':')[0] if t.startswith(('cp:',)) else t] += 1
    print('requests: %d' % len(entries))
    for key in sorted(c):
        print('  %-26s %d' % (key, c[key]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out')
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=2400)
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
