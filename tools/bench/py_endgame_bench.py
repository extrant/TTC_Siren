# -*- coding: utf-8 -*-
"""
Phase 5 benchmark of the Python reference (CPython): the information-set robust endgame pipeline of ai_server on
deterministic endgame positions at several occupancies with 0, 1 and 2 unknown opponent cards, plus the exact solver.
Read-only use of the reference (imports and calls only). The positions and Python's results are written to a JSON
file that the C# benchmark (`TTC.Siren.Bench endgame <file>`) replays, so both sides run identical inputs and the
C# side can verify that it produces the same selected moves.

Per position: the time of _build_endgame_scenarios, and of select_endgame_robust_move in normal and aggressive mode
(median of --runs runs each), the number of scenarios, and the selected move code (slot * 9 + cell).

Usage:
  python tools/bench/py_endgame_bench.py --out tools/bench/endgame_bench_positions.json [--per-cell 8] [--runs 3]
  python tools/bench/py_endgame_bench.py --exact     # _solve_endgame_exact on the section 3.6 positions
"""
import argparse
import json
import os
import platform
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'parity'))
os.chdir(ROOT)

import py_endgame_trace as T  # noqa: E402  (position builder and the stubbed console; imports ai_server)
import py_rules_trace as R  # noqa: E402
import ai_server as S  # noqa: E402

RULE_SETS = [
    [],
    ['同数'],
    ['同数', '加算'],
    ['逆转', '王牌杀手'],
    ['同类强化'],
    ['同数', '加算', '逆转', '王牌杀手', '同类强化'],
]
OCCUPANCIES = [5, 6, 7, 8]
UNKNOWNS = [0, 1, 2]


def timed_runs(fn, runs):
    """(median, min, max, last result) of `runs` timed calls (review L5-3: print the spread, not only the median)."""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times), result


def make_positions(per_cell, seed):
    db = R.load_db_rows()
    positions = []
    index = 0
    for occupancy in OCCUPANCIES:
        for unknown in UNKNOWNS:
            made = 0
            attempt = 0
            while made < per_cell:
                attempt += 1
                rng = random.Random(seed * 1_000_003 + occupancy * 1000 + unknown * 100 + attempt)
                rules = RULE_SETS[(index + made) % len(RULE_SETS)]
                state = T.build_endgame_position(rng, db, rules, occupancy, unknown)
                opp = 1 - state.current_player_idx
                hand = state.players[opp].hand
                n_unknown = len(S._get_unknown_hand_indices(hand))
                if n_unknown != unknown:
                    continue
                if unknown and not S._build_legal_unknown_candidates(state, hand, state.board, T.COLOR[opp], None):
                    continue
                positions.append((occupancy, unknown, rules, state, opp))
                made += 1
            index += per_cell
    return positions


def bench_pipeline(args):
    positions = make_positions(args.per_cell, args.seed)
    out = {'python': platform.python_version(), 'seed': args.seed, 'runs': args.runs, 'positions': []}
    totals = {'median': 0.0, 'min': 0.0, 'max': 0.0}
    for n, (occupancy, unknown, rules, state, opp) in enumerate(positions):
        sample = T.route_sample_count(unknown)
        hand = state.players[opp].hand
        ai = state.current_player_idx

        t_build, b_min, b_max, scenarios = timed_runs(lambda: S._build_endgame_scenarios(
            state, hand, set(), rules, state.board, T.COLOR[opp], opp, sample), args.runs)
        t_normal, n_min, n_max, normal = timed_runs(lambda: S.select_endgame_robust_move(state, scenarios, ai, aggressive=False), args.runs)
        t_aggr, a_min, a_max, aggr = timed_runs(lambda: S.select_endgame_robust_move(state, scenarios, ai, aggressive=True), args.runs)
        rec = {
            'label': 'occ%d_unk%d_%d' % (occupancy, unknown, n),
            'occupancy': occupancy, 'unknown': unknown, 'rules': rules,
            'state': R.export_state(state), 'gen': T.export_gen(state), 'opp': opp, 'sample': sample,
            'scenarios': len(scenarios),
            'py': {
                'build_s': t_build, 'select_normal_s': t_normal, 'select_aggressive_s': t_aggr,
                'build_min_s': b_min, 'select_normal_min_s': n_min, 'select_aggressive_min_s': a_min,
                'build_max_s': b_max, 'select_normal_max_s': n_max, 'select_aggressive_max_s': a_max,
                'best_normal': None if normal[0] is None else T.move_code(state, normal[0]),
                'best_aggressive': None if aggr[0] is None else T.move_code(state, aggr[0]),
            },
        }
        out['positions'].append(rec)
        totals['median'] += t_build + t_normal + t_aggr
        totals['min'] += b_min + n_min + a_min
        totals['max'] += b_max + n_max + a_max
        print('%s: scenarios=%d build=%.4fs normal=%.4fs aggressive=%.4fs' %
              (rec['label'], len(scenarios), t_build, t_normal, t_aggr), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    out['totals'] = totals
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps({'summary': 'endgame', 'runs': args.runs, 'positions': len(out['positions']),
                      'total_min_s': totals['min'], 'total_median_s': totals['median'], 'total_max_s': totals['max']}))
    print('wrote %d positions -> %s' % (len(out['positions']), args.out))


def bench_exact(args):
    import py_search_trace as PS  # the section 3.3 benchmark positions
    positions = PS.bench_positions()
    for name in ['end5_same_plus', 'mid4_rev_ace', 'mid3_ascension']:
        base = positions[name][0]
        times = []
        for _ in range(args.runs):
            s = base.copy()
            n = [0]
            orig = s.make_move

            def counted(*a, **k):
                n[0] += 1
                return orig(*a, **k)
            s.make_move = counted
            cache = {}
            t0 = time.perf_counter()
            v = S._solve_endgame_exact(s, s.current_player_idx, cache)
            times.append(time.perf_counter() - t0)
        print(json.dumps({'scenario': name, 'value': v, 'make_moves': n[0], 'cache': len(cache), 'runs': args.runs,
                          'min_s': min(times), 'median_s': statistics.median(times), 'max_s': max(times)}))


def main():
    import pin  # tools/bench/pin.py
    print('# python=%s affinity=%s cpus=%s' % (platform.python_version(), pin.describe(pin.current_affinity()), os.cpu_count()), flush=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'endgame_bench_positions.json'))
    ap.add_argument('--per-cell', type=int, default=8)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--exact', action='store_true')
    args = ap.parse_args()
    if args.exact:
        bench_exact(args)
    else:
        bench_pipeline(args)


if __name__ == '__main__':
    main()
