# -*- coding: utf-8 -*-
"""
Work comparison of the robust endgame pipeline (review section 6 / B-4): counts the make_move calls that
select_endgame_robust_move performs on the 72 benchmark positions, in a NOT timed, instrumented run. The C# counterpart
is `TTC.Siren.Bench endgame-work`. A speed-up ratio is only meaningful next to a work ratio: Python shares one exact-solver
cache per move across the scenarios, C# keeps one memo per scenario across the moves, so the two make a slightly
different number of moves.

Counted: every call of GameState.make_move made while select_endgame_robust_move runs (this includes each scenario's
first move; scenario construction makes no moves), over both modes (normal and aggressive); and the top-level calls of
_solve_endgame_exact (one per evaluated (move, scenario) pair).

Read-only use of the reference: only wraps GameState.make_move and ai_server._solve_endgame_exact with counters in this
process.

Usage:
  python tools/bench/count_robust_work.py [--per-cell 6] [--seed 20261010] [--positions tools/bench/endgame_bench_positions.json]
  dotnet run -c Release --project CSharp/src/TTC.Siren.Bench -- endgame-work tools/bench/endgame_bench_positions.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'parity'))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import py_endgame_bench as B  # noqa: E402  (position generator; imports ai_server)
import py_endgame_trace as T  # noqa: E402
import py_rules_trace as R  # noqa: E402
import ai_server as S  # noqa: E402
from core.game_state import GameState  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-cell', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--positions', default=os.path.join(HERE, 'endgame_bench_positions.json'),
                    help='if the file exists, the generated positions are checked against it (same inputs as the C# side)')
    args = ap.parse_args()

    positions = B.make_positions(args.per_cell, args.seed)
    if os.path.exists(args.positions):
        with open(args.positions, encoding='utf-8') as f:
            stored = json.load(f)['positions']
        assert len(stored) == len(positions), 'position count differs from %s' % args.positions
        for rec, (occ, unk, rules, state, opp) in zip(stored, positions):
            assert rec['state'] == R.export_state(state), 'position %s differs from the stored file' % rec['label']
        print('positions identical to', args.positions)

    makes = [0]
    solves = [0]
    depth = [0]
    orig_make = GameState.make_move
    orig_solve = S._solve_endgame_exact

    def counted_make(self, *a, **k):
        makes[0] += 1
        return orig_make(self, *a, **k)

    def counted_solve(*a, **k):
        if depth[0] == 0:
            solves[0] += 1
        depth[0] += 1
        try:
            return orig_solve(*a, **k)
        finally:
            depth[0] -= 1

    by_cell = {}
    for occupancy, unknown, rules, state, opp in positions:
        sample = T.route_sample_count(unknown)
        hand = state.players[opp].hand
        ai = state.current_player_idx
        scenarios = S._build_endgame_scenarios(state, hand, set(), rules, state.board, T.COLOR[opp], opp, sample)
        m0, s0 = makes[0], solves[0]
        GameState.make_move = counted_make
        S._solve_endgame_exact = counted_solve
        try:
            for aggressive in (False, True):
                S.select_endgame_robust_move(state, scenarios, ai, aggressive=aggressive)
        finally:
            GameState.make_move = orig_make
            S._solve_endgame_exact = orig_solve
        cell = by_cell.setdefault((occupancy, unknown), [0, 0, 0])
        cell[0] += makes[0] - m0
        cell[1] += solves[0] - s0
        cell[2] += 1

    print('%9s %7s %9s %16s %22s' % ('occupancy', 'unknown', 'positions', 'make_move calls', 'top-level exact solves'))
    for (occ, unk), (m, s, n) in sorted(by_cell.items()):
        print('%9d %7d %9d %16d %22d' % (occ, unk, n, m, s))
    print('%9s %7s %9d %16d %22d' % ('total', '', len(positions), makes[0], solves[0]))
    print('RESULT python-work total_make_moves=%d pairs=%d positions=%d' % (makes[0], solves[0], len(positions)))


if __name__ == '__main__':
    main()
