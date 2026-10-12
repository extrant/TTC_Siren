# Usage: python py_bench.py <repo_root> [fixed|exact|server|all]
# Phase 1 baseline benchmark (CSHARP_MIGRATION.md appendix A). Read-only use of the Python reference.
import sys, os, time, json
ROOT = sys.argv[1]; sys.path.insert(0, ROOT); os.chdir(ROOT)
import ai.ai as A
from ai_server import get_all_cards, _solve_endgame_exact
from core.board import Board
from core.player import Player
from core.game_state import GameState

DB = {c.card_id: c for c in get_all_cards()}
def card(cid, owner):
    c = DB[cid].copy(); c.owner = owner; return c
def state(board_items, red_ids, blue_ids, cur, rules):
    b = Board()
    for (r, c), cid, owner in board_items: b.place_card(r, c, card(cid, owner))
    players = [Player('red', [card(i, 'red') for i in red_ids]),
               Player('blue', [card(i, 'blue') for i in blue_ids])]
    s = GameState(b, players, current_player_idx=cur, rules=list(rules))
    if '同类强化' in rules or '同类弱化' in rules: s.recalculate_type_modifiers()
    return s

RED5 = [56, 64, 27, 46, 7]; BLUE5 = [59, 51, 35, 48, 8]
SCEN = {
  'open_none':      lambda: state([], RED5, BLUE5, 0, []),
  'open_same_plus': lambda: state([], RED5, BLUE5, 0, ['同数', '加算']),
  'mid3_ascension': lambda: state([((1,1),40,'red'), ((0,0),41,'blue'), ((2,2),52,'red')],
                                  [42,43,31,15], [53,54,47,14], 1, ['同类强化']),
  'mid4_rev_ace':   lambda: state([((0,1),1,'red'), ((1,0),2,'blue'), ((1,2),3,'red'), ((2,1),4,'blue')],
                                  [56,64,27], [59,51,35], 0, ['逆转', '王牌杀手']),
  'end5_same_plus': lambda: state([((0,0),56,'red'), ((0,1),59,'blue'), ((1,1),64,'red'),
                                   ((2,0),51,'blue'), ((2,2),27,'red')], [46,7], [35,48], 1, ['同数', '加算']),
}
def fmt_move(m):
    if m is None: return None
    c, (r, col) = m; return [c.card_id, r, col]
def run_fixed_depth(name, depth):
    s = SCEN[name](); A.TRANSPOSITION_TABLE = {}; A.SEARCH_STATS.reset()
    A.START_TIME = time.time(); A.TIME_LIMIT = 1e9
    t0 = time.perf_counter()
    res = A.minimax(s, depth, float('-inf'), float('inf'), True, s.current_player_idx, False, True, {})
    dt = time.perf_counter() - t0; n = A.SEARCH_STATS.nodes_searched
    return dict(scenario=name, depth=depth, nodes=n, sec=dt, score_hex=float(res.eval_score).hex(),
                move=fmt_move(res.best_move), cutoffs=A.SEARCH_STATS.alpha_beta_cutoffs, tt_hits=A.SEARCH_STATS.tt_hits)
def run_server_like(name, max_time=10, max_depth=10):
    s = SCEN[name](); t0 = time.perf_counter()
    mv, _ = A.find_best_move_parallel(s, max_depth=max_depth, max_time=max_time)
    return dict(scenario=name, depth_completed=A.SEARCH_STATS.depth_completed, nodes=A.SEARCH_STATS.nodes_searched,
                sec=time.perf_counter() - t0, move=fmt_move(mv), nodes_per_depth=A.SEARCH_STATS.nodes_per_depth)
def run_exact(name):
    s = SCEN[name](); n = [0]; orig = s.make_move
    def counted(*a, **k): n[0] += 1; return orig(*a, **k)
    s.make_move = counted; cache = {}; t0 = time.perf_counter()
    v = _solve_endgame_exact(s, s.current_player_idx, cache)
    return dict(scenario=name, make_moves=n[0], cache=len(cache), sec=time.perf_counter() - t0, score=v)

def repeated(fn, runs):
    """Runs fn() `runs` times; the returned record keeps the last run's data, with `sec` = the median and the spread
    in `sec_min` / `sec_max` (review L5-3: a min/median/max printout exposes outliers). runs=1 is the old behaviour."""
    secs = []
    rec = None
    for _ in range(runs):
        rec = fn()
        secs.append(rec['sec'])
    secs.sort()
    rec['sec'] = secs[len(secs) // 2]
    rec['sec_min'] = secs[0]
    rec['sec_max'] = secs[-1]
    rec['runs'] = runs
    return rec

if __name__ == '__main__':
    import pin  # tools/bench/pin.py
    print('# python=%s affinity=%s cpus=%s' % (sys.version.split()[0], pin.describe(pin.current_affinity()), os.cpu_count()), flush=True)
    # usage: py_bench.py <repo_root> [fixed|exact|server|all] [runs=1]
    which = sys.argv[2] if len(sys.argv) > 2 else 'all'
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if which in ('all', 'fixed'):
        run_fixed_depth('end5_same_plus', 1)  # untimed: the first call loads the card database lazily
        tot = {'nodes': 0, 'sec_min': 0.0, 'sec_median': 0.0, 'sec_max': 0.0}
        for name, depths in [('open_none',[1,2,3,4]), ('open_same_plus',[1,2,3,4]), ('mid3_ascension',[1,2,3,4,5]),
                             ('mid4_rev_ace',[1,2,3,4,5]), ('end5_same_plus',[1,2,3,4])]:
            for d in depths:
                rec = repeated(lambda: run_fixed_depth(name, d), runs)
                tot['nodes'] += rec['nodes']; tot['sec_min'] += rec['sec_min']; tot['sec_median'] += rec['sec']; tot['sec_max'] += rec['sec_max']
                print(json.dumps(rec, ensure_ascii=False), flush=True)
        print(json.dumps(dict(summary='fixed', runs=runs, **tot)), flush=True)  # sums of per-item min / median / max
    if which in ('all', 'exact'):
        for name in ['end5_same_plus', 'mid4_rev_ace']: print(json.dumps(repeated(lambda: run_exact(name), runs)), flush=True)
    if which in ('all', 'server'):
        for name in ['open_none', 'open_same_plus', 'mid3_ascension']:
            print(json.dumps(run_server_like(name), ensure_ascii=False), flush=True)
