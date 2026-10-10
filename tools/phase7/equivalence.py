# -*- coding: utf-8 -*-
"""
Engine-off equivalence (Phase 7): do the responses of ``/ai_move`` stay exactly the same?

  # run in a checkout of the commit BEFORE Phase 7 (a git worktree) and in the current tree; the engine is switched off by
  # TTC_CS_ENGINE=off (the old code does not know the variable and ignores it)
  python tools/phase7/equivalence.py capture --root <repo or worktree> --corpus corpus.json --out a.json
  TTC_CS_ENGINE=off python tools/phase7/equivalence.py capture --root <repo> --corpus corpus.json --out b.json
  python tools/phase7/equivalence.py compare a.json b.json

``capture`` imports ``ai_server`` from ``--root`` (with that directory as cwd, because the server loads ``data/`` relative to it),
silences the console UI, and for every corpus entry seeds ``random`` with the entry's seed, posts the request through Flask's test
client and stores the status code and the JSON body. Search settings in the corpus are deterministic (fixed ``max_depth``,
``max_time`` = 10^6), so everything except the fields that measure time must be identical.

Fields excluded from the comparison (they measure time by nature): search_progress[*].time_elapsed / time_remaining /
nodes_per_second, search_summary.total_time_seconds / search_efficiency (derived from the speed), and the intermediate
search_progress entries without a best move that minimax emits every PROGRESS_INTERVAL seconds (their number and content depend
on the wall clock: the same request gives 0, 1 or 2 of them from run to run).
"""
import argparse
import json
import os
import random
import sys

TIMING_PROGRESS_KEYS = ('time_elapsed', 'time_remaining', 'nodes_per_second')
TIMING_SUMMARY_KEYS = ('total_time_seconds', 'search_efficiency')
NO_MOVE_TEXT = '无移动'


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


def capture(root, corpus_path, out_path, limit=None):
    root = os.path.abspath(root)
    os.chdir(root)
    sys.path.insert(0, root)
    import ai_server  # noqa: E402
    ai_server.console_ui = _NullUi()
    ai_server.ensure_handler_initialized()
    with open(corpus_path, encoding='utf-8') as f:
        corpus = json.load(f)
    entries = corpus['entries'][:limit] if limit else corpus['entries']
    client = ai_server.app.test_client()
    results = []
    for entry in entries:
        random.seed(entry['seed'])
        response = client.post('/ai_move', json=entry['request'])
        results.append({'i': entry['i'], 'status': response.status_code, 'body': response.get_json()})
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'root': root, 'engine_env': os.environ.get('TTC_CS_ENGINE'), 'results': results}, f, ensure_ascii=False)
    print('captured %d responses from %s -> %s' % (len(results), root, out_path))


def normalise(body):
    if not isinstance(body, dict):
        return body
    body = json.loads(json.dumps(body))
    if body.get('search_progress'):
        # minimax itself calls the progress callback every PROGRESS_INTERVAL seconds with best_move None ("phase": "searching"):
        # how many such intermediate entries exist, and where, depends on the wall clock. Only the entries added after a
        # completed depth (they carry a best move) are compared.
        body['search_progress'] = [item for item in body['search_progress'] if item.get('best_move') != NO_MOVE_TEXT]
    for item in body.get('search_progress', []) or []:
        for key in TIMING_PROGRESS_KEYS:
            item.pop(key, None)
    summary = body.get('search_summary')
    if isinstance(summary, dict):
        for key in TIMING_SUMMARY_KEYS:
            summary.pop(key, None)
    return body


def compare(a_path, b_path):
    with open(a_path, encoding='utf-8') as f:
        a = json.load(f)
    with open(b_path, encoding='utf-8') as f:
        b = json.load(f)
    ra, rb = a['results'], b['results']
    if len(ra) != len(rb):
        raise SystemExit('different number of results: %d vs %d' % (len(ra), len(rb)))
    different = []
    kinds = {'200': 0, 'no_move': 0, 'other_status': 0}
    for x, y in zip(ra, rb):
        assert x['i'] == y['i']
        if x['status'] != y['status'] or normalise(x['body']) != normalise(y['body']):
            different.append(x['i'])
        if x['status'] == 200 and isinstance(x['body'], dict) and x['body'].get('move', 1) is None:
            kinds['no_move'] += 1
        elif x['status'] == 200:
            kinds['200'] += 1
        else:
            kinds['other_status'] += 1
    print('compared %d responses (%s); excluded timing fields: search_progress[*].{%s}, search_summary.{%s}' %
          (len(ra), ', '.join('%s=%d' % kv for kv in kinds.items()), ', '.join(TIMING_PROGRESS_KEYS), ', '.join(TIMING_SUMMARY_KEYS)))
    if different:
        print('DIFFERENT: %d responses, first indices %s' % (len(different), different[:20]))
        return 1
    print('IDENTICAL')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--root', required=True)
    c.add_argument('--corpus', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--limit', type=int)
    d = sub.add_parser('compare')
    d.add_argument('a')
    d.add_argument('b')
    args = ap.parse_args()
    if args.cmd == 'capture':
        capture(args.root, os.path.abspath(args.corpus), os.path.abspath(args.out), args.limit)
    else:
        sys.exit(compare(args.a, args.b))


if __name__ == '__main__':
    main()
