"""Phase 7: the C# engine client and the dispatch in ``ai_move``.

A stub HTTP server plays the engine, so every fallback reason, the circuit breaker, the ``needs_fallback_candidates`` round trip,
the response mapping and the random-number bookkeeping can be tested without the real engine. With the engine off (or on and
falling back) the Python path must produce exactly the response it produced before Phase 7.
"""
import copy
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))

import ai_server  # noqa: E402
from ai import csharp_engine  # noqa: E402
import gen_corpus  # noqa: E402


class NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


# --------------------------------------------------------------------------- stub engine

class StubEngine:
    """Answers GET /v1/health and POST /v1/solve as programmed by the test."""

    def __init__(self):
        self.health = {'protocol': 1, 'status': 'ok', 'cardDbSha256': csharp_engine.card_db_sha256()}
        self.health_status = 200
        self.solve = lambda payload: (200, first_legal_move_response(payload), 0.0)
        self.posts = []
        self.health_calls = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):  # silence
                pass

            def _send(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                stub.health_calls += 1
                self._send(stub.health_status, stub.health)

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                stub.posts.append(payload)
                status, body, delay = stub.solve(payload)
                if delay:
                    time.sleep(delay)
                try:
                    self._send(status, body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return 'http://127.0.0.1:%d' % self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def first_legal_move_response(payload):
    """slot 0 on the first empty cell: legal unless the rules filter the card out (秩序/混乱 can_use)."""
    cur = payload['state']['cur']
    hand = payload['state']['hands'][cur]
    slot = next((i for i, card in enumerate(hand) if card[12] == 1), 0)
    cell = next(i for i, c in enumerate(payload['state']['board']) if c is None)
    move = {'slot': slot, 'row': cell // 3, 'col': cell % 3, 'cardId': -1 if hand[slot][0] is None else hand[slot][0]}
    return ok_response(move, payload)


def ok_response(move, payload, robust=None):
    depth = 2
    return {
        'protocol': 1, 'status': 'ok', 'move': move,
        'standard': {'move': move, 'score': 1.5, 'scoreBits': '3FF8000000000000'},
        'depths': [{'depth': d, 'move': move, 'score': 1.5, 'scoreBits': '3FF8000000000000', 'nodes': 10 * d,
                    'nodesTotal': 10 * d * (d + 1) // 2, 'ttHitsTotal': d, 'cutoffsTotal': 2 * d, 'elapsed': 0.001 * d}
                   for d in range(1, depth + 1)],
        'totals': {'nodes': 30, 'depthCompleted': depth, 'ttHits': 2, 'cutoffs': 6, 'rootMoves': 12, 'elapsed': 0.002},
        'robust': robust,
        'timings': {'totalMs': 1.0},
        'worker': 0,
    }


@pytest.fixture
def stub(monkeypatch):
    csharp_engine.reset()
    s = StubEngine()
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    monkeypatch.setenv('TTC_CS_ENGINE_URL', s.url)
    monkeypatch.setenv('TTC_CS_ENGINE_BACKOFF', '30')
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '0')
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    ai_server.ensure_handler_initialized()
    yield s
    s.close()
    csharp_engine.reset()


@pytest.fixture(scope='module')
def corpus():
    return gen_corpus.generate(20261010, 360)


def pick(corpus, *required, avoid=()):
    for entry in corpus:
        tags = set(entry['tags'])
        if all(t in tags for t in required) and not any(t in tags for t in avoid):
            return entry
    raise AssertionError('no corpus entry with tags %r' % (required,))


def post(entry, depth=2, **overrides):
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = depth
    request.update(overrides)
    random.seed(entry['seed'])
    client = ai_server.app.test_client()
    response = client.post('/ai_move', json=request)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def strip_engine_fields(body):
    body = copy.deepcopy(body)
    for key in ('engine', 'fallback_reason', 'engine_info'):
        body.pop(key, None)
    body.get('performance_stats', {}).pop('engine', None)
    body.get('performance_stats', {}).pop('fallback_reason', None)
    return body


def comparable(body):
    """The deterministic part of a response (see tools/phase7/equivalence.py for the excluded timing fields)."""
    import equivalence
    return equivalence.normalise(body)


# --------------------------------------------------------------------------- engine off

def test_engine_off_never_contacts_the_engine_and_adds_no_fields(stub, monkeypatch, corpus):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    body = post(pick(corpus, 'k:3', avoid=('full_board',)))
    assert stub.posts == [] and stub.health_calls == 0
    assert 'engine' not in body and 'fallback_reason' not in body
    assert 'engine' not in body['performance_stats']


@pytest.mark.parametrize('value', ['off', 'OFF', '0', 'false', 'no'])
def test_all_off_spellings_switch_the_engine_off(monkeypatch, value):
    monkeypatch.setenv('TTC_CS_ENGINE', value)
    assert not csharp_engine.is_enabled()


def test_the_engine_is_on_by_default(monkeypatch):
    monkeypatch.delenv('TTC_CS_ENGINE', raising=False)
    assert csharp_engine.is_enabled()


# --------------------------------------------------------------------------- the engine answers

def test_a_legal_engine_answer_is_used_and_reported(stub, monkeypatch, corpus):
    entry = pick(corpus, 'k:2', avoid=('full_board', 'rule:秩序'))
    body = post(entry, show_search_progress=True)
    assert body['engine'] == 'csharp'
    assert body['performance_stats']['engine'] == 'csharp'
    assert body['performance_stats']['nodes_searched'] == 30
    assert body['performance_stats']['search_depth'] == 2
    assert body['performance_stats']['tt_hit_rate'] == pytest.approx(2 / 30 * 100)
    assert body['performance_stats']['cutoff_rate'] == pytest.approx(6 / 30 * 100)
    assert len(stub.posts) == 1
    payload = stub.posts[0]
    assert payload['protocol'] == 1 and payload['robust'] is None and payload['maxDepth'] == 2
    move = first_legal_move_response(payload)['move']
    assert body['pos'] == [move['row'], move['col']]
    assert body['card_id'] == (None if move['cardId'] == -1 else move['cardId']) or body['card_id'] == move['cardId']
    assert [p['depth'] for p in body['search_progress']] == [1, 2]
    assert body['search_progress'][0]['best_score'] == 1.5
    assert body['search_progress'][1]['nodes_searched'] == 30 and body['search_progress'][1]['branching_factor'] == 12
    assert 'search_summary' in body


def test_the_request_is_built_before_any_python_search_in_request_hand_order(stub, corpus):
    entry = pick(corpus, 'k:3', avoid=('full_board',))
    post(entry)
    payload = stub.posts[0]
    # the hand order of the request: known cards in the order of the client's items (unknown slots are sampled in place)
    cur = payload['state']['cur']
    request = entry['request']
    my_known = [it for it in request['myHand'] if it.get('inHand', True)]
    color = 'b' if request['myOwner'] == 1 else 'r'
    hands = payload['state']['hands']
    mine = hands[1] if request['myOwner'] == 1 else hands[0]
    assert len(mine) == len(my_known)
    for card, item in zip(mine, my_known):
        if item['numU'] or item['numL'] or item['numD'] or item['numR']:
            assert card[2] == color
            assert card[4:8] == [item['numU'], item['numL'], item['numD'], item['numR']]
    assert payload['state']['cur'] == cur


def test_the_robust_section_describes_the_unknown_slots(stub, corpus):
    entry = pick(corpus, 'robust_candidate', 'unknown:1', avoid=('npc_blocked',))
    post(entry)
    robust = stub.posts[0]['robust']
    assert robust is not None
    assert robust['aiPlayer'] == stub.posts[0]['state']['cur']
    assert robust['sampleCount'] == 16  # one unknown card: min(24, max(16, 12))
    assert sum(robust['generated'][robust['opponent']]) >= 1
    assert robust['handler'] is True and robust['fallbackCandidates'] is None


# --------------------------------------------------------------------------- the fallback matrix

def run_with_stub(stub, monkeypatch, corpus, behaviour, expected_reason, entry=None, **kw):
    entry = entry or pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    reference = post(entry)
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    csharp_engine.reset()
    behaviour(stub)
    body = post(entry, **kw)
    assert body['engine'] == 'python', body
    assert body['fallback_reason'] == expected_reason, body.get('fallback_reason')
    assert body['performance_stats']['fallback_reason'] == expected_reason
    # exactly the Python-path answer
    assert comparable(strip_engine_fields(body)) == comparable(reference)
    return body


def test_fallback_busy(stub, monkeypatch, corpus):
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (503, {'protocol': 1, 'status': 'busy', 'reason': 'queue_full'}, 0)), 'busy')
    assert not csharp_engine.breaker_open()  # a busy engine is not a broken engine


def test_fallback_invalid_request(stub, monkeypatch, corpus):
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (400, {'protocol': 1, 'status': 'invalid_request', 'message': 'x'}, 0)), 'invalid_request')
    assert not csharp_engine.breaker_open()


def test_fallback_engine_error(stub, monkeypatch, corpus):
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (500, {'protocol': 1, 'status': 'error', 'message': 'boom'}, 0)), 'engine_error')


def test_fallback_malformed_json_opens_the_breaker(stub, monkeypatch, corpus):
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (200, b'this is not json', 0)), 'malformed_response')
    assert csharp_engine.breaker_open()


def test_fallback_malformed_fields(stub, monkeypatch, corpus):
    def broken(payload):
        r = first_legal_move_response(payload)
        del r['totals']
        return 200, r, 0
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', broken), 'malformed_response')


def test_fallback_unknown_status(stub, monkeypatch, corpus):
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (200, {'protocol': 1, 'status': 'weird'}, 0)), 'http_error')


def test_fallback_timeout_opens_the_breaker(stub, monkeypatch, corpus):
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '0')
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', lambda p: (200, first_legal_move_response(p), 3.5)), 'timeout', max_time=0.2)
    assert csharp_engine.breaker_open()


def test_fallback_connection_refused_opens_the_breaker_and_skips_the_engine_afterwards(stub, monkeypatch, corpus):
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    reference = post(entry)
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    csharp_engine.reset()
    stub.close()  # nothing listens on that port any more
    body = post(entry)
    # a refused connection (on Windows the refusal can take longer than the 0.5 s connect timeout: then it is a timeout)
    assert body['fallback_reason'] in ('connection_failed', 'timeout')
    assert comparable(strip_engine_fields(body)) == comparable(reference)
    assert csharp_engine.breaker_open()
    again = post(entry)
    assert again['fallback_reason'] == 'circuit_open'  # no new connection attempt
    # after the back-off the engine is tried again
    monkeypatch.setenv('TTC_CS_ENGINE_BACKOFF', '0.05')
    csharp_engine.reset()
    post(entry)
    csharp_engine._disabled_until = time.monotonic() - 1
    assert not csharp_engine.breaker_open()


def test_the_breaker_closes_after_the_backoff(stub, monkeypatch, corpus):
    entry = pick(corpus, 'k:2', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    monkeypatch.setenv('TTC_CS_ENGINE_BACKOFF', '0.3')
    stub.solve = lambda p: (200, b'garbage', 0)
    assert post(entry)['fallback_reason'] == 'malformed_response'
    assert post(entry)['fallback_reason'] == 'circuit_open'
    time.sleep(0.4)
    stub.solve = lambda p: (200, first_legal_move_response(p), 0)
    assert post(entry)['engine'] == 'csharp'


def test_fallback_protocol_mismatch(stub, monkeypatch, corpus):
    def setup(s):
        s.health = dict(s.health, protocol=2)
    run_with_stub(stub, monkeypatch, corpus, setup, 'protocol_mismatch')
    assert stub.posts == []  # never asked to solve
    assert csharp_engine.breaker_open()


def test_fallback_card_database_mismatch(stub, monkeypatch, corpus):
    def setup(s):
        s.health = dict(s.health, cardDbSha256='0' * 64)
    run_with_stub(stub, monkeypatch, corpus, setup, 'card_db_mismatch')
    assert stub.posts == []


def test_fallback_health_failure(stub, monkeypatch, corpus):
    def setup(s):
        s.health_status = 503
    run_with_stub(stub, monkeypatch, corpus, setup, 'health_failed')


@pytest.mark.parametrize('kind', ['slot_out_of_range', 'wrong_card_id', 'occupied_cell', 'cell_off_board', 'bad_types'])
def test_fallback_illegal_move(stub, monkeypatch, corpus, kind):
    entry = pick(corpus, 'k:4', avoid=('full_board', 'rule:秩序', 'rule:混乱'))

    def bad(payload):
        r = first_legal_move_response(payload)
        m = r['move']
        if kind == 'slot_out_of_range':
            m['slot'] = 9
        elif kind == 'wrong_card_id':
            m['cardId'] = 99999
        elif kind == 'occupied_cell':
            cell = next(i for i, c in enumerate(payload['state']['board']) if c is not None)
            m['row'], m['col'] = cell // 3, cell % 3
        elif kind == 'cell_off_board':
            m['row'] = 3
        else:
            m['slot'] = '0'
        return 200, r, 0
    expected = 'malformed_response' if kind == 'bad_types' else 'illegal_move'
    run_with_stub(stub, monkeypatch, corpus, lambda s: setattr(s, 'solve', bad), expected, entry=entry)


def test_fallback_a_card_that_the_order_rule_forbids(stub, monkeypatch, corpus):
    entry = pick(corpus, 'rule:秩序', 'k:3', avoid=('full_board',))
    request = entry['request']
    # make the first card unplayable and let the stub choose exactly that one
    for item in request['myHand'] + request['oppHand']:
        item['canUse'] = False
    entry = dict(entry, request=request)

    def bad(payload):
        cur = payload['state']['cur']
        hand = payload['state']['hands'][cur]
        cell = next(i for i, c in enumerate(payload['state']['board']) if c is None)
        return 200, ok_response({'slot': 0, 'row': cell // 3, 'col': cell % 3, 'cardId': -1 if hand[0][0] is None else hand[0][0]}, payload), 0
    csharp_engine.reset()
    stub.solve = bad
    result = post(entry)
    # every card is unusable under the order rule: the stub still names slot 0, which is not playable -> fallback to
    # the Python path, which finds no move at all
    assert result['move'] is None
    assert result['engine'] == 'python' and result['fallback_reason'] == 'illegal_move'


@pytest.mark.parametrize('name,value', [('max_depth', 'ten'), ('max_depth', True), ('max_time', '10'), ('max_time', None)])
def test_unsupported_parameters_are_left_to_the_python_path(stub, monkeypatch, corpus, name, value):
    entry = pick(corpus, 'k:3', avoid=('full_board',))
    overrides = {name: value}
    request = copy.deepcopy(entry['request'])
    request.update(overrides)
    request.setdefault('max_depth', 2)
    random.seed(entry['seed'])
    # whatever the Python path does with such a value (maybe a 500), the engine must not have been asked
    ai_server.app.test_client().post('/ai_move', json=request)
    assert stub.posts == []


def test_a_nan_or_infinite_time_limit_is_sent_as_no_limit():
    assert csharp_engine.wire_max_time(float('nan')) is None
    assert csharp_engine.wire_max_time(float('inf')) is None
    assert csharp_engine.wire_max_time(float('-inf')) < 0
    assert csharp_engine.wire_max_time(10) == 10
    assert csharp_engine.wire_max_depth(1000) == 100
    assert csharp_engine.wire_max_depth(float('nan')) == 0
    assert csharp_engine.wire_max_depth(float('-inf')) == -float('inf') or csharp_engine.wire_max_depth(float('-inf')) < 0
    assert csharp_engine.wire_max_depth(2.7) == 2.7


# --------------------------------------------------------------------------- the random fallback round trip

def fallback_stub(stub, second):
    """First call: needs_fallback_candidates; second call: ``second(payload)``."""
    calls = {'n': 0}

    def solve(payload):
        calls['n'] += 1
        if calls['n'] == 1:
            return 200, {'protocol': 1, 'status': 'needs_fallback_candidates', 'unknownCount': 1}, 0
        return second(payload)
    stub.solve = solve
    return calls


def test_needs_fallback_candidates_round_trip_consumes_random_numbers_like_python(stub, monkeypatch, corpus):
    entry = pick(corpus, 'fallback_path')
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    calls = fallback_stub(stub, lambda p: (200, first_legal_move_response(p), 0))

    random.seed(entry['seed'])
    body = ai_server.app.test_client().post('/ai_move', json=request).get_json()
    assert body['engine'] == 'csharp' and calls['n'] == 2
    sent = stub.posts[1]['robust']['fallbackCandidates']
    assert sent is not None and all(len(card) == 13 for card in sent)
    after_engine = random.random()

    # the Python path: the same request with the engine off draws the handler's random cards at the same point
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    random.seed(entry['seed'])
    ai_server.app.test_client().post('/ai_move', json=request)
    assert random.random() == after_engine


def test_a_failure_after_the_fallback_candidates_restores_the_random_state(stub, monkeypatch, corpus):
    entry = pick(corpus, 'fallback_path')
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    fallback_stub(stub, lambda p: (503, {'protocol': 1, 'status': 'busy', 'reason': 'queue_full'}, 0))

    random.seed(entry['seed'])
    body = ai_server.app.test_client().post('/ai_move', json=request).get_json()
    assert body['engine'] == 'python' and body['fallback_reason'] == 'busy'
    after_fallback = random.random()

    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    random.seed(entry['seed'])
    reference = ai_server.app.test_client().post('/ai_move', json=request).get_json()
    assert random.random() == after_fallback  # the engine attempt left no trace in the generator
    assert comparable(strip_engine_fields(body)) == comparable(reference)


# --------------------------------------------------------------------------- the opt-in recorder

def test_the_request_recorder_is_off_by_default_and_records_when_asked(tmp_path, monkeypatch, corpus):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    entry = pick(corpus, 'k:2', avoid=('full_board',))
    monkeypatch.delenv('TTC_RECORD_REQUESTS', raising=False)
    post(entry)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv('TTC_RECORD_REQUESTS', str(tmp_path))
    post(entry)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding='utf-8'))
    assert saved['request']['myOwner'] == entry['request']['myOwner']
    state = saved['random_state']
    random.setstate((state[0], tuple(state[1]), state[2]))  # a valid generator state: what a replay needs


# --------------------------------------------------------------------------- the remaining reasons

def test_fallback_unexpected_status_after_the_fallback_candidates(stub, monkeypatch, corpus):
    entry = pick(corpus, 'fallback_path')
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    fallback_stub(stub, lambda p: (200, {'protocol': 1, 'status': 'needs_fallback_candidates', 'unknownCount': 1}, 0))
    random.seed(entry['seed'])
    body = ai_server.app.test_client().post('/ai_move', json=request).get_json()
    assert body['engine'] == 'python' and body['fallback_reason'] == 'unexpected_status'
    after = random.random()
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    random.seed(entry['seed'])
    ai_server.app.test_client().post('/ai_move', json=request)
    assert random.random() == after  # the generator was restored for the Python path


def test_fallback_client_error_never_fails_the_request(stub, monkeypatch, corpus):
    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(csharp_engine, 'state_payload', boom)
    run_with_stub(stub, monkeypatch, corpus, lambda s: None, 'client_error')
