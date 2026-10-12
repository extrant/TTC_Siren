"""Phase 7 (with the Phase 7 hardening, decision A-20): the C# engine client and the dispatch in ``ai_move``.

A stub HTTP server plays the engine, so every engine problem, the ``needs_fallback_candidates`` round trip, the response mapping and the
random-number bookkeeping can be tested without the real engine. While the engine is on it is the ONLY minimax solver: every problem
is answered with an HTTP error (503 + ``engine_error``) and the Python solver is never called (the ``python_solver`` spy counts calls
to ``find_best_move_parallel``, ``select_endgame_robust_move`` and ``_solve_minimax_python``). With the engine off by hand
(``TTC_CS_ENGINE=off``) the Python path must produce exactly the response it produced before Phase 7
(tests/test_engine_off_equivalence.py).
"""
import copy
import json
import os
import random
import socket
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
    """Answers GET /v1/health and POST /v1/solve as programmed by the test; counts the connections it accepts."""

    def __init__(self, port=0, queue_wait_ms=1000):
        self.health = {'protocol': 1, 'status': 'ok', 'cardDbSha256': csharp_engine.card_db_sha256(), 'queueWaitMs': queue_wait_ms}
        self.health_status = 200
        self.solve = lambda payload: (200, first_legal_move_response(payload), 0.0)
        self.posts = []
        self.health_calls = 0
        self.accepted = 0
        self.sockets = []
        self.keep_alive = True
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
                if not stub.keep_alive:
                    self.send_header('Connection', 'close')
                    self.close_connection = True
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

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self):
                request = super().get_request()
                stub.accepted += 1
                stub.sockets.append(request[0])
                return request

        self.server = Server(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    @property
    def url(self):
        return 'http://127.0.0.1:%d' % self.port

    def close(self):
        """Stop listening and drop every connection, as a killed engine process would."""
        self.server.shutdown()
        self.server.server_close()
        self.drop_connections()

    def drop_connections(self):
        """Close the accepted connections (the engine restarted, or closed an idle keep-alive connection)."""
        sockets, self.sockets = self.sockets, []
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


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
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '0')
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    ai_server.ensure_handler_initialized()
    yield s
    s.close()
    csharp_engine.reset()


class PythonSolverSpy:
    """Counts calls of the Python solver entry points; calls through, so that the engine-off path still works."""

    def __init__(self):
        self.calls = []


@pytest.fixture
def python_solver(monkeypatch):
    spy = PythonSolverSpy()
    for name in ('find_best_move_parallel', 'select_endgame_robust_move', '_solve_minimax_python'):
        real = getattr(ai_server, name)

        def wrapper(*args, _real=real, _name=name, **kwargs):
            spy.calls.append(_name)
            return _real(*args, **kwargs)
        monkeypatch.setattr(ai_server, name, wrapper)
    return spy


@pytest.fixture(scope='module')
def corpus():
    return gen_corpus.generate(20261010, 360)


def pick(corpus, *required, avoid=()):
    for entry in corpus:
        tags = set(entry['tags'])
        if all(t in tags for t in required) and not any(t in tags for t in avoid):
            return entry
    raise AssertionError('no corpus entry with tags %r' % (required,))


def post_raw(entry, depth=2, **overrides):
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = depth
    request.update(overrides)
    random.seed(entry['seed'])
    return ai_server.app.test_client().post('/ai_move', json=request)


def post(entry, depth=2, **overrides):
    response = post_raw(entry, depth, **overrides)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


# --------------------------------------------------------------------------- engine off (the manual switch)

def test_engine_off_never_contacts_the_engine_and_adds_no_fields(stub, monkeypatch, corpus):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    body = post(pick(corpus, 'k:3', avoid=('full_board',)))
    assert stub.posts == [] and stub.health_calls == 0
    assert 'engine' not in body and 'engine_error' not in body
    assert 'engine' not in body['performance_stats']


def test_engine_off_runs_the_python_solver(stub, monkeypatch, corpus, python_solver):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    post(pick(corpus, 'k:3', avoid=('full_board',)))
    assert '_solve_minimax_python' in python_solver.calls and 'find_best_move_parallel' in python_solver.calls


@pytest.mark.parametrize('value', ['off', 'OFF', '0', 'false', 'no'])
def test_all_off_spellings_switch_the_engine_off(monkeypatch, value):
    monkeypatch.setenv('TTC_CS_ENGINE', value)
    assert not csharp_engine.is_enabled()


def test_the_engine_is_on_by_default(monkeypatch):
    monkeypatch.delenv('TTC_CS_ENGINE', raising=False)
    assert csharp_engine.is_enabled()


def test_nothing_switches_the_engine_off_automatically(stub, monkeypatch, corpus, python_solver):
    """After every kind of failure the engine is still tried by the next request: ``is_enabled`` depends on the environment only."""
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    stub.solve = lambda p: (500, {'protocol': 1, 'status': 'error', 'message': 'boom'}, 0)
    for _ in range(3):
        assert post_raw(entry).status_code == 503
        assert csharp_engine.is_enabled()
    stub.solve = lambda p: (200, first_legal_move_response(p), 0)
    assert post(entry)['engine'] == 'csharp'
    assert python_solver.calls == []


# --------------------------------------------------------------------------- the engine answers

def test_a_legal_engine_answer_is_used_and_reported(stub, monkeypatch, corpus, python_solver):
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
    assert python_solver.calls == []


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


# --------------------------------------------------------------------------- the error matrix (A-20): 503 + engine_error, no Python solver

def expect_engine_error(stub, corpus, python_solver, behaviour, reason, status=503, entry=None, **kw):
    """Program the stub with ``behaviour``, post a request and check the A-20 answer. Returns the response body."""
    entry = entry or pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    behaviour(stub)
    response = post_raw(entry, **kw)
    body = response.get_json()
    assert response.status_code == status, body
    assert body['engine'] == 'csharp'
    assert body['engine_error'] == reason, body
    assert isinstance(body['error'], str) and reason in body['error']
    assert 'card' not in body and 'pos' not in body and 'move' not in body  # no answer was computed
    assert python_solver.calls == [], 'the Python solver ran on the engine path: %r' % python_solver.calls
    return body


def test_error_busy(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (503, {'protocol': 1, 'status': 'busy', 'reason': 'queue_full'}, 0)), 'busy')


def test_error_invalid_request(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (400, {'protocol': 1, 'status': 'invalid_request', 'message': 'x'}, 0)), 'invalid_request')


def test_error_engine_error(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (500, {'protocol': 1, 'status': 'error', 'message': 'boom'}, 0)), 'engine_error')


def test_error_malformed_json(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (200, b'this is not json', 0)), 'malformed_response')


def test_error_malformed_fields(stub, corpus, python_solver):
    def broken(payload):
        r = first_legal_move_response(payload)
        del r['totals']
        return 200, r, 0
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', broken), 'malformed_response')


def test_error_unknown_status(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (200, {'protocol': 1, 'status': 'weird'}, 0)), 'http_error')


def test_error_timeout(stub, monkeypatch, corpus, python_solver):
    # read timeout = max_time 0.2 + the engine's queue wait (health: 0.3 s) + margin 0 = 0.5 s; the engine takes 2 s
    stub.health['queueWaitMs'] = 300
    body = expect_engine_error(stub, corpus, python_solver,
                               lambda s: setattr(s, 'solve', lambda p: (200, first_legal_move_response(p), 2.0)), 'timeout', max_time=0.2)
    assert 'timeout' in body['error']


def test_error_connection_refused_then_the_engine_comes_back(stub, corpus, python_solver):
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    port = stub.port
    post(entry)  # verified and working
    stub.close()  # nothing listens on that port any more
    # on Windows a refused connection can take longer than the 0.5 s connect timeout: that is reported as connection_failed too
    body = expect_engine_error(stub, corpus, python_solver, lambda s: None, 'connection_failed', entry=entry)
    assert csharp_engine.is_enabled()
    again = expect_engine_error(stub, corpus, python_solver, lambda s: None, 'connection_failed', entry=entry)  # still tried, no 'circuit_open'
    assert again['engine_error'] == 'connection_failed' and body['engine_error'] == 'connection_failed'
    # the engine is started again (on the same port): the next request is answered by it, with no back-off in between
    revived = StubEngine(port=port)
    try:
        assert post(entry)['engine'] == 'csharp'
        assert revived.health_calls >= 1  # it was verified again
    finally:
        revived.close()


def test_error_protocol_mismatch(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: s.health.update(protocol=2), 'protocol_mismatch')
    assert stub.posts == []  # never asked to solve


def test_error_card_database_mismatch(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: s.health.update(cardDbSha256='0' * 64), 'card_db_mismatch')
    assert stub.posts == []


def test_error_health_failure(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'health_status', 503), 'health_failed')


def test_error_health_without_queue_wait(stub, corpus, python_solver):
    expect_engine_error(stub, corpus, python_solver, lambda s: s.health.pop('queueWaitMs'), 'health_failed')


def test_a_failed_verification_is_not_cached_so_a_corrected_engine_is_accepted_at_once(stub, corpus, python_solver):
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    expect_engine_error(stub, corpus, python_solver, lambda s: s.health.update(cardDbSha256='0' * 64), 'card_db_mismatch', entry=entry)
    stub.health['cardDbSha256'] = csharp_engine.card_db_sha256()
    assert post(entry)['engine'] == 'csharp'  # no breaker, no waiting


@pytest.mark.parametrize('kind', ['slot_out_of_range', 'wrong_card_id', 'occupied_cell', 'cell_off_board', 'bad_types'])
def test_error_illegal_move(stub, corpus, python_solver, kind):
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
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', bad), expected, entry=entry)


def test_error_a_card_that_the_order_rule_forbids(stub, corpus, python_solver):
    entry = pick(corpus, 'rule:秩序', 'k:3', avoid=('full_board',))
    request = entry['request']
    # make every card unusable and let the stub choose slot 0 anyway
    for item in request['myHand'] + request['oppHand']:
        item['canUse'] = False
    entry = dict(entry, request=request)

    def bad(payload):
        cur = payload['state']['cur']
        hand = payload['state']['hands'][cur]
        cell = next(i for i, c in enumerate(payload['state']['board']) if c is None)
        return 200, ok_response({'slot': 0, 'row': cell // 3, 'col': cell % 3, 'cardId': -1 if hand[0][0] is None else hand[0][0]}, payload), 0
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', bad), 'illegal_move', entry=entry)


@pytest.mark.parametrize('name,value', [('max_depth', 'ten'), ('max_time', '10'), ('max_time', None), ('max_depth', [3])])
def test_a_parameter_that_is_not_a_number_is_a_400_and_the_engine_is_not_asked(stub, corpus, python_solver, name, value):
    entry = pick(corpus, 'k:3', avoid=('full_board',))
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    request[name] = value
    random.seed(entry['seed'])
    response = ai_server.app.test_client().post('/ai_move', json=request)
    body = response.get_json()
    assert response.status_code == 400 and body['engine_error'] == 'unsupported_parameter' and body['engine'] == 'csharp'
    assert stub.posts == [] and python_solver.calls == []


def test_a_bool_is_the_integer_it_is_in_python(stub, corpus):
    entry = pick(corpus, 'k:3', avoid=('full_board',))
    body = post(entry, depth=True, max_time=True)
    assert body['engine'] == 'csharp'
    assert stub.posts[0]['maxDepth'] == 1 and stub.posts[0]['maxTime'] == 1


def test_a_defect_of_the_client_code_is_a_500_with_a_reason(stub, monkeypatch, corpus, python_solver):
    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(csharp_engine, 'state_payload', boom)
    body = expect_engine_error(stub, corpus, python_solver, lambda s: None, 'client_error', status=500)
    assert 'RuntimeError' in body['error']


def test_every_error_is_logged_at_error_level(stub, corpus, python_solver, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger='ttc.engine')
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (503, {'protocol': 1, 'status': 'busy', 'reason': 'queue_full'}, 0)), 'busy')
    records = [r for r in caplog.records if r.name == 'ttc.engine']
    assert records and records[-1].levelno == logging.ERROR and 'busy' in records[-1].getMessage()


def test_a_failed_request_is_recorded_when_recording_is_on(stub, monkeypatch, tmp_path, corpus, python_solver):
    monkeypatch.setenv('TTC_RECORD_REQUESTS', str(tmp_path))
    expect_engine_error(stub, corpus, python_solver, lambda s: setattr(s, 'solve', lambda p: (500, {'protocol': 1, 'status': 'error', 'message': 'boom'}, 0)), 'engine_error')
    assert len(list(tmp_path.iterdir())) == 1


def test_a_nan_or_infinite_time_limit_is_sent_as_no_limit():
    assert csharp_engine.wire_max_time(float('nan')) is None
    assert csharp_engine.wire_max_time(float('inf')) is None
    assert csharp_engine.wire_max_time(float('-inf')) < 0
    assert csharp_engine.wire_max_time(10) == 10
    assert csharp_engine.wire_max_depth(1000) == 100
    assert csharp_engine.wire_max_depth(float('nan')) == 0
    assert csharp_engine.wire_max_depth(float('-inf')) == -float('inf') or csharp_engine.wire_max_depth(float('-inf')) < 0
    assert csharp_engine.wire_max_depth(2.7) == 2.7


def test_the_http_status_of_each_reason():
    assert csharp_engine.EngineError('busy').http_status == 503
    assert csharp_engine.EngineError('timeout').http_status == 503
    assert csharp_engine.EngineError('illegal_move').http_status == 503
    assert csharp_engine.EngineError('unsupported_parameter').http_status == 400
    assert csharp_engine.EngineError('client_error').http_status == 500


# --------------------------------------------------------------------------- a slow engine is waited for (review M7-1)

def test_the_read_timeout_is_max_time_plus_the_engines_queue_wait_plus_a_margin(stub, monkeypatch):
    stub.health['queueWaitMs'] = 20_000
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '2')
    csharp_engine._verify()
    assert csharp_engine.read_timeout_for(10) == pytest.approx(10 + 20 + 2)
    assert csharp_engine.read_timeout_for(0) == pytest.approx(22)
    assert csharp_engine.read_timeout_for(None) > 3600


def test_an_engine_that_answers_after_the_old_timeout_is_still_waited_for(stub, monkeypatch, corpus, python_solver):
    """
    The old client gave up after max_time + margin + 1 s even when the engine was merely queueing; that opened the breaker and moved the
    load to Python (M7-1). Now: max_time 0.2 + queue wait 1.5 s + margin 0.3 s = 2.0 s; the engine answers after 1.6 s (> the old
    0.2 + 0.3 + 1 = 1.5 s) and the answer is used.
    """
    stub.health['queueWaitMs'] = 1500
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '0.3')
    stub.solve = lambda p: (200, first_legal_move_response(p), 1.6)
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    started = time.monotonic()
    body = post(entry, max_time=0.2)
    assert time.monotonic() - started > 1.5
    assert body['engine'] == 'csharp'
    assert python_solver.calls == []


def test_a_timeout_does_not_disable_the_engine(stub, monkeypatch, corpus, python_solver):
    stub.health['queueWaitMs'] = 100
    entry = pick(corpus, 'k:3', avoid=('full_board', 'rule:秩序', 'rule:混乱'))
    stub.solve = lambda p: (200, first_legal_move_response(p), 1.5)
    expect_engine_error(stub, corpus, python_solver, lambda s: None, 'timeout', entry=entry, max_time=0.1)
    stub.solve = lambda p: (200, first_legal_move_response(p), 0)
    assert post(entry, max_time=0.1)['engine'] == 'csharp'  # the very next request is answered by the engine


# --------------------------------------------------------------------------- the random fallback round trip (not a solver fallback)

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


def test_needs_fallback_candidates_round_trip_consumes_random_numbers_like_python(stub, monkeypatch, corpus, python_solver):
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
    assert python_solver.calls == []

    # the Python path: the same request with the engine off draws the handler's random cards at the same point
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    random.seed(entry['seed'])
    ai_server.app.test_client().post('/ai_move', json=request)
    assert random.random() == after_engine


def random_state_after(stub, entry, request, behaviour):
    stub.solve = behaviour
    random.seed(entry['seed'])
    response = ai_server.app.test_client().post('/ai_move', json=request)
    return response, random.random()


def test_a_failure_after_the_fallback_candidates_leaves_no_trace_in_the_random_generator(stub, corpus, python_solver):
    """A failed request must not consume random numbers, so that a retry of the same request samples the same cards."""
    entry = pick(corpus, 'fallback_path')
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    busy = (503, {'protocol': 1, 'status': 'busy', 'reason': 'queue_full'}, 0)

    # reference: the engine fails at once, before the handler's random cards are drawn
    response, expected = random_state_after(stub, entry, request, lambda p: busy)
    assert response.status_code == 503 and response.get_json()['engine_error'] == 'busy'

    # the engine asks for the fallback candidates (the handler draws random cards) and then fails
    fallback_stub(stub, lambda p: busy)
    response, after_failure = random_state_after(stub, entry, request, stub.solve)
    assert response.status_code == 503 and response.get_json()['engine_error'] == 'busy'
    assert after_failure == expected
    assert python_solver.calls == []


def test_error_unexpected_status_after_the_fallback_candidates(stub, corpus, python_solver):
    entry = pick(corpus, 'fallback_path')
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = 2
    fallback_stub(stub, lambda p: (200, {'protocol': 1, 'status': 'needs_fallback_candidates', 'unknownCount': 1}, 0))
    random.seed(entry['seed'])
    response = ai_server.app.test_client().post('/ai_move', json=request)
    assert response.status_code == 503 and response.get_json()['engine_error'] == 'unexpected_status'
    assert python_solver.calls == []


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
