"""Review M7-1 / M7-2 reproduced automatically: the real engine process, the real Flask app behind a real threaded HTTP server.

The reviewer's overload probe: an engine with ONE worker and 12 concurrent opening requests. The old client gave up after
``max_time + margin + 1`` seconds while its requests were still queued in the engine, fell back to Python (about 100 times slower), and the
timeouts opened a 30 s circuit breaker that sent even cheap requests to Python. Now (A-20) a queued request waits for the engine:
every request is answered by the engine, with no client timeout and no Python search; only a full queue answers 503 ``busy``.
The searches are short (``max_time`` 1 s) so that the test runs in about 15 s, but the queue is still longer than the old read timeout.
"""
import copy
import http.client
import json
import os
import random
import sys
import threading
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_server  # noqa: E402
from ai import csharp_engine  # noqa: E402
import gen_corpus  # noqa: E402
from engine_process import engine_process, flask_server, get_json, tcp_connections_to  # noqa: E402


class NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture(scope='module')
def corpus():
    return gen_corpus.generate(20261010, 360)


@pytest.fixture
def python_solver_calls(monkeypatch):
    """The Python solver entry points are replaced by recorders: any call is a failure of the test (A-20)."""
    calls = []
    for name in ('find_best_move_parallel', 'select_endgame_robust_move', '_solve_minimax_python'):
        def recorder(*a, _n=name, **k):
            calls.append(_n)
            raise AssertionError('the Python solver ran on the engine path: ' + _n)
        monkeypatch.setattr(ai_server, name, recorder)
    return calls


@pytest.fixture
def server_env(monkeypatch):
    csharp_engine.reset()
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '1')  # the old read timeout was max_time + 1 + 1 = 3 s here
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    ai_server.ensure_handler_initialized()
    yield
    csharp_engine.reset()


def openings(corpus, n):
    entries = [e for e in corpus if 'opening' in e['tags']]
    assert len(entries) >= 1
    return [entries[i % len(entries)] for i in range(n)]


def post_json(base, request, timeout=120):
    host, port = base.replace('http://', '').split(':')
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
    try:
        t0 = time.monotonic()
        conn.request('POST', '/ai_move', body=json.dumps(request), headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        body = json.loads(response.read().decode('utf-8'))
        return response.status, body, time.monotonic() - t0
    finally:
        conn.close()


def burst(base, entries, max_time):
    results = [None] * len(entries)

    def one(i):
        request = copy.deepcopy(entries[i]['request'])
        request['max_depth'] = 10
        request['max_time'] = max_time
        results[i] = post_json(base, request)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(entries))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert all(r is not None for r in results), 'a request did not finish'
    return results


def test_one_worker_and_a_dozen_concurrent_openings_are_all_answered_by_the_engine(corpus, server_env, python_solver_calls, monkeypatch):
    max_time = 1.0
    old_read_timeout = max_time + 1 + 1  # the pre-hardening client: max_time + margin + 1
    with engine_process(workers=1, queue=64, queue_wait_ms=120000) as engine:
        monkeypatch.setenv('TTC_CS_ENGINE_URL', engine['url'])
        with flask_server(ai_server.app) as base:
            random.seed(1)
            results = burst(base, openings(corpus, 12), max_time)
        health = get_json(engine['url'] + '/v1/health')

    statuses = [r[0] for r in results]
    assert statuses == [200] * 12, [(r[0], r[1].get('engine_error')) for r in results]
    assert all(r[1]['engine'] == 'csharp' for r in results)
    latencies = sorted(r[2] for r in results)
    assert latencies[-1] > old_read_timeout, 'the queue was not longer than the old read timeout (%.1f s): not a reproduction' % latencies[-1]
    assert python_solver_calls == []  # 0 Python solver calls
    assert health['served'] == 12 and health['busyAnswers'] == 0 and health['workersCreated'] == 1  # all by the one engine worker
    print('latencies (s):', ['%.1f' % x for x in latencies], 'old read timeout %.1f s' % old_read_timeout)


def test_a_full_queue_answers_503_busy_and_nothing_times_out(corpus, server_env, python_solver_calls, monkeypatch):
    with engine_process(workers=1, queue=2, queue_wait_ms=120000) as engine:
        monkeypatch.setenv('TTC_CS_ENGINE_URL', engine['url'])
        with flask_server(ai_server.app) as base:
            random.seed(2)
            results = burst(base, openings(corpus, 8), 1.0)
        health = get_json(engine['url'] + '/v1/health')

    ok = [r for r in results if r[0] == 200]
    busy = [r for r in results if r[0] == 503]
    assert len(ok) + len(busy) == 8, [(r[0], r[1].get('engine_error')) for r in results]  # no other outcome: no timeout, no 500
    assert 1 <= len(ok) <= 4 and len(busy) >= 4  # one running + two queued (+ at most a search that finished meanwhile)
    for status, body, _ in busy:
        assert body['engine'] == 'csharp' and body['engine_error'] == 'busy'
        assert 'queue_full' in body['error']
    assert all(r[1]['engine'] == 'csharp' for r in ok)
    assert python_solver_calls == []
    assert health['busyAnswers'] == len(busy)


def test_sixty_sequential_requests_through_threaded_flask_leave_at_most_the_pool_size_of_connections(corpus, server_env, python_solver_calls, monkeypatch):
    """
    The reviewer's M7-2 probe. Before the fix: 61 client-side connections to the engine (TIME_WAIT) after 60 requests.
    Counted with ``netstat``, every TCP state, before and after.
    """
    pool_size = 4
    monkeypatch.setenv('TTC_CS_ENGINE_POOL_SIZE', str(pool_size))
    entries = [e for e in corpus if 'k:3' in e['tags'] and 'full_board' not in e['tags']]
    with engine_process(workers=2) as engine:
        monkeypatch.setenv('TTC_CS_ENGINE_URL', engine['url'])
        before = tcp_connections_to(engine['port'])
        with flask_server(ai_server.app) as base:
            for i in range(60):
                request = copy.deepcopy(entries[i % len(entries)]['request'])
                request['max_depth'] = 3
                random.seed(i)
                status, body, _ = post_json(base, request)
                assert status == 200 and body['engine'] == 'csharp', body
            after = tcp_connections_to(engine['port'])
        print('connections to the engine (netstat, all states): before %d, after 60 requests %d' % (before, after))
        assert after - before <= pool_size
        assert csharp_engine.pool_idle() <= pool_size
        assert get_json(engine['url'] + '/v1/health')['served'] == 60
    assert python_solver_calls == []
