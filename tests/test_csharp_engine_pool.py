"""Review M7-2: the engine client keeps one process-wide, thread-safe pool of keep-alive connections.

Flask's threaded server runs every request on a new thread. The old client kept its connection in ``threading.local``, so each request
opened (and abandoned) a TCP connection: 61 TIME_WAIT sockets after 60 requests. Here a stub engine counts the connections it accepts.
"""
import http.client
import os
import sys
import threading
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai import csharp_engine  # noqa: E402
from test_csharp_engine_client import StubEngine  # noqa: E402

PAYLOAD = {'protocol': 1}
OK = {'protocol': 1, 'status': 'ok'}


@pytest.fixture
def engine(monkeypatch):
    csharp_engine.reset()
    stub = StubEngine()
    stub.solve = lambda payload: (200, OK, 0)
    monkeypatch.setenv('TTC_CS_ENGINE_URL', stub.url)
    monkeypatch.setenv('TTC_CS_ENGINE_POOL_SIZE', '4')
    monkeypatch.setenv('TTC_CS_ENGINE_READ_MARGIN', '1')
    yield stub
    stub.close()
    csharp_engine.reset()


@pytest.fixture
def created(monkeypatch):
    """Every HTTPConnection the client creates, so that a test can check that none is left open."""
    made = []
    real = http.client.HTTPConnection

    class Recording(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(http.client, 'HTTPConnection', Recording)
    return made


def run_on_new_thread(target):
    box = {}

    def body():
        try:
            box['result'] = target()
        except BaseException as e:  # noqa: BLE001
            box['error'] = e
    t = threading.Thread(target=body)
    t.start()
    t.join(30)
    assert not t.is_alive()
    if 'error' in box:
        raise box['error']
    return box['result']


def test_requests_on_short_lived_threads_share_the_pooled_connection(engine, created):
    for _ in range(40):
        assert run_on_new_thread(lambda: csharp_engine.solve(PAYLOAD, 1))['status'] == 'ok'
    assert len(engine.posts) == 40
    assert engine.health_calls == 1  # verified once
    assert engine.accepted <= 4  # never more than the pool size; in fact one connection serves them all
    assert engine.accepted == 1
    assert csharp_engine.pool_idle() == 1
    csharp_engine.close_pool()
    assert csharp_engine.pool_idle() == 0
    assert all(c.sock is None for c in created), 'a connection was left open outside the pool'


def test_more_concurrent_requests_than_the_pool_size_keep_at_most_the_pool_size_afterwards(engine, created):
    engine.solve = lambda payload: (200, OK, 0.3)
    errors = []

    def one():
        try:
            csharp_engine.solve(PAYLOAD, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=one) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert engine.accepted > 4  # the burst needed more connections than the pool keeps ...
    assert csharp_engine.pool_idle() == 4  # ... and only the pool size is kept
    open_now = [c for c in created if c.sock is not None]
    assert len(open_now) == 4, 'the surplus connections must be closed, not left to the garbage collector'
    accepted = engine.accepted
    engine.solve = lambda payload: (200, OK, 0)
    for _ in range(10):
        csharp_engine.solve(PAYLOAD, 1)
    assert engine.accepted == accepted  # the pooled connections serve the following requests
    csharp_engine.close_pool()
    assert all(c.sock is None for c in created)


def test_a_pool_of_size_zero_closes_every_connection_after_use(engine, created, monkeypatch):
    monkeypatch.setenv('TTC_CS_ENGINE_POOL_SIZE', '0')
    for _ in range(3):
        csharp_engine.solve(PAYLOAD, 1)
    assert csharp_engine.pool_idle() == 0
    assert all(c.sock is None for c in created)


def test_the_pool_is_last_in_first_out(engine):
    engine.solve = lambda payload: (200, OK, 0.2)
    threads = [threading.Thread(target=lambda: csharp_engine.solve(PAYLOAD, 1)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert csharp_engine.pool_idle() == 3
    newest = csharp_engine._pool[-1][1]
    conn, reused = csharp_engine._checkout(csharp_engine.engine_url(), False)
    assert reused and conn is newest
    csharp_engine._close(conn)


def test_an_idle_connection_closed_by_the_server_is_replaced_once_and_the_engine_is_verified_again(engine):
    csharp_engine.solve(PAYLOAD, 1)
    assert engine.accepted == 1 and engine.health_calls == 1
    engine.drop_connections()  # the server closed the idle keep-alive connection (or the engine restarted)
    assert csharp_engine.solve(PAYLOAD, 1)['status'] == 'ok'
    assert engine.accepted == 2  # one new connection, no failure reported
    assert engine.health_calls == 2  # a restarted engine may have loaded another card database: verified again before the retry
    assert len(engine.posts) == 2  # the solve was not sent twice (the first attempt never reached the server)


def test_a_connection_that_fails_is_discarded(engine, created):
    csharp_engine.solve(PAYLOAD, 1)
    engine.solve = lambda payload: (200, OK, 3.0)
    with pytest.raises(csharp_engine.EngineError) as e:
        csharp_engine.solve(PAYLOAD, 0.1)  # read timeout 0.1 + the engine's queue wait 1 + margin 1 = 2.1 s < the engine's 3 s
    assert e.value.reason == 'timeout'
    assert csharp_engine.pool_idle() == 0  # nothing from the failed exchange went back to the pool
    assert all(c.sock is None for c in created)
    engine.solve = lambda payload: (200, OK, 0)
    assert csharp_engine.solve(PAYLOAD, 1)['status'] == 'ok'  # a new connection works at once
    assert csharp_engine.pool_idle() == 1


def test_a_server_that_closes_after_each_answer_is_not_pooled(engine, created):
    engine.keep_alive = False
    for _ in range(3):
        assert csharp_engine.solve(PAYLOAD, 1)['status'] == 'ok'
    assert csharp_engine.pool_idle() == 0
    assert all(c.sock is None for c in created)


def test_a_pooled_connection_is_not_used_for_another_url(engine, created, monkeypatch):
    csharp_engine.solve(PAYLOAD, 1)
    other = StubEngine()
    other.solve = lambda payload: (200, OK, 0)
    try:
        monkeypatch.setenv('TTC_CS_ENGINE_URL', other.url)
        assert csharp_engine.solve(PAYLOAD, 1)['status'] == 'ok'
        assert other.accepted == 1 and len(other.posts) == 1
        assert csharp_engine.pool_idle() == 1  # the connection to the old URL was closed, the new one is pooled
        assert len([c for c in created if c.sock is not None]) == 1
    finally:
        csharp_engine.close_pool()
        other.close()


def test_reset_closes_the_pooled_connections(engine, created):
    csharp_engine.solve(PAYLOAD, 1)
    assert any(c.sock is not None for c in created)
    csharp_engine.reset()
    assert csharp_engine.pool_idle() == 0
    assert all(c.sock is None for c in created)
