"""Phase 7: Flask ``ai_server`` + the real C# engine process (built from csharp/src/TTC.Siren.Engine).

The engine is started as ``dotnet TTC.Siren.Engine.dll`` on a free loopback port. These tests need the .NET 10 SDK/runtime; if the engine
has not been built yet it is built first (``dotnet build -c Release``). Nothing here is skipped: without .NET the fixture fails.
"""
import copy
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))

import ai_server  # noqa: E402
from ai import csharp_engine  # noqa: E402
import gen_corpus  # noqa: E402

ENGINE_DIR = os.path.join(ROOT, 'csharp', 'src', 'TTC.Siren.Engine')
ENGINE_DLL = os.path.join(ENGINE_DIR, 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')


class NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def engine():
    if not os.path.exists(ENGINE_DLL):
        subprocess.run(['dotnet', 'build', os.path.join(ENGINE_DIR, 'TTC.Siren.Engine.csproj'), '-c', 'Release'], check=True,
                       capture_output=True)
    port = free_port()
    proc = subprocess.Popen(['dotnet', ENGINE_DLL, '--port', str(port), '--workers', '3', '--tt-entries', '262144',
                             '--card-db', os.path.join(ROOT, 'data', '幻卡数据库.csv')],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = 'http://127.0.0.1:%d' % port
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url + '/v1/health', timeout=2) as r:
                    health = json.loads(r.read().decode('utf-8'))
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.3)
        else:
            raise AssertionError('the engine did not start')
        yield {'url': url, 'proc': proc, 'health': health}
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture
def client(engine, monkeypatch):
    csharp_engine.reset()
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    monkeypatch.setenv('TTC_CS_ENGINE_URL', engine['url'])
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    ai_server.ensure_handler_initialized()
    yield ai_server.app.test_client()
    csharp_engine.reset()


@pytest.fixture(scope='module')
def corpus():
    return gen_corpus.generate(20261010, 360)


def request_of(entry, depth=6):
    request = copy.deepcopy(entry['request'])
    request['max_depth'] = depth
    request['max_time'] = 20
    return request


def cell_is_empty(request, pos):
    return not any(item['pos'] == list(pos) for item in request['board'])


def test_health_reports_the_same_card_database_as_this_server(engine):
    assert engine['health']['protocol'] == csharp_engine.PROTOCOL
    assert engine['health']['cardDbSha256'] == csharp_engine.card_db_sha256()


def pick(corpus, *required, avoid=()):
    for entry in corpus:
        tags = set(entry['tags'])
        if all(t in tags for t in required) and not any(t in tags for t in avoid):
            return entry
    raise AssertionError('no corpus entry with tags %r' % (required,))


@pytest.mark.parametrize('tags', [
    ('opening',), ('k:3',), ('k:4', 'progress'), ('robust_candidate', 'unknown:0'), ('robust_candidate', 'unknown:2'),
    ('fallback_path',), ('dup_ids', 'k:5'), ('full_board',),
])
def test_the_engine_answers_through_flask(client, corpus, tags):
    entry = pick(corpus, *tags)
    random.seed(entry['seed'])
    request = request_of(entry)
    response = client.post('/ai_move', json=request)
    assert response.status_code == 200
    body = response.get_json()
    if 'full_board' in tags:
        assert body['move'] is None and body['engine'] == 'csharp'
        return
    assert body['engine'] == 'csharp', body
    assert body['performance_stats']['engine'] == 'csharp'
    assert body['performance_stats']['nodes_searched'] > 0
    assert 'engine_error' not in body
    row, col = body['pos']
    assert cell_is_empty(request, (row, col))
    if 'progress' in tags:
        assert body['search_progress'] and body['search_summary']['max_depth_reached'] == body['performance_stats']['search_depth']
    if 'robust_candidate' in tags or 'fallback_path' in tags:
        assert body['engine_info']['robust'] is True and body['engine_info']['scenarios'] >= 1


def test_the_engine_answer_is_the_same_whichever_flask_thread_asks(client, corpus):
    """Two requests for the same position (no unknown cards: nothing random) give the same move."""
    entry = pick(corpus, 'k:6', 'unknown:0', avoid=('full_board',))
    request = request_of(entry)
    first = client.post('/ai_move', json=request).get_json()
    second = client.post('/ai_move', json=request).get_json()
    assert (first['card_id'], first['pos']) == (second['card_id'], second['pos'])
    assert first['engine'] == second['engine'] == 'csharp'


def test_a_stopped_engine_is_a_503_and_the_python_solver_does_not_run(corpus, monkeypatch):
    """A-20 (replaces the test of the automatic fallback): no engine, no answer, and above all no Python search."""
    csharp_engine.reset()
    monkeypatch.setenv('TTC_CS_ENGINE', 'on')
    monkeypatch.setenv('TTC_CS_ENGINE_URL', 'http://127.0.0.1:%d' % free_port())  # nothing listens there
    monkeypatch.setenv('TTC_CS_ENGINE_CONNECT_TIMEOUT', '0.3')
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    calls = []
    for name in ('find_best_move_parallel', 'select_endgame_robust_move', '_solve_minimax_python'):
        monkeypatch.setattr(ai_server, name, lambda *a, _n=name, **k: calls.append(_n))
    entry = pick(corpus, 'k:3', avoid=('full_board',))
    random.seed(entry['seed'])
    response = ai_server.app.test_client().post('/ai_move', json=request_of(entry, depth=2))
    body = response.get_json()
    assert response.status_code == 503
    assert body['engine'] == 'csharp' and body['engine_error'] == 'connection_failed'
    assert 'card' not in body and calls == []
    csharp_engine.reset()
