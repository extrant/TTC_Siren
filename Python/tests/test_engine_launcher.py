# -*- coding: utf-8 -*-
"""The server starts the C# engine by itself when none runs (decision A-30, ai/engine_launcher.py)."""
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import engine_launcher  # noqa: E402
from engine_process import ENGINE_DIR, ensure_engine_built, free_port, get_json  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('TTC_CS_ENGINE', 'TTC_CS_ENGINE_URL', 'TTC_CS_ENGINE_AUTOSTART', 'TTC_CS_ENGINE_EXE'):
        monkeypatch.delenv(name, raising=False)
    yield
    engine_launcher.stop()


class FakeProc:
    def __init__(self, exits_with=None):
        self.returncode = exits_with
        self.args = None
        self._handle = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def recording_popen(proc):
    calls = []

    def popen(args, **kwargs):
        proc.args = args
        calls.append((args, kwargs))
        return proc
    return popen, calls


def no_popen(*args, **kwargs):
    raise AssertionError('nothing may be started here')


def test_switched_off_starts_nothing(monkeypatch):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    assert engine_launcher.ensure_running(popen=no_popen)[0] == 'off'


def test_a_running_engine_is_used_as_it_is(monkeypatch):
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: True)
    assert engine_launcher.ensure_running(popen=no_popen) == ('running', 'http://127.0.0.1:5300')


def test_autostart_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: False)
    monkeypatch.setenv('TTC_CS_ENGINE_AUTOSTART', 'off')
    assert engine_launcher.ensure_running(popen=no_popen)[0] == 'disabled'


def test_a_remote_engine_is_never_started_here(monkeypatch):
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: False)
    monkeypatch.setenv('TTC_CS_ENGINE_URL', 'http://10.1.2.3:5300')
    assert engine_launcher.ensure_running(popen=no_popen)[0] == 'remote'


def test_a_missing_engine_is_reported_with_the_places_looked_at(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: False)
    monkeypatch.setattr(engine_launcher, 'find_engine_exe', lambda: None)
    state, detail = engine_launcher.ensure_running(popen=no_popen)
    assert state == 'missing' and 'build_engine.bat' in detail and os.path.join('dist', 'engine') in detail


def test_candidates_put_the_explicit_path_first_then_the_folder_of_the_packaged_exe(monkeypatch, tmp_path):
    explicit = str(tmp_path / 'my' / engine_launcher.EXE_NAME)
    monkeypatch.setenv('TTC_CS_ENGINE_EXE', explicit)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'dist' / 'TTC_Siren.exe'))
    paths = engine_launcher.candidate_paths()
    assert paths[0] == explicit
    assert paths[1] == os.path.normpath(str(tmp_path / 'dist' / 'engine' / engine_launcher.EXE_NAME))
    assert os.path.normpath(os.path.join(engine_launcher.ROOT, 'dist', 'engine', engine_launcher.EXE_NAME)) in paths
    assert len(paths) == len(set(os.path.normcase(p) for p in paths))


def test_the_engine_is_started_on_the_port_of_the_url_with_this_servers_card_database(monkeypatch, tmp_path):
    exe = tmp_path / engine_launcher.EXE_NAME
    exe.write_bytes(b'')
    monkeypatch.setenv('TTC_CS_ENGINE_EXE', str(exe))
    monkeypatch.setenv('TTC_CS_ENGINE_URL', 'http://127.0.0.1:5399')
    answers = iter([False, False, True])
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: next(answers))
    proc = FakeProc()
    popen, calls = recording_popen(proc)
    state, detail = engine_launcher.ensure_running(popen=popen)
    assert state == 'started' and detail == str(exe)
    args, kwargs = calls[0]
    assert args[0] == str(exe) and args[1:5] == ['--host', '127.0.0.1', '--port', '5399']
    assert args[5] == '--card-db' and args[6] == engine_launcher.csharp_engine.card_db_path()
    assert kwargs['cwd'] == str(tmp_path)
    assert (tmp_path / 'engine.log').exists()


def test_an_engine_that_exits_at_start_is_reported(monkeypatch, tmp_path):
    exe = tmp_path / engine_launcher.EXE_NAME
    exe.write_bytes(b'')
    monkeypatch.setenv('TTC_CS_ENGINE_EXE', str(exe))
    monkeypatch.setattr(engine_launcher, 'engine_answers', lambda url, timeout=1.0: False)
    popen, _ = recording_popen(FakeProc(exits_with=3))
    state, detail = engine_launcher.ensure_running(popen=popen)
    assert state == 'failed' and 'code 3' in detail and 'engine.log' in detail


def test_the_real_engine_is_started_answers_and_is_stopped_again(monkeypatch):
    ensure_engine_built()
    exe = os.path.join(ENGINE_DIR, 'bin', 'Release', 'net10.0', engine_launcher.EXE_NAME)
    if not os.path.exists(exe):
        subprocess.run(['dotnet', 'build', os.path.join(ENGINE_DIR, 'TTC.Siren.Engine.csproj'), '-c', 'Release'], check=True, capture_output=True)
    port = free_port()
    monkeypatch.setenv('TTC_CS_ENGINE_EXE', exe)
    monkeypatch.setenv('TTC_CS_ENGINE_URL', 'http://127.0.0.1:%d' % port)
    monkeypatch.chdir(ROOT)  # card_db_path() of a source checkout is ./data
    state, detail = engine_launcher.ensure_running(wait_seconds=60)
    assert state == 'started', detail
    proc = engine_launcher._process
    health = get_json('http://127.0.0.1:%d/v1/health' % port)
    assert health['status'] == 'ok' and health['cardDbSha256'] == engine_launcher.csharp_engine.card_db_sha256()
    # a second start finds it running and starts nothing
    assert engine_launcher.ensure_running(popen=no_popen)[0] == 'running'
    engine_launcher.stop()
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    assert proc.poll() is not None
    assert not engine_launcher.engine_answers('http://127.0.0.1:%d' % port, timeout=0.5)
