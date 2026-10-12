"""serve_waitress.py: the optional waitress launcher. waitress itself may not be installed on the machine that runs the tests, so the wiring
is tested with a stand-in module that records the call of ``serve``; the 'not installed' path is tested for real."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import serve_waitress  # noqa: E402


class FakeWaitress:
    def __init__(self):
        self.calls = []

    def serve(self, app, **kwargs):
        self.calls.append((app, kwargs))


def test_missing_waitress_exits_with_a_clear_message(capsys):
    code = serve_waitress.main(['--no-console'], waitress_module=None) if serve_waitress.load_waitress() is None else None
    if code is None:  # waitress is installed here: simulate its absence
        original = serve_waitress.load_waitress
        serve_waitress.load_waitress = lambda: None
        try:
            code = serve_waitress.main(['--no-console'])
        finally:
            serve_waitress.load_waitress = original
    assert code == 2
    assert 'pip install waitress' in capsys.readouterr().err


def test_the_launcher_serves_the_unified_app_with_a_fixed_thread_pool():
    fake = FakeWaitress()
    assert serve_waitress.main(['--no-console', '--host', '127.0.0.1', '--port', '5055', '--threads', '7'], waitress_module=fake) == 0
    (app, kwargs), = fake.calls
    assert kwargs == {'host': '127.0.0.1', 'port': 5055, 'threads': 7}
    rules = {r.rule for r in app.url_map.iter_rules()}
    # the routes of ai_server AND of app.py, i.e. what ``python app.py`` serves
    assert {'/ai_move', '/search_progress', '/api/recommend', '/api/draft/analyze', '/api/winrate'} <= rules


def test_defaults_and_invalid_thread_count(capsys):
    args = serve_waitress.build_parser().parse_args([])
    assert (args.host, args.port, args.threads) == ('127.0.0.1', 5000, 16)
    assert serve_waitress.main(['--threads', '0'], waitress_module=FakeWaitress()) == 2


def test_the_script_runs_as_a_program_and_reports_a_missing_waitress():
    code = ('import sys, builtins\n'
            'real = builtins.__import__\n'
            'def blocked(name, *a, **k):\n'
            '    if name == "waitress": raise ImportError(name)\n'
            '    return real(name, *a, **k)\n'
            'builtins.__import__ = blocked\n'
            'sys.argv = ["serve_waitress.py", "--no-console"]\n'
            'import runpy; runpy.run_path("serve_waitress.py", run_name="__main__")\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and 'pip install waitress' in result.stderr
