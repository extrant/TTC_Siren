"""Helpers for the tests that need the real thing: the C# engine as its own process and the Flask app behind a real threaded HTTP server.

The engine is the Release build of CSharp/src/TTC.Siren.Engine (built first when it is missing); Flask is served by werkzeug's server with
``threaded=True`` (one new thread per request, the model the production script ``app.run(threaded=True)`` uses).
"""
import contextlib
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENGINE_DIR = os.path.join(ROOT, '..', 'CSharp', 'src', 'TTC.Siren.Engine')
ENGINE_DLL = os.path.join(ENGINE_DIR, 'bin', 'Release', 'net10.0', 'TTC.Siren.Engine.dll')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def ensure_engine_built():
    if not os.path.exists(ENGINE_DLL):
        subprocess.run(['dotnet', 'build', os.path.join(ENGINE_DIR, 'TTC.Siren.Engine.csproj'), '-c', 'Release'], check=True,
                       capture_output=True)


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


@contextlib.contextmanager
def engine_process(workers=2, queue=256, queue_wait_ms=60000, tt_entries=262144, extra=()):
    """Start the engine on a free loopback port and yield ``{'url', 'port', 'proc', 'health'}``; kill it afterwards."""
    ensure_engine_built()
    port = free_port()
    proc = subprocess.Popen(['dotnet', ENGINE_DLL, '--port', str(port), '--workers', str(workers), '--queue', str(queue),
                             '--queue-wait-ms', str(queue_wait_ms), '--tt-entries', str(tt_entries),
                             '--card-db', os.path.join(ROOT, 'data', '幻卡数据库.csv'), *extra],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = 'http://127.0.0.1:%d' % port
    try:
        deadline = time.time() + 60
        while True:
            try:
                health = get_json(url + '/v1/health', timeout=2)
                break
            except Exception:  # noqa: BLE001
                if time.time() > deadline or proc.poll() is not None:
                    raise AssertionError('the engine did not start')
                time.sleep(0.3)
        yield {'url': url, 'port': port, 'proc': proc, 'health': health}
    finally:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def flask_server(app):
    """Serve ``app`` with a real threaded HTTP server on a free loopback port; yield its base URL."""
    from werkzeug.serving import make_server
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d' % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


_ENDPOINT = re.compile(r'^\[?[0-9a-fA-F:.]+\]?[:.]\d+$')


def tcp_connections_to(port):
    """
    Client-side TCP connections to ``port`` on this machine, every state (ESTABLISHED, TIME_WAIT, ...), counted with ``netstat``:
    the entries whose FOREIGN address has that port (the engine's own entries have it as the local port).
    """
    out = subprocess.run(['netstat', '-an', '-p', 'TCP'] if os.name == 'nt' else ['netstat', '-tan'],
                         capture_output=True, text=True, check=True).stdout
    count = 0
    for line in out.splitlines():
        endpoints = [t for t in line.split() if _ENDPOINT.match(t) and ':' in t]
        if len(endpoints) >= 2 and endpoints[1].rsplit(':', 1)[1] == str(port):
            count += 1
    return count
