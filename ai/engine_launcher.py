# -*- coding: utf-8 -*-
"""
Starts the C# engine (csharp/src/TTC.Siren.Engine) next to the Python server, so that ``python app.py``, ``scripts\\run_pypy.bat`` and the packaged
``TTC_Siren.exe`` work as before with one start: the server answers every endpoint itself and hands the minimax search of ``/ai_move`` to the
engine (decision A-30). Standard library only (the interpreter may be PyPy).

``ensure_running()`` is called once by the server's entry points before they listen:
  * the engine is switched off (``TTC_CS_ENGINE=off``)             -> nothing is started (the Python solver answers);
  * an engine already answers at ``TTC_CS_ENGINE_URL``             -> it is used as it is (for example one started by hand);
  * ``TTC_CS_ENGINE_AUTOSTART=off``, or the URL is not loopback    -> nothing is started;
  * otherwise ``TTC.Siren.Engine.exe`` is looked for, started on the port of the URL with the card database this server loads, and stopped
    again when the server exits. Its output goes to ``engine.log`` next to it.

Where the engine is looked for, first match wins: ``TTC_CS_ENGINE_EXE``; then, from the folder of the packaged executable (when frozen) and
from the repository root: ``engine\\``, ``..\\engine\\`` and ``dist\\engine\\`` (what ``scripts\\build_engine.bat`` produces).
"""
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from ai import csharp_engine

EXE_NAME = 'TTC.Siren.Engine.exe' if os.name == 'nt' else 'TTC.Siren.Engine'
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_process = None


def autostart_enabled():
    """``TTC_CS_ENGINE_AUTOSTART`` is ``on`` by default; ``off`` / ``0`` / ``false`` / ``no`` leave the engine to the operator."""
    return os.environ.get('TTC_CS_ENGINE_AUTOSTART', 'on').strip().lower() not in ('off', '0', 'false', 'no')


def candidate_paths():
    """Where the engine executable may be, in the order they are tried."""
    paths = []
    explicit = os.environ.get('TTC_CS_ENGINE_EXE', '').strip().strip('"')
    if explicit:
        paths.append(explicit)
    bases = []
    if getattr(sys, 'frozen', False):
        bases.append(os.path.dirname(os.path.abspath(sys.executable)))
    bases.append(ROOT)
    for base in bases:
        for relative in ('engine', os.path.join('..', 'engine'), os.path.join('dist', 'engine')):
            paths.append(os.path.normpath(os.path.join(base, relative, EXE_NAME)))
    seen, unique = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def find_engine_exe():
    for path in candidate_paths():
        if os.path.isfile(path):
            return path
    return None


def is_loopback_url(url):
    host = (urllib.parse.urlparse(url).hostname or '').lower()
    return host in ('127.0.0.1', 'localhost', '::1')


def engine_answers(url, timeout=1.0):
    """True when something at ``url`` answers ``GET /v1/health`` with ``status: ok`` (any engine, also one started by hand)."""
    try:
        with urllib.request.urlopen(url.rstrip('/') + '/v1/health', timeout=timeout) as response:
            return response.status == 200 and json.loads(response.read().decode('utf-8')).get('status') == 'ok'
    except Exception:  # noqa: BLE001 - not reachable, not ours, not JSON: all mean "no engine there"
        return False


_job = None  # Windows job object that holds the engine; kept open for the life of this process


def _bind_to_this_process(proc):
    """
    Windows: puts the engine into a job object that is killed when its last handle closes. This process holds the only handle, so the engine
    ends whenever the server ends -- also when the console window is closed or the process is killed, when ``atexit`` does not run.
    Best effort: on failure the engine is still stopped by ``atexit`` on a normal exit.
    """
    global _job
    if os.name != 'nt':
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                                 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters), ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # JobObjectExtendedLimitInformation
            return False
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            return False
        _job = job
        return True
    except Exception:  # noqa: BLE001
        return False


def stop():
    """Stops the engine this process started (registered with ``atexit``; safe to call more than once)."""
    global _process
    proc, _process = _process, None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def ensure_running(wait_seconds=30.0, popen=subprocess.Popen):
    """
    Makes sure the engine runs when the server needs it. Returns ``(state, detail)``; ``state`` is one of ``off``, ``disabled``, ``running``,
    ``remote``, ``missing``, ``started`` or ``failed``. Never raises: a server without an engine still starts, and ``/ai_move`` then answers 503
    with the reason (decision A-20: no silent fallback to the Python solver).
    """
    global _process
    if not csharp_engine.is_enabled():
        return 'off', 'TTC_CS_ENGINE=off: the Python minimax solver answers'
    url = csharp_engine.engine_url()
    if engine_answers(url):
        return 'running', url
    if not autostart_enabled():
        return 'disabled', 'TTC_CS_ENGINE_AUTOSTART=off and no engine answers at ' + url
    if not is_loopback_url(url):
        return 'remote', 'no engine answers at ' + url + ' (only a local engine is started automatically)'
    exe = find_engine_exe()
    if exe is None:
        return 'missing', 'TTC.Siren.Engine.exe not found (run scripts\\build_engine.bat); looked in: ' + '; '.join(candidate_paths())

    port = urllib.parse.urlparse(url).port or 80
    folder = os.path.dirname(os.path.abspath(exe))
    log_path = os.path.join(folder, 'engine.log')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        log_file = open(log_path, 'wb')
    except OSError:
        log_file = subprocess.DEVNULL
    try:
        proc = popen([exe, '--host', '127.0.0.1', '--port', str(port), '--card-db', csharp_engine.card_db_path()],
                     cwd=folder, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags)
    except OSError as e:
        return 'failed', 'could not start %s: %s' % (exe, e)
    finally:
        if log_file is not subprocess.DEVNULL:
            log_file.close()
    _process = proc
    _bind_to_this_process(proc)
    atexit.register(stop)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _process = None
            return 'failed', 'the engine exited at start (code %s); see %s' % (proc.returncode, log_path)
        if engine_answers(url, timeout=0.5):
            return 'started', exe
        time.sleep(0.2)
    stop()
    return 'failed', 'the engine did not answer within %d s; see %s' % (wait_seconds, log_path)


def describe(state, detail):
    """One line for the server's start-up banner."""
    text = {
        'off': '关闭（TTC_CS_ENGINE=off，使用 Python 求解器）',
        'disabled': '未自动启动（TTC_CS_ENGINE_AUTOSTART=off）',
        'running': '已在运行',
        'remote': '未连接（远程地址不会自动启动）',
        'missing': '未找到（先运行 scripts\\build_engine.bat）',
        'started': '已启动',
        'failed': '启动失败',
    }.get(state, state)
    return 'C# 求解引擎: %s  %s' % (text, detail)
