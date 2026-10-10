# -*- coding: utf-8 -*-
"""
Client of the C# engine (Phase 7): a local HTTP service (csharp/src/TTC.Siren.Engine) that runs the minimax solver of
``/ai_move`` (Fast iterative deepening and the information-set robust endgame) much faster than the Python code.

Standard library only (``http.client``, ``json``, ``hashlib``): the production interpreter may be PyPy.

Since the Phase 7 hardening (decision A-20) the engine is the ONLY minimax solver while it is switched on: every problem is raised
as :class:`EngineError` and ``ai_server.ai_move`` answers it with an HTTP error. There is no automatic fallback to the Python solver.

Behaviour
  * Configuration by environment variables, read at call time:
      TTC_CS_ENGINE                   ``on`` (default) | ``off``. ``off`` is a MANUAL switch (set by the operator, never automatically):
                                      the server then runs the unchanged Python solver and never contacts the engine.
      TTC_CS_ENGINE_URL               default ``http://127.0.0.1:5300``
      TTC_CS_ENGINE_CONNECT_TIMEOUT   seconds, default 0.5
      TTC_CS_ENGINE_READ_MARGIN       seconds added to ``max_time`` + the engine's queue wait for the read timeout, default 5
      TTC_CS_ENGINE_POOL_SIZE         idle keep-alive connections kept in the process-wide pool, default 32
  * The first use (and the first use after a connection-level failure) checks ``GET /v1/health``: the protocol version must match,
    the engine's card-database SHA-256 must equal the hash of the CSV this process loads, and the health reports the engine's
    ``queueWaitMs``. Otherwise :class:`EngineError` (``protocol_mismatch`` / ``card_db_mismatch`` / ``health_failed``). Nothing is
    cached about a failed verification, so a corrected engine is accepted by the next request.
  * Read timeout of a solve = ``max_time`` + the engine's queue wait + a margin: a request that waits in the engine's queue is answered by
    the engine (a result or ``busy``) before the client gives up. There is no circuit breaker.
  * Connections come from one process-wide, thread-safe LIFO pool, so they are reused however many short-lived threads (Flask's
    ``threaded=True``) ask. A connection that fails is closed and never returned; an idle one that the server has closed is replaced
    once. Connections are closed explicitly (never left to the garbage collector: PyPy).

Nothing here touches the Python search state (``SEARCH_STATS``, ``TRANSPOSITION_TABLE``) or the random generator.
"""
import atexit
import hashlib
import http.client
import json
import logging
import math
import os
import socket
import sys
import threading
import urllib.parse

PROTOCOL = 1
DEFAULT_URL = 'http://127.0.0.1:5300'
DEFAULT_POOL_SIZE = 32

log = logging.getLogger('ttc.engine')

# What ``ai_server.ai_move`` answers for each reason: a problem with the request itself is 400, a defect of this server's own client
# code 500, everything the engine (or the way to it) does wrong 503.
_HTTP_STATUS = {'unsupported_parameter': 400, 'client_error': 500}


class EngineError(Exception):
    """The engine cannot answer this request. ``reason`` is a short machine-readable code, ``detail`` a human-readable addition."""

    def __init__(self, reason, detail=''):
        super().__init__('%s%s' % (reason, (': ' + detail) if detail else ''))
        self.reason = reason
        self.detail = detail

    @property
    def http_status(self):
        return _HTTP_STATUS.get(self.reason, 503)


class _StaleConnection(Exception):
    """A reused idle connection had been closed by the server before the request was sent (internal; never leaves this module)."""


# --------------------------------------------------------------------------- configuration

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def is_enabled():
    """``TTC_CS_ENGINE`` is ``on`` by default; ``off`` / ``0`` / ``false`` / ``no`` switch the engine off completely (manual switch)."""
    return os.environ.get('TTC_CS_ENGINE', 'on').strip().lower() not in ('off', '0', 'false', 'no')


def engine_url():
    return os.environ.get('TTC_CS_ENGINE_URL', DEFAULT_URL).strip() or DEFAULT_URL


def pool_limit():
    return max(0, _env_int('TTC_CS_ENGINE_POOL_SIZE', DEFAULT_POOL_SIZE))


# --------------------------------------------------------------------------- card database hash

_db_lock = threading.Lock()
_db_sha = None
_db_path_override = None


def card_db_path():
    """The CSV the Python server loads (``ai_server.get_card_db``): next to the executable when frozen, else ``./data``."""
    if _db_path_override:
        return _db_path_override
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'data/幻卡数据库.csv')
    return os.path.join(os.path.abspath('.'), 'data/幻卡数据库.csv')


def card_db_sha256():
    global _db_sha
    if _db_sha is None:
        with _db_lock:
            if _db_sha is None:
                with open(card_db_path(), 'rb') as f:
                    _db_sha = hashlib.sha256(f.read()).hexdigest()
    return _db_sha


# --------------------------------------------------------------------------- verification state and the connection pool

_state_lock = threading.Lock()
_verified_url = None
_queue_wait_s = 0.0
_pool = []  # idle connections, newest last: a LIFO of (url, HTTPConnection)


def reset():
    """Forget the verification and close every pooled connection (tests, and after reconfiguration)."""
    global _verified_url, _queue_wait_s, _db_sha
    with _state_lock:
        _verified_url = None
        _queue_wait_s = 0.0
    close_pool()
    _db_sha = None


def set_card_db_path(path):
    """Tests: hash another file instead of ``data/幻卡数据库.csv``."""
    global _db_path_override, _db_sha
    _db_path_override = path
    _db_sha = None


def _invalidate_verification():
    global _verified_url
    with _state_lock:
        _verified_url = None


def _close(conn):
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def close_pool():
    """Close every idle pooled connection."""
    with _state_lock:
        idle = list(_pool)
        del _pool[:]
    for _, conn in idle:
        _close(conn)


atexit.register(close_pool)


def pool_idle():
    """Number of idle connections in the pool (tests, monitoring)."""
    with _state_lock:
        return len(_pool)


def _checkout(url, fresh):
    """An idle connection to ``url`` (newest first) with ``True`` when it has been used before, otherwise a new unconnected one."""
    if not fresh:
        wrong_url = []
        found = None
        with _state_lock:
            while _pool:
                pooled_url, conn = _pool.pop()
                if pooled_url == url:
                    found = conn
                    break
                wrong_url.append(conn)
        for conn in wrong_url:
            _close(conn)
        if found is not None:
            return found, True
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'http' or not parsed.hostname:
        raise EngineError('bad_url', url)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                                      timeout=_env_float('TTC_CS_ENGINE_CONNECT_TIMEOUT', 0.5)), False


def _checkin(url, conn):
    """Return a healthy connection to the pool, or close it when the pool is full."""
    with _state_lock:
        if len(_pool) < pool_limit():
            _pool.append((url, conn))
            return
    _close(conn)


# --------------------------------------------------------------------------- one HTTP exchange

def _request(method, path, body, read_timeout, retry_stale=True, fresh=False):
    """
    One HTTP exchange. Returns (status, parsed JSON). A reused idle connection that the server had closed is replaced once when
    ``retry_stale`` is true; otherwise :class:`_StaleConnection` is raised so that the caller can verify the engine again first.
    """
    url = engine_url()
    while True:
        conn, reused = _checkout(url, fresh)
        connecting = False
        try:
            if conn.sock is None:
                connecting = True
                conn.connect()
                connecting = False
            conn.sock.settimeout(read_timeout)
            headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive'} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
        except (http.client.RemoteDisconnected, http.client.CannotSendRequest, http.client.ResponseNotReady,
                BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            _close(conn)
            if reused:
                if retry_stale:
                    fresh = True
                    continue
                raise _StaleConnection()
            raise EngineError('connection_failed', type(e).__name__)
        except (socket.timeout, TimeoutError):
            _close(conn)
            if connecting:
                raise EngineError('connection_failed', 'connect timeout')
            raise EngineError('timeout', path)
        except (OSError, http.client.HTTPException) as e:
            _close(conn)
            raise EngineError('connection_failed', type(e).__name__)
        except BaseException:
            _close(conn)
            raise
        if response.will_close:
            _close(conn)
        else:
            _checkin(url, conn)
        break
    try:
        payload = json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise EngineError('malformed_response', 'HTTP %d' % response.status)
    if not isinstance(payload, dict):
        raise EngineError('malformed_response', 'not an object')
    return response.status, payload


def _verify():
    """Health check: protocol version, card-database hash and the engine's queue wait. Done once per URL, and again after a failure."""
    global _verified_url, _queue_wait_s
    url = engine_url()
    with _state_lock:
        if _verified_url == url:
            return
    status, health = _request('GET', '/v1/health', None, max(_env_float('TTC_CS_ENGINE_CONNECT_TIMEOUT', 0.5), 2.0))
    if status != 200 or health.get('status') != 'ok':
        raise EngineError('health_failed', 'HTTP %d' % status)
    if health.get('protocol') != PROTOCOL:
        raise EngineError('protocol_mismatch', 'engine speaks %r, client %r' % (health.get('protocol'), PROTOCOL))
    if health.get('cardDbSha256') != card_db_sha256():
        raise EngineError('card_db_mismatch', 'engine %s, client %s' % (health.get('cardDbSha256'), card_db_sha256()))
    wait_ms = health.get('queueWaitMs')
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, (int, float)) or not 0 <= wait_ms < 1e9:
        raise EngineError('health_failed', 'queueWaitMs missing or invalid: %r' % (wait_ms,))
    with _state_lock:
        _verified_url = url
        _queue_wait_s = wait_ms / 1000.0


# --------------------------------------------------------------------------- the call

def read_timeout_for(max_time):
    """``max_time`` + the engine's queue wait (from its health) + a margin: the engine answers (a result or ``busy``) before this."""
    margin = _env_float('TTC_CS_ENGINE_READ_MARGIN', 5.0)
    with _state_lock:
        queue_wait = _queue_wait_s
    if max_time is None or (isinstance(max_time, float) and (math.isinf(max_time) or math.isnan(max_time))):
        return 3600.0 + queue_wait + margin
    return max(0.0, float(max_time)) + queue_wait + margin


def solve(payload, max_time):
    """
    POST ``payload`` to ``/v1/solve`` and return the response dict (status ``ok`` or ``needs_fallback_candidates``).
    Raises :class:`EngineError` for everything else. ``max_time`` is the search limit in seconds (None = unlimited); together with the
    engine's queue wait it sets the read timeout.
    """
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    for attempt in range(3):
        try:
            _verify()
            # the last attempt opens a new connection whatever the pool holds, so that a stale one can never fail it
            status, response = _request('POST', '/v1/solve', body, read_timeout_for(max_time), retry_stale=False, fresh=attempt == 2)
            break
        except _StaleConnection:
            # the idle connection was closed by the server: probably an engine restart, so verify it again before the next try
            _invalidate_verification()
            close_pool()
            if attempt == 2:  # cannot happen (the last attempt opens a new connection); never loop
                raise EngineError('connection_failed', 'stale connection')
        except EngineError as e:
            if e.reason in ('connection_failed', 'timeout'):
                _invalidate_verification()  # the engine may be replaced by another build: verify again
                close_pool()
            raise
    state = response.get('status')
    if state in ('ok', 'needs_fallback_candidates') and status == 200:
        return response
    if state == 'busy':
        raise EngineError('busy', str(response.get('reason', '')))
    if state == 'invalid_request':
        raise EngineError('invalid_request', str(response.get('message', '')))
    if state == 'error':
        raise EngineError('engine_error', str(response.get('message', '')))
    raise EngineError('http_error', 'HTTP %d status %r' % (status, state))


# --------------------------------------------------------------------------- request building helpers

def card_array(card):
    """The 13-value card form of the protocol: [id|null, type|null, "r"|"b"|null, mod, base4, display4, canUse]."""
    owner = {'red': 'r', 'blue': 'b'}.get(card.owner)
    card_id = None if card.card_id is None else int(card.card_id)
    return [card_id, card.card_type, owner, int(card.type_modifier),
            int(card.base_up), int(card.base_right), int(card.base_down), int(card.base_left),
            int(card.up), int(card.right), int(card.down), int(card.left), 1 if card.can_use else 0]


def state_payload(game_state):
    board = []
    for r in range(3):
        for c in range(3):
            card = game_state.board.get_card(r, c)
            board.append(None if card is None else card_array(card))
    return {
        'cur': int(game_state.current_player_idx),
        'board': board,
        'hands': [[card_array(card) for card in player.hand] for player in game_state.players],
    }


def number_or_error(value, name):
    """
    ``max_depth`` / ``max_time`` as Python would use them: a real number. A bool is the integer it is in Python (``True`` is 1);
    anything else (string, null, list ...) makes the Python solver raise a TypeError, so the request is refused (HTTP 400).
    """
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, (int, float)):
        raise EngineError('unsupported_parameter', '%s=%r is not a number' % (name, value))
    return value


def wire_max_depth(value):
    """Python: ``min(max_depth, 100)``; ``depth <= limit`` for depth 1, 2, ... so NaN and anything below 1 mean no level."""
    value = number_or_error(value, 'max_depth')
    if isinstance(value, float) and math.isnan(value):
        return 0
    return min(value, 100)


def wire_max_time(value):
    """Python: time is up once ``elapsed > max_time``. NaN and +inf never; -inf always; None on the wire = no limit."""
    value = number_or_error(value, 'max_time')
    if isinstance(value, float) and (math.isnan(value) or value == math.inf):
        return None
    if isinstance(value, float) and value == -math.inf:
        return -1.0
    return value


def hex_to_float(bits):
    """Exact double from the 16 hex digits of its IEEE-754 bit pattern."""
    import struct
    return struct.unpack('>d', bytes.fromhex(bits))[0]
