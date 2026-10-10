# -*- coding: utf-8 -*-
"""
Client of the C# engine (Phase 7): a local HTTP service (csharp/src/TTC.Siren.Engine) that runs the minimax solver of
``/ai_move`` (Fast iterative deepening and the information-set robust endgame) much faster than the Python code.

Standard library only (``http.client``, ``json``, ``hashlib``): the production interpreter may be PyPy.

Behaviour
  * Configuration by environment variables, read at call time:
      TTC_CS_ENGINE                   ``on`` (default) | ``off``
      TTC_CS_ENGINE_URL               default ``http://127.0.0.1:5300``
      TTC_CS_ENGINE_CONNECT_TIMEOUT   seconds, default 0.5
      TTC_CS_ENGINE_READ_MARGIN       seconds added to ``max_time`` for the read timeout, default 5
      TTC_CS_ENGINE_BACKOFF           seconds the engine is skipped after a connection-level failure, default 30
  * The first use (and the first use after a failure) checks ``GET /v1/health``: the protocol version must match and the
    engine's card-database SHA-256 must equal the hash of the CSV this process loads. Otherwise: fall back.
  * Every problem raises :class:`EngineFallback` with a short ``reason``; the caller then runs the unchanged Python solver.
  * One keep-alive connection per thread. Connection-level failures (refused, timeout, reset) open a circuit breaker for
    ``TTC_CS_ENGINE_BACKOFF`` seconds so that every request does not pay the timeout again.

Nothing here touches the Python search state (``SEARCH_STATS``, ``TRANSPOSITION_TABLE``) or the random generator.
"""
import hashlib
import http.client
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import urllib.parse

PROTOCOL = 1
DEFAULT_URL = 'http://127.0.0.1:5300'

log = logging.getLogger('ttc.engine')


class EngineFallback(Exception):
    """The engine cannot answer this request; run the Python solver. ``reason`` is a short machine-readable code."""

    def __init__(self, reason, detail=''):
        super().__init__('%s%s' % (reason, (': ' + detail) if detail else ''))
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------- configuration

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def is_enabled():
    """``TTC_CS_ENGINE`` is ``on`` by default; ``off`` / ``0`` / ``false`` / ``no`` switch the engine off completely."""
    return os.environ.get('TTC_CS_ENGINE', 'on').strip().lower() not in ('off', '0', 'false', 'no')


def engine_url():
    return os.environ.get('TTC_CS_ENGINE_URL', DEFAULT_URL).strip() or DEFAULT_URL


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


# --------------------------------------------------------------------------- circuit breaker and connections

_state_lock = threading.Lock()
_disabled_until = 0.0
_verified_url = None
_local = threading.local()


def reset():
    """Forget the breaker, the verification and this thread's connection (tests, and after reconfiguration)."""
    global _disabled_until, _verified_url, _db_sha
    with _state_lock:
        _disabled_until = 0.0
        _verified_url = None
    _drop_connection()
    _db_sha = None


def set_card_db_path(path):
    """Tests: hash another file instead of ``data/幻卡数据库.csv``."""
    global _db_path_override, _db_sha
    _db_path_override = path
    _db_sha = None


def breaker_open():
    with _state_lock:
        return time.monotonic() < _disabled_until


def _open_breaker():
    global _disabled_until, _verified_url
    backoff = _env_float('TTC_CS_ENGINE_BACKOFF', 30.0)
    with _state_lock:
        _disabled_until = time.monotonic() + max(0.0, backoff)
        _verified_url = None
    _drop_connection()


def _drop_connection():
    conn = getattr(_local, 'conn', None)
    _local.conn = None
    _local.conn_url = None
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _connection(url, read_timeout):
    """This thread's keep-alive connection to ``url`` (created on first use)."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'http' or not parsed.hostname:
        raise EngineFallback('bad_url', url)
    conn = getattr(_local, 'conn', None)
    if conn is None or getattr(_local, 'conn_url', None) != url:
        _drop_connection()
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                                          timeout=_env_float('TTC_CS_ENGINE_CONNECT_TIMEOUT', 0.5))
        _local.conn = conn
        _local.conn_url = url
    return conn


def _request(method, path, body, read_timeout):
    """One HTTP exchange. Returns (status, parsed JSON). A reused keep-alive connection that the server closed is retried once."""
    url = engine_url()
    connect_timeout = _env_float('TTC_CS_ENGINE_CONNECT_TIMEOUT', 0.5)
    for attempt in (0, 1):
        conn = _connection(url, read_timeout)
        reused = conn.sock is not None
        try:
            if conn.sock is None:
                conn.timeout = connect_timeout
                conn.connect()
            conn.sock.settimeout(read_timeout)
            headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive'} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            break
        except (http.client.RemoteDisconnected, http.client.CannotSendRequest, http.client.ResponseNotReady,
                BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            _drop_connection()
            if reused and attempt == 0:
                continue  # the idle keep-alive connection had been closed by the server: reconnect once
            raise EngineFallback('connection_failed', type(e).__name__)
        except (socket.timeout, TimeoutError):
            _drop_connection()
            raise EngineFallback('timeout', path)
        except (ConnectionRefusedError, OSError) as e:
            _drop_connection()
            raise EngineFallback('connection_failed', type(e).__name__)
    try:
        payload = json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise EngineFallback('malformed_response', 'HTTP %d' % response.status)
    if not isinstance(payload, dict):
        raise EngineFallback('malformed_response', 'not an object')
    return response.status, payload


def _verify():
    """Health check: protocol version and card-database hash. Done once per URL, and again after a failure."""
    global _verified_url
    url = engine_url()
    with _state_lock:
        if _verified_url == url:
            return
    status, health = _request('GET', '/v1/health', None, max(_env_float('TTC_CS_ENGINE_CONNECT_TIMEOUT', 0.5), 2.0))
    if status != 200 or health.get('status') != 'ok':
        raise EngineFallback('health_failed', 'HTTP %d' % status)
    if health.get('protocol') != PROTOCOL:
        raise EngineFallback('protocol_mismatch', 'engine speaks %r, client %r' % (health.get('protocol'), PROTOCOL))
    if health.get('cardDbSha256') != card_db_sha256():
        raise EngineFallback('card_db_mismatch', 'engine %s, client %s' % (health.get('cardDbSha256'), card_db_sha256()))
    with _state_lock:
        _verified_url = url


# --------------------------------------------------------------------------- the call

def read_timeout_for(max_time):
    margin = _env_float('TTC_CS_ENGINE_READ_MARGIN', 5.0)
    if max_time is None or (isinstance(max_time, float) and (math.isinf(max_time) or math.isnan(max_time))):
        return 3600.0
    return max(0.0, float(max_time)) + margin + 1.0


def solve(payload, max_time):
    """
    POST ``payload`` to ``/v1/solve`` and return the response dict (status ``ok`` or ``needs_fallback_candidates``).
    Raises :class:`EngineFallback` for everything else. ``max_time`` is the search limit in seconds (None = unlimited); it
    sets the read timeout.
    """
    if breaker_open():
        raise EngineFallback('circuit_open')
    try:
        _verify()
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        status, response = _request('POST', '/v1/solve', body, read_timeout_for(max_time))
    except EngineFallback as e:
        if e.reason in ('connection_failed', 'timeout', 'health_failed', 'protocol_mismatch', 'card_db_mismatch',
                        'malformed_response'):
            _open_breaker()
        raise
    state = response.get('status')
    if state in ('ok', 'needs_fallback_candidates') and status == 200:
        return response
    if state == 'busy':
        raise EngineFallback('busy', str(response.get('reason', '')))
    if state == 'invalid_request':
        raise EngineFallback('invalid_request', str(response.get('message', '')))
    if state == 'error':
        raise EngineFallback('engine_error', str(response.get('message', '')))
    raise EngineFallback('http_error', 'HTTP %d status %r' % (status, state))


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


def number_or_fallback(value, name):
    """``max_depth`` / ``max_time`` as Python would use them: a real number (bool and strings are left to the Python path)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineFallback('unsupported_parameter', '%s=%r' % (name, value))
    return value


def wire_max_depth(value):
    """Python: ``min(max_depth, 100)``; ``depth <= limit`` for depth 1, 2, ... so NaN and anything below 1 mean no level."""
    value = number_or_fallback(value, 'max_depth')
    if isinstance(value, float) and math.isnan(value):
        return 0
    return min(value, 100)


def wire_max_time(value):
    """Python: time is up once ``elapsed > max_time``. NaN and +inf never; -inf always; None on the wire = no limit."""
    value = number_or_fallback(value, 'max_time')
    if isinstance(value, float) and (math.isnan(value) or value == math.inf):
        return None
    if isinstance(value, float) and value == -math.inf:
        return -1.0
    return value


def hex_to_float(bits):
    """Exact double from the 16 hex digits of its IEEE-754 bit pattern."""
    import struct
    return struct.unpack('>d', bytes.fromhex(bits))[0]
