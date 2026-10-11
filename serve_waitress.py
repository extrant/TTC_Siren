# -*- coding: utf-8 -*-
"""
Optional production front end: the unified TTC_Siren service (``app.py``: /ai_move, /search_progress, /api/recommend,
/api/draft/analyze, /api/winrate) served by waitress, a WSGI server with a fixed thread pool, instead of Flask's development server.

  pip install waitress
  python serve_waitress.py --host 127.0.0.1 --port 5000 --threads 16

Why (Phase 7 review M7-2): ``app.run(threaded=True)`` starts a new thread per request. The engine client keeps a process-wide connection
pool, so that works too, but a fixed pool of threads bounds the number of requests in flight and is the supported production setup.
The thread count is the number of requests a server process works on at once; the C# engine queues what its workers cannot take.

It performs the same initialisation as ``python app.py``: it imports ``app`` (which registers the extra routes on the ``ai_server``
Flask app), silences the werkzeug request log, sets up the console panel (``--no-console`` leaves it out) and starts it. ``app.py``
itself and its start-up are not changed. Run it from any directory: it changes to the repository root (the card database is read
from ``data/`` relative to it).
"""
import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_waitress():
    """The waitress module, or ``None`` when it is not installed."""
    try:
        import waitress
    except ImportError:
        return None
    return waitress


def build_parser():
    ap = argparse.ArgumentParser(description='TTC_Siren service on waitress', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='Environment: TTC_CS_ENGINE, TTC_CS_ENGINE_URL, TTC_CS_ENGINE_POOL_SIZE ... see docs/DEPLOY_WINDOWS.md')
    ap.add_argument('--host', default='127.0.0.1', help='listen address (default 127.0.0.1)')
    ap.add_argument('--port', type=int, default=5000, help='listen port (default 5000)')
    ap.add_argument('--threads', type=int, default=16, help='size of the fixed worker-thread pool (default 16)')
    ap.add_argument('--no-console', action='store_true', help='do not start the terminal panel (services, redirected output)')
    return ap


def main(argv=None, waitress_module=None, start_engine=None):
    """``start_engine``: start the C# engine when none runs (decision A-30). Default: only when serving for real, i.e. with the real waitress."""
    args = build_parser().parse_args(argv)
    if start_engine is None:
        start_engine = waitress_module is None
    waitress = waitress_module if waitress_module is not None else load_waitress()
    if waitress is None:
        print('waitress is not installed. Install it with:  pip install waitress\n'
              '(python app.py keeps working without it.)', file=sys.stderr)
        return 2
    if args.threads < 1:
        print('--threads must be at least 1', file=sys.stderr)
        return 2

    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import app as service  # registers the unified routes on the ai_server Flask app, exactly as ``python app.py`` does

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('waitress').setLevel(logging.ERROR)
    console = None
    if not args.no_console:
        service.init_console_window(120, 30)
        service.detect_terminal()
        console = service.console_ui
        console.start()
    if start_engine:
        from ai import engine_launcher
        print('   ' + engine_launcher.describe(*engine_launcher.ensure_running()), flush=True)
    print(' * TTC_Siren service on http://%s:%d (waitress, %d threads)' % (args.host, args.port, args.threads), flush=True)
    try:
        waitress.serve(service.app, host=args.host, port=args.port, threads=args.threads)
    except KeyboardInterrupt:
        pass
    finally:
        if console is not None:
            console.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
