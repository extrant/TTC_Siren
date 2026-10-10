# -*- coding: utf-8 -*-
"""
Runs the Flask ai_server the way the production script does (``app.run(threaded=True)``), on a loopback port, without the Rich console
panel (it is a terminal UI; its redraw thread would distort a load test). Used by the Phase 7 load test and the integration test.

  python tools/phase7/serve_flask.py --port 5001
Environment: TTC_CS_ENGINE (on|off), TTC_CS_ENGINE_URL, ... as documented in ai/csharp_engine.py.
"""
import argparse
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


class _NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=5001)
    args = ap.parse_args()
    import ai_server
    ai_server.console_ui = _NullUi()
    ai_server.ensure_handler_initialized()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print('flask ai_server on http://127.0.0.1:%d (engine=%s)' % (args.port, os.environ.get('TTC_CS_ENGINE', 'on')), flush=True)
    ai_server.app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
