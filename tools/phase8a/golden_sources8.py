# -*- coding: utf-8 -*-
"""
Phase 8A golden data is bound to the Python reference source exactly like the older goldens (tools/parity/golden_sources.py), but with
its own file sets: this module deliberately does NOT edit golden_sources.py (whose hash is part of every older golden header, so editing
it would make all of them stale). The sets are mirrored by csharp/tests/TTC.Siren.Tests/GoldenSources8.cs; a mismatch makes the C# check
fail, it never passes silently. Read-only: this module only reads files.
"""
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

_RNG = [
    'tools/phase8a/golden_sources8.py',
    'tools/phase8a/py_rng_trace.py',
]
# Everything that decides what ``ai_move`` parses and samples: the request parser, the unknown-card handler and its configuration,
# the rules/state model it builds, the card database, and the corpus + generator that produce the requests.
_PARSE = [
    'core/*.py',
    'csv_compat.py',
    'data/幻卡数据库.csv',
    'ai_server.py',
    'ai/unknown_card_handler.py',
    'config/unknown_card_config.py',
    'tools/parity/py_rules_trace.py',
    'tools/phase7/gen_corpus.py',
    'tools/phase8a/golden_sources8.py',
    'tools/phase8a/gen_requests.py',
    'tools/phase8a/py_parse_trace.py',
]
# The end-to-end reference additionally depends on the engine client, the search code and its own generator.
_E2E = _PARSE + [
    'ai/ai.py',
    'ai/csharp_engine.py',
    'tools/phase8a/py_e2e_trace.py',
]

# The sampler cases need the handler, its configuration and the rules model, but not the request corpora.
_SAMPLING = [
    'core/*.py',
    'csv_compat.py',
    'data/幻卡数据库.csv',
    'ai_server.py',
    'ai/unknown_card_handler.py',
    'config/unknown_card_config.py',
    'tools/parity/py_rules_trace.py',
    'tools/phase8a/golden_sources8.py',
    'tools/phase8a/py_sampling_trace.py',
]

_FMT = [
    'tools/phase8a/golden_sources8.py',
    'tools/phase8a/py_format_trace.py',
]

KIND_PATTERNS = {'fmt': _FMT, 'rng': _RNG, 'parse': _PARSE, 'sampling': _SAMPLING, 'e2e': _E2E}


def expand(kind):
    files = set()
    for pattern in KIND_PATTERNS[kind]:
        if any(ch in pattern for ch in '*?['):
            matches = glob.glob(os.path.join(ROOT, pattern))
            if not matches:
                raise SystemExit('golden_sources8: pattern matches nothing: ' + pattern)
            for m in matches:
                files.add(os.path.relpath(m, ROOT).replace(os.sep, '/'))
        else:
            if not os.path.exists(os.path.join(ROOT, pattern)):
                raise SystemExit('golden_sources8: missing file: ' + pattern)
            files.add(pattern)
    return sorted(files)


def sha256_of(rel):
    with open(os.path.join(ROOT, rel), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def header_fields(kind):
    return {
        'python_version': '%d.%d' % sys.version_info[:2],
        'source_sha256': {rel: sha256_of(rel) for rel in expand(kind)},
    }
