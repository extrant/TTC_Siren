# -*- coding: utf-8 -*-
"""
Binds golden data to the Python reference source (review finding M-1 / M4-2).

Every golden file written by the generators in tools/parity/ carries, in its header, the SHA-256 of every file of
the reference implementation (and of the generator itself) that can influence its content, plus the Python
major.minor version. The C# tests (GoldenFiles.cs / GoldenSources) recompute the same hashes from the working tree
and regenerate a golden file whose header does not match, so a golden file can never silently outlive the Python
code it was generated from.

Read-only: this module only reads files.

The file sets below are mirrored by GoldenSources.cs; keep both in step (a mismatch of the set makes the C# check
fail, it never passes silently).
"""
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

_RULES = [
    'core/*.py',
    'csv_compat.py',
    'data/幻卡数据库.csv',
    'tools/parity/golden_sources.py',
    'tools/parity/py_rules_trace.py',
]
_SEARCH = _RULES + [
    'ai/ai.py',
    'tools/parity/py_search_trace.py',
]
_ENDGAME = _SEARCH + [
    'ai_server.py',
    'ai/unknown_card_handler.py',
    'config/unknown_card_config.py',
    'tools/parity/py_endgame_trace.py',
]

_E2E = _ENDGAME + [
    'ai/csharp_engine.py',
    'tools/phase7/gen_corpus.py',
    'tools/phase7/e2e_reference.py',
]

KIND_PATTERNS = {
    'rules': _RULES,
    'carddb': _RULES,
    'search': _SEARCH,
    'endgame': _ENDGAME,
    'e2e': _E2E,
}


def expand(kind):
    """Sorted repository-relative paths (forward slashes) of the files a golden file of this kind is bound to."""
    files = set()
    for pattern in KIND_PATTERNS[kind]:
        if any(ch in pattern for ch in '*?['):
            matches = glob.glob(os.path.join(ROOT, pattern))
            if not matches:
                raise SystemExit('golden_sources: pattern matches nothing: ' + pattern)
            for m in matches:
                files.add(os.path.relpath(m, ROOT).replace(os.sep, '/'))
        else:
            if not os.path.exists(os.path.join(ROOT, pattern)):
                raise SystemExit('golden_sources: missing file: ' + pattern)
            files.add(pattern)
    return sorted(files)


def sha256_of(rel):
    with open(os.path.join(ROOT, rel), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def header_fields(kind):
    """Extra header entries: the Python major.minor version and the SHA-256 of every bound source file."""
    return {
        'python_version': '%d.%d' % sys.version_info[:2],
        'source_sha256': {rel: sha256_of(rel) for rel in expand(kind)},
    }
