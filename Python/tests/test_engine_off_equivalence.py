"""Phase 7: with the C# engine switched off, ``/ai_move`` answers exactly as it did before Phase 7.

``tests/data/engine_off_expected.json`` holds the (normalised) responses that commit 03b4704, the last commit before Phase 7, gave for a
sample of the 360-request corpus (tools/phase7/gen_corpus.py, seed 20261010) with fixed random seeds and deterministic search settings
(max_depth 4, max_time 10^6). The full 360-request comparison is tools/phase7/equivalence.py (see docs/migration/reports/PHASE7_INTEGRATION.md);
this is its regression sample. Fields that measure time are excluded (see equivalence.normalise).
"""
import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'phase7'))

import ai_server  # noqa: E402
from ai import csharp_engine  # noqa: E402
import equivalence  # noqa: E402
import gen_corpus  # noqa: E402

with open(os.path.join(ROOT, 'tests', 'data', 'engine_off_expected.json'), encoding='utf-8') as f:
    EXPECTED = json.load(f)


class NullUi:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture(scope='module')
def corpus():
    entries = gen_corpus.generate(EXPECTED['corpus']['seed'], EXPECTED['corpus']['count'])
    return {e['i']: e for e in entries}


@pytest.mark.parametrize('index', sorted(int(i) for i in EXPECTED['responses']))
def test_engine_off_response_equals_the_pre_phase7_response(index, corpus, monkeypatch):
    monkeypatch.setenv('TTC_CS_ENGINE', 'off')
    monkeypatch.setattr(ai_server, 'console_ui', NullUi())
    csharp_engine.reset()
    ai_server.ensure_handler_initialized()
    entry = corpus[index]
    random.seed(entry['seed'])
    response = ai_server.app.test_client().post('/ai_move', json=entry['request'])
    expected = EXPECTED['responses'][str(index)]
    assert response.status_code == expected['status']
    assert equivalence.normalise(response.get_json()) == expected['body'], 'request %d (tags %s)' % (index, entry['tags'])
