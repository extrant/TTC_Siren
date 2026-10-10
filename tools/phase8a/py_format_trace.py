# -*- coding: utf-8 -*-
"""
Golden values for the number formatting of the C# compatibility layer (Phase 8A): Python's ``round(x, n)`` and ``repr(float)``.

  python tools/phase8a/py_format_trace.py --out tools/parity/golden/p8_format_g1_s20261011_n6000.jsonl.gz --seed 20261011 --count 6000

Records ``{"x": float.hex, "n": ndigits, "round": float.hex, "repr": repr(x), "repr_round": repr(round(x, n))}``. The values include exact
ties (k/16 and k/2000 multiples), tiny and huge magnitudes, negatives, round-trip edge cases and the typical response values (win
probabilities, scores, rates).
"""
import argparse
import gzip
import json
import math
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_sources8 as GS  # noqa: E402

GENERATOR_VERSION = 'p8_format/1'


def values(rng, count):
    out = []
    fixed = [0.0, -0.0, 1.0, 0.5, 0.0625, 0.1875, 0.3125, 2.5, -2.5, 0.0005, 0.0015, 0.0025, 1e-4, 1e-5, 1.5e-5, 1e15, 1e16, 1.5e16,
             123456789.125, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, 5e-324, 1.7976931348623157e308, 100.0, 12345.0, 1e22, 1e21, 123456789012345680.0]
    out.extend((x, n) for x in fixed for n in (0, 1, 2, 3))
    while len(out) < count:
        kind = rng.randrange(7)
        if kind == 0:
            x = rng.random()
        elif kind == 1:
            x = rng.uniform(-30, 30)
        elif kind == 2:
            x = rng.randrange(-20000, 20000) / 16.0           # exact ties at 3 digits
        elif kind == 3:
            x = rng.randrange(-5000, 5000) / 2000.0           # near-ties
        elif kind == 4:
            x = rng.uniform(0, 5e6)                            # node counts per second
        elif kind == 5:
            x = struct.unpack('>d', struct.pack('>Q', rng.getrandbits(64)))[0]
            if not math.isfinite(x):
                continue
        else:
            x = 1 / (1 + math.exp(-rng.uniform(-40, 40) / 5))  # sigmoid values
        out.append((x, rng.choice([0, 1, 2, 3, 3, 3])))
    return out[:count]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=6000)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf-8', newline='\n') as f:
        header = dict({'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count}, **GS.header_fields('fmt'))
        f.write(json.dumps({'header': header}, separators=(',', ':')) + '\n')
        for x, n in values(rng, args.count):
            r = round(x, n)
            f.write(json.dumps({'x': x.hex(), 'n': n, 'round': r.hex(), 'repr': repr(x), 'repr_round': repr(r)}, separators=(',', ':')) + '\n')
    print('format trace: %d records -> %s' % (args.count, args.out))


if __name__ == '__main__':
    main()
