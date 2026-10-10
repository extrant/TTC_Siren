# -*- coding: utf-8 -*-
"""
Golden sequences of CPython's ``random`` module for the C# ``PythonRandom`` (Phase 8A, decision A-23).

  python tools/phase8a/py_rng_trace.py --out tools/parity/golden/p8_rng_g1_s20261011_n600.jsonl.gz --seed 20261011 --count 600

Every record is ``{"seed": <int as string>, "ops": [...], "out": [...]}``: ``random.seed(seed)`` and then the operations in order. The
operations are drawn from the module-level functions the unknown-card handler uses (``sample``, ``choice``) and from every other primitive
the C# port implements (``random``, ``getrandbits``, ``_randbelow``, ``randrange``, ``randint``, ``shuffle``, ``uniform``), in
random mixes, so that the generator state is checked as a whole after arbitrary call sequences. Seeds cover 0, small values, the 32-bit
and 64-bit boundaries, integers wider than 64 bits and negative integers. Floats are written as ``float.hex()``, large ints as strings.
"""
import argparse
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import golden_sources8 as GS  # noqa: E402

GENERATOR_VERSION = 'p8_rng/1'

SPECIAL_SEEDS = [0, 1, 2, 3, 5, 42, 255, 256, 65535, 65536, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1, 2 ** 53,
                 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1, 2 ** 64, 2 ** 64 + 12345, 2 ** 96 + 7, 2 ** 128 - 1, 10 ** 30,
                 -1, -2, -(2 ** 32), -(2 ** 63), -(2 ** 64) - 5, 20261011, 123456789012345678]


def make_ops(meta, count):
    ops = []
    for _ in range(count):
        kind = meta.choice(['random', 'bits', 'bits', 'below', 'below', 'range', 'randint', 'choice', 'sample', 'sample', 'sample_big',
                            'shuffle', 'uniform'])
        if kind == 'random':
            ops.append(['random'])
        elif kind == 'bits':
            ops.append(['bits', meta.choice([1, 2, 3, 7, 8, 16, 31, 32, 33, 48, 64, 65, 100, 127])])
        elif kind == 'below':
            ops.append(['below', meta.choice([1, 2, 3, 5, 9, 10, 45, 100, 475, 1000, 4096, 9500, 65536, 2 ** 31 - 1, 2 ** 32 - 1])])
        elif kind == 'range':
            a = meta.randrange(-50, 50)
            ops.append(['range', a, a + meta.randrange(1, 1000)])
        elif kind == 'randint':
            a = meta.randrange(-50, 50)
            ops.append(['randint', a, a + meta.randrange(0, 500)])
        elif kind == 'choice':
            ops.append(['choice', meta.choice([1, 2, 3, 5, 17, 475])])
        elif kind == 'sample':          # small populations: the pool algorithm
            n = meta.randrange(1, 60)
            ops.append(['sample', n, meta.randrange(0, min(n, 12) + 1)])
        elif kind == 'sample_big':      # populations around and above CPython's set-size threshold
            k = meta.choice([1, 2, 3, 5, 6, 8, 12, 30, 150])
            n = meta.choice([k, k + 1, 22, 25, 40, 100, 475, 1045, 1046, 1100, 5000, 9500])
            ops.append(['sample', max(n, k), k])
        elif kind == 'shuffle':
            ops.append(['shuffle', meta.randrange(0, 40)])
        else:
            ops.append(['uniform', meta.choice([0.0, -3.5, 1.0]), meta.choice([1.0, 10.0, 7.25])])
    return ops


def run(seed, ops):
    r = random.Random()
    r.seed(seed)
    out = []
    for op in ops:
        k = op[0]
        if k == 'random':
            out.append(r.random().hex())
        elif k == 'bits':
            out.append(str(r.getrandbits(op[1])))
        elif k == 'below':
            out.append(r._randbelow(op[1]))
        elif k == 'range':
            out.append(r.randrange(op[1], op[2]))
        elif k == 'randint':
            out.append(r.randint(op[1], op[2]))
        elif k == 'choice':
            out.append(r.choice(list(range(op[1]))))
        elif k == 'sample':
            out.append(r.sample(list(range(op[1])), op[2]))
        elif k == 'shuffle':
            x = list(range(op[1]))
            r.shuffle(x)
            out.append(x)
        elif k == 'uniform':
            out.append(r.uniform(op[1], op[2]).hex())
        else:
            raise SystemExit('unknown op ' + k)
    return out


def generate(seed, count):
    meta = random.Random(seed)
    records = []
    for i in range(count):
        if i < len(SPECIAL_SEEDS):
            s = SPECIAL_SEEDS[i]
        elif i % 5 == 0:
            s = meta.getrandbits(64)
        elif i % 7 == 0:
            s = meta.getrandbits(meta.choice([40, 80, 130]))
        else:
            s = meta.randrange(1, 10 ** 9)
        ops = make_ops(meta, meta.randrange(5, 60))
        records.append({'seed': str(s), 'ops': ops, 'out': run(s, ops)})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--count', type=int, default=600)
    args = ap.parse_args()
    records = generate(args.seed, args.count)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf-8', newline='\n') as f:
        header = dict({'generator': GENERATOR_VERSION, 'seed': args.seed, 'count': args.count}, **GS.header_fields('rng'))
        f.write(json.dumps({'header': header}, ensure_ascii=False, separators=(',', ':')) + '\n')
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
    print('rng trace: %d records -> %s' % (len(records), args.out))


if __name__ == '__main__':
    main()
