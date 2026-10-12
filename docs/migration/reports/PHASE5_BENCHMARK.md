# Phase 5 benchmark report — Python (CPython 3.11.4) versus C# (net10.0, Release)

Date: 2026-10-08. Machine: Intel Core i7-13650HX (14C/20T), 32 GB, Windows 11 (`CSHARP_MIGRATION.md` section 3.1).
CPython 3.11.4, no PyPy (Q-6 default: CPython baseline only). Single thread on both sides. Every number below was
measured in this session; nothing is estimated. Run-to-run noise on this machine is about ±10% (more under load), so
medians of at least 3 runs are reported and ratios are taken inside one batch.

Reproduce:

```
python tools/bench/py_endgame_bench.py --out tools/bench/endgame_bench_positions.json --per-cell 6 --runs 3
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- endgame tools/bench/endgame_bench_positions.json 5
python tools/bench/py_endgame_bench.py --exact --runs 3
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- exact 5
python tools/bench/py_bench.py . fixed        # and: server
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search 5
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- fast 5
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- server 10
```

## 1. Information-set robust endgame pipeline (scenario construction + robust selection, both modes)

72 deterministic positions (seed 20261010): occupancy 5, 6, 7, 8 × 0, 1, 2 unknown opponent cards × 6 positions,
rule sets cycling through none / 同数 / 同数+加算 / 逆转+王牌杀手 / 同类强化 / all five. For each position: one call of
`_build_endgame_scenarios` and of `select_endgame_robust_move` in normal and aggressive mode (the production scenario
count: 1 for 0 unknown, 16 for 1, 24 for 2). Python: median of 3 runs per call. C#: median of 5 runs per call.
**C# selected the same move as Python in all 72 positions, in both modes (72 identical, 0 different), and built the same
number of scenarios.** At occupancy 5 or more the robust selection solves every (move, scenario) pair exactly.

| occupancy | unknown | positions | Python s | C# s | speed-up |
|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 6 | 0.1474 | 0.00463 | 31.8× |
| 5 | 1 | 6 | 3.8978 | 0.04304 | 90.6× |
| 5 | 2 | 6 | 3.3700 | 0.01609 | 209.4× |
| 6 | 0 | 6 | 0.0178 | 0.00034 | 51.9× |
| 6 | 1 | 6 | 0.3730 | 0.00296 | 125.8× |
| 6 | 2 | 6 | 0.3601 | 0.00356 | 101.1× |
| 7 | 0 | 6 | 0.0076 | 0.00009 | 89.5× |
| 7 | 1 | 6 | 0.1469 | 0.00163 | 90.0× |
| 7 | 2 | 6 | 0.2556 | 0.00219 | 116.5× |
| 8 | 0 | 6 | 0.0022 | 0.00005 | 41.8× |
| 8 | 1 | 6 | 0.0555 | 0.00143 | 38.9× |
| 8 | 2 | 6 | 0.0805 | 0.00184 | 43.8× |
| **total** | | 72 | **8.7144** | **0.07786** | **111.9×** |

Three independent C# runs gave totals 0.07786 / 0.07989 / 0.07969 s (median 0.07969 s, 109.4× faster than Python's
8.7144 s). The largest absolute cost is occupancy 5 with 1 or 2 unknown cards: Python spends about 0.6 s per position there
(all three calls), C# about 3–7 ms.

## 2. Exact endgame solver (`_solve_endgame_exact`, section 3.6 positions plus one larger position)

Root value for the side to move. Python: median of 3 runs; C#: median of 5 runs in each of 3 process runs (the median
over the 3 run medians is shown). The values are identical.

| position | value | Python make_move calls | C# make_move calls | Python s | C# s | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| end5_same_plus | 1 | 217 | 217 | 0.00381 | 0.00004 | ≈ 95× |
| mid4_rev_ace | −2 | 4,707 | 4,707 | 0.05012 | 0.00082 | ≈ 61× |
| mid3_ascension (6 empty cells) | −1 | 297,935 | 241,264 | 15.58 | 0.05918 | ≈ 263× |

C# allocates 0 bytes during the solve. The C# memo is keyed by the Zobrist hash, which ignores hand order, so it makes
fewer moves than Python's order-sensitive key on `mid3_ascension` (241,264 versus 297,935); on the two smaller
positions the counts are equal. (The memo key is an implementation detail; the value is a pure function of the position.)

## 3. Baseline items of CSHARP_MIGRATION.md section 3

Fixed-depth minimax, the 22 items of section 3.4 (43,003 nodes in both implementations; ReferenceCompat reproduces
Python's nodes, scores and moves exactly):

| | run 1 | run 2 | run 3 | median | NPS |
|---|---:|---:|---:|---:|---:|
| Python | 2.592 s | 2.603 s | 2.732 s | **2.603 s** | 16.5 k |
| C# ReferenceCompat (`Bench search`) | 0.0528 s | 0.0516 s | 0.0524 s | **0.0524 s** | 820 k |

C# ReferenceCompat is about **49.7×** faster than CPython on this suite.

Server-equivalent call (`find_best_move_parallel`, max depth 10, 10 s limit; wall clock, three runs each):

| position | Python depth / nodes (3 runs) | C# ReferenceCompat depth / nodes | C# Fast depth / nodes / time |
|---|---|---|---|
| open_none | 5 / 187,537; 187,078; 187,870 | 10 / 15,944,848 (8.8 s) | 9 / 2,566,250 (0.93 s) |
| open_same_plus | 5 / 178,099; 176,681; 178,213 | 9 / 16.5–16.7 M (10 s) | 9 / 3,915,260 (1.54 s) |
| mid3_ascension | 10 / 30,500 (2.6 s) | 10 / 30,500 (0.03 s) | 6 / 8,651 (0.01 s) |

The move is the same in every run and every implementation (`27` at (2,2), `27` at (2,2), `53` at (1,2)). Fast stops
when the whole game tree is searched (depth = number of empty cells, D-014), so it finishes the open boards in about a
second and one to two seconds, where Python reaches depth 5 in ten seconds.

Phase 4 Fast search versus ReferenceCompat (table off / on), fixed depth, d ≥ 4 items, 3 runs: Fast 1.3272 / 1.3304 /
1.3388 s, TT off 2.0227 / 2.0234 / 2.0222 s, TT on 2.1175 / 2.1230 / 2.1352 s (node counts 3,625,620 / 3,771,758 /
3,431,185): Fast is about 1.52× faster than the table-off oracle and 1.60× faster than Python-semantics ReferenceCompat.

## 4. Caveats

- Python timings are CPython 3.11.4 only (Q-6). A PyPy build, which the production package may use, is faster than
  CPython and would narrow the ratios; it was not measured because PyPy is not installed.
- Section 1 uses the deterministic part of the pipeline only. The random fallback that Python uses when no legal
  candidate exists (`handler.generate_opponent_cards`) is not part of the C# port (A-7) and is excluded from both sides.
- The robust selection is only used by the Flask handler when at least five cells are occupied; the heuristic
  one-ply branch that the golden tests also cover is not reachable from production (CSHARP_MIGRATION.md section 4, D-016).

---

# Corrections after the Opus benchmark review (2026-10-08)

Source: `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` (verdict BENCHMARK_VALID_WITH_CAVEATS: correctness is fine, some
numbers above are biased). The tables above are kept as they were; where they disagree with this section, **this section
wins**. Nothing in `TTC.Siren.Core` / `TTC.Siren.Search`, in any test assertion, or in any Python business code was
changed for this correction, and no timing change below is a solver improvement. Node counts, score bits, moves and the
72/72 endgame selections are identical to before in every table (checked item by item, see "Verification").

## C1. What was wrong, and the status of each headline figure

| Figure in the original report | Status | Replaced by |
|---|---|---|
| Fixed suite "≈ 49.7×" (section 3) | **Cold-JIT figure; relabelled** | cold **53–63×**, warm **119×** (C3) |
| Exact solver "≈ 263×" for `mid3_ascension` (section 2) | **Withdrawn** (it used one contaminated Python timing of 15.58 s) | **161–167×** total, **131×** per `make_move`, with 1.235× of work saved by the memo key (C4) |
| Robust pipeline "111.9×" / "109.4×" (section 1) | Cold-JIT figure; relabelled | cold **111–112×**, warm **178–189×**; 1.074× of it is less work (C5) |
| Per-cell speed-ups of section 1 (31.8× … 209.4×) | Order artefact, **do not use** | the occupancy-5 cells ran first in the process, on tier-0 code (C5) |
| Server-equivalent table (section 3) | Not a speed-up measurement; **no ratio is derived from it** | NPS ratio ≈ 95–100× (C6) |
| Fast "1.52× / 1.60×" (section 3) | Holds | warm **1.49× / 1.60×**, cold **1.49× / 1.58×**, rotating variant order (C7) |

Why the numbers moved:

- **A-1, JIT.** With tiered compilation the hot methods start as tier-0/OSR code. The old `Bench` warmed up each item once
  and then timed it, so the first items of every process ran slow code. The old figures are therefore cold-start figures.
  The harness now warms the **whole item set** for 1.5 s first (default, labelled *warm*) and keeps the old behaviour as
  `--cold`. Example, same binary, `open_none` d4 (15,807 nodes): cold median 25.6 ms (min 8.6, max 36.2 ms: the run
  that crossed the tier-1 promotion), 0.62 M NPS; warm median 8.8 ms (min 8.8, max 9.3 ms), 1.80 M NPS. In the same cold run
  `open_same_plus` d4 (later in the process, already warm) is at 1.70 M NPS, which is why per-item ratios of the old
  report swung with the position of the item in the run order.
- **A-2, the 15.58 s.** My first Python timing of `mid3_ascension` overlapped with a background process that was generating
  golden data on the same machine. Measured alone (3 runs: 10.35 / 10.47 / 10.49 s) and in 6 interleaved rounds
  (10.24–10.67 s), the Python figure is **10.2–10.7 s**. The reviewer measured 9.95–10.21 s; the difference to my figures
  is 1–5%, inside the 20% bound the remediation goal sets, so no human decision is needed.
- **A-3, batches.** The old Python and C# columns were measured in separate batches whose load differed by up to ~40%.
  The new runner alternates them.
- **A-4, display.** The old `Bench search` printed seconds with 4 decimals, so items under 0.1 ms showed `0.0000`. Times are
  now printed in microseconds with one decimal, and items under 50 µs are marked `*` (timer-limited).

## C2. Harness and reproduction

```
# C# harness: warm (default) or cold; every table says which; min / median / max everywhere
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search 5            # warm
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- search 5 --cold     # cold (old configuration)
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- fast 5 | exact 5 | endgame <positions.json> 5 | server 10 3
# Python and C# interleaved, >= 3 rounds, per-round totals and ratios, min / median / max over the rounds
python tools/bench/run_interleaved.py --rounds 3            # warm C#
python tools/bench/run_interleaved.py --rounds 3 --cold     # cold C#
# Work comparison of the robust pipeline (not timed)
python tools/bench/count_robust_work.py
dotnet run -c Release --project csharp/src/TTC.Siren.Bench -- endgame-work tools/bench/endgame_bench_positions.json
```

`py_bench.py <root> fixed|exact|server [runs]` and `py_endgame_bench.py [--exact] --runs N` now print min / median / max of
their runs (L5-3). Nothing else may run on the machine while these scripts run (wall-clock figures).

## C3. Fixed-depth suite (the 22 items of section 3.4; 43,003 nodes on both sides)

C# `Bench search`, three separate process runs per mode, sum of the per-item median (min–max) in seconds:

| mode | run 1 | run 2 | run 3 |
|---|---|---|---|
| warm | 0.02333 (0.02202–0.02535) | 0.02289 (0.02145–0.02444) | 0.02306 (0.02205–0.02424) |
| cold | 0.05104 (0.02902–0.05715) | 0.05177 (0.02947–0.05923) | 0.05287 (0.02882–0.05667) |

The warm total agrees with the reviewer's steady state (0.024–0.025 s). The `*` marks: all depth-1 items and the
cheapest depth-2/3 items are under 50 µs (timer-limited).

Interleaved with Python (3 rounds, CPython 3.11.4, `run_interleaved.py`):

| C# mode | Python s (min / median / max) | C# s (min / median / max) | ratio (min / median / max) |
|---|---|---|---|
| warm | 2.6816 / 2.7109 / 2.7451 | 0.02252 / 0.02280 / 0.02300 | **118.9 / 119.1 / 119.3×** |
| cold | 2.7300 / 2.7554 / 2.7984 | 0.04404 / 0.04684 / 0.05237 | **53.4 / 58.3 / 62.6×** |

The old "49.7×" is the cold figure (ratio of 2.603 s to 0.0524 s measured in separate batches). The per-item cold
timings are not stable (a run that crosses the tier-1 promotion is up to 4× slower than the next one), hence the wide
cold C# range.

## C4. Exact endgame solver (replaces the 263× row of section 2)

Interleaved, 3 rounds, C# warm; Python = single run per round (min / median / max over rounds):

| position | make_move Python / C# (work ratio) | Python s | C# s | ratio |
|---|---|---|---|---|
| end5_same_plus | 217 / 217 (1.000) | 0.00344 / 0.00348 / 0.00357 | 0.000014 / 0.000015 / 0.000017 (timer-limited) | 205 / 236 / 251× |
| mid4_rev_ace | 4,707 / 4,707 (1.000) | 0.05476 / 0.05502 / 0.05524 | 0.000179 / 0.000187 / 0.000225 | 245 / 293 / 308× |
| **mid3_ascension** | **297,935 / 241,264 (1.235)** | **10.239 / 10.325 / 10.393** | **0.0636 / 0.0639 / 0.0646** | **161 / 161 / 162×** |

Cold C# (3 more rounds): `mid3_ascension` Python 10.36 / 10.61 / 10.67 s, C# 0.0622 / 0.0649 / 0.0651 s, ratio
**163 / 164 / 167×**; `mid4_rev_ace` 65 / 70 / 73×; `end5_same_plus` 86 / 98 / 99× (both tiny items are dominated by
tier-0 code and are timer-limited, so only the large position is a meaningful ratio).

Decomposition for `mid3_ascension` (warm): per `make_move` Python 10.325 s / 297,935 = 34.7 µs, C# 0.0639 s / 241,264 =
0.265 µs, i.e. **≈ 131× runtime and data-layout**; times **1.235× algorithmic** (the C# memo is keyed by the Zobrist
hash, which ignores hand order, while Python's `_endgame_state_key` includes it, so C# makes fewer moves; the value is the
same). 131 × 1.235 = 161. **The 263× figure is withdrawn.**

## C5. Robust endgame pipeline (replaces the ratios of section 1)

Interleaved, 3 rounds, 72 positions, both modes, 72/72 identical selections in every round:

| C# mode | Python s (min / median / max) | C# s (min / median / max) | ratio (min / median / max) |
|---|---|---|---|
| warm | 9.1822 / 9.2005 / 9.5797 | 0.04987 / 0.05081 / 0.05161 | **177.9 / 184.5 / 188.5×** |
| cold | 9.0262 / 9.1422 / 9.2154 | 0.08035 / 0.08243 / 0.08264 | **110.9 / 111.5 / 112.3×** |

Work comparison (`count_robust_work.py` and `Bench endgame-work`, not timed; 72 positions × 2 modes; first moves
included): Python **418,990** `make_move` calls (209,495 per mode), C# **389,990** (194,995 per mode): **1.074×**. Python
shares one exact-solver cache per move across the scenarios, C# keeps one memo per scenario across the moves (the
context rule of the Fast table), so only the occupancy-5 cells differ (the other cells make the same number of moves).
Both sides evaluate the same **10,834** (move, scenario) pairs. At equal work the ratios are about 172× (warm) and
104× (cold).

A single warm run of `Bench endgame` (per-cell figures; the occupancy-5 cells are no longer penalised by running first):

| occupancy | unknown | Python s | C# s | ratio |
|---:|---:|---:|---:|---:|
| 5 | 0 / 1 / 2 | 0.1474 / 3.8978 / 3.3700 | 0.00269 / 0.02131 / 0.01704 | 54.9 / 182.9 / 197.7× |
| 6 | 0 / 1 / 2 | 0.0178 / 0.3730 / 0.3601 | 0.00041 / 0.00251 / 0.00218 | 43.9 / 148.7 / 165.1× |
| 7 | 0 / 1 / 2 | 0.0076 / 0.1469 / 0.2556 | 0.00012 / 0.00124 / 0.00134 | 64.1 / 118.1 / 191.1× |
| 8 | 0 / 1 / 2 | 0.0022 / 0.0555 / 0.0805 | 0.00004 / 0.00091 / 0.00106 | 53.5 / 60.9 / 76.0× |

The 0-unknown cells (a single scenario, a few hundred microseconds or less) are partly timer- and overhead-limited.

## C6. Server-equivalent call (section 3.5)

Wall-clock limited, so the depth reached depends on speed: **do not derive a speed-up from the depth**. What can be said is
the node rate: Python 17.8–18.8 k NPS (178–188 k nodes in 10 s), ReferenceCompat 1.68–1.83 M NPS, i.e. **≈ 95–100× per
node** (consistent with the warm fixed-suite ratio). Fast reaches depth 9 (the whole tree) because of the bound-aware table
and the early stop at depth = empty cells (D-014); that is an algorithmic effect and has no ratio.

## C7. Fast search (Phase 4)

Warm, rotating variant order, d ≥ 4, 5 runs (sums of per-item medians, min–max): TT on 2.2288 s (2.1865–2.3197), TT off
2.0791 s (2.0504–2.1300), Fast 1.3911 s (1.3521–1.4625): Fast is **1.49×** faster than the table-off oracle and
**1.60×** faster than Python-semantics ReferenceCompat; cold: 2.1897 / 2.0652 / 1.3830 s, **1.49× / 1.58×**. All 40 rows
have the oracle's exact score (`same`). **Fast node counts and NPS are not comparable with Python's or ReferenceCompat's**:
Fast explores a different tree (bound-aware table, stored move first, no sorting at depth 1) and counts the unsorted
depth-1 leaves as nodes. Only the time to the same score is comparable.

## C8. Trusted ranges (versus CPython 3.11.4, single thread, this machine, 2026-10-08)

The "ranges" are the observed min–max of 3 interleaved rounds (not statistical confidence intervals; with 3 rounds
nothing more is defensible). Python variation was ±1–4% except for the contaminated figure above.

| Workload | Cold process (tier-0 start) | Warm (resident engine) | Of which algorithmic |
|---|---|---|---|
| Python-semantics search (same nodes) | 53–63× | **119×** | none |
| Robust endgame pipeline | 111–112× | 178–189× | 1.074× |
| Exact solver, large endgame | 163–167× | 161–162× | 1.235× (memo key) |
| Exact solver, tiny endgames | 65–99× (dominated by cold code) | 205–308× (timer-limited, not meaningful) | 1.0× |
| Fast over the table-off oracle | 1.49× | 1.49× | 1.49× is algorithmic/ordering (different tree) |

Single headline, if one is needed: "**≈ 55–190× versus CPython 3.11.4 depending on workload and JIT warm-up (cold
per-request process at the low end, warm resident engine at the high end), PyPy not measured**". Do not quote 263× or
49.7× at all, and do not quote any ratio without "versus CPython 3.11.4".

## C9. Caveats that still apply

- Every ratio is versus **CPython 3.11.4** only. The production package may run PyPy (Q-6 default: not measured, not
  installed). PyPy is faster than CPython, so every ratio above would be lower there, by an unknown factor.
- Python's first call includes the lazy load of the card database (< 1% of the suite); the `py_bench.py fixed` runs now
  perform one untimed call first.
- The Fast table in `Bench fast` is the 96 MiB default; the exact-solver memo is 32 K entries per `Select` call
  (≈ 768 KiB allocated per call; not measured by `Bench endgame`, review L5-2).
- Tier-1 code with PGO is another few percent faster than `TieredCompilation=0` on long runs; "warm" here means the
  1.5 s whole-set warm-up, not an infinitely long one.

## C10. Verification of this correction

- `dotnet build -c Release --no-incremental`: 0 warnings, 0 errors. `dotnet test`: 78/78 (no test changed).
  `python -m pytest -q`: 29 passed.
- Bench output compared with the output before the harness change: `search` 22 items, identical node counts and score
  bits; `fast` 40 rows, identical node counts, every row `same`; `exact` identical values, make_move, node and memo-hit
  counts; `endgame` 72/72 identical selections. Allocation columns 0 for `search`, `fast`, `exact`.

---

## A-7: CPU core placement (2026-10-08, after the benchmark re-review)

Source: `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` §12.4. **Every trusted ratio in this report is for both sides pinned
to the same CPU set, the P-cores (`0xFFF`), and for CPython 3.11.4.** Earlier unpinned figures (C1–C10 above) agree with the
pinned ones only because both sides happened to land on P-cores; unpinned runs are not reproducible on this machine.

### A7.1 The problem and the mechanism

The i7-13650HX is a hybrid CPU: 6 P-cores with SMT (logical processors 0–11, mask `0xFFF`, Windows efficiency class 1) and
8 E-cores (12–19, mask `0xFF000`, class 0). Windows places a long single-thread run on either class, non-deterministically.
Observed today: the same Python fixed suite took 2.68–2.75 s in one batch, 3.04–3.10 s pinned to the P-cores, 3.40–3.47 s
unpinned in another batch, and 4.72–4.74 s pinned to the E-cores. An unpinned interleaved run whose Python half lands on
E-cores and whose C# half lands on P-cores reports an inflated ratio (the review saw 271× and 301× where the P-core-pinned
values are about 162× and 173× in this re-measurement). Pinning .NET to a single logical CPU is no remedy (its JIT and GC threads share that CPU and
the result is distorted by 2–4×), so the mask is always a CPU **set**.

Mechanism (`tools/bench/pin.py`, used by `tools/bench/run_interleaved.py --affinity`):

- The child process is created **suspended**, `SetProcessAffinityMask` is applied, then it is resumed
  (`NtResumeProcess`). It therefore never runs on another CPU, the .NET runtime sizes its worker threads from the
  restricted set (`Environment.ProcessorCount` = 12 for `0xFFF`, 8 for `0xFF000`, 20 unpinned) and its children inherit
  the mask. The C# harness is started as the built `TTC.Siren.Bench.exe` (no `dotnet run` wrapper that would spawn the real
  process outside the mask).
- The P-core set is **detected**, not hard-coded: `--affinity auto` reads the efficiency classes from
  `GetLogicalProcessorInformationEx` (highest class = P-cores). `--affinity FFF` passes an explicit hex mask (`FFF` is this
  machine's value; `python tools/bench/pin.py` prints the detected classes). Without `--affinity` the runner prints a
  warning on a hybrid CPU that the results are not reproducible.
- `Bench` (C#) and `py_bench.py` / `py_endgame_bench.py` print the affinity mask they **actually run with**, read back
  from the OS, in their header (`affinity=0xFFF cpus=12`); the runner checks that every child reports the requested mask
  and prints the mask in its header, in every result line and in the summary.

Reproduce:

```
python tools/bench/pin.py                                            # detected core classes and masks
python tools/bench/run_interleaved.py --rounds 3 --affinity auto     # warm C#, both sides on the P-cores (0xFFF here)
python tools/bench/run_interleaved.py --rounds 3 --affinity auto --cold
python tools/bench/run_interleaved.py --rounds 3 --affinity FF000    # E-cores, to document the effect
```

### A7.2 Pinned results, both sides on the P-cores (`0xFFF`), 3 interleaved rounds, min / median / max

Warm C# (whole item set warmed for 1.5 s):

| suite | Python s | C# s | ratio |
|---|---|---|---|
| fixed, 22 items (43,003 nodes) | 3.0420 / 3.0725 / 3.1000 | 0.02625 / 0.02625 / 0.02635 | **115.9 / 116.6 / 118.1×** |
| exact, three positions | 11.3794 / 11.4229 / 11.5766 | 0.07024 / 0.07052 / 0.07186 | **161.1 / 162.0 / 162.0×** |
| exact `mid3_ascension` | 11.312 / 11.358 / 11.509 | 0.06999 / 0.07018 / 0.07163 | 161 / 162 / 162× |
| exact `mid4_rev_ace` (C# 0.22–0.33 ms) | 0.0609 / 0.0633 / 0.0633 | 0.000218 / 0.000230 / 0.000331 | 184 / 275 / 291× |
| exact `end5_same_plus` (C# 16–17 µs, timer-limited) | 0.0037 / 0.0042 / 0.0043 | 0.000016 / 0.000017 / 0.000017 | 214 / 247 / 261× |
| robust pipeline, 72 positions | 9.7953 / 10.0004 / 10.0500 | 0.05730 / 0.05788 / 0.05823 | **169.2 / 172.6 / 174.5×** |

Cold C# (`--cold`, one untimed run per item):

| suite | Python s | C# s | ratio |
|---|---|---|---|
| fixed, 22 items | 2.8286 / 2.9547 / 3.0894 | 0.03396 / 0.03591 / 0.04685 | 60.4 / 82.3 / 91.0× (see A7.4) |
| exact, three positions | 10.7674 / 10.9325 / 11.1752 | 0.06266 / 0.06566 / 0.06751 | **159.5 / 170.2 / 174.5×** |
| exact `mid3_ascension` | 10.707 / 10.870 / 11.112 | 0.06176 / 0.06482 / 0.06664 | 161 / 171 / 176× |
| robust pipeline, 72 positions | 9.1942 / 9.4752 / 9.5828 | 0.07799 / 0.08463 / 0.08564 | **108.6 / 110.6 / 122.9×** |

In every round the node counts (43,003), the exact values (1 / −2 / −1) and the 72/72 selections were identical to Python
(the runner asserts them). The `mid3_ascension` decomposition of C4 still holds: work ratio 1.235 (297,935 / 241,264
`make_move`), so about 161 / 1.235 ≈ 131× per `make_move`.

### A7.3 Effect of the core class (both sides on the E-cores, `0xFF000`, warm, 3 rounds)

| suite | Python s | C# s | ratio |
|---|---|---|---|
| fixed | 4.7197 / 4.7386 / 4.7398 | 0.03666 / 0.03679 / 0.03726 | 126.7 / 128.8 / 129.2× |
| exact (three positions) | 16.4582 / 16.6698 / 17.4180 | 0.09320 / 0.09331 / 0.09345 | 176.4 / 178.4 / 186.9× |
| exact `mid3_ascension` | 16.351 / 16.560 / 17.309 | 0.09284 / 0.09296 / 0.09307 | 176 / 178 / 186× |
| robust pipeline | 15.4046 / 15.4759 / 15.7849 | 0.08032 / 0.08238 / 0.08300 | 186.4 / 187.0 / 196.5× |

E-cores slow both sides down: Python by 1.46–1.55× (fixed 3.07 → 4.74 s, `mid3_ascension` 11.36 → 16.56 s, pipeline 10.00 →
15.48 s) and C# by 1.32–1.42× (fixed 0.0263 → 0.0368 s, `mid3_ascension` 0.0702 → 0.0930 s, pipeline 0.0579 → 0.0824 s).
With both sides on the same class the ratios stay close (E-core ratios are 8–10% higher because Python loses a little more).
**Mixed placement is what breaks the ratio**, not the core class itself. It also bounds the noise of a C#-versus-C#
before/after comparison: one side landing on E-cores moves a time by about 1.3–1.4×, larger than most optimisations, so
all Baseline→Compare runs (CLAUDE.md §6) must be pinned to the same set.

### A7.4 Comparison with the reviewer's pinned figures, and one open point

| ratio (both on P-cores) | reviewer (§12.6) | measured here | within 15%? |
|---|---|---|---|
| fixed, warm | 118–121× | 115.9–118.1× | yes |
| robust pipeline, warm | 178–189× | 169.2–174.5× (−5% against the lower bound) | yes |
| robust pipeline, cold | 109–112× | 108.6–122.9× (+10% against the upper bound) | yes |
| exact `mid3_ascension` | 150–180× | 159.5–176× (warm and cold) | yes |
| **fixed, cold** | **55–73×** | **60.4 / 82.3 / 91.0× in the 3-round run** | **median yes; round 1 (91.0×) is 25% above 73×** |

The cold fixed-suite ratio is the one figure outside the 15% band in a single round. It is a cold-JIT measurement and
bimodal: the C# cold total is either about 0.035 s or about 0.048 s depending on when the tier-1 code arrives, while
Python is steady (±3%). Eight further pinned cold rounds (fixed suite only) gave 83.6 / 57.5 / 59.4 / 81.9 / 81.0 / 70.1 /
63.0 / 63.8× (median 66.9×); over all 11 pinned cold rounds the ratio is **57.5–91.0×, median 70.1×** (C# cold total
0.034–0.050 s). The median is inside the reviewer's range; individual rounds are not. **PROVISIONAL, awaiting the user's confirmation. The user answered the open question (median or whole range?) only with "continue"; the assistant applied the most conservative reading, which is NOT an approval of a figure: the cold fixed-suite ratio is quoted as the whole observed range together with its median, "57–91× (median 70×) over 11 pinned cold rounds", never as a single number. One sentence changes if the user prefers the median alone or the range alone.** This states the observed spread; it is not a narrowed "trusted" interval, and the reviewer's 55–73× is consistent with its lower and central part. The warm figure (116–118×) is stable and unaffected.

The earlier statement in C8 for the cold fixed suite ("53–63×") was a 3-round observation and is superseded by the 11-round
distribution above.
