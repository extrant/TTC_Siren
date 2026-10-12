# Phase 6 — profiler-driven single-thread optimisation (2026-10-09/10)

Scope: `docs/migration/goals/phase6_goal.md` step C. Single thread only: no `unsafe`, no SIMD, no multithreading, no root parallelism, no change
of any search or rule semantics. Every kept change is its own local commit (not pushed). C3 was reverted **twice** and re-applied twice after re-measurements (`6b61630`; revert `17c07c7` / re-apply `290b15e`; revert `7878d1f` / re-apply `a4d86ad`; see §5). The net code of C3 is `6b61630`.

**All timings in this report are pinned to the P-core set `0xFFF`** (i7-13650HX, logical processors 0–11; both Python and C#),
measured with `tools/bench/run_cs_pinned.py` (C# only, independent process runs, min/median/max) and
`tools/bench/run_interleaved.py --affinity auto` (Python and C# alternating). The mask is read back from the OS by every child and checked.
Python is CPython 3.11.4. Raw data: `docs/migration/reports/phase6/` (JSON with all per-process values, profile summaries, the candidate log).

**How to read the numbers.** This machine drifts by up to ~6% over tens of minutes (the stored baseline of the start of the session
measured `open_none` make/undo at 0.2439 s, the same code an hour later 0.259 s). So every keep/revert decision compares a candidate with a
**baseline build measured in the same time window** (a copy of the previous commit's `Bench` folder, `--exe`), never with an older file.
The shipped comparison (original commit vs final commit) was also taken in one window with the original built in a separate worktree.

## 1. Baseline (commit `a0eb193` + tooling, no optimisation) — pinned `0xFFF`

Warm C# (whole item set warmed 1.5 s), 3 process runs, median of the per-process medians (min–max in the JSON):

| mode | metric | median s |
|---|---|---|
| `search` | 22 fixed-depth items, 43,003 nodes | 0.02338 |
| `fast` | compat TT-on / TT-off (oracle) / Fast, d≥4 totals | 2.3056 / 2.1934 / 1.4378 |
| `exact` | `end5_same_plus` / `mid4_rev_ace` / `mid3_ascension` | 0.000013 / 0.000194 / 0.06447 |
| `endgame` | robust pipeline, 72 positions | 0.05238 |
| `makeundo` (2M pairs) | open_none / open_same_plus / mid3_ascension / mid4_rev_ace / end5_same_plus / open_all_rules | 0.2439 / 0.3176 / 0.7239 / 0.1107 / 0.1389 / 1.0257 |

Cold (`--cold`, one untimed run per item), 5 process runs: `search` 0.0409 (0.0363–0.0425), `fast` 2.320 / 2.206 / 1.461, `exact mid3` 0.0652, `endgame` 0.0777.

Interleaved Python/C# (3 rounds, `0xFFF` both sides), warm: fixed **114.3 / 117.7 / 118.9×**, exact **166.7 / 170.5 / 170.9×**, robust pipeline
**180.9 / 181.9 / 186.3×**; cold: fixed 61.8 / 68.9 / 80.2×, exact 159.8 / 164.5 / 169.7×, robust 114.9 / 118.8 / 120.6× (files `baseline_interleaved_*.txt`).

## 2. Profile before (`dotnet-trace`, sampled thread time, 12 s per workload, pinned, warm; `Bench loop fast|compat|exact`)

Inclusive shares (AggressiveInlining methods are attributed to their callers):

| workload | main inclusive costs |
|---|---|
| Fast (`open_none` d5, `open_all_rules` d5, `mid3_ascension` d6, `mid4_rev_ace` d6) | `MakeMove` 44.0% (`RecalculateTypeModifiers` 29.4%, `ApplyTypeModifier` 9.3%, `ResolveFlips` 11.1%); `EvaluateState` 30.6% (`PlacementEdgeScore` 21.7%); `OrderMoves` 12.7% (`EvaluateMove` 10.4%) |
| ReferenceCompat (§3.4 d3–d5 items) | `OrderMoves` 56.6% (`EvaluateMove` 44.7%, exclusive 41.5%); `PlacementEdgeScore` 27.4%; `EvaluateState` 18.3%; `MakeMove` 8.6%; Python TT key 4.5% + `TryGet` 5.0% |
| Exact solver (`mid3_ascension`) | `MakeMove` 71.5% (`RecalculateTypeModifiers` 58.6%, `ApplyTypeModifier` 20.9%, `ResolveFlips` 7.5%); `Node` exclusive 16% |

This reproduces the review's evidence (§8 of `OPUS_PHASE5_BENCHMARK_REVIEW.md`). The three hotspots attempted, in the order of the evidence:
(H1) same-type modifier maintenance, (H2) `PlacementEdgeScore`/`EvaluateState`, (H3) `EvaluateMove`/ordering; plus the small items.

## 3. Candidates

Keep rule: a repeatable improvement of at least 3% on the targeted workload and no regression above 3% elsewhere; otherwise revert and record.
"base" is always a build of the previous kept commit measured in the same window (5–9 process runs, medians).

| # | change | result | decision |
|---|---|---|---|
| **C1** `3a77dbe` | `RecalculateTypeModifiers`: one pass over the instance table instead of a board count plus, per present type, a board scan and two hand scans (every instance is on the board or in a hand, so the same cards are touched once each; order is irrelevant: XOR hash, per-card display refresh; all present types are still re-applied on every typed placement, D-017, and display edges still refresh and are not restored by undo, D-003) | warm: exact `mid3_ascension` 0.0645 → 0.0431 s (**−33%**), Fast 1.438 → 1.298 s (−9.7%), robust pipeline −6.3%, make/undo with type rules **−43..−48%**; cold: exact mid3 −37%, Fast −12%; non-type scenarios ±4% (same code path, noise) | **kept** |
| C2 | snapshot/restore of the 19 modifiers with `Span.CopyTo` | vs a stored file: mid3 exact +4.9%, but vs a fresh base in the same window −8.9%, Fast −1.7%/+1.0%, search −0.7%: the sign flips with the baseline run | reverted (no repeatable 3%) |
| **C3** `6b61630` / re-applied `290b15e` | table-driven `PlacementEdgeScore`: terms per effective value 1..10 for exposed/hidden edges and the exposed sides per cell are tables, replacing a `switch` and compare chains with hard-to-predict branches. All terms are multiples of 0.5, so the sums are exact in any order: bit-identical (1.6M `evaluate_move` + 400k `evaluate_state` golden checks) | warm: compat suite 0.02395 → 0.01909 s (**−20%**), Fast 1.328 → 1.073 s (**−19%**), compat TT-on/off −14%/−16%; exact/robust/make-undo unchanged (±3%). Cold, alternating-build sampling (§5): 22-item suite 0.0445 → 0.0412 s (−7.5%), Fast −28%, no cold regression | **kept** |
| **C4a** `2c226bb` | `EvaluateMove` walks the board's precomputed neighbour table (new `GameState.NeighborCount/Cell/MySide/OppSide`) instead of computing offsets and bounds in two loops; the accumulation order is unchanged | warm: compat suite −6.8% / −5.8% (two runs), TT-on/off −3.3..−4.0%, Fast −2.4..−2.8%; others unchanged | **kept** |
| C5a | `EvaluateState`: per-cell weight table and `PlacementEdgeScoreAt(inst, cell)` (no row/col arithmetic) | compat +4.0%, TT-on −1.0%, TT-off −0.3%, Fast +0.4% | reverted (no gain) |
| C6 | `stackalloc double[n]` instead of `[MaxMoves]` for the order scores (less zeroing per node; the no-`unsafe` substitute for `SkipLocalsInit`) | compat −0.7%, TT-on −2.6%, TT-off −1.4%, Fast −1.7% | reverted (< 3%) |
| C8a | `ApplyTypeModifier` skips the Zobrist XOR and store when the modifier is unchanged | Fast +1.4..+1.7%, exact mid3 +1.1%, make/undo ±2% | reverted (no gain) |
| C9 | `ApplyTypeModifier` refreshes the display edges through one slice per array instead of eight indexed accesses | exact mid3 −3.7%, −2.4%, −2.8% (median −2.8%), Fast −0.6..−1.0%, make/undo −0.2..−2.7% | reverted (median below 3%) |
| **C7** `782a3b5` | `EndgameRobust.Select` reuses one exact-solver memo per thread (it allocated and zeroed a ~768 KiB table per call, review L5-2). Every scenario still starts its own generation (`Bind`), so nothing can leak between scenarios, sides or calls | warm: robust pipeline 0.0508 → 0.0401 s (**−21%**) and 0.0506 → 0.0393 s (**−22%**); cold −20% in that run | **kept** |

Not attempted, with the reason: **`[SkipLocalsInit]`** (O-1) needs `AllowUnsafeBlocks`; the nearest no-unsafe variant (C6, zeroing less) gave under 3%, so there is no
evidence it would pay. **Per-node clock read in ReferenceCompat** (O-2): fixed-depth and node-budget runs (all parity and benchmark workloads) never read the clock
(`IsTimeUp` reads it only when a time limit is set), so there is nothing to measure; the wall-clock `server` mode is not parity-testable and was left alone.
Incremental leaf evaluation or an `(inst, cell, modifier)` score cache for `EvaluateState` would be a design change with an invalidation problem (D-003/D-017
make modifiers and display edges history-dependent); it is recorded as a remaining hotspot, not attempted.

The remaining part of the scope rule ("until the top three hotspots have each been attempted, or no candidate shows 3% in a quick pinned measurement") is met: H1 (C1 kept, C2/C8a/C9 reverted),
H2 (C3 kept, C5a reverted), H3 (C4a kept, C6 reverted), plus L5-2 (C7 kept).

## 4. Result: original commit vs final commit, same window, pinned `0xFFF`

7 process runs each, medians (original = `48e38cd`, tooling only, built in a separate worktree; final = `782a3b5`). `docs/migration/reports/phase6/ab_*.json` hold every run.

Warm:

| workload | original | final | change |
|---|---|---|---|
| ReferenceCompat, 22 fixed-depth items | 0.02439 s | 0.01839 s | **−24.6%** |
| Fast, d≥4 totals (compat TT-on / TT-off / Fast) | 2.3004 / 2.1804 / 1.4431 s | 1.7768 / 1.6585 / 1.0257 s | −22.8% / −23.9% / **−28.9%** |
| Exact solver `mid3_ascension` | 0.06755 s | 0.04191 s | **−38.0%** |
| Exact `mid4_rev_ace` / `end5_same_plus` (timer-limited, 14–200 µs) | 0.000207 / 0.000016 s | 0.000205 / 0.000014 s | −0.7% / −14% (noise) |
| Robust pipeline, 72 positions | 0.05356 s | 0.03859 s | **−28.0%** |
| Make/undo, type rules (mid3_ascension / open_all_rules) | 0.7473 / 1.0503 s | 0.3745 / 0.5853 s | −49.9% / −44.3% |
| Make/undo, no type rules (open_none / open_same_plus / mid4_rev_ace / end5_same_plus) | 0.2555 / 0.3293 / 0.1210 / 0.1349 s | 0.2530 / 0.3247 / 0.1193 / 0.1349 s | −1.0 / −1.4 / −1.4 / 0.0% (unchanged) |

Cold (`--cold`):

| workload | original | final | change |
|---|---|---|---|
| ReferenceCompat 22 items (total) | 0.03621 s | 0.04511 s | +24.6% in this **block-order** run — an ordering artifact, see §5; with alternating builds (25 samples) 0.04450 → 0.04116 s, **−7.5%** |
| Fast d≥4 totals | 2.315 / 2.212 / 1.441 s | 1.820 / 1.696 / 1.054 s | −21.4% / −23.3% / −26.8% |
| Exact `mid3_ascension` | 0.06821 s | 0.04503 s | −34.0% |
| Robust pipeline | 0.07927 s | 0.07851 s | −1.0% in this block-order run; alternating builds (15 samples): 0.08187 → 0.06325 s (**−22.7%**) |

Cold, **alternating builds** (`tools/bench/ab_interleaved.py`, original / final, `docs/migration/reports/phase6/ab_interleaved_cold_*.txt`): ReferenceCompat 22 items median 0.04450 → 0.04116 s (−7.5%, 25 samples); Fast 1.4006 → 1.0087 s (−28%, 7); exact `mid3_ascension` 0.0640 → 0.0421 s (−34%, 7); robust pipeline 0.08187 → 0.06325 s (−23%, 15).

**Python vs C#, interleaved, 3 rounds, `0xFFF` on both sides** (CPython 3.11.4), min / median / max. The "final" columns are a fresh `run_interleaved.py --rounds 3 --affinity auto [--cold]`
run on the **final commit `427d67f`** (`final_427d67f_interleaved_*.txt`; every child reported `0xFFF`, 72/72 selections identical in every round, node counts and exact values asserted by the runner).
An earlier run taken before C3 was reverted and re-applied is superseded and is not quoted.

| suite | warm: baseline | warm: final `427d67f` | cold: baseline | cold: final `427d67f` |
|---|---|---|---|---|
| fixed-depth search (same nodes) | 114.3 / 117.7 / 118.9× | **156.0 / 162.4 / 165.7×** | 61.8 / 68.9 / 80.2× | 62.9 / 63.4 / 66.6× |
| exact solver (3 positions) | 166.7 / 170.5 / 170.9× | **249.4 / 255.2 / 263.4×** | 159.8 / 164.5 / 169.7× | 241.8 / 256.6 / 259.1× |
| robust pipeline (72 positions) | 180.9 / 181.9 / 186.3× | **235.9 / 239.7 / 254.8×** | 114.9 / 118.8 / 120.6× | 140.9 / 142.7 / 150.3× |

The cold fixed-suite ratio does not improve in this protocol: the 22-item cold suite is 40 ms of mostly tier-0 code, its C# total is bimodal (62.9–80.2× over the baseline runs, 62.9–66.6× now; the
ranges overlap), and the like-for-like alternating-build measurement of §5 shows the C# time itself 7.5% lower. Python's own timings drift a few percent between sessions (fixed 2.80 → 2.72 s here), which is
why the ratios of one session are only comparable with each other.

Of the exact-solver ratio, 1.235× is still the algorithmic memo-key gain (297,935 vs 241,264 `make_move`), and the robust pipeline still does 1.074× less work. The Python timings
of these runs are the same as in the baseline (exact `mid3_ascension` 10.8–11.4 s), so the change is on the C# side only.

## 5. Caveats and one judgement call

- **Cold 22-item suite and the keep decision for C3.**
  - *First measurement.* A block comparison (`run_cs_pinned.py --cold`, all baseline runs and then all candidate runs) showed C3 about 12% slower on the cold 22-item suite. I wrongly kept C3
    anyway and called it a deviation; the Stop-hook check pointed out that the rule is not mine to relax, so C3 was reverted (`17c07c7`) and re-measured.
  - *Goal protocol.* `run_cs_pinned.py --modes search --runs 7 --cold --affinity auto`, medians, run order alternated, four repetitions (`phase6/c3_cold_protocol.txt`): the candidate against its
    predecessor is −7.6%, +4.4%, +7.7%, +1.8% (median of the four medians +2.2%). The sign changes with the repetition; the predecessor's own median moves by 16% (0.0423–0.0489 s), the candidate's by 3%.
    A regression above 3% is therefore not repeatable, and the keep rule (a repeatable gain of at least 3% on the targeted workload, no regression above 3% elsewhere) is met:
    warm −20%, Fast cold −18..−28%, cold 22-item suite within noise. C3 was re-applied as `290b15e`.
  - *Reverting C3 does not restore the cold figure either.* The tree without C3 (`17c07c7`, 11 process runs, `run_cs_pinned.py --cold`, original build first) measured the cold 22-item suite at
    0.0500 s against 0.0405 s for the original (+23.5%), so the cold number of this micro-suite is not attributable to C3. A second strict-reading trial (revert `7878d1f`, four repetitions,
    `phase6/c3_cold_protocol.txt`) gave a median of medians of +4.6% for the tree without C3, always at the slow mode of the original; it was undone by `a4d86ad`.
  - *Deciding evidence (independent reviewer, `OPUS_PHASE6_REVIEW.md` §6; builds alternating sample by sample, pinned):* cold compat 22 items, 21 samples: original `48e38cd` **0.04793 s**,
    final **0.04021 s** (−16.1%), build without C3 (`17c07c7`) **0.04883 s**, so C3 itself is −17.6% cold. My own alternating measurement agrees in sign (`tools/bench/ab_interleaved.py`: original 0.0431 s,
    with C3 0.0423 s, without C3 0.0500 s). Removing C3 is what makes the cold suite slower.
  - The block-order figures of §4 are kept as measured and labelled.
- Cold figures are bimodal (JIT timing); read them as ranges and compare builds only with alternating samples (`ab_interleaved.py`). Block-order cold comparisons in this report (e.g. robust pipeline −1% in the first cold table) disagree with the alternating ones (−23%) for that reason.
- Short items (µs) are timer-limited; `exact end5_same_plus` / `mid4_rev_ace` changes of ±10% are noise.
- Everything is versus CPython 3.11.4 (Q-6); PyPy is unmeasured.

## 6. Invariants (acceptance)

- **Build** `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental`: 0 warnings, 0 errors. **Tests** `dotnet test`: 78/78 (0 failed, 0 skipped); no assertion weakened.
  **pytest**: 34 passed (29 + the 5 of `tests/test_duplicate_hand_ids.py`).
- **ReferenceCompat node-for-node identical to Python**: all golden differential tests (fixed depth 4,507 runs incl. the 22 benchmark items, ID, evaluation 1.6M `evaluate_move` + 400k `evaluate_state`) pass on the
  regenerated golden data; additionally the `SIG` lines of `Bench search` (node count, score bits, move per item) are **identical** between the original and the final build (23 lines, hash `8b2fbbb55400a735`).
- **Fast**: root scores bit-identical to the oracle (`Bench fast`: 40/40 `same`, 0 `DIFF`); node counts, scores and moves of all three variants are **identical** to the original build (41 `SIG` lines, hash `457de1e07cd244b8`),
  so no tie-set analysis is needed.
- **Exact solver**: values 1 / −2 / −1 and `make_move` counts 217 / 4,707 / 241,264 identical (`a14d89aa07fe733c`). **Endgame**: 72/72 selections identical (`4cf59289094219cd`); robust parity tests (8 differential tests) pass.
- **Make/undo checksums** of the six playout scenarios identical (`575d814747eec483`).
- **Same-type semantics preserved**: D-017 (every present type re-applied on every typed placement, scenario cards catch up) and D-003 (display edges refreshed each time, not restored by undo) are exercised by the
  golden rule traces (20,000 sequences; tens of thousands of ascension/descension recalculations and of undos that restore changed modifiers, counted in the test output), the Zobrist sufficiency tests and the incremental-hash tests; all pass.
- **Zero allocation** in the hot paths: the existing `AllocationTests`, `SearchAllocationTests`, `ExactEndgameSolverTests` (8,684,790 solver nodes, 0 bytes) pass; `Bench` prints `alloc 0` for every search item.
- No `unsafe`, SIMD or threads; no change of search/rule semantics.

## 7. Profile after (same workloads)

Fast: `MakeMove` 38.4% (`RecalculateTypeModifiers` 17.9%), `EvaluateState` 29.4% (exclusive 29.1%: the inlined `PlacementEdgeScore` table lookups), `ResolveFlips` 15.7%, `EvaluateMove` 10.4%.
ReferenceCompat: `OrderMoves` 54.8% (`EvaluateMove` 39.0%, exclusive 38.9%), `EvaluateState` 15.5%, `MakeMove` 10.2%, Python TT key 5.9% + `TryGet` 5.3%.
Exact solver: `MakeMove` 62.8% (`RecalculateTypeModifiers` 41.9%, `ApplyTypeModifier` 23.1%), `Node` exclusive 21.3%, `ResolveFlips` 12.1%.

## 8. Remaining hotspots

1. Same-type maintenance is still 42% of the exact solver and 18% of Fast: `ApplyTypeModifier` is called for every instance of every present type on every typed placement (Python semantics); a delta/undo-log
   scheme must reproduce the D-017 catch-up and the history-dependent display edges exactly. C2, C8a and C9 show that micro-changes do not pay; a structural change is needed.
2. `EvaluateState` (29% of Fast): nine `PlacementEdgeScore` lookups per leaf; an incremental or cached evaluation needs an invalidation scheme for modifier changes.
3. `EvaluateMove` (39% of ReferenceCompat): the remaining cost is arithmetic that must stay in Python's order; relevant only if compat stays on the production path (D-015 → Phase 7 uses Fast).
4. `ResolveFlips` (11–16%).
