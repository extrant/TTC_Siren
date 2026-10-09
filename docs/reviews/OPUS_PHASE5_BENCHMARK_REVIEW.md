# OPUS Phase 5 Benchmark Review — are the Python-vs-C# numbers comparable?

Reviewer: independent benchmark-validity and correctness auditor (Opus). Review only: no optimisation, no Phase 6 work.
Date: 2026-10-08. Severity scale as in the earlier Opus reviews.

## 1. Scope and inputs

- **Documents:** `CLAUDE.md`; `CSHARP_MIGRATION.md` (§3 baseline, §7 Phase 5, §10 benchmark history); `docs/reports/PHASE5_BENCHMARK.md` (the report under audit); the previous reviews `OPUS_PHASE3_REVIEW.md` and `OPUS_PHASE4_REVIEW.md`.
- **Python harnesses:** `tools/bench/py_bench.py` (fixed-depth suite of §3.4 and the server-equivalent call) and `tools/bench/py_endgame_bench.py` (robust endgame pipeline and exact solver).
- **C# harness:** `csharp/src/TTC.Siren.Bench/Program.cs`, modes `search`, `fast`, `exact`, `endgame` and `server`. It is a hand-written `Stopwatch` harness. **BenchmarkDotNet is not used.**
- **Code read for semantics:** `ExactEndgameSolver.cs` and `EndgameRobust.Select`, to see how the memo is created, shared and counted.

## 2. Environment and build hygiene (verified)

| Item | Finding |
|---|---|
| Python | CPython 3.11.4, 64-bit, AMD64, release interpreter (no `gettotalrefcount`), `-O` off. **No PyPy** (Q-6 default). |
| .NET | SDK 10.0.401, RID `win-x64`, process x64; workstation GC, as printed by the Bench header. |
| Build | All C# numbers come from `-c Release` (`dotnet run -c Release --project …Bench`), and the test suite also runs in Release. No Debug build is mixed in. `Directory.Build.props` and the csproj files contain no `Optimize`, `TieredCompilation` or GC overrides. |
| Debugger | Runs go through `dotnet run`/`dotnet test` from a shell, with no debugger attached. An untracked `csharp/.vs/` folder shows Visual Studio was opened at some point, but nothing indicates measurements were taken under VS (L5-6). |
| JIT | Default tiered compilation plus tiered PGO. **This matters a lot here; see A-1.** |
| CPU | Same machine for both sides (i7-13650HX). Frequency scaling and power plan cannot be controlled from this session. The reviewer interleaved Python and C# runs to share load conditions. |

## 3. Correctness under benchmark conditions (no regression)

- `dotnet build -c Release --no-incremental`: 0 warnings, 0 errors.
- `dotnet test`: **78/78 passed**.
- `pytest`: 29 passed. On the Python side only the two parity-generator scripts differ from HEAD; no Python business code changed.
- **Fixed-depth suite (22 items):** in three interleaved rounds, every item has **identical node counts and bit-identical root scores** in Python and C# ReferenceCompat (checked item by item by the reviewer).
- **Robust endgame pipeline (72 positions × 2 modes):** C# selected **the same move as Python in 72/72 positions**, with the same scenario counts, on freshly regenerated Python data. The positions are byte-identical to the committed `endgame_bench_positions.json`.
- **Exact solver:** values are identical (1, −2, −1).
- **Server-equivalent call:** the chosen move is identical in all implementations (`27@(2,2)`, `27@(2,2)`, `53@(1,2)`).

**There is no correctness regression.**

## 4. Equivalence of inputs and semantics

| Dimension | Fixed-depth suite | Robust pipeline | Exact solver | Server-equivalent |
|---|---|---|---|---|
| Starting state, cards, owners, rules, side to move | Same 5 positions (§3.3), built by the same recipe (`py_bench.state` versus `ReferenceScenarios`); confirmed by identical nodes and score bits | Python builds positions and exports them to JSON; C# loads exactly that state (`StateSnapshot`), including hand order, unknown-slot flags and sample count | Same §3.3 positions | Same |
| Depth / terminal condition | Same fixed depth; heuristic evaluation at the end of the game (D-007) on both sides | Exact solve to the end of the game for every (move, scenario) pair (occupancy ≥ 5) | Full solve | Wall clock, 10 s, `max_depth` 10 |
| Time policy | None (fixed depth, unlimited time) | None | None | **Wall clock: depth reached depends on speed (not a like-for-like work comparison)** |
| Move ordering / TT | ReferenceCompat reproduces Python's ordering, history and TT (D-005/D-006) node for node | n/a (exact solver, no ordering) | n/a | Compat equals Python semantics; **Fast differs by design** (bound TT, early ID stop D-014) |
| Unknown cards | none | deterministic candidates and scenarios only; positions with no legal candidate are excluded on both sides (Python's random fallback is not ported, A-7) | none | none |
| Evaluation / endgame semantics | `evaluate_state`, bit-identical | integer card difference, identical | identical | compat identical; Fast exact minimax (it can differ from Python's D-006 values) |

**Node-counter semantics.**
- Python `nodes_searched` and C# ReferenceCompat `NodesSearched` count the same thing: every recursive call, including the root, leaves and calls that end in a TT cutoff. Proven by the node-for-node parity.
- Fast counts calls the same way, but it explores a different tree (bound TT, stored move first, no sorting at depth 1), so **Fast node counts and NPS are not comparable** with Python or compat. Only time to the same score is.
- In the exact solver, Python counts `make_move` calls through a wrapper and C# counts `MakeMove` inside the solver: the same meaning.
- The `RobustSelection.ExactMakeMoves` of the robust pipeline excludes each scenario's first move. The reviewer added it back when comparing work (§6).

**Timed region.** Both sides time solver work only (search or solve, or scenario build plus selection). State construction and JSON parsing are outside the timed region on both sides. Two small asymmetries:
1. Python's first fixed-depth item includes the lazy CSV load of `get_card_star_map()`. That is under 1% of the suite.
2. The Python exact harness times a `make_move` counting wrapper. The reviewer measured it: 10.09–10.28 s without the wrapper versus 10.22 s with it, i.e. noise.

## 5. Benchmark artifacts found

### A-1 (MEDIUM) — C# timings are dominated by JIT warm-up on short items; the "49.7×" headline is a cold-start figure

`Bench search` runs one warm-up per item and then 5 timed runs. With tiered compilation, the hot methods are still tier-0 or OSR code during the first items of the process. Evidence:

| Run mode | C# suite total (22 items) | `open_none` d4 | `open_same_plus` d4 |
|---|---:|---:|---:|
| default, 5 runs per item (as in the report) | 0.0521–0.0541 s | 0.0322–0.0325 s | 0.0072–0.0073 s |
| `DOTNET_TieredCompilation=0` | 0.0247–0.0254 s | 0.0097–0.0100 s | 0.0082–0.0085 s |
| default, **50** runs per item (steady state) | **0.0240–0.0252 s** | 0.0086–0.0087 s | 0.0072–0.0073 s |

`open_none` d4 (15.8k nodes) runs at 0.49M NPS in the report's configuration and 1.8M NPS when warm. `open_same_plus` d4, measured later in the same process, is already warm. Per-item ratios in the report's configuration therefore range from 19× to 243× depending only on the **position in the run order**.

- The suite ratio is about **50× cold versus about 110× warm** (Python 2.70 s, stable at ±0.4% over 3 rounds).
- The same effect appears in `Bench endgame`: the 114× total with the default JIT becomes **170×** with `TieredCompilation=0`. The occupancy-5 cells run first, which is why their ratios are 90× versus 226× for nearly equal work.
- In `Bench exact`, `mid4_rev_ace` is 0.00090 s with the default JIT and 0.00021 s with `TieredCompilation=0`.
- Fully optimised JIT is *not* the true steady state either: tier-1 code with PGO was about 8% faster than `TieredCompilation=0` on long runs (`Bench fast` totals 1.38 s versus 1.50–1.53 s).

The artifact makes C# look slower, so the report's ratios are conservative lower bounds for a warm resident engine (Phase 7) and roughly realistic for a cold per-request process. **Both should be reported, labelled.**

### A-2 (MEDIUM) — the exact-solver "≈ 263×" rests on an outlier Python timing

The report gives Python 15.58 s for the `mid3_ascension` exact solve. The reviewer re-measured it four times, interleaved with C#: **9.95, 10.07, 10.21 and 10.09 s** (median of 3 inside each). C# measured 0.0597–0.0661 s. The supported ratio is **about 160–170×**. The 263× figure is not reproducible and should be withdrawn. Its decomposition:
- C# does 1.235× less work (241,264 versus 297,935 make_move calls). Its Zobrist memo ignores hand order, while Python's `_endgame_state_key` includes it. That is an **algorithmic** gain from the memo key, not a runtime gain.
- Per make_move: Python about 33.8 µs, C# about 0.25 µs, so **about 135× runtime/data layout**.

### A-3 (LOW) — cross-batch ratios

The report's Python and C# columns were measured in separate batches, at different times. Machine load varies by up to about 40% between batches on this machine (documented in §10), which is how A-2 happened. The reviewer's interleaved re-runs agree with the report within about 5% on the fixed suite (Python 2.70 s versus 2.60 s reported) and the pipeline (9.11 s versus 8.71 s), but not for the exact solver.

### A-4 (LOW) — display precision

`Bench search` prints per-item times with `F4`, so items under 0.1 ms print as `0.0000`. Totals are computed from full-precision medians, so the suite total is correct, but per-item NPS cannot be read from the output.

### A-5 (LOW) — server-equivalent table is not a speedup measurement

That table compares *depth reached in 10 s*. Fast's "depth 9 in 0.94 s" for `open_none` is a full solve, enabled by the bound-aware TT and the early stop once depth equals the empty cells (D-014). It mixes algorithmic and runtime effects. The report does not compute a ratio from it, which is correct. Read it only as "compat runs at about 1.65–1.79M NPS versus Python's 17–18k NPS", which is about 95–100×, consistent with A-1's warm figure.

### A-6 (LOW) — measurement order inside `Bench fast`

Each row measures TT-on, then TT-off, then Fast, so later columns benefit from more warm-up on the first items. The totals are dominated by later, larger items, and the reviewer measured the Fast/TT-off ratio at 1.49–1.52× in both JIT modes, so the 1.5× claim holds.

## 6. Speed-up decomposition (evidence-based)

| Component | Evidence | Factor |
|---|---|---|
| **Language/runtime + data layout** (same algorithm, same nodes) | Fixed suite, identical node counts: 50× cold, 110× warm. Server-equivalent NPS: 95–100×. Exact solver per make_move: 135×. Robust pipeline at an equal-work ratio of 1.07: 107× (default JIT) to 160× (warm) | **≈ 95–135× warm; ≈ 45–55× cold** |
| **Algorithmic: memo key** (exact solver) | 297,935 / 241,264 make_moves | 1.24× (`mid3_ascension`); 1.00× on small positions |
| **Algorithmic: memo sharing pattern** (robust pipeline; C# shares per scenario across moves, Python per move across scenarios) | Reviewer's make_move count over the 72 positions: Python 209,495 versus C# 194,995 (1.07×; only the occupancy-5 cells differ, 1.07–1.14×) | ≈ 1.07× |
| **TT / move ordering** (Fast versus the oracle, compat TT-off, same exact score) | `Bench fast` d ≥ 4 totals, both JIT modes | 1.49–1.52× |
| **Early ID termination + bound-aware TT** (Fast versus Python at equal wall time) | Full-depth solves become feasible (depth 9 in under 1 s versus depth 5 in 10 s) | not expressible as a ratio; algorithmic |
| **Benchmark artifacts** | A-1 (warm-up, understates C# by about 2×), A-2 (Python outlier, overstates by about 1.6×) | — |

## 7. Representative positions (reviewer measurements, interleaved, 2026-10-08)

| Position | Python (CPython 3.11.4) | C# ReferenceCompat | C# Fast | Score / move |
|---|---|---|---|---|
| Opening `open_none` d4 | 15,807 nodes, 1.034 s, 15.3k NPS | 15,807 nodes; 0.0323 s cold (32×), **0.0086 s warm (120×)**, 1.8M NPS | 20,497 nodes, 0.0043 s | −2.44 (bits `C003851EB851EB85`) / 27@(2,2): all identical |
| Midgame `mid3_ascension` d5 (同类强化) | 2,404 nodes, 0.199 s, 12.1k NPS | 2,404 nodes, 0.0019 s (≈ 105×) | 2,048 nodes, 0.0014 s | 11.61 / 53@(1,2): identical |
| Midgame `mid4_rev_ace` d5 | 587 nodes, 0.035 s | 587 nodes, ≈ 0.0002 s (timer-limited) | 582 nodes, ≈ 0.0001 s | 0.165 / 27@(2,2): identical |
| Endgame `end5_same_plus` d4 | 111 nodes, 0.0067 s | 111 nodes, < 0.1 ms | 111 nodes | 2.76 / 48@(1,0): identical |
| Endgame exact solve, `mid3_ascension` (6 empty cells) | 297,935 makes, **10.1 s** | — | exact solver: 241,264 makes, 0.061 s | value −1 identical; **≈ 165× total, 135× per make** |
| Robust pipeline, 72 positions | 9.11 s | — | 0.0793 s default JIT / 0.0535 s warm | 72/72 identical; **114–170×** |
| Server-equivalent `open_none`, 10 s | depth 5, 183,141 nodes | depth 10, 15.94M nodes, 8.89 s | depth 9 (full tree), 2.57M nodes, 0.94 s | 27@(2,2) all |

"Completed depth" is only meaningful in the server-equivalent row. The other rows are fixed-depth.

## 8. Where the C# time goes (Phase 6 evidence)

The profile is `dotnet-trace` sampled thread time (about 100 Hz, 12 s per workload, warm process, default JIT). Methods marked `AggressiveInlining` are attributed to their callers. Inclusive shares:

| Workload | Main inclusive costs |
|---|---|
| **Fast** (`open_none` d5, `open_all_rules` d5, `mid3_ascension` d6, `mid4_rev_ace` d6) | `MakeMove` 43%, of which `RecalculateTypeModifiers` is 29% and `ApplyTypeModifier` 10%; `EvaluateState` 31%; `PlacementEdgeScore` 21% (shared by state and move evaluation); `OrderMoves` 14.5% (`EvaluateMove` 12%); `ResolveFlips` 11%; `UndoMove` 2.4% |
| **ReferenceCompat** (the §3.4 d4–d5 items) | `OrderMoves` 58.5% (`EvaluateMove` 45%); `PlacementEdgeScore` 25%; `EvaluateState` 17%; `MakeMove` 9%; Python TT key (`PythonStateKey.Compute`) 4.6% plus `TryGet` 4.5% |
| **Exact solver** (`mid3_ascension` full solve) | `MakeMove` 70%, of which `RecalculateTypeModifiers` is 57% and `ApplyTypeModifier` 21% |

The share of `RecalculateTypeModifiers` depends on the workload (同类 positions only). The figures above are for these workloads, not universal.

## 9. Conclusion

# **BENCHMARK_VALID_WITH_CAVEATS**

Inputs, depths, rules, evaluation and endgame semantics are equivalent, and the results (scores, moves and node counts where the algorithm is the same) are identical. Node counters have equal meaning where they are compared. The numbers are therefore comparable, with three caveats:
- **A-1:** the C# figures are cold-JIT figures and understate a warm engine by about 2× on short workloads.
- **A-2:** the 263× exact-solver ratio is not reproducible; about 165× is.
- **Baseline:** everything is versus CPython 3.11.4. Production may run PyPy (Q-6), which was not measured and would lower every ratio by an unknown factor.

### Trusted speed-up range (versus CPython 3.11.4, single thread)

- **Python-semantics search (same nodes):** **45–55×** in a cold process; **95–120×** warm.
- **Robust endgame pipeline:** **110–170×** (default to warm JIT); about 1.07× of it is algorithmic.
- **Exact solver (large endgame):** **130–170×**; about 1.24× of it is the memo key.
- **Fast versus Python at a fixed depth to the same exact score:** roughly the compat range × 1.5 (**≈ 70–180×**), but part of that is algorithmic, and Fast's values follow exact minimax rather than Python's D-006 TT.
- **Single headline, if one is needed:** "**≈ 100× (50–170× depending on workload and JIT warm-up) versus CPython; unmeasured versus PyPy**". Do not quote 263× or 49.7× without these qualifiers.

### Current actual bottleneck (evidence §8)

- **In the Fast and exact paths:** `MakeMove` under 同类 rules. `RecalculateTypeModifiers` rescans the board and both hands and calls `ApplyTypeModifier` (which refreshes the display edges) on every typed placement, plus a 19-entry snapshot. That is 29% of Fast time and 57% of exact-solver time on these workloads.
- **Second:** leaf and move evaluation through `PlacementEdgeScore` (21–25%) and `EvaluateState` (17–31%).
- **In Python-compatible search:** move ordering (`EvaluateMove`, 45%) dominates.

### Recommended Phase 6 profiling targets

1. **同类 modifier maintenance in `MakeMove`/`UndoMove`.** Profile-driven candidates: a delta undo log instead of the full 19-entry snapshot; updating only changed instances; deferring display-edge refresh. **Semantic hazard:** Python recalculates *all* present types on *any* typed placement, and scenario cards start with modifier 0 without recalculation (D-017). An incremental scheme must reproduce the first-recalculation catch-up exactly; the golden data covers this.
2. **`PlacementEdgeScore` / `EvaluateState`.** For example a per-(instance, cell, modifier) score table, or incremental leaf evaluation. Results must stay bit-identical, and the 0.5-multiple argument for summation order only covers this term.
3. **`EvaluateMove`/ordering cost.** This matters only if ReferenceCompat stays on the production path (D-015 suggests Phase 7 may need compat for move-level parity with Python). It is a measured hot spot.
4. **Benchmark infrastructure first** (see the remediation goal): Phase 6's Baseline → Compare process (CLAUDE.md §6) is not trustworthy while short-item timings swing by 3× with run order.

## 10. Other findings

| ID | Severity | Finding |
|---|---|---|
| L5-1 | LOW | The report's Python fixed-suite figure (§3 of the report, 2.603 s) and the reviewer's 2.70 s agree within noise; fine. The report should state that both Python and C# per-item timings are medians of independent runs, and that C# items under 0.1 ms are timer-limited. |
| L5-2 | LOW | `Bench endgame` does not report allocations. `EndgameRobust.Select` allocates a new 32 K-entry memo (≈ 768 KiB) per call; the cost is small in these measurements but should be visible in Phase 6 baselines. |
| L5-3 | LOW | There is no Python-side repeated-measurement statistic beyond 3 medians. Python's spread was small (±0.4%) except for the A-2 outlier, which a min/median/max printout would have exposed. |
| L5-6 | LOW | Untracked `csharp/.vs/` (Visual Studio state) is in the working tree; add it to `csharp/.gitignore`. |

## 11. Recommended next action

Run `.claude/sonnet_review_fix_goal.md`. It covers benchmark-harness and report corrections only (A-1, A-2, A-4 and the report text), and no solver optimisation. Then start Phase 6 with a trustworthy baseline. Phase 6 still needs the user's explicit authorization.

---

## 12. Remediation Verification (2026-10-08)

This verifies the B-1 to B-5 remediation of `.claude/sonnet_review_fix_goal.md`. Sonnet's summary was not taken as evidence; every item below was re-run by the reviewer.

### 12.1 Files inspected

Phase 5 is still uncommitted, so the remediation delta was isolated by modification time against this review (09:13). The files changed afterwards are:
- `csharp/src/TTC.Siren.Bench/Program.cs`
- `tools/bench/py_bench.py`, `tools/bench/py_endgame_bench.py`, and the new `tools/bench/run_interleaved.py` and `tools/bench/count_robust_work.py`
- `csharp/.gitignore`
- `docs/reports/PHASE5_BENCHMARK.md`
- `CSHARP_MIGRATION.md`

All are on the goal's §2 allow-list. **No file under `TTC.Siren.Core` or `TTC.Siren.Search`, no test, no Python business code and no parity generator was touched after the review, and no solver optimisation was introduced.** The harness was read in full: `Harness.WarmUp`, `TimeRuns` and `Timed`, the `Opts` options `--cold`/`--warmup-ms`, the rotation in `Fast`, and the `endgame-work` mode. The timed regions still cover only the search, solve or pipeline call.

### 12.2 Commands run and results

| # | Command | Result |
|---|---|---|
| R1 | `dotnet build -c Release --no-incremental`; `dotnet test`; `pytest` | 0 warnings, 0 errors; **78/78**; 29 passed |
| R2 | `Bench search 5`: warm ×3 and `--cold` ×3, separate processes | warm totals **0.02344 / 0.02362 / 0.02425 s** (within ±10% of the 0.024–0.025 s target); cold **0.04381 / 0.04760 / 0.05181 s** (≈ 0.05 s; the report's configuration). Warm `open_none` d4 runs at 1.65M NPS and `open_same_plus` d4 at 1.58M NPS (**4.7% apart**; cold was 3.7×). Tables are labelled warm/cold and show min/median/max in µs. No `0.0000` remains; `*` marks items under 50 µs. **All 22 items have the same node counts and score bits as the pre-remediation output from the morning of 2026-10-08, warm and cold.** Allocation is 0 everywhere |
| R3 | `Bench fast 3`, warm and `--cold` | Rotation code verified (`k = (j + r) % 3`). Totals: nodes 3,431,185 / 3,771,758 / 3,625,620; **40/40 `same`, 0 `DIFF`**. All 40 rows (nodes, score, moves) are identical to the pre-remediation output. Fast/TT-off is 1.51× warm and 1.52× cold |
| R4 | `Bench exact 5`, warm | values 1 / −2 / −1, make_moves 217 / 4,707 / 241,264, allocation 0 |
| R5 | `Bench endgame tools/bench/endgame_bench_positions.json 5` | **72/72 same** |
| R6 | `Bench endgame-work` and `count_robust_work.py` | Python **418,990** versus C# **389,990** make_move calls (1.074×). **The same 10,834 (move, scenario) pairs** on both sides. Python confirmed its positions are identical to the stored file. This matches the review's own probe (209,495 versus 194,995 per mode) |
| R7 | `run_interleaved.py --rounds 3` (all suites, warm) | It interleaves correctly and asserts node counts, values and 72/72 every round. **But the ratios were 124 / 271 / 301×** (fixed / exact / endgame), because Python ran **1.7× slower** than in every earlier measurement (`mid3_ascension` 18.0 s instead of about 10.3 s; fixed 3.1 s instead of 2.7 s) while C# was unchanged. See A-7 |
| R8 | Core-placement experiment (Windows process affinity; i7-13650HX = 6 P-cores on LP 0–11 plus 8 E-cores on LP 12–19) | Python `mid3_ascension` exact: **P-core 10.9–12.6 s, E-core 23.4–23.7 s**, unpinned 10.1 / 16.8 / 18.0 s on the same day. With the P-core set `FFF`: Python 11.0–12.3 s, C# 0.068–0.072 s. With the E-core set `FF000`: Python 17.0 s, C# 0.097 s. Pinning .NET to a *single* CPU distorts it by 2–4× (the runtime's JIT and GC threads share that CPU), so only multi-CPU sets are meaningful |
| R9 | Both sides pinned to the P-core set `FFF` | fixed: Python **2.86 / 2.91 s**, C# warm **0.0242 / 0.0241 s** (**118–121×**), C# cold 0.0395 / 0.0441 s (**65–73×**). Endgame: Python 9.48 s, C# warm 0.0523 s (**181×**), cold 0.0868 s (**109×**), 72/72. Exact `mid3_ascension`: **152–180×** |

### 12.3 Verdicts

| Item | Verdict | Basis |
|---|---|---|
| **B-1** warm/cold harness, min/median/max, rotation in `fast` | **RESOLVED** | R2, R3: every acceptance figure met; results unchanged |
| **B-2** precision, NPS per item, timer-limited marks | **RESOLVED** | R2 |
| **B-3** interleaved runner, Python min/median/max, exact re-measurement | **RESOLVED as specified**: it interleaves at least 3 rounds and prints per-round ratios and min/median/max; exact re-measured at 10.2–10.7 s, within the 20% bound | But see the new finding A-7: interleaving does not control core placement |
| **B-4** report corrections | **RESOLVED** | 263× withdrawn; 161–167× derived from interleaved data with the 131 × 1.235 decomposition; 49.7× labelled cold, warm 119× added; work comparison scripted and reproduced (R6); trusted ranges, CPython-only / PyPy unmeasured, and "Fast nodes/NPS not comparable" all stated; original tables kept; §10 rows are measurements. The corrected figures **agree with the reviewer's P-core-pinned re-measurement (R9)**. One remark: the report attributes the 15.58 s figure to a background golden generation. R8 shows that E-core placement alone reproduces 16.8–23.7 s, so that attribution is plausible but not the only explanation |
| **B-5** `.vs/` in `csharp/.gitignore` | **RESOLVED** | present |

### 12.4 New finding

**A-7 (MEDIUM): hybrid-CPU core placement is uncontrolled.** On this i7-13650HX, Windows places long single-thread runs on P- or E-cores non-deterministically. The same Python solve takes 10.1–18.0 s unpinned and 23.4 s on an E-core; C# is about 1.4–1.5× slower on E-cores. When Python lands on E-cores and C# on P-cores, the interleaved runner reports inflated ratios. R7 gave 271× and 301× where the P-core-pinned truth is about 165× and 181×. Interleaving alone, which removes load differences, does not remove this. It also matters for Phase 6: a C#-versus-C# Baseline→Compare can swing about 1.45× if one side lands on E-cores, which is larger than typical optimisation effects. **Fix:** run both sides with an explicit multi-CPU P-core affinity mask and record the mask in every table, for example via `start /affinity FFF` or `ProcessorAffinity` set from the runner. The machine's P-core set must be read or confirmed per machine, not hard-coded silently.

### 12.5 Final benchmark verdict

# **BENCHMARK_VALID_WITH_CAVEATS**

Correctness is unaffected. The corrected numbers in `PHASE5_BENCHMARK.md` are confirmed under controlled (P-core-pinned) conditions. Remaining caveats:
- A-7 (unpinned runs, including `run_interleaved.py`, are not reproducible on this hybrid CPU);
- CPython 3.11.4 only; PyPy is unmeasured.

### 12.6 Trusted speed-up ranges (versus CPython 3.11.4, single thread, both sides on P-cores)

| Workload | Cold C# process | Warm C# | Algorithmic share |
|---|---|---|---|
| Python-semantics search (same nodes) | **55–73×** | **118–121×** | none |
| Robust endgame pipeline | **≈ 109–112×** | **≈ 178–189×** | 1.074× |
| Exact solver, large endgame | **≈ 150–180×** | **≈ 150–180×** | 1.235× (memo key) |
| Fast over the oracle (same exact score) | 1.5× | 1.5× | algorithmic (different tree) |

The single headline remains: **"≈ 55–190× versus CPython 3.11.4, depending on workload and JIT warm-up; PyPy not measured."**

### 12.7 Phase 6

**Not yet.** Fix A-7 first; it is small. `.claude/sonnet_review_fix_goal.md` has been rewritten to contain only that. After it is verified, Phase 6 can start once the user authorizes it.

---

## 13. Remediation Verification — A-7 (CPU core placement), and regression check of B-1 to B-5 (2026-10-08)

The goal file executed by Sonnet was the A-7 goal of §12.7. The user's acceptance prompt still listed B-1 to B-5. Both were checked: A-7 as the new work, and B-1 to B-5 for regressions. Sonnet's summary was not taken as evidence.

### 13.1 Files inspected

Files modified after §12 (15:51), by modification time:
- `tools/bench/pin.py` (new)
- `tools/bench/run_interleaved.py`, `tools/bench/py_bench.py`, `tools/bench/py_endgame_bench.py`
- `csharp/src/TTC.Siren.Bench/Program.cs` (+3 lines: it reads back and prints `affinity=` and `cpus=`)
- `docs/reports/PHASE5_BENCHMARK.md` (new "A-7: CPU core placement" section)
- `CSHARP_MIGRATION.md`

All are on the A-7 goal's allow-list. No change to `TTC.Siren.Core`, `TTC.Siren.Search`, tests, Python business code or parity generators; no optimisation. Code reviewed:
- **`pin.py`:** the child is created `CREATE_SUSPENDED`, `SetProcessAffinityMask` is applied, then `NtResumeProcess`, so the mask is in force before the first instruction, the .NET runtime sizes itself from it (`cpus=12`), and children inherit it. The `GetLogicalProcessorInformationEx` parsing (`EfficiencyClass` at +9, `GroupCount` at +30, the first `GROUP_AFFINITY` at +32) is correct for x64. The P-core set is detected, not hard-coded, and multi-group systems are refused.
- **The runner:** it starts the built `TTC.Siren.Bench.exe` directly (not `dotnet run`), checks that every child reports the requested mask, prints the mask everywhere, and warns when unpinned on a hybrid CPU.

### 13.2 Commands run and results

| # | Check | Result |
|---|---|---|
| V1 | `dotnet build -c Release --no-incremental`; `dotnet test`; `pytest` | 0 warnings, 0 errors; **78/78**; 29 passed |
| V2 | `python tools/bench/pin.py` | class 1 = `0xFFF` (P-cores), class 0 = `0xFF000` (E-cores), hybrid = True; this matches the reviewer's own affinity experiment (§12 R8) |
| V3 | `run_interleaved.py --rounds 3 --affinity auto` (warm) | every child reported `0xFFF`. fixed: Python 2.63–2.68 s, C# 0.0227–0.0233 s, **112.8 / 116.0 / 117.5×**. exact: **166.5 / 168.1 / 169.1×** (`mid3_ascension` 166–169×; work ratio 1.235). endgame: **176.5 / 178.4 / 179.8×**, 72/72 every round. The runner's assertions (43,003 nodes, values 1/−2/−1, 72/72) passed |
| V4 | the same with `--cold` | fixed: Python 2.62–2.70 s, C# 0.0495–0.0522 s, **51.7 / 52.6 / 53.2×**. exact **164.8 / 166.9 / 168.0×**. endgame **109.6 / 112.0 / 112.8×**, 72/72 |
| V5 | `Bench fast 3`, `Bench search 5` (regression check of B-1/B-2) | all 40 `fast` rows identical to the pre-remediation output (nodes TT-on/off/Fast, score, moves), 40/40 `same`. All 22 `search` items have identical nodes, score bits and allocation 0 |

Sonnet's pinned figures (report A7.2) and the reviewer's agree:
- **Warm fixed:** 115.9–118.1× versus 112.8–117.5×.
- **Exact:** 161–176× versus 165–169×.
- **Warm endgame:** 169–175× versus 176–180×.
- **Cold endgame:** 109–123× versus 110–113×.

The cold fixed suite is **bimodal**. The C# cold total is either about 0.035 s or about 0.050 s, depending on when tier-1 code arrives. Sonnet observed 57.5–91.0× over 11 pinned rounds (median 70×). The reviewer's 3 rounds all fell in the slow mode (52–53×), consistent with the lower end. The report's "PROVISIONAL" range statement in A7.4 is therefore correct as a range. The reviewer endorses quoting the cold fixed-suite ratio only as a range, **≈ 52–91× (bimodal, cold JIT)**, never as a single number. The earlier "55–73×" of §12.6 was based on two runs and is superseded. Whether to keep or delete the "PROVISIONAL" sentence in the report is for the user to decide; it no longer blocks anything.

### 13.3 Verdicts

| Item | Verdict | Basis |
|---|---|---|
| **A-7** core placement | **RESOLVED** | the mechanism is correct (13.1); pinned results are reproducible and agree between two independent measurers (V3, V4); the E-core control run is documented (report A7.3: both sides 1.3–1.55× slower, so mixed placement is what inflates ratios); masks are printed and checked |
| B-1 warm/cold | **RESOLVED** (no regression) | V3–V5 |
| B-2 precision | **RESOLVED** (no regression) | V5 |
| B-3 interleaved runner | **RESOLVED**, now also core-controlled | V3, V4 |
| B-4 report | **RESOLVED** | A-7 section appended; earlier tables kept; trusted ratios now stated for both sides on the P-cores |
| B-5 `.gitignore` | **RESOLVED** | unchanged |

**New findings:**
- **L5-7 (LOW):** the label "A-7" for the core-placement finding collides with ADR **A-7** ("random sampling is not reproduced bit for bit") in `CSHARP_MIGRATION.md` §8, which §7 Phase 5 also cites. Readers should resolve "A-7" by context: §8 is the ADR, and the "5 A-7" rows of §10 and the review are the core-placement finding. A future cleanup may rename the review finding (for example "BR-7"). No action required.
- No other new finding.

### 13.4 Final benchmark verdict

# **BENCHMARK_VALID_WITH_CAVEATS**

The remaining caveats are inherent, not defects:
1. Every ratio is versus CPython 3.11.4; PyPy is unmeasured (Q-6).
2. Cold-JIT figures are bimodal and must be quoted as ranges.
3. On this hybrid CPU every timing comparison, including Phase 6's C#-versus-C# Baseline→Compare, must be run pinned to one core class (`run_interleaved.py --affinity auto`, or `pin.run_pinned` for C#-only runs).

### 13.5 Trusted speed-up ranges (versus CPython 3.11.4, single thread, both sides on the P-cores `0xFFF`; union of the two independent pinned measurements)

| Workload | Cold C# | Warm C# | Algorithmic share |
|---|---|---|---|
| Python-semantics search (same nodes) | **≈ 52–91×** (bimodal) | **113–118×** | none |
| Robust endgame pipeline | **109–123×** | **169–180×** | 1.074× |
| Exact solver, large endgame (`mid3_ascension`) | **159–176×** | **161–169×** | 1.235× (memo key); ≈ 131–137× per make_move |
| Fast over the oracle (same exact score) | 1.5× | 1.5× | algorithmic |

**Headline:** "**≈ 50–180× versus CPython 3.11.4** (cold per-request process at the low end, warm resident engine at the high end; both on P-cores; PyPy not measured)".

### 13.6 Phase 6

**Technically ready from the benchmark point of view (YES)**, with one standing rule: every Phase 6 baseline and comparison runs pinned to the P-core set and records the mask (CLAUDE.md §6, plus A-7). Phase 6 still needs the user's explicit authorization. There is no remaining remediation work; `.claude/sonnet_review_fix_goal.md` is marked completed.
