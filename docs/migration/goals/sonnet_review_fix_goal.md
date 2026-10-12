# TTC_Siren — Review Remediation Goal (Phase 5 benchmark re-review → fix A-7 only: CPU core placement)

> **STATUS: COMPLETED AND VERIFIED (2026-10-08).** The re-review found A-7 **RESOLVED** and B-1 to B-5 without regression (`docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` §13). Verdict: `BENCHMARK_VALID_WITH_CAVEATS`. **Do not run this goal again.** There is no remaining remediation work; it is kept for the record only.

Run with: `/goal @docs/migration/goals/sonnet_review_fix_goal.md`

This goal replaces the previous one (B-1 to B-5, all verified **RESOLVED** in `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` §12). It authorizes **only** finding **A-7** of that review (§12.4). It does **not** authorize Phase 6, any solver optimisation, or any change to search, rules, evaluation or endgame code.

## 1. Start-up

1. Read `CLAUDE.md` (§6 performance process, §7 tests, §8 git).
2. Read `docs/migration/reviews/OPUS_PHASE5_BENCHMARK_REVIEW.md` §12 completely (especially R7–R9 and A-7) and `docs/migration/reports/PHASE5_BENCHMARK.md` (the "Corrections" section).
3. Baseline in your first message: `dotnet build csharp/TTC.Siren.sln -c Release --no-incremental` (0 warnings, 0 errors), `dotnet test csharp/TTC.Siren.sln -c Release` (78/78), `python -m pytest -q` (29 passed).

## 2. Scope

- Allowed files: `tools/bench/run_interleaved.py`, other scripts under `tools/bench/` if strictly needed, `csharp/src/TTC.Siren.Bench/Program.cs` (reporting only), `docs/migration/reports/PHASE5_BENCHMARK.md` (append only), `CSHARP_MIGRATION.md` (§10 rows and a short note under §7 Phase 5).
- No change to `TTC.Siren.Core`, `TTC.Siren.Search`, tests, Python business code or parity generators. No optimisation. No commit unless the user asks; never push; never touch `main`. Do not start Phase 6.

## 3. Problem (A-7)

The machine has a hybrid CPU (i7-13650HX: 6 P-cores on logical processors 0–11, 8 E-cores on 12–19). Windows places long single-thread runs on P- or E-cores non-deterministically:
- Python `mid3_ascension` exact solve: P-core 10.9–12.6 s, E-core 23.4–23.7 s, unpinned 10.1–18.0 s on the same day.
- C# is about 1.4–1.5× slower on the E-core set.

The reviewer's unpinned `run_interleaved.py --rounds 3` therefore reported 124 / 271 / 301×, where the P-core-pinned truth is about 120 / 165 / 181×. Pinning .NET to a **single** logical CPU distorts it by 2–4× (its JIT and GC threads share the CPU), so pinning must use a **multi-CPU set**.

## 4. Required work and acceptance criteria

- [ ] `run_interleaved.py` gets an option `--affinity <hex mask>` that runs **both** the Python and the C# subprocesses restricted to that CPU set. On Windows set it at process start (for example `cmd /c start /wait /b /affinity <mask> <exe> …` with the interpreter's full path, or `psutil`/`SetProcessAffinityMask` in a small launcher). Avoid `dotnet run` wrappers that spawn the real process outside the mask: launch the built `TTC.Siren.Bench.exe` directly, or verify that the mask is inherited. Print the mask in the header and in every result line.
- [ ] A default or documented way to obtain the P-core set. Either detect it (for example via `GetLogicalProcessorInformationEx` efficiency classes, or `psutil` plus documentation), or require the user to pass it, with this machine's value `FFF` documented. Do not hard-code it silently. If no mask is given, print a clear warning that results on a hybrid CPU are not reproducible.
- [ ] `Bench` and the Python harnesses print, in their header, the process affinity mask they actually run with (read it back from the OS), so every table shows whether it was pinned.
- [ ] Re-measure with `run_interleaved.py --rounds 3 --affinity FFF` (warm) and `--cold`, all suites. The results must be consistent with the reviewer's pinned figures: fixed about 118–121× warm and about 55–73× cold; endgame about 178–189× warm and about 109–112× cold; exact `mid3_ascension` about 150–180×. If your pinned results fall outside these ranges by more than 15%, stop and report to the user instead of choosing a number.
- [ ] Optionally also run with the E-core set (`FF000`) once and record it, to document the effect.
- [ ] Append a dated "A-7: CPU core placement" subsection to the "Corrections" section of `docs/migration/reports/PHASE5_BENCHMARK.md`. It records the measurements, states that every trusted ratio is for **both sides on the P-core set**, and adds the reproduction command with `--affinity`. Record the rows in `CSHARP_MIGRATION.md` §10 (measured numbers only) and a one-line note under §7 Phase 5. Do not rewrite earlier rows.
- [ ] Correctness is unchanged: identical node counts and score bits for the 22 fixed items, 40/40 `same` in `fast`, exact values 1/−2/−1, 72/72 endgame selections. Build: 0 warnings, 0 errors; tests 78/78; pytest 29.

## 5. Stop conditions

Stop and report if any benchmark shows a different node count, score, value or selection; if pinned results disagree with the ranges above by more than 15%; if a fix would require solver changes; or if a destructive or remote git operation seems necessary. Stop when §4 is satisfied. Do not start Phase 6.

## 6. Final report

Report: the affinity mechanism and how the mask is obtained, pinned warm and cold tables, the optional E-core run, correctness checks, exact build/test/pytest counts, `git diff --stat`, and that Phase 6 has not been started.
