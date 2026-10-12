# TTC_Siren — Autonomous Migration Phase Goal

> **STATUS: RETIRED (2026-10-10). Do not run this goal.** Phases are now authorized one by one through dedicated goal files (`docs/migration/goals/phase6_goal.md`, `docs/migration/goals/phase7_goal.md`, `docs/migration/goals/phase7_hardening_goal.md`, …). Running this file would pick "the next unfinished phase" (Phase 8), which the user has **not** authorized. If you were started with this goal, stop and tell the user. Kept for the record only.

Reusable goal for the Python → C# migration. Starting this goal is the user's authorization for exactly ONE phase: the next unfinished phase.

## 1. Start-up (mandatory, in this order)

1. Read `CLAUDE.md` completely. It holds the permanent project rules.
2. Read `CSHARP_MIGRATION.md` completely. It is the authoritative migration state.
3. From §0 (status table), identify the **next unfinished phase**: the lowest-numbered phase that is not DONE.
4. Re-read that phase's scope and acceptance criteria in §7 and its open questions in §9.
5. State the phase number, its scope and its completion criteria in one short message, then begin work.

## 2. Scope discipline

- Complete **exactly one phase**: the next unfinished one. Never start, prepare or stub work for the following phase, even if this phase finishes early or looks easy.
- Treat `CSHARP_MIGRATION.md` as the single source of truth for migration state, and the existing Python implementation (`core/`, `ai/ai.py`, `ai_server.py`) as the behavioral reference. Python's actual behavior outranks README text, comments, official game rules and what "should" happen.
- Never modify Python business code. `tools/parity/` and `tools/bench/` scripts only import and call Python modules.
- Follow every rule in `CLAUDE.md`. If this file and `CLAUDE.md` seem to conflict, the stricter rule wins.

## 3. Definition of done

The phase is complete **only when every completion criterion recorded for it in `CSHARP_MIGRATION.md` is satisfied**, with evidence. Beyond those, the following apply whenever relevant to the phase:

- [ ] `dotnet build -c Release` succeeds with 0 warnings and 0 errors.
- [ ] All relevant C# tests pass. None are skipped, deleted, commented out or weakened.
- [ ] The corresponding Python reference tests pass (`python -m pytest -q` stays at 26 passed).
- [ ] Python and C# agree on the applicable reference scenarios: legal moves (and their order), resulting board state, ownership, hand state (including order), score, and best move. Floating-point scores are compared **bit-for-bit** (Python `float.hex()` vs C# `BitConverter.DoubleToInt64Bits`). No epsilon comparisons.
- [ ] When the phase involves mutable search state, `MakeMove` followed by `UndoMove` restores the complete logical state exactly, apart from deliberately reproduced Python quirks listed in the discrepancy table (for example D-001 hand order, D-002, D-003).
- [ ] There is no unresolved Triple Triad rule-semantic discrepancy. Known Python quirks recorded in the discrepancy table, and reproduced on purpose with a `// PY-QUIRK(Dxxx)` comment, are not "unresolved". An unexplained difference is.
- [ ] Required benchmark or baseline results are recorded in the benchmark history.
- [ ] `CSHARP_MIGRATION.md` is updated with: the phase status; files changed or added; design decisions; tests added and their results; benchmark results; known limitations; new discrepancies found; and next-phase information.

If any item above is not satisfied, the phase is not complete. Do not report it as complete.

## 4. Working cycle (repeat autonomously)

1. **Inspect**: read the relevant Python and existing C# code and the current test state.
2. **Implement** the smallest coherent unit.
3. **Build**: `dotnet build -c Release`.
4. **Test**: run the C# tests, and the Python tests when applicable.
5. **Compare against Python** when applicable (golden data or differential tests).
6. **Diagnose** any failure down to the root cause.
7. **Fix** the root cause. Never mask it.
8. Repeat.

Do not ask for approval for ordinary implementation details that can be determined from the repository, the tests, `CLAUDE.md` or `CSHARP_MIGRATION.md`. Choose the option that follows the existing conventions and record it as a design decision.

## 5. Failures and Python bugs

- Never skip, delete, `[Skip]`, comment out or loosen a failing test. Never tune values until output matches. Fix the cause. A test assertion may change only if the assertion itself is proven wrong, and the reason must be recorded.
- Never silently change existing Python behavior.
- If Python appears to contain a bug, record all of the following in `CSHARP_MIGRATION.md`:
  1. Python behavior.
  2. C# behavior.
  3. The difference.
  4. Evidence (reproduction input and output).
  5. The likely correct behavior.
- Do not silently fix such behavior if the fix changes compatibility. C# reproduces Python's actual behavior, marked `// PY-QUIRK(Dxxx)`, until the user approves a fix. A fix is applied to both Python and C#, with a regression test, and the table entry becomes `FIXED-BOTH`.
- Where `CSHARP_MIGRATION.md` §9 lists a default for an open question, use that default and do not wait for an answer.

## 6. Performance work

Use this process every time, one kind of change at a time:

**Baseline → Change → Build → Correctness tests → Benchmark → Compare**

- Take the baseline on the same inputs, in Release, on the same machine. Record nodes, time, NPS, score and move.
- Benchmark at least 3 runs and compare the median.
- Revert the change if any of these holds:
  - it gives no meaningful, repeatable improvement;
  - it unexpectedly increases allocations;
  - it harms correctness.
- Do not introduce `unsafe` code, SIMD, aggressive parallelism or complicated bit packing unless profiler or benchmark evidence shows an actual need. Record that evidence.
- Prefer high single-thread NPS and good cache locality before any parallel search.
- Hot-path code (MakeMove, UndoMove, Evaluate, OrderMoves, AlphaBeta, TT probe) must allocate nothing in steady state, and tests assert this with `GC.GetAllocatedBytesForCurrentThread()`.
- Record only measured numbers. Never write performance claims without data.

## 7. Stop conditions

### 7.1 Stop immediately and mark the phase BLOCKED when

- an unresolved Python-versus-C# rule discrepancy needs a human decision;
- the Python reference behavior is ambiguous and cannot be resolved from the repository code or tests;
- the work requires destructive or remote Git operations;
- an architectural assumption in `CSHARP_MIGRATION.md` is proven fundamentally incorrect.

When blocked:

1. Add an entry to the BLOCKED table in `CSHARP_MIGRATION.md` with: the reproduction input, the Python output, the C# output, and the hypotheses already ruled out.
2. Set the phase status to BLOCKED.
3. Report to the user and stop. Do not work around it, and do not start another phase.

### 7.2 Stop when

- the current phase is fully complete (§3 all satisfied, `CSHARP_MIGRATION.md` updated), or
- 20 autonomous turns have been used and the phase is still incomplete. Count one pass of the §4 cycle as one turn. In that case set the phase status to IN PROGRESS, record exactly what is done, what is failing and what remains, then report.

Do not continue into a later phase merely because this one finished early.

## 8. Git rules

- **Never push** to any remote. Never force-push, rewrite pushed history, tag, release or upload artifacts.
- Never modify `main`. Work only on the `csharp-solver-migration` branch.
- A **local** checkpoint commit may be created when useful, such as after a green build and test run. Each commit holds one logical change, and the message states what and why.
- Destructive local operations (`reset --hard`, `clean -fd`, `checkout --`, branch deletion) require the user's explicit approval. If the work seems to need one, treat it as a BLOCKED condition.

## 9. Final report

When stopping, report briefly:

- the phase worked on and its final status (DONE / BLOCKED / IN PROGRESS);
- the evidence for each completion criterion: build, tests, parity, benchmarks;
- the files changed, design decisions and known limitations;
- any new discrepancies or BLOCKED items;
- the next phase, which is not started and awaits the user's authorization.
