import { describe, expect, it } from 'vitest'
import type { FlipStep } from '../domain/timeline'
import type { Cause } from '../domain/types'
import { EFFECTS, PLACE_MS, planFlips, specOf, totalMs } from './flipEffects'

const flip = (row: number, col: number, cause: Cause | null, extra: Partial<FlipStep> = {}): FlipStep => ({ row, col, cardId: 1, cause, ...extra })

describe('flip effects', () => {
  it('has a distinct effect for each cause', () => {
    const causes: Cause[] = ['basic', 'same', 'plus', 'combo', 'reverse', 'fallen_ace']
    expect(Object.keys(EFFECTS).sort()).toEqual([...causes].sort())
    const signature = (c: Cause) => JSON.stringify([EFFECTS[c].axis, EFFECTS[c].halfTurns, EFFECTS[c].ring?.color ?? null, EFFECTS[c].sprite, EFFECTS[c].shake, EFFECTS[c].arc, EFFECTS[c].color])
    expect(new Set(causes.map(signature)).size).toBe(causes.length)
  })

  it('shows a flip whose cause is unknown as a plain flip', () => {
    expect(specOf(null)).toBe(EFFECTS.basic)
  })

  it('starts after the placement, in the order of the facts, with a stagger', () => {
    const plan = planFlips([flip(0, 0, 'basic'), flip(0, 1, 'same'), flip(0, 2, 'plus')])
    expect(plan.map((p) => p.cell)).toEqual([0, 1, 2])
    expect(plan[0].startMs).toBe(PLACE_MS)
    expect(plan[1].startMs).toBeGreaterThan(plan[0].startMs)
    expect(plan[2].startMs).toBeGreaterThan(plan[1].startMs)
  })

  it('makes a combo wait until the flip that set it off has ended', () => {
    const plan = planFlips([flip(0, 0, 'same'), flip(1, 0, 'combo', { viaRow: 0, viaCol: 0 })])
    expect(plan[1].viaCell).toBe(0)
    expect(plan[1].startMs).toBeGreaterThanOrEqual(plan[0].startMs + plan[0].spec.durationMs)
  })

  it('does not wait for a combo whose trigger is not in the step', () => {
    const plan = planFlips([flip(1, 0, 'combo', { viaRow: 2, viaCol: 2 })])
    expect(plan[0].startMs).toBe(PLACE_MS)
  })

  it('is empty for a move that flips nothing, and ends with the last effect', () => {
    expect(planFlips([])).toEqual([])
    expect(totalMs([])).toBe(PLACE_MS)
    const plan = planFlips([flip(0, 0, 'fallen_ace')])
    expect(totalMs(plan)).toBe(plan[0].startMs + plan[0].spec.durationMs)
  })
})
