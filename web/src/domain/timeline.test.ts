import { describe, expect, it } from 'vitest'
import demo from '../demo/demo-game.json'
import { buildTimeline, flipOrder, immediateEffect, type FlipStep } from './timeline'
import type { LogRecord } from './types'

const records = demo.records as unknown as LogRecord[]
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('timeline of the sample game', () => {
  const tl = buildTimeline(records)

  it('reads the header of the game', () => {
    expect(tl.gameId).toBe('demo-game')
    expect(tl.myOwner).toBe('blue')
    expect(tl.mine).toBe('b')
    expect(tl.rules).toEqual(['同数', '加算', '王牌杀手'])
    expect(tl.schema).toBe(2)
    expect(tl.ended).not.toBeNull()
  })

  it('has a step per position, per move of ours, per move of the opponent, and an end', () => {
    const kinds = tl.steps.map((s) => s.kind)
    expect(kinds[0]).toBe('position')
    expect(kinds[kinds.length - 1]).toBe('end')
    expect(kinds.filter((k) => k === 'position')).toHaveLength(records.filter((r) => r.type === 'request').length)
    expect(kinds.filter((k) => k === 'opponent_move')).toHaveLength(records.filter((r) => r.type === 'opponent_move').length)
    expect(kinds.filter((k) => k === 'our_move').length).toBeGreaterThan(0)
    expect(tl.steps.map((s) => s.index)).toEqual(tl.steps.map((_, i) => i))
  })

  it('takes the flips of our move from the facts of the engine, with the cause the facts give', () => {
    for (const step of tl.steps.filter((s) => s.kind === 'our_move')) {
      const fact = immediateEffect(step.explanation)!
      expect(step.flips).toHaveLength(fact.flips.length)
      expect(step.flips.map((f) => f.cause)).toEqual(fact.flips.map((f) => f.cause))
      expect(step.flips.every((f) => f.cause !== null)).toBe(true)
      expect(step.placed).toMatchObject({ row: fact.move.row, col: fact.move.col, by: 'mine' })
    }
  })

  it('takes the flips of the opponent from the opponent_move record', () => {
    const opp = records.filter((r) => r.type === 'opponent_move')
    const steps = tl.steps.filter((s) => s.kind === 'opponent_move')
    expect(steps.map((s) => s.flips.length)).toEqual(opp.map((r) => (r.type === 'opponent_move' ? r.flips.length : -1)))
  })

  it('gives every flipped cell to the mover and keeps the counts consistent with the board', () => {
    for (const step of tl.steps) {
      const owners = step.board.filter((c) => c !== null)
      expect(step.counts.mine).toBe(owners.filter((c) => c!.owner === tl.mine).length)
      expect(step.counts.theirs).toBe(owners.filter((c) => c!.owner !== tl.mine).length)
      if (step.placed && step.kind !== 'position') {
        for (const f of step.flips) expect(step.board[f.row * 3 + f.col]!.owner).toBe(step.placed.card.owner)
      }
    }
  })

  it('does not change the records it reads', () => {
    const before = JSON.stringify(records)
    buildTimeline(records)
    expect(JSON.stringify(records)).toBe(before)
  })

  it('shows every cause of a flip in this game (the demo is meant to)', () => {
    const causes = new Set(tl.steps.flatMap((s) => s.flips.map((f) => f.cause)))
    for (const c of ['basic', 'same', 'plus', 'fallen_ace', 'combo']) expect(causes.has(c as never), c).toBe(true)
  })
})

describe('timeline of damaged or old logs', () => {
  it('reads a log without a version as schema 1', () => {
    const old = clone(records).map((r) => {
      const { v: _v, ...rest } = r as unknown as Record<string, unknown>
      return rest as unknown as LogRecord
    })
    expect(buildTimeline(old).schema).toBe(1)
  })

  it('shows a request without an explanation as a position only', () => {
    const stripped = clone(records).map((r) => {
      if (r.type === 'request') delete (r as { explanation?: unknown }).explanation
      return r
    })
    const tl = buildTimeline(stripped)
    expect(tl.steps.some((s) => s.kind === 'our_move')).toBe(false)
    expect(tl.steps.some((s) => s.kind === 'position')).toBe(true)
  })

  it('marks an opponent move that the log cannot place as inconsistent, without inventing a card', () => {
    const broken = clone(records)
    const opp = broken.find((r) => r.type === 'opponent_move') as unknown as Record<string, unknown>
    opp.row = null
    opp.col = null
    opp.card_edges = null
    const step = buildTimeline(broken).steps.find((s) => s.kind === 'opponent_move')!
    expect(step.inconsistent).toBe(true)
    expect(step.placed).toBeNull()
    expect(step.flips).toEqual([])
  })

  it('passes on the ambiguity of an opponent card: the edges are known, the identity is one of several', () => {
    const amb = clone(records)
    const opp = amb.find((r) => r.type === 'opponent_move') as unknown as Record<string, unknown>
    opp.ambiguous = true
    opp.candidate_ids = [40, 41, 42]
    opp.card_id = null
    const step = buildTimeline(amb).steps.find((s) => s.kind === 'opponent_move')!
    expect(step.placed).toMatchObject({ ambiguous: true, candidateIds: [40, 41, 42] })
    expect(step.placed!.card.id).toBeNull()
    expect(step.placed!.card.base).toEqual((opp.card_edges as number[]))
  })

  it('does not read a log with no records', () => {
    const tl = buildTimeline([])
    expect(tl.steps).toEqual([])
    expect(tl.gameId).toBeNull()
    expect(tl.ended).toBeNull()
  })
})

describe('flipOrder', () => {
  const f = (row: number, col: number, cause: FlipStep['cause'], via?: [number, number]): FlipStep => ({ row, col, cardId: null, cause, viaRow: via?.[0], viaCol: via?.[1] })

  it('makes a combo come after the flip that set it off', () => {
    const order = flipOrder([f(0, 0, 'same'), f(1, 0, 'combo', [0, 0])])
    expect(order.map((o) => o.after)).toEqual([null, 0])
  })

  it('leaves a combo with a trigger outside the step unanchored', () => {
    expect(flipOrder([f(1, 0, 'combo', [2, 2])])[0].after).toBeNull()
  })
})
