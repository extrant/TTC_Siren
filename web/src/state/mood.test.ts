import { describe, expect, it } from 'vitest'
import type { LogRecord } from '../domain/types'
import { en } from '../i18n/en'
import { zh } from '../i18n/zh-CN'
import type { LiveGame, LiveRequest } from './live'
import { moodForLive, neutralMood } from './mood'

const request = (extra: Partial<LiveRequest> = {}): LiveRequest => ({ requestNo: 2, state: null, receivedAt: '', depths: [], robust: null, decision: null, record: null, ...extra })
const game = (extra: Partial<LiveGame> = {}): LiveGame => ({ id: 'g', idSource: null, myOwner: 'blue', rules: [], records: [], current: null, previous: null, ended: null, lastEventAt: null, ...extra })
const decision = { move: { card_id: 1, row: 0, col: 0, slot: 0 }, briefs: {}, brief: null, depthReached: 6, nodes: 10 }
const withOutcome = (outcome: Record<string, unknown>): LiveRequest =>
  request({ decision, record: { type: 'request', explanation: { facts: [{ type: 'outcome', ...outcome }] } } as unknown as LiveRequest['record'] })

describe('mascot mood', () => {
  it('is calm with no game or no request', () => {
    expect(moodForLive(null)).toEqual(neutralMood())
    expect(moodForLive(game())).toEqual(neutralMood())
  })

  it('thinks while a request has no decision, and says how deep the search is', () => {
    const m = moodForLive(game({ current: request({ depths: [{ depth: 1 } as never, { depth: 2 } as never] }) }))
    expect(m).toMatchObject({ expression: 'thinking', caption: 'mascot.thinking', params: { depth: 2 } })
  })

  it('says solved only for an exact result', () => {
    expect(moodForLive(game({ current: withOutcome({ kind: 'exact', result: 'win', difference: 3 }) }))).toMatchObject({ expression: 'confident', caption: 'mascot.solvedWin', params: { diff: 3 } })
    expect(moodForLive(game({ current: withOutcome({ kind: 'exact', result: 'loss', difference: -2 }) }))).toMatchObject({ expression: 'worried', caption: 'mascot.solvedLoss', params: { diff: 2 } })
    expect(moodForLive(game({ current: withOutcome({ kind: 'exact', result: 'draw', difference: 0 }) })).caption).toBe('mascot.solvedDraw')
  })

  it('never claims a win from a heuristic result', () => {
    for (const result of ['ahead', 'level', 'behind']) {
      const m = moodForLive(game({ current: withOutcome({ kind: 'heuristic', result, score: 5 }) }))
      expect(m.caption).not.toMatch(/solved/i)
    }
    expect(moodForLive(game({ current: withOutcome({ kind: 'heuristic', result: 'behind', score: -1.5 }) }))).toMatchObject({ expression: 'worried', caption: 'mascot.behind', params: { score: '-1.50' } })
  })

  it('is surprised when the opponent just made a combo', () => {
    const opp = { type: 'opponent_move', between: [1, 2], flips: [{ row: 0, col: 0, card_id: 1, cause: 'combo' }, { row: 1, col: 0, card_id: 2, cause: 'basic' }] } as unknown as LogRecord
    expect(moodForLive(game({ current: request({ decision }), records: [opp] }))).toMatchObject({ expression: 'surprised', params: { n: 2 } })
  })

  it('does not take an older combo for the last one', () => {
    const opp = { type: 'opponent_move', between: [0, 1], flips: [{ row: 0, col: 0, card_id: 1, cause: 'combo' }] } as unknown as LogRecord
    expect(moodForLive(game({ current: request({ decision }), records: [opp] })).expression).toBe('confident')
  })

  it('reacts to the end of the game', () => {
    expect(moodForLive(game({ ended: { winner: 'mine', mine: 6, theirs: 4 } }))).toMatchObject({ expression: 'happy', caption: 'mascot.won', params: { mine: 6, theirs: 4 } })
    expect(moodForLive(game({ ended: { winner: 'theirs', mine: 4, theirs: 6 } })).expression).toBe('disappointed')
    expect(moodForLive(game({ ended: { winner: 'draw', mine: 5, theirs: 5 } })).caption).toBe('mascot.draw')
  })

  it('always names a caption that exists in both languages', () => {
    const games: LiveGame[] = [
      game(),
      game({ current: request() }),
      game({ current: withOutcome({ kind: 'exact', result: 'win', difference: 1 }) }),
      game({ current: withOutcome({ kind: 'exact', result: 'loss', difference: -1 }) }),
      game({ current: withOutcome({ kind: 'exact', result: 'draw', difference: 0 }) }),
      game({ current: withOutcome({ kind: 'heuristic', result: 'behind', score: -1 }) }),
      game({ current: request({ decision }) }),
      game({ ended: { winner: 'mine', mine: 5, theirs: 4 } }),
    ]
    for (const g of games) {
      const { caption } = moodForLive(g)
      expect(zh[caption], caption).toBeTruthy()
      expect(en[caption], caption).toBeTruthy()
    }
  })
})
