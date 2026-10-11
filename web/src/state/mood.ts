import type { LiveGame } from './live'
import type { OutcomeFact } from '../domain/types'
import type { MessageKey } from '../i18n'

// Which expression the mascot shows, and the neutral caption beside it, follows from the events (docs/design/MASCOT.md section 4). A caption states only what the facts say:
// "solved" is shown only for an exact result, a win only for an exact win.

export type Expression = 'neutral' | 'thinking' | 'confident' | 'surprised' | 'worried' | 'happy' | 'explaining' | 'disappointed'

export interface Mood {
  expression: Expression
  caption: MessageKey
  params: Record<string, string | number>
}

export const neutralMood = (): Mood => ({ expression: 'neutral', caption: 'mascot.idle', params: {} })

export function outcomeOf(game: LiveGame): OutcomeFact | null {
  const f = game.current?.record?.explanation?.facts?.find((x) => (x as { type?: string }).type === 'outcome')
  return (f as OutcomeFact | undefined) ?? null
}

/** The mood of the live view for a game. */
export function moodForLive(game: LiveGame | null): Mood {
  if (!game) return neutralMood()
  if (game.ended) {
    const won = game.ended.winner === 'mine'
    const draw = game.ended.winner === 'draw'
    return {
      expression: won ? 'happy' : draw ? 'neutral' : 'disappointed',
      caption: won ? 'mascot.won' : draw ? 'mascot.draw' : 'mascot.lost',
      params: { mine: game.ended.mine, theirs: game.ended.theirs },
    }
  }
  const cur = game.current
  if (!cur) return neutralMood()
  if (!cur.decision) {
    const depth = cur.depths.length ? cur.depths[cur.depths.length - 1].depth : 0
    return { expression: 'thinking', caption: 'mascot.thinking', params: { depth } }
  }
  const outcome = outcomeOf(game)
  if (outcome?.kind === 'exact') {
    if (outcome.result === 'win') return { expression: 'confident', caption: 'mascot.solvedWin', params: { diff: Math.abs(outcome.difference ?? 0) } }
    if (outcome.result === 'loss') return { expression: 'worried', caption: 'mascot.solvedLoss', params: { diff: Math.abs(outcome.difference ?? 0) } }
    return { expression: 'neutral', caption: 'mascot.solvedDraw', params: {} }
  }
  if (outcome?.kind === 'heuristic' && outcome.result === 'behind') return { expression: 'worried', caption: 'mascot.behind', params: { score: (outcome.score ?? 0).toFixed(2) } }
  // a decision: when the opponent's last move flipped many cards or used a combo, the mascot is surprised
  const lastOpp = [...game.records].reverse().find((r) => r.type === 'opponent_move')
  if (lastOpp && lastOpp.type === 'opponent_move' && lastOpp.between[1] === cur.requestNo && lastOpp.flips.some((f) => f.cause === 'combo')) {
    return { expression: 'surprised', caption: 'mascot.combo', params: { n: lastOpp.flips.length } }
  }
  return { expression: 'confident', caption: 'mascot.decided', params: { depth: cur.decision.depthReached } }
}
