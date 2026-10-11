import { boardFromWire } from '../domain/board'
import { buildTimeline } from '../domain/timeline'
import type { OutcomeFact, Owner } from '../domain/types'
import { ghostsFrom, type TableView } from '../table/boardModel'
import type { LiveGame } from './live'

/**
 * What the table of the live view shows for a game: the position of the request that is being solved; once the decision is there, our move with the flips the facts give
 * and the principal variation as ghost cards; right after the opponent's move was inferred, that move with its flips. Everything comes from the events and the log records.
 */
export function liveTableView(game: LiveGame | null): TableView | null {
  if (!game || !game.current) return null
  const cur = game.current
  const mine: Owner = game.myOwner === 'red' ? 'r' : game.myOwner === 'blue' ? 'b' : null
  const timeline = buildTimeline(game.records)

  if (cur.decision && cur.record) {
    const step = [...timeline.steps].reverse().find((s) => s.kind === 'our_move' && s.requestNo === cur.requestNo)
    if (step) {
      const pv = (step.explanation?.facts.find((f) => (f as { type?: string }).type === 'outcome') as OutcomeFact | undefined)?.pv ?? []
      return {
        board: step.board, mine, animKey: `${game.id}:${step.index}`, placed: step.placed, flips: step.flips, ghosts: ghostsFrom(pv.slice(1), step.board),
        marked: step.placed ? step.placed.row * 3 + step.placed.col : null,
      }
    }
  }
  if (!cur.decision) {
    const opp = [...timeline.steps].reverse().find((s) => s.kind === 'opponent_move' && s.requestNo === cur.requestNo && s.placed)
    const best = cur.depths.length ? cur.depths[cur.depths.length - 1].move : null
    if (opp) return { board: opp.board, mine, animKey: `${game.id}:${opp.index}`, placed: opp.placed, flips: opp.flips, ghosts: [], marked: best ? best.row * 3 + best.col : null }
    if (cur.state) {
      return { board: boardFromWire(cur.state.board), mine, animKey: `${game.id}:req${cur.requestNo}`, placed: null, flips: [], ghosts: [], marked: best ? best.row * 3 + best.col : null }
    }
  }
  if (cur.state) {
    const m = cur.decision?.move
    return { board: boardFromWire(cur.state.board), mine, animKey: `${game.id}:req${cur.requestNo}:d`, placed: null, flips: [], ghosts: [], marked: m ? m.row * 3 + m.col : null }
  }
  return null
}
