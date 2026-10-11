import { effectiveEdges, type Board } from '../domain/board'
import { cardInfo } from '../domain/cards'
import type { Placed, FlipStep } from '../domain/timeline'
import type { Owner, PvPly } from '../domain/types'
import type { CardArt, Side } from './cardArt'

/** What the table (flat or 3D) is told to show. */
export interface TableView {
  board: Board
  /** the owner letter of the user's cards */
  mine: Owner
  /** changes whenever a new step is shown: the animations of `placed` and `flips` play once per key */
  animKey: string
  placed: Placed | null
  flips: FlipStep[]
  /** cards of the principal variation, drawn as ghosts on the empty cells */
  ghosts: Ghost[]
  /** the mark of the cell the move goes to (the decision of the live view) */
  marked: number | null
}

export interface Ghost {
  cell: number
  ply: number
  side: 'self' | 'opponent'
  cardId: number
}

export function ghostsFrom(pv: readonly PvPly[], board: Board): Ghost[] {
  const out: Ghost[] = []
  for (const p of pv) {
    const cell = p.row * 3 + p.col
    if (board[cell] === null && !out.some((g) => g.cell === cell)) out.push({ cell, ply: p.ply, side: p.side, cardId: p.card_id })
  }
  return out
}

export function sideOf(owner: Owner, mine: Owner): Side {
  return owner === null ? 'none' : owner === mine ? 'mine' : 'theirs'
}

export function artFor(board: Board, cell: number, mine: Owner, placed: Placed | null): CardArt | null {
  const c = board[cell]
  if (!c) return null
  const info = cardInfo(c.id)
  const eff = effectiveEdges(c)
  return {
    edges: eff, side: sideOf(c.owner, mine), star: info?.star ?? 0, type: c.type ?? info?.type ?? null, mod: c.mod, id: c.id,
    ambiguous: placed !== null && placed.row * 3 + placed.col === cell && placed.ambiguous,
  }
}

export function ghostArt(g: Ghost): CardArt {
  const info = cardInfo(g.cardId)
  return { edges: info?.edges ?? [0, 0, 0, 0], side: g.side === 'self' ? 'mine' : 'theirs', star: info?.star ?? 0, type: info?.type ?? null, mod: 0, id: g.cardId, ghost: true, ply: g.ply }
}
