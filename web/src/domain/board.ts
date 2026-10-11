import type { Owner, WireCard } from './types'

/** A card on the board or in a hand, as the log describes it. Nothing here is computed by the rules: it is what the server sent. */
export interface BoardCard {
  id: number | null
  type: string | null
  owner: Owner
  /** the bonus a same-type rule (同类强化 / 同类弱化) gives this card */
  mod: number
  base: [number, number, number, number]
  display: [number, number, number, number]
}

export type Board = (BoardCard | null)[]

export const emptyBoard = (): Board => Array.from({ length: 9 }, () => null)

export function cardFromWire(w: WireCard): BoardCard {
  return {
    id: w[0],
    type: w[1],
    owner: w[2],
    mod: w[3],
    base: [w[4], w[5], w[6], w[7]],
    display: [w[8], w[9], w[10], w[11]],
  }
}

export function boardFromWire(cells: (WireCard | null)[]): Board {
  const board = emptyBoard()
  for (let i = 0; i < 9; i++) {
    const c = cells[i]
    board[i] = c ? cardFromWire(c) : null
  }
  return board
}

/** The values the rules compare: the base value plus the type bonus, kept within 1..10. */
export function effectiveEdges(card: BoardCard): [number, number, number, number] {
  const clamp = (v: number) => Math.max(1, Math.min(10, v + card.mod))
  return [clamp(card.base[0]), clamp(card.base[1]), clamp(card.base[2]), clamp(card.base[3])]
}

/** 10 is written A, as in the game. */
export function edgeText(v: number): string {
  return v === 10 ? 'A' : String(v)
}

export function copyBoard(board: Board): Board {
  return board.map((c) => (c ? { ...c, base: [...c.base] as BoardCard['base'], display: [...c.display] as BoardCard['display'] } : null))
}

export const cellOf = (row: number, col: number): number => row * 3 + col
export const rowOf = (cell: number): number => Math.floor(cell / 3)
export const colOf = (cell: number): number => cell % 3

export interface Counts {
  mine: number
  theirs: number
}

export function countOwners(board: Board, mine: Owner): Counts {
  let m = 0
  let t = 0
  for (const c of board) {
    if (!c || c.owner === null) continue
    if (c.owner === mine) m++
    else t++
  }
  return { mine: m, theirs: t }
}
