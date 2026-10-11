import { boardFromWire, copyBoard, countOwners, emptyBoard, type Board, type BoardCard, type Counts } from './board'
import type {
  Cause,
  Explanation,
  GameEndRecord,
  GameStartRecord,
  ImmediateEffectFact,
  LogRecord,
  OpponentMoveRecord,
  Owner,
  RequestRecord,
} from './types'

/** One card flipped by a step, with the reason the log or the facts give. Never worked out here. */
export interface FlipStep {
  row: number
  col: number
  cardId: number | null
  cause: Cause | null
  /** for a combo: the cell whose flip set it off */
  viaRow?: number
  viaCol?: number
  /** the values compared (when the facts say them) */
  side?: string
  myValue?: number
  theirValue?: number
  sum?: number
}

export type StepKind = 'position' | 'our_move' | 'opponent_move' | 'end'

export interface Placed {
  row: number
  col: number
  card: BoardCard
  by: 'mine' | 'theirs'
  /** the opponent's card was identified by its edges and several database cards share them */
  ambiguous: boolean
  candidateIds: number[]
}

export interface Step {
  index: number
  kind: StepKind
  /** server time of the record the step comes from */
  t: string
  requestNo: number | null
  /** the board after the step */
  board: Board
  counts: Counts
  placed: Placed | null
  flips: FlipStep[]
  explanation: Explanation | null
  request: RequestRecord | null
  opponent: OpponentMoveRecord | null
  end: GameEndRecord | null
  /** the log could not explain the board (an opponent move that the rules do not account for, a missing record): the flips have no cause */
  inconsistent: boolean
  note: string | null
}

export interface Timeline {
  gameId: string | null
  /** the owner letter of the user's cards on the wire */
  mine: Owner
  myOwner: 'red' | 'blue' | null
  rules: string[]
  npcName: string | null
  steps: Step[]
  ended: GameEndRecord | null
  schema: number
}

export function immediateEffect(explanation: Explanation | null | undefined): ImmediateEffectFact | null {
  const f = explanation?.facts?.find((x) => (x as { type?: string }).type === 'immediate_effect')
  return (f as ImmediateEffectFact | undefined) ?? null
}

function other(owner: Owner): Owner {
  return owner === 'r' ? 'b' : owner === 'b' ? 'r' : null
}

/**
 * Turns the records of a game log into the steps of a replay. The board of a request is taken from the request; our move and the flips it causes come from the
 * explanation facts of that request (the engine's own account of the move); the opponent's move and its flips come from the `opponent_move` record. No rule is applied
 * here: the replay shows what the server recorded, and where the record is incomplete the step says so.
 */
export function buildTimeline(records: readonly LogRecord[]): Timeline {
  let start: GameStartRecord | null = null
  let ended: GameEndRecord | null = null
  let schema = 1
  const steps: Step[] = []
  let board = emptyBoard()
  let mine: Owner = null

  const push = (s: Omit<Step, 'index' | 'counts'>) => {
    steps.push({ ...s, index: steps.length, counts: countOwners(s.board, mine) })
  }

  for (const rec of records) {
    schema = Math.max(schema, rec.v ?? 1)
    switch (rec.type) {
      case 'game_start':
        start = rec
        mine = rec.my_owner === 'red' ? 'r' : 'b'
        break
      case 'request': {
        if (!rec.state) break
        board = boardFromWire(rec.state.board)
        push({
          kind: 'position', t: rec.t, requestNo: rec.request_no, board: copyBoard(board), placed: null, flips: [], explanation: null, request: rec, opponent: null, end: null,
          inconsistent: false, note: rec.error ? String(typeof rec.error === 'string' ? rec.error : JSON.stringify(rec.error)) : null,
        })
        const move = immediateEffect(rec.explanation)
        if (move && rec.result?.move) {
          const next = copyBoard(board)
          const m = move.move
          const card: BoardCard = { id: m.card_id, type: m.type, owner: mine, mod: 0, base: [...m.edges], display: [...m.edges] }
          const flips: FlipStep[] = []
          for (const f of move.flips) {
            const cell = next[f.row * 3 + f.col]
            if (cell) cell.owner = mine
            flips.push({
              row: f.row, col: f.col, cardId: f.card_id, cause: f.cause, viaRow: f.via_row, viaCol: f.via_col, side: f.my_side, myValue: f.my_value, theirValue: f.their_value, sum: f.sum,
            })
          }
          next[m.row * 3 + m.col] = card
          // a same-type rule changes the bonus of every card of that type; the facts say which type and what the bonus becomes
          for (const ch of move.modifier_changes) for (const c of next) if (c && c.type === ch.type) c.mod = ch.to
          board = next
          push({
            kind: 'our_move', t: rec.t, requestNo: rec.request_no, board: copyBoard(board), placed: { row: m.row, col: m.col, card, by: 'mine', ambiguous: false, candidateIds: [] },
            flips, explanation: rec.explanation ?? null, request: rec, opponent: null, end: null, inconsistent: false, note: null,
          })
        }
        break
      }
      case 'opponent_move': {
        if (rec.row === null || rec.col === null || !rec.card_edges) {
          push({
            kind: 'opponent_move', t: rec.t, requestNo: rec.between?.[1] ?? null, board: copyBoard(board), placed: null, flips: [], explanation: null, request: null, opponent: rec, end: null,
            inconsistent: true, note: rec.note,
          })
          break
        }
        const next = copyBoard(board)
        const theirs = other(mine)
        const edges = rec.card_edges as [number, number, number, number]
        const card: BoardCard = { id: rec.card_id, type: null, owner: theirs, mod: 0, base: [...edges], display: [...edges] }
        const flips: FlipStep[] = []
        for (const f of rec.flips) {
          const cell = next[f.row * 3 + f.col]
          if (cell) cell.owner = theirs
          flips.push({ row: f.row, col: f.col, cardId: f.card_id, cause: f.cause })
        }
        next[rec.row * 3 + rec.col] = card
        board = next
        push({
          kind: 'opponent_move', t: rec.t, requestNo: rec.between?.[1] ?? null, board: copyBoard(board),
          placed: { row: rec.row, col: rec.col, card, by: 'theirs', ambiguous: rec.ambiguous === true, candidateIds: rec.candidate_ids ?? [] },
          flips, explanation: null, request: null, opponent: rec, end: null, inconsistent: !rec.consistent, note: rec.note,
        })
        break
      }
      case 'game_end':
        ended = rec
        push({
          kind: 'end', t: rec.t, requestNo: null, board: boardFromWire(rec.board), placed: null, flips: [], explanation: null, request: null, opponent: null, end: rec, inconsistent: false, note: null,
        })
        break
    }
  }

  return { gameId: start?.game_id ?? null, mine, myOwner: start?.my_owner ?? null, rules: start?.rules ?? [], npcName: start?.npc_name ?? null, steps, ended, schema }
}

/** The flips of a step in the order they should play: a combo waits for the flip that set it off; everything else plays in the order of the facts. */
export function flipOrder(flips: readonly FlipStep[]): { flip: FlipStep; after: number | null }[] {
  const index = (r: number | undefined, c: number | undefined) => flips.findIndex((f) => f.row === r && f.col === c)
  return flips.map((flip) => ({ flip, after: flip.cause === 'combo' && flip.viaRow !== undefined ? (index(flip.viaRow, flip.viaCol) >= 0 ? index(flip.viaRow, flip.viaCol) : null) : null }))
}
