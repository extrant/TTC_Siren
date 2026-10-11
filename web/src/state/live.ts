import type { DepthRow, Lang, LogRecord, MoveRef, RequestRecord, WireState } from '../domain/types'

/** What the server says about one request while it is being solved, and afterwards. */
export interface LiveRequest {
  requestNo: number
  state: WireState | null
  receivedAt: string
  depths: DepthRow[]
  robust: { done: number; total: number } | null
  decision: { move: MoveRef; briefs: Partial<Record<Lang, string>>; brief: string | null; depthReached: number; nodes: number } | null
  /** the record the log wrote for the request (facts, explanation, result): arrives after the decision */
  record: RequestRecord | null
}

export interface LiveGame {
  id: string
  idSource: string | null
  myOwner: 'red' | 'blue' | null
  rules: string[]
  records: LogRecord[]
  current: LiveRequest | null
  previous: LiveRequest | null
  ended: { winner: string; mine: number; theirs: number } | null
  lastEventAt: string | null
}

export type ConnectionStatus = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface LiveState {
  status: ConnectionStatus
  /** the id of the newest event applied; events with an id at or below it are repeats after a resume and are ignored */
  lastEventId: number
  /** events the server told us it dropped because we read too slowly (lagged), summed over the connection */
  lagged: number
  /** the server could not resume the stream (events lost, or it restarted): the games have to be read again */
  resyncNeeded: boolean
  resyncReason: string | null
  games: Record<string, LiveGame>
  order: string[]
  /** the game the view follows; with `follow`, the newest one that is active */
  selected: string | null
  follow: boolean
  eventCount: number
}

export const initialLiveState = (): LiveState => ({
  status: 'idle', lastEventId: 0, lagged: 0, resyncNeeded: false, resyncReason: null, games: {}, order: [], selected: null, follow: true, eventCount: 0,
})

export type LiveAction =
  | { type: 'connecting' }
  | { type: 'opened' }
  | { type: 'connection_lost' }
  | { type: 'closed' }
  | { type: 'event'; id: number | null; name: string; data: Record<string, unknown> }
  | { type: 'select'; gameId: string | null }
  | { type: 'follow'; follow: boolean }
  | { type: 'resynced' }
  | { type: 'games_loaded'; games: { id: string; records: LogRecord[] }[] }

const newGame = (id: string): LiveGame => ({ id, idSource: null, myOwner: null, rules: [], records: [], current: null, previous: null, ended: null, lastEventAt: null })

const freshRequest = (requestNo: number, state: WireState | null, receivedAt: string): LiveRequest => ({ requestNo, state, receivedAt, depths: [], robust: null, decision: null, record: null })

function touch(s: LiveState, gameId: string): LiveGame {
  let g = s.games[gameId]
  if (!g) {
    g = newGame(gameId)
    s.games = { ...s.games, [gameId]: g }
    s.order = [...s.order, gameId]
  }
  return g
}

/**
 * What the events would have said about a request, read from the record the log wrote for it: the depths and the decision. Used for a game read from the log (a game that
 * finished before this page was opened has no events), and to fill in what a missed event would have brought. Events that did arrive are not overwritten.
 */
function withRecord(cur: LiveRequest, rec: RequestRecord): LiveRequest {
  const result = rec.result
  const next: LiveRequest = { ...cur, record: rec }
  if (!result) return next
  if (next.depths.length === 0 && result.depths.length > 0) {
    next.depths = result.depths.map((d) => ({ depth: d.depth, score: d.score, move: d.move, nodes: d.nodes_total, elapsed_ms: d.elapsed_ms }))
  }
  if (next.decision === null && result.move) {
    const briefs: Partial<Record<Lang, string>> = {}
    for (const l of ['zh-CN', 'en'] as const) {
      const brief = rec.explanation?.texts?.[l]?.brief
      if (typeof brief === 'string') briefs[l] = brief
    }
    next.decision = { move: result.move, briefs, brief: null, depthReached: result.depth_reached, nodes: result.nodes }
  }
  return next
}

/** Adds a record of the log to a game (a record with the same seq is replaced, so a record that arrives twice is stored once). */
function addRecord(g: LiveGame, rec: LogRecord): LiveGame {
  const records = g.records.filter((r) => r.seq !== rec.seq).concat(rec).sort((a, b) => a.seq - b.seq)
  const next: LiveGame = { ...g, records }
  if (rec.type === 'game_start') {
    next.idSource = rec.id_source
    next.myOwner = rec.my_owner
    next.rules = rec.rules
  } else if (rec.type === 'request') {
    if (next.current && next.current.requestNo === rec.request_no) next.current = withRecord(next.current, rec)
    else if (!next.current || next.current.requestNo < rec.request_no) {
      next.previous = next.current
      next.current = withRecord(freshRequest(rec.request_no, rec.state, rec.t), rec)
    }
  } else if (rec.type === 'game_end') {
    next.ended = { winner: rec.winner, mine: rec.final.mine, theirs: rec.final.theirs }
  }
  return next
}

function selectFollowed(s: LiveState): LiveState {
  if (!s.follow) return s
  // the newest game that has not ended; if all have, the newest one
  const active = [...s.order].reverse().find((id) => !s.games[id].ended)
  const target = active ?? s.order[s.order.length - 1] ?? null
  return target === s.selected ? s : { ...s, selected: target }
}

export function liveReducer(state: LiveState, action: LiveAction): LiveState {
  switch (action.type) {
    case 'connecting':
      return { ...state, status: state.status === 'open' || state.status === 'reconnecting' ? 'reconnecting' : 'connecting' }
    case 'opened':
      return { ...state, status: 'open' }
    case 'connection_lost':
      return { ...state, status: 'reconnecting' }
    case 'closed':
      return { ...state, status: 'closed' }
    case 'select':
      return { ...state, selected: action.gameId, follow: action.gameId === null }
    case 'follow':
      return selectFollowed({ ...state, follow: action.follow })
    case 'resynced':
      return { ...state, resyncNeeded: false, resyncReason: null }
    case 'games_loaded': {
      let s: LiveState = { ...state, games: { ...state.games }, order: [...state.order] }
      for (const { id, records } of action.games) {
        let g = touch(s, id)
        for (const r of records) g = addRecord(g, r)
        s.games = { ...s.games, [id]: g }
      }
      return selectFollowed(s)
    }
    case 'event':
      return applyEvent(state, action)
  }
}

function applyEvent(state: LiveState, action: Extract<LiveAction, { type: 'event' }>): LiveState {
  const { id, name, data } = action
  if (name === 'lagged') return { ...state, lagged: state.lagged + (typeof data.dropped === 'number' ? data.dropped : 0) }
  if (name === 'resync') return { ...state, resyncNeeded: true, resyncReason: typeof data.reason === 'string' ? data.reason : 'unknown', lastEventId: data.reason === 'stream_restarted' ? 0 : state.lastEventId }
  // a repeat after a resume (the id is at or below the newest we applied) changes nothing; an id of 0 or null (not numbered) is always applied
  if (id !== null && id > 0 && id <= state.lastEventId) return state
  const gameId = typeof data.game_id === 'string' ? data.game_id : null
  let s: LiveState = { ...state, lastEventId: id !== null && id > state.lastEventId ? id : state.lastEventId, eventCount: state.eventCount + 1 }
  if (gameId === null) return s
  s.games = { ...s.games }
  s.order = [...s.order]
  let g = touch(s, gameId)
  const at = typeof data.t === 'string' ? data.t : null
  g = { ...g, lastEventAt: at ?? g.lastEventAt }
  switch (name) {
    case 'request.received': {
      const requestNo = Number(data.request_no)
      g = {
        ...g, idSource: (data.id_source as string) ?? g.idSource, rules: (data.rules as string[]) ?? g.rules,
        previous: g.current, current: freshRequest(requestNo, (data.state as WireState) ?? null, at ?? ''),
      }
      break
    }
    case 'search.depth': {
      const requestNo = Number(data.request_no)
      if (!g.current || g.current.requestNo !== requestNo) g = { ...g, previous: g.current, current: freshRequest(requestNo, null, at ?? '') }
      const row: DepthRow = {
        depth: Number(data.depth), score: Number(data.score), move: (data.move as MoveRef | null) ?? null, nodes: Number(data.nodes ?? 0),
        tt_hits: Number(data.tt_hits ?? 0), cutoffs: Number(data.cutoffs ?? 0), elapsed_ms: Number(data.elapsed_ms ?? 0),
      }
      const depths = g.current!.depths.filter((d) => d.depth !== row.depth).concat(row).sort((a, b) => a.depth - b.depth)
      g = { ...g, current: { ...g.current!, depths } }
      break
    }
    case 'robust.progress': {
      if (g.current) g = { ...g, current: { ...g.current, robust: { done: Number(data.done), total: Number(data.total) } } }
      break
    }
    case 'decision': {
      if (g.current) {
        g = {
          ...g,
          current: {
            ...g.current,
            decision: {
              move: data.move as MoveRef, briefs: (data.briefs as Partial<Record<Lang, string>>) ?? {}, brief: (data.brief as string) ?? null,
              depthReached: Number(data.depth_reached ?? 0), nodes: Number(data.nodes ?? 0),
            },
          },
        }
      }
      break
    }
    case 'game.updated': {
      const rec = data.record as LogRecord | undefined
      if (rec) g = addRecord(g, rec)
      break
    }
    case 'game.ended': {
      const final = data.final as { mine: number; theirs: number } | undefined
      g = { ...g, ended: { winner: String(data.winner), mine: final?.mine ?? 0, theirs: final?.theirs ?? 0 } }
      break
    }
  }
  s.games[gameId] = g
  return selectFollowed(s)
}

export const gameOf = (s: LiveState): LiveGame | null => (s.selected ? s.games[s.selected] ?? null : null)
