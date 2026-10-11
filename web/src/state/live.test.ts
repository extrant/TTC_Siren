import { describe, expect, it } from 'vitest'
import type { LogRecord } from '../domain/types'
import { gameOf, initialLiveState, liveReducer, type LiveAction, type LiveState } from './live'

const event = (id: number | null, name: string, data: Record<string, unknown>): LiveAction => ({ type: 'event', id, name, data })
const run = (actions: LiveAction[], from: LiveState = initialLiveState()) => actions.reduce(liveReducer, from)

const received = (game: string, no: number, id = no * 10) => event(id, 'request.received', { game_id: game, request_no: no, t: `2026-10-11T09:00:0${no}Z`, id_source: 'client', rules: ['加算'], state: null })
const depth = (game: string, no: number, d: number, id: number) => event(id, 'search.depth', { game_id: game, request_no: no, depth: d, score: d / 2, move: { card_id: 1, row: 0, col: 0, slot: 0 }, nodes: 100 * d })

const record = (seq: number, extra: Record<string, unknown>): LogRecord => ({ v: 2, seq, t: '2026-10-11T09:00:00Z', ...extra }) as unknown as LogRecord

describe('live reducer', () => {
  it('tracks the connection', () => {
    let s = run([{ type: 'connecting' }])
    expect(s.status).toBe('connecting')
    s = liveReducer(s, { type: 'opened' })
    expect(s.status).toBe('open')
    s = liveReducer(s, { type: 'connection_lost' })
    expect(s.status).toBe('reconnecting')
    s = liveReducer(s, { type: 'connecting' })
    expect(s.status).toBe('reconnecting') // a reconnect is not a first connect
    s = liveReducer(s, { type: 'closed' })
    expect(s.status).toBe('closed')
  })

  it('starts a request, collects the depths in order of depth even if they arrive out of order, and replaces a repeated depth', () => {
    const s = run([received('g', 1), depth('g', 1, 2, 11), depth('g', 1, 1, 12), depth('g', 1, 3, 13), depth('g', 1, 2, 14)])
    const cur = gameOf(s)!.current!
    expect(cur.requestNo).toBe(1)
    expect(cur.depths.map((d) => d.depth)).toEqual([1, 2, 3])
    expect(s.eventCount).toBe(5)
  })

  it('moves the finished request to previous when the next one arrives', () => {
    const s = run([received('g', 1), depth('g', 1, 1, 11), received('g', 2)])
    const g = gameOf(s)!
    expect(g.current!.requestNo).toBe(2)
    expect(g.current!.depths).toEqual([])
    expect(g.previous!.requestNo).toBe(1)
    expect(g.previous!.depths).toHaveLength(1)
  })

  it('creates the request from a depth event when the received event was missed', () => {
    const s = run([depth('g', 4, 1, 40)])
    expect(gameOf(s)!.current!.requestNo).toBe(4)
  })

  it('applies the decision and the progress of the endgame to the current request', () => {
    const s = run([
      received('g', 1),
      event(11, 'robust.progress', { game_id: 'g', done: 3, total: 10 }),
      event(12, 'decision', { game_id: 'g', move: { card_id: 7, row: 1, col: 1, slot: 0 }, briefs: { en: 'x', 'zh-CN': 'y' }, brief: 'x', depth_reached: 5, nodes: 1234 }),
    ])
    const cur = gameOf(s)!.current!
    expect(cur.robust).toEqual({ done: 3, total: 10 })
    expect(cur.decision).toMatchObject({ depthReached: 5, nodes: 1234, brief: 'x' })
    expect(cur.decision!.briefs).toEqual({ en: 'x', 'zh-CN': 'y' })
  })

  it('ignores a progress or decision event with no request to attach it to', () => {
    const s = run([event(1, 'robust.progress', { game_id: 'g', done: 1, total: 2 }), event(2, 'decision', { game_id: 'g', move: null })])
    expect(gameOf(s)!.current).toBeNull()
  })

  describe('order and repeats', () => {
    it('applies an event only once: ids at or below the newest are repeats after a resume', () => {
      const once = run([received('g', 1), depth('g', 1, 1, 11)])
      const again = run([received('g', 1), depth('g', 1, 1, 11), received('g', 1), depth('g', 1, 1, 11), depth('g', 1, 1, 5)])
      expect(again.games).toEqual(once.games)
      expect(again.lastEventId).toBe(11)
      expect(again.eventCount).toBe(2)
    })

    it('does not go backwards: a lower id arriving late is dropped, not applied over the newer state', () => {
      const s = run([received('g', 1), depth('g', 1, 2, 20), depth('g', 1, 1, 15)])
      expect(gameOf(s)!.current!.depths.map((d) => d.depth)).toEqual([2])
      expect(s.lastEventId).toBe(20)
    })

    it('always applies an event without a number', () => {
      const s = run([depth('g', 1, 1, 0), depth('g', 1, 2, 0), event(null, 'search.depth', { game_id: 'g', request_no: 1, depth: 3, score: 1, move: null })])
      expect(gameOf(s)!.current!.depths).toHaveLength(3)
      expect(s.lastEventId).toBe(0)
    })

    it('keeps the numbering across a reconnect, so what the server replays after Last-Event-ID is not applied twice', () => {
      let s = run([received('g', 1), depth('g', 1, 1, 11), { type: 'connection_lost' }, { type: 'connecting' }, { type: 'opened' }])
      s = run([depth('g', 1, 1, 11), depth('g', 1, 2, 12)], s) // the server replays from 11 and then goes on
      expect(gameOf(s)!.current!.depths.map((d) => d.depth)).toEqual([1, 2])
      expect(s.status).toBe('open')
      expect(s.lastEventId).toBe(12)
    })
  })

  describe('lag and resync', () => {
    it('adds up the events the server says it dropped, and still applies later events', () => {
      let s = run([received('g', 1), event(null, 'lagged', { dropped: 3 }), event(null, 'lagged', { dropped: 4 })])
      expect(s.lagged).toBe(7)
      s = run([depth('g', 1, 1, 30)], s)
      expect(gameOf(s)!.current!.depths).toHaveLength(1)
    })

    it('ignores a lagged event without a number', () => {
      expect(run([event(null, 'lagged', {})]).lagged).toBe(0)
    })

    it('asks for the games to be read again when the server cannot resume, and clears the request when that is done', () => {
      let s = run([received('g', 1), depth('g', 1, 1, 11), event(null, 'resync', { reason: 'events_lost' })])
      expect(s.resyncNeeded).toBe(true)
      expect(s.resyncReason).toBe('events_lost')
      expect(s.lastEventId).toBe(11) // events were lost: the numbering goes on
      s = liveReducer(s, { type: 'resynced' })
      expect(s.resyncNeeded).toBe(false)
      expect(s.resyncReason).toBeNull()
    })

    it('forgets the numbering when the server restarted, so that the new stream (which starts again at 1) is not taken for repeats', () => {
      let s = run([received('g', 1), depth('g', 1, 1, 11), event(null, 'resync', { reason: 'stream_restarted' })])
      expect(s.lastEventId).toBe(0)
      s = run([depth('g', 1, 2, 1)], s)
      expect(gameOf(s)!.current!.depths.map((d) => d.depth)).toEqual([1, 2])
    })
  })

  describe('records and games', () => {
    it('stores a record that arrives twice once, in order of seq', () => {
      const start = record(0, { type: 'game_start', game_id: 'g', id_source: 'client', my_owner: 'blue', rules: ['同数'] })
      const req = record(1, { type: 'request', request_no: 1, state: null, t: '2026-10-11T09:00:01Z' })
      const s = run([
        event(1, 'game.updated', { game_id: 'g', record: req }),
        event(2, 'game.updated', { game_id: 'g', record: start }),
        event(3, 'game.updated', { game_id: 'g', record: req }),
      ])
      const g = gameOf(s)!
      expect(g.records.map((r) => r.seq)).toEqual([0, 1])
      expect(g.myOwner).toBe('blue')
      expect(g.rules).toEqual(['同数'])
      expect(g.current!.requestNo).toBe(1)
    })

    it('attaches the record of a request to the request that is being solved, and does not start a new one', () => {
      const req = record(1, { type: 'request', request_no: 1, state: null, t: 'x', explanation: { facts: [] } })
      const s = run([received('g', 1), depth('g', 1, 1, 11), event(12, 'game.updated', { game_id: 'g', record: req })])
      const cur = gameOf(s)!.current!
      expect(cur.requestNo).toBe(1)
      expect(cur.depths).toHaveLength(1)
      expect(cur.record).not.toBeNull()
      expect(gameOf(s)!.previous).toBeNull()
    })

    it('reads the depths and the decision from the record of a game that finished before the page was opened', () => {
      const req = record(1, {
        type: 'request', request_no: 1, state: null, t: 'x',
        result: { move: { card_id: 9, row: 1, col: 2, slot: 0 }, depth_reached: 2, nodes: 77, depths: [
          { depth: 1, score: 1.5, move: null, nodes_total: 10, elapsed_ms: 0.1 }, { depth: 2, score: 2.5, move: { card_id: 9, row: 1, col: 2, slot: 0 }, nodes_total: 77, elapsed_ms: 0.4 }] },
        explanation: { texts: { en: { brief: 'in English' }, 'zh-CN': { brief: '中文' } } },
      })
      const cur = gameOf(liveReducer(initialLiveState(), { type: 'games_loaded', games: [{ id: 'old', records: [req] }] }))!.current!
      expect(cur.depths.map((d) => [d.depth, d.nodes])).toEqual([[1, 10], [2, 77]])
      expect(cur.decision).toMatchObject({ depthReached: 2, nodes: 77, briefs: { en: 'in English', 'zh-CN': '中文' } })
      expect(cur.decision!.move.card_id).toBe(9)
    })

    it('does not overwrite what the events brought with what the record says', () => {
      const req = record(1, { type: 'request', request_no: 1, state: null, t: 'x', result: { move: { card_id: 9, row: 0, col: 0, slot: 0 }, depth_reached: 9, nodes: 1, depths: [] } })
      const s = run([received('g', 1), event(12, 'decision', { game_id: 'g', move: { card_id: 7, row: 1, col: 1, slot: 0 }, briefs: {}, brief: null, depth_reached: 5, nodes: 5 }), event(13, 'game.updated', { game_id: 'g', record: req })])
      expect(gameOf(s)!.current!.decision!.depthReached).toBe(5)
      expect(gameOf(s)!.current!.decision!.move.card_id).toBe(7)
    })

    it('marks the game ended', () => {
      const s = run([received('g', 1), event(13, 'game.ended', { game_id: 'g', winner: 'mine', final: { mine: 6, theirs: 4 } })])
      expect(gameOf(s)!.ended).toEqual({ winner: 'mine', mine: 6, theirs: 4 })
    })

    it('loads games from the log and merges them with what the stream already brought', () => {
      const req = record(1, { type: 'request', request_no: 1, state: null, t: 'x' })
      let s = run([received('b', 1)])
      s = liveReducer(s, { type: 'games_loaded', games: [{ id: 'a', records: [req] }, { id: 'b', records: [req] }] })
      expect(s.order).toEqual(['b', 'a'])
      expect(s.games.a.records).toHaveLength(1)
      expect(s.games.b.records).toHaveLength(1)
    })
  })

  describe('which game is shown', () => {
    it('follows the newest game that has not ended, and the newest of all when every game has ended', () => {
      let s = run([received('a', 1, 1), received('b', 1, 2)])
      expect(s.selected).toBe('b')
      s = run([event(90, 'game.ended', { game_id: 'b', winner: 'mine', final: { mine: 5, theirs: 5 } })], s)
      expect(s.selected).toBe('a')
      s = run([event(91, 'game.ended', { game_id: 'a', winner: 'theirs', final: { mine: 4, theirs: 6 } })], s)
      expect(s.selected).toBe('b')
    })

    it('stays on the chosen game when the user picks one, until following is switched on again', () => {
      let s = run([received('a', 1, 1), received('b', 1, 2)])
      s = liveReducer(s, { type: 'select', gameId: 'a' })
      expect(s.follow).toBe(false)
      s = run([received('c', 1, 3)], s)
      expect(s.selected).toBe('a')
      s = liveReducer(s, { type: 'follow', follow: true })
      expect(s.selected).toBe('c')
    })

    it('has no game before the first event', () => {
      expect(gameOf(initialLiveState())).toBeNull()
    })
  })

  it('does not change its input', () => {
    const before = run([received('g', 1)])
    const snapshot = JSON.stringify(before)
    run([depth('g', 1, 1, 11), event(12, 'game.ended', { game_id: 'g', winner: 'draw', final: { mine: 5, theirs: 5 } })], before)
    expect(JSON.stringify(before)).toBe(snapshot)
  })
})
