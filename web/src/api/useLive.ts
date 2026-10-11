import { useCallback, useEffect, useReducer, useRef } from 'react'
import { api } from './client'
import type { GameSummary, Lang, LogRecord } from '../domain/types'
import { initialLiveState, liveReducer, type LiveAction, type LiveState } from '../state/live'

const EVENT_NAMES = ['request.received', 'search.depth', 'robust.progress', 'decision', 'game.updated', 'game.ended', 'lagged', 'resync'] as const

export async function fetchGame(id: string, lang: Lang): Promise<{ id: string; records: LogRecord[] }> {
  const data = await api.get<{ records: LogRecord[] }>(`/v1/games/${encodeURIComponent(id)}?lang=${lang}`)
  return { id, records: data.records }
}

export async function fetchRecentGames(lang: Lang, limit = 5): Promise<{ id: string; records: LogRecord[] }[]> {
  const list = await api.get<{ games: GameSummary[] }>(`/v1/games?limit=${limit}`)
  const games = await Promise.all(list.games.map((g) => fetchGame(g.game_id, lang)))
  return games.reverse() // oldest first: the reducer follows the newest
}

/**
 * The live view's data: the newest games are read from the log once, then the event stream (Server-Sent Events) keeps them current. When the connection breaks the browser
 * reconnects by itself and sends Last-Event-ID, so the server replays what was missed; if it cannot (events lost, or it restarted) it says so (resync) and the games are read again.
 */
export function useLive(enabled: boolean, lang: Lang): { state: LiveState; dispatch: (a: LiveAction) => void; reload: () => Promise<void> } {
  const [state, dispatch] = useReducer(liveReducer, undefined, initialLiveState)
  const ref = useRef(state)
  ref.current = state

  const reload = useCallback(async () => {
    try {
      dispatch({ type: 'games_loaded', games: await fetchRecentGames(lang) })
    } catch {
      // not logged in any more, or the server is away: the status shows it when the stream fails too
    }
  }, [lang])

  useEffect(() => {
    if (enabled) void reload()
  }, [enabled, reload])

  useEffect(() => {
    if (!enabled) return
    dispatch({ type: 'connecting' })
    const last = ref.current.lastEventId
    const url = `/v1/live?lang=${encodeURIComponent(lang)}${last > 0 ? `&last_event_id=${last}` : ''}`
    const source = new EventSource(url)
    source.onopen = () => dispatch({ type: 'opened' })
    source.onerror = () => dispatch(source.readyState === EventSource.CLOSED ? { type: 'closed' } : { type: 'connection_lost' })
    for (const name of EVENT_NAMES) {
      source.addEventListener(name, (e) => {
        const message = e as MessageEvent<string>
        let data: Record<string, unknown>
        try {
          data = JSON.parse(message.data) as Record<string, unknown>
        } catch {
          return
        }
        const id = message.lastEventId ? Number(message.lastEventId) : null
        dispatch({ type: 'event', id: id !== null && Number.isFinite(id) ? id : null, name, data })
      })
    }
    return () => source.close()
  }, [enabled, lang])

  useEffect(() => {
    if (!state.resyncNeeded) return
    void reload().then(() => dispatch({ type: 'resynced' }))
  }, [state.resyncNeeded, reload])

  return { state, dispatch, reload }
}
