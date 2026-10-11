import { useEffect, useMemo } from 'react'
import { useLive } from '../api/useLive'
import { BoardView } from '../components/BoardView'
import { DepthTable } from '../components/DepthTable'
import { ExplanationPanel } from '../components/ExplanationPanel'
import { useMood } from '../components/Mascot'
import { Link } from '../components/router'
import { useSession, useSettings } from '../components/context'
import { cardInfo } from '../domain/cards'
import { edgeText } from '../domain/board'
import { gameOf } from '../state/live'
import { liveTableView } from '../state/liveView'
import { moodForLive } from '../state/mood'
import type { MessageKey } from '../i18n'

/** Real-time solve status and live game follow: the table, the search as it progresses, the decision and its explanation. */
export function LivePage() {
  const { t, lang } = useSettings()
  const { status } = useSession()
  const { state, dispatch } = useLive(status === 'user', lang)
  const game = gameOf(state)
  const view = useMemo(() => liveTableView(game), [game])
  useMood(moodForLive(game))
  useEffect(() => {
    document.title = `${t('nav.live')} · TTC Siren`
  }, [t])

  if (status === 'unknown') return <p className="muted">{t('common.loading')}</p>
  if (status === 'anonymous') {
    return (
      <section className="page narrow" data-testid="live-login-needed">
        <h1>{t('live.title')}</h1>
        <p>{t('live.loginNeeded')}</p>
        <Link to="/login" className="button">{t('nav.login')}</Link>
      </section>
    )
  }

  const cur = game?.current ?? null
  const unknownAssumed = (() => {
    const f = cur?.record?.explanation?.facts.find((x) => (x as { type?: string }).type === 'unknown_cards') as { assumed: { card_id: number; star: number }[] } | undefined
    return f?.assumed ?? []
  })()

  return (
    <section className="page live" data-testid="live-page">
      <h1>{t('live.title')}</h1>
      <div className="live-bar">
        <span className={`status status-${state.status}`} data-testid="connection-status" role="status">{t(`live.status.${state.status}` as MessageKey)}</span>
        {state.lagged > 0 ? <span className="note" data-testid="lagged-note">{t('live.lagged', { n: state.lagged })}</span> : null}
        {state.resyncNeeded ? <span className="note" data-testid="resync-note">{t('live.resync')}</span> : null}
        <label className="inline">
          <input type="checkbox" checked={state.follow} onChange={(e) => dispatch({ type: 'follow', follow: e.target.checked })} data-testid="follow-toggle" />
          {t('live.follow')}
        </label>
        <label className="inline">
          {t('live.game')}
          <select
            value={state.selected ?? ''} onChange={(e) => dispatch({ type: 'select', gameId: e.target.value || null })} data-testid="game-picker" aria-label={t('live.game')}
          >
            <option value="">{t('live.newest')}</option>
            {state.order.map((id) => (
              <option key={id} value={id}>{id}{state.games[id].ended ? ` · ${t('games.ended')}` : ''}</option>
            ))}
          </select>
        </label>
      </div>

      {!game ? (
        <p className="muted" data-testid="live-empty">{t('live.waiting')}</p>
      ) : (
        <div className="live-grid">
          <div>
            {view ? <BoardView view={view} /> : <p className="muted">{t('live.noPosition')}</p>}
            <p className="muted" data-testid="live-game-id">
              {t('live.gameInfo', { id: game.id, no: cur?.requestNo ?? 0, rules: game.rules.length ? game.rules.join(', ') : t('live.noRules') })}
            </p>
            {game.ended ? <p className="badge" data-testid="game-ended">{t('live.ended', { mine: game.ended.mine, theirs: game.ended.theirs, winner: t(`winner.${game.ended.winner}` as MessageKey) })}</p> : null}
          </div>
          <div>
            <DepthTable depths={cur?.depths ?? []} robust={cur?.robust ?? null} />
            {cur?.decision ? (
              <p className="decision" data-testid="decision">
                {t('live.decision', {
                  id: cur.decision.move.card_id, cell: t(`cell.${cur.decision.move.row * 3 + cur.decision.move.col}` as MessageKey), depth: cur.decision.depthReached,
                })}
              </p>
            ) : (
              <p className="muted" data-testid="searching">{cur && !game.ended ? t('live.searching') : ''}</p>
            )}
            {unknownAssumed.length > 0 ? (
              <p className="muted" data-testid="live-assumed">
                {t('live.assumed')}: {unknownAssumed.map((a) => { const i = cardInfo(a.card_id); return `#${a.card_id} (${i ? i.edges.map(edgeText).join(' ') : '?'})` }).join(', ')}
              </p>
            ) : null}
            <ExplanationPanel explanation={cur?.record?.explanation ?? null} />
          </div>
        </div>
      )}
    </section>
  )
}
