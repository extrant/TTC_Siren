import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import { fetchGame } from '../api/useLive'
import { ReplayPlayer } from '../components/ReplayPlayer'
import { Link, useRouter } from '../components/router'
import { useSession, useSettings } from '../components/context'
import type { GameSummary, LogRecord } from '../domain/types'

const PAGE = 15

function useLoginGuard(): 'ok' | 'wait' | 'no' {
  const { status } = useSession()
  return status === 'user' ? 'ok' : status === 'unknown' ? 'wait' : 'no'
}

function LoginNeeded({ title }: { title: string }) {
  const { t } = useSettings()
  return (
    <section className="page narrow" data-testid="games-login-needed">
      <h1>{title}</h1>
      <p>{t('games.loginNeeded')}</p>
      <Link to="/login" className="button">{t('nav.login')}</Link>
    </section>
  )
}

/** A confirmation in the page (not the browser's dialog): says what will be deleted and asks. */
function ConfirmDelete({ id, onYes, onNo }: { id: string; onYes: () => void; onNo: () => void }) {
  const { t } = useSettings()
  return (
    <div className="dialog-backdrop">
      <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" data-testid="confirm-delete">
        <h2 id="confirm-title">{t('games.deleteTitle')}</h2>
        <p>{t('games.deleteText', { id })}</p>
        <div className="row">
          <button type="button" className="danger" onClick={onYes} data-testid="confirm-yes" autoFocus>{t('games.deleteYes')}</button>
          <button type="button" onClick={onNo} data-testid="confirm-no">{t('common.cancel')}</button>
        </div>
      </div>
    </div>
  )
}

export function GamesPage() {
  const { t, lang } = useSettings()
  const guard = useLoginGuard()
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<{ total: number; games: GameSummary[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setData(await api.get<{ total: number; games: GameSummary[] }>(`/v1/games?offset=${offset}&limit=${PAGE}`))
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }, [offset])
  useEffect(() => {
    if (guard === 'ok') void load()
  }, [guard, load])
  useEffect(() => {
    document.title = `${t('nav.games')} · TTC Siren`
  }, [t])

  if (guard === 'wait') return <p className="muted">{t('common.loading')}</p>
  if (guard === 'no') return <LoginNeeded title={t('games.title')} />

  const remove = async (id: string) => {
    setDeleting(null)
    try {
      await api.delete(`/v1/games/${encodeURIComponent(id)}`)
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  return (
    <section className="page" data-testid="games-page">
      <h1>{t('games.title')}</h1>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {data && data.games.length === 0 ? <p className="muted" data-testid="games-empty">{t('games.empty')}</p> : null}
      {data && data.games.length > 0 ? (
        <table className="games-table" data-testid="games-table">
          <thead>
            <tr>
              <th scope="col">{t('games.id')}</th>
              <th scope="col">{t('games.started')}</th>
              <th scope="col">{t('games.requests')}</th>
              <th scope="col">{t('games.state')}</th>
              <th scope="col">{t('games.rules')}</th>
              <th scope="col"><span className="sr-only">{t('games.actions')}</span></th>
            </tr>
          </thead>
          <tbody>
            {data.games.map((g) => (
              <tr key={g.game_id} data-game={g.game_id}>
                <th scope="row"><Link to={`/games/${encodeURIComponent(g.game_id)}`} data-testid={`open-${g.game_id}`}>{g.game_id}</Link></th>
                <td>{new Date(g.started).toLocaleString(lang)}</td>
                <td>{g.requests}</td>
                <td>{g.ended ? `${t('games.ended')}${g.winner ? ` · ${t(`winner.${g.winner}` as never)}` : ''}` : t('games.active')}</td>
                <td>{g.rules || '–'}</td>
                <td className="actions">
                  <a href={`/v1/games/${encodeURIComponent(g.game_id)}/download`} download={`${g.game_id}.jsonl`} data-testid={`download-${g.game_id}`}>{t('games.download')}</a>
                  <button type="button" className="link-button danger" onClick={() => setDeleting(g.game_id)} data-testid={`delete-${g.game_id}`}>{t('games.delete')}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {data && data.total > PAGE ? (
        <nav className="pager" aria-label={t('games.pages')}>
          <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} data-testid="page-prev">{t('games.prev')}</button>
          <span>{t('games.range', { from: offset + 1, to: Math.min(offset + PAGE, data.total), total: data.total })}</span>
          <button type="button" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)} data-testid="page-next">{t('games.next')}</button>
        </nav>
      ) : null}
      {deleting ? <ConfirmDelete id={deleting} onYes={() => void remove(deleting)} onNo={() => setDeleting(null)} /> : null}
    </section>
  )
}

export function GameViewPage({ id }: { id: string }) {
  const { t, lang } = useSettings()
  const guard = useLoginGuard()
  const { navigate } = useRouter()
  const [records, setRecords] = useState<LogRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)
  const shownId = useRef(id)

  useEffect(() => {
    if (guard !== 'ok') return
    let cancelled = false
    // another game starts from nothing; the same game read again (the language changed: the texts come in the new language) keeps what is shown, so the replay stays where it is
    if (shownId.current !== id) {
      shownId.current = id
      setRecords(null)
      setError(null)
    }
    fetchGame(id, lang)
      .then((g) => {
        if (!cancelled) setRecords(g.records)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError && e.status === 404 ? t('games.notFound') : e instanceof ApiError ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [guard, id, lang, t])
  useEffect(() => {
    document.title = `${id} · TTC Siren`
  }, [id])

  if (guard === 'wait') return <p className="muted">{t('common.loading')}</p>
  if (guard === 'no') return <LoginNeeded title={t('games.title')} />

  return (
    <section className="page" data-testid="game-page">
      <p><Link to="/games">← {t('games.title')}</Link></p>
      <h1>{id}</h1>
      <div className="row">
        <a className="button" href={`/v1/games/${encodeURIComponent(id)}/download`} download={`${id}.jsonl`} data-testid="game-download">{t('games.download')}</a>
        <button type="button" className="danger" onClick={() => setConfirming(true)} data-testid="game-delete">{t('games.delete')}</button>
      </div>
      {error ? <p className="error" role="alert" data-testid="game-error">{error}</p> : null}
      {records ? <ReplayPlayer records={records} /> : error ? null : <p className="muted">{t('common.loading')}</p>}
      {confirming ? (
        <ConfirmDelete
          id={id}
          onNo={() => setConfirming(false)}
          onYes={() => {
            setConfirming(false)
            api.delete(`/v1/games/${encodeURIComponent(id)}`).then(() => navigate('/games'), (e) => setError(e instanceof ApiError ? e.message : String(e)))
          }}
        />
      ) : null}
    </section>
  )
}
