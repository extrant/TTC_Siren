import { useSettings } from './context'
import type { DepthRow } from '../domain/types'
import type { MessageKey } from '../i18n'

/** Progress of the search, one row per finished depth, as the events arrive. */
export function DepthTable({ depths, robust }: { depths: DepthRow[]; robust: { done: number; total: number } | null }) {
  const { t } = useSettings()
  if (depths.length === 0 && !robust) return <p className="muted" data-testid="depth-none">{t('live.noDepth')}</p>
  return (
    <div data-testid="depth-table-wrap">
      <table className="depth-table" data-testid="depth-table">
        <caption>{t('live.depths')}</caption>
        <thead>
          <tr>
            <th scope="col">{t('live.depth')}</th>
            <th scope="col">{t('live.score')}</th>
            <th scope="col">{t('live.bestMove')}</th>
            <th scope="col">{t('live.nodes')}</th>
            <th scope="col">{t('live.nps')}</th>
            <th scope="col">{t('live.elapsed')}</th>
          </tr>
        </thead>
        <tbody>
          {depths.map((d) => (
            <tr key={d.depth} data-depth={d.depth}>
              <th scope="row">{d.depth}</th>
              <td>{d.score.toFixed(2)}</td>
              <td>{d.move ? t('live.moveText', { id: d.move.card_id, cell: t(`cell.${d.move.row * 3 + d.move.col}` as MessageKey) }) : '–'}</td>
              <td>{d.nodes.toLocaleString('en-US')}</td>
              <td>{d.elapsed_ms > 0 ? Math.round(d.nodes / (d.elapsed_ms / 1000)).toLocaleString('en-US') : '–'}</td>
              <td>{Math.round(d.elapsed_ms)} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
      {robust ? (
        <p className="robust-progress" data-testid="robust-progress">
          <progress value={robust.done} max={Math.max(1, robust.total)} aria-label={t('live.robust')} /> {t('live.robustText', { done: robust.done, total: robust.total })}
        </p>
      ) : null}
    </div>
  )
}
