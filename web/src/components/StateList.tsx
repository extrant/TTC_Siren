import { effectiveEdges, edgeText } from '../domain/board'
import { useSettings } from './context'
import type { TableView } from '../table/boardModel'
import type { MessageKey } from '../i18n'

/** The table as text: one line per cell. Screen readers get the position without the picture, and the tests read it. */
export function StateList({ view }: { view: TableView }) {
  const { t } = useSettings()
  return (
    <ol className="state-list" aria-label={t('table.textList')} data-testid="state-list">
      {view.board.map((c, cell) => {
        const name = t(`cell.${cell}` as MessageKey)
        if (!c) return <li key={cell} data-cell={cell}>{t('table.empty', { cell: name })}</li>
        const e = effectiveEdges(c).map(edgeText)
        const owner = c.owner === null ? 'none' : c.owner === view.mine ? 'mine' : 'theirs'
        return (
          <li key={cell} data-cell={cell} data-owner={owner}>
            {t(owner === 'mine' ? 'table.mine' : owner === 'theirs' ? 'table.theirs' : 'table.neutral', {
              cell: name, id: c.id ?? '?', up: e[0], right: e[1], down: e[2], left: e[3],
            })}
            {c.mod !== 0 ? ` ${t('table.bonus', { mod: c.mod > 0 ? `+${c.mod}` : String(c.mod) })}` : ''}
          </li>
        )
      })}
    </ol>
  )
}
