import { useEffect, useMemo, useState } from 'react'
import { CardCanvas } from './CardCanvas'
import { useSettings } from './context'
import { planFlips } from '../table/flipEffects'
import { artFor, ghostArt, type TableView } from '../table/boardModel'
import type { MessageKey } from '../i18n'

/** The flat table: the same information as the 3D one (card faces, owners, bonus badges, the principal variation as ghost cards), with CSS animations per cause. */
export function Board2D({ view }: { view: TableView }) {
  const { t, animate } = useSettings()
  const plan = useMemo(() => planFlips(view.flips), [view.flips])
  // the animation classes are set for the time of the animation only, so that the same step can be played again
  const [playing, setPlaying] = useState<string | null>(null)
  useEffect(() => {
    if (!animate) {
      setPlaying(null)
      return
    }
    setPlaying(view.animKey)
    const end = Math.max(...plan.map((p) => p.startMs + p.spec.durationMs), 700) + 100
    const id = window.setTimeout(() => setPlaying(null), end)
    return () => window.clearTimeout(id)
  }, [view.animKey, animate, plan])

  return (
    <div className="board2d" role="group" aria-label={t('table.label')} data-testid="board2d" data-anim-key={view.animKey}>
      {Array.from({ length: 9 }, (_, cell) => {
        const art = artFor(view.board, cell, view.mine, view.placed)
        const ghost = view.ghosts.find((g) => g.cell === cell)
        const flip = plan.find((p) => p.cell === cell)
        const isPlaced = view.placed !== null && view.placed.row * 3 + view.placed.col === cell
        const classes = ['cell']
        if (art) classes.push('has-card', `owner-${art.side}`)
        if (view.marked === cell) classes.push('marked')
        if (playing && flip) classes.push('flip', `flip-${flip.flip.cause ?? 'basic'}`)
        if (playing && isPlaced) classes.push('placed')
        return (
          <div
            key={cell}
            className={classes.join(' ')}
            data-testid={`cell-${cell}`}
            data-owner={art?.side ?? 'empty'}
            data-card-id={art?.id ?? ''}
            data-flip={flip && playing ? (flip.flip.cause ?? 'basic') : undefined}
            style={flip && playing ? { animationDelay: `${flip.startMs}ms` } : isPlaced && playing ? { animationDelay: '0ms' } : undefined}
          >
            {art ? <CardCanvas art={art} /> : ghost ? <CardCanvas art={ghostArt(ghost)} /> : <span className="cell-name">{t(`cell.${cell}` as MessageKey)}</span>}
          </div>
        )
      })}
    </div>
  )
}
