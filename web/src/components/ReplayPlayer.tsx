import { useEffect, useMemo, useRef, useState } from 'react'
import { BoardView } from './BoardView'
import { ExplanationPanel } from './ExplanationPanel'
import { useSettings } from './context'
import { useMood } from './Mascot'
import { effectiveEdges, edgeText } from '../domain/board'
import { buildTimeline, immediateEffect, type Step } from '../domain/timeline'
import type { LogRecord, PvPly, OutcomeFact } from '../domain/types'
import type { MessageKey } from '../i18n'
import type { Mood } from '../state/mood'
import { ghostsFrom, type TableView } from '../table/boardModel'
import { planFlips, totalMs } from '../table/flipEffects'

function pvOf(step: Step | undefined): PvPly[] {
  const f = step?.explanation?.facts.find((x) => (x as { type?: string }).type === 'outcome') as OutcomeFact | undefined
  return f?.pv ?? []
}

/** What the mascot shows for a step of a replay (neutral captions that say what the record says). */
function moodForStep(step: Step | undefined): Mood | null {
  if (!step) return null
  if (step.kind === 'end' && step.end) {
    const w = step.end.winner
    return { expression: w === 'mine' ? 'happy' : w === 'draw' ? 'neutral' : 'disappointed', caption: w === 'mine' ? 'mascot.won' : w === 'draw' ? 'mascot.draw' : 'mascot.lost', params: { mine: step.end.final.mine, theirs: step.end.final.theirs } }
  }
  if (step.kind === 'opponent_move' && step.flips.some((f) => f.cause === 'combo')) return { expression: 'surprised', caption: 'mascot.combo', params: { n: step.flips.length } }
  if (step.kind === 'our_move' && step.explanation) return { expression: 'explaining', caption: 'mascot.explaining', params: {} }
  return { expression: 'neutral', caption: 'mascot.idle', params: {} }
}

export function ReplayPlayer({ records, autoPlay = false }: { records: readonly LogRecord[]; autoPlay?: boolean }) {
  const { t, animate, lang } = useSettings()
  const timeline = useMemo(() => buildTimeline(records), [records])
  const steps = timeline.steps
  const [index, setIndex] = useState(0)
  const [nonce, setNonce] = useState(0)
  const [playing, setPlaying] = useState(autoPlay)
  const [speed, setSpeed] = useState(1)
  const count = steps.length
  const clamped = Math.min(index, Math.max(0, count - 1))
  const step = steps[clamped]

  // a new log (another game) starts at its first step
  const gameKey = timeline.gameId ?? ''
  const lastGame = useRef(gameKey)
  useEffect(() => {
    if (lastGame.current !== gameKey) {
      lastGame.current = gameKey
      setIndex(0)
      setPlaying(autoPlay)
    }
  }, [gameKey, autoPlay])

  // autoplay: the next step comes when the animation of this one has played
  useEffect(() => {
    if (!playing) return
    if (clamped >= count - 1) {
      setPlaying(false)
      return
    }
    const wait = (animate ? totalMs(planFlips(step?.flips ?? [])) + 700 : 900) / speed
    const id = window.setTimeout(() => {
      setIndex(clamped + 1)
      setNonce((n) => n + 1)
    }, wait)
    return () => window.clearTimeout(id)
  }, [playing, clamped, count, step, animate, speed])

  useMood(moodForStep(step))

  const go = (i: number) => {
    setIndex(Math.max(0, Math.min(count - 1, i)))
    setNonce((n) => n + 1)
  }

  const view: TableView | null = useMemo(() => {
    if (!step) return null
    const pv = pvOf(step)
    return {
      board: step.board, mine: timeline.mine, animKey: `${timeline.gameId}:${step.index}:${nonce}`, placed: step.placed, flips: step.flips,
      ghosts: step.kind === 'our_move' ? ghostsFrom(pv.slice(1), step.board) : [], marked: step.placed ? step.placed.row * 3 + step.placed.col : null,
    }
  }, [step, timeline.mine, timeline.gameId, nonce])

  if (!step || !view) return <p className="muted" data-testid="replay-empty">{t('replay.empty')}</p>

  const label = (s: Step): string => {
    const cell = s.placed ? t(`cell.${s.placed.row * 3 + s.placed.col}` as MessageKey) : ''
    switch (s.kind) {
      case 'position': return t('replay.step.position', { no: s.requestNo ?? '?' })
      case 'our_move': return t('replay.step.ours', { id: s.placed?.card.id ?? '?', cell, n: s.flips.length })
      case 'opponent_move':
        return s.placed ? t('replay.step.theirs', { id: s.placed.ambiguous ? '?' : s.placed.card.id ?? '?', cell, n: s.flips.length }) : t('replay.step.theirsUnknown')
      case 'end': return t('replay.step.end', { mine: s.end?.final.mine ?? 0, theirs: s.end?.final.theirs ?? 0 })
    }
  }

  const effect = immediateEffect(step.explanation)
  const result = step.request?.result

  return (
    <div className="replay" data-testid="replay" data-step={clamped} data-step-kind={step.kind}>
      <div className="replay-table">
        <BoardView view={view} />
        <div className="replay-controls" role="group" aria-label={t('replay.controls')}>
          <button type="button" onClick={() => go(0)} disabled={clamped === 0} aria-label={t('replay.first')} data-testid="replay-first">⏮</button>
          <button type="button" onClick={() => go(clamped - 1)} disabled={clamped === 0} aria-label={t('replay.prev')} data-testid="replay-prev">◀</button>
          <button type="button" onClick={() => setPlaying(!playing)} aria-pressed={playing} aria-label={playing ? t('replay.pause') : t('replay.play')} data-testid="replay-play">
            {playing ? '⏸' : '▶'}
          </button>
          <button type="button" onClick={() => go(clamped + 1)} disabled={clamped >= count - 1} aria-label={t('replay.next')} data-testid="replay-next">▶|</button>
          <button type="button" onClick={() => go(count - 1)} disabled={clamped >= count - 1} aria-label={t('replay.last')} data-testid="replay-last">⏭</button>
          <label className="speed">
            {t('replay.speed')}
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} data-testid="replay-speed">
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
          </label>
        </div>
        <input
          type="range" min={0} max={count - 1} value={clamped} onChange={(e) => go(Number(e.target.value))} className="scrubber" data-testid="replay-scrubber"
          aria-label={t('replay.scrubber')} aria-valuetext={t('replay.of', { i: clamped + 1, n: count })}
        />
        <p className="muted" data-testid="replay-position">{t('replay.of', { i: clamped + 1, n: count })} · {step.counts.mine}–{step.counts.theirs}</p>
      </div>

      <div className="replay-side">
        <section aria-label={t('replay.detail')} data-testid="step-detail">
          <h3>{label(step)}</h3>
          <p className="muted">{step.t}</p>
          {step.placed ? (
            <p data-testid="placed-card">
              {t(step.placed.by === 'mine' ? 'replay.placedMine' : 'replay.placedTheirs', {
                cell: t(`cell.${step.placed.row * 3 + step.placed.col}` as MessageKey),
                edges: effectiveEdges(step.placed.card).map(edgeText).join(' '),
              })}
            </p>
          ) : null}
          {step.placed?.ambiguous ? (
            <p className="note" data-testid="ambiguous-note">{t('replay.ambiguous', { ids: step.placed.candidateIds.join(', '), edges: step.placed.card.base.map(edgeText).join(' ') })}</p>
          ) : null}
          {step.inconsistent ? <p className="note" data-testid="inconsistent-note">{t('replay.inconsistent')}{step.note ? ` (${step.note})` : ''}</p> : null}
          {step.flips.length > 0 ? (
            <ul className="flips" data-testid="flip-list">
              {step.flips.map((f, i) => (
                <li key={i} data-cause={f.cause ?? 'none'}>
                  {t('replay.flip', { cell: t(`cell.${f.row * 3 + f.col}` as MessageKey), cause: f.cause ? t(`cause.${f.cause}` as MessageKey) : t('cause.unknown') })}
                  {f.myValue !== undefined && f.theirValue !== undefined ? ` (${f.myValue} : ${f.theirValue})` : ''}
                </li>
              ))}
            </ul>
          ) : step.kind === 'our_move' || step.kind === 'opponent_move' ? (
            <p className="muted">{t('replay.noFlips')}</p>
          ) : null}
          {effect && effect.modifier_changes.length > 0 ? <p className="muted">{t('replay.modifier', { n: effect.modifier_changes.length })}</p> : null}
          {result ? (
            <dl className="solver" data-testid="solver-data">
              <dt>{t('live.depth')}</dt><dd>{result.depth_reached}</dd>
              <dt>{t('live.score')}</dt><dd>{result.standard_score === null ? '–' : result.standard_score.toFixed(2)}</dd>
              <dt>{t('live.nodes')}</dt><dd>{result.nodes.toLocaleString('en-US')}</dd>
              <dt>{t('live.elapsed')}</dt><dd>{Math.round(result.search_ms)} ms</dd>
            </dl>
          ) : null}
        </section>
        {step.kind === 'our_move' ? <ExplanationPanel explanation={step.explanation} /> : null}
        <section aria-label={t('replay.timeline')}>
          <h3>{t('replay.timeline')}</h3>
          <ol className="timeline" data-testid="timeline" data-lang={lang}>
            {steps.map((s) => (
              <li key={s.index}>
                <button type="button" className={s.index === clamped ? 'current' : ''} aria-current={s.index === clamped ? 'step' : undefined} onClick={() => go(s.index)} data-testid={`timeline-${s.index}`}>
                  <span className="kind">{label(s)}</span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  )
}
