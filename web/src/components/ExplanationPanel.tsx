import { useState } from 'react'
import { useSettings } from './context'
import { cardInfo } from '../domain/cards'
import { edgeText } from '../domain/board'
import type { ConfidenceFact, Explanation, ExplanationText, Lang, OutcomeFact, RobustFact, UnknownFact } from '../domain/types'
import type { MessageKey } from '../i18n'

/** The text of an explanation in the language of the page: the log keeps every language (texts), an older record only the one it was written in. */
export function textIn(explanation: Explanation | null | undefined, lang: Lang): ExplanationText | null {
  if (!explanation) return null
  return explanation.texts?.[lang] ?? explanation.text ?? null
}

function fact<T>(explanation: Explanation, type: string): T | null {
  return (explanation.facts.find((f) => (f as { type?: string }).type === type) as T | undefined) ?? null
}

/** Every string below is shown as text (React escapes it): nothing the server sends is ever put into the page as HTML. */
export function ExplanationPanel({ explanation }: { explanation: Explanation | null | undefined }) {
  const { lang, t } = useSettings()
  const [full, setFull] = useState(false)
  if (!explanation) return <p className="muted" data-testid="explanation-none">{t('explain.none')}</p>
  const text = textIn(explanation, lang)
  const outcome = fact<OutcomeFact>(explanation, 'outcome')
  const robust = fact<RobustFact>(explanation, 'robust_endgame')
  const unknown = fact<UnknownFact>(explanation, 'unknown_cards')
  const confidence = fact<ConfidenceFact>(explanation, 'confidence')
  return (
    <section className="explanation" data-testid="explanation" aria-label={t('explain.title')}>
      <h3>{t('explain.title')}</h3>
      {text ? <p className="brief" data-testid="explanation-brief">{text.brief}</p> : null}
      <div className="badges">
        {outcome ? (
          <span className={`badge outcome-${outcome.kind}-${outcome.result}`} data-testid="outcome-badge">
            {outcome.kind === 'exact'
              ? t(`outcome.exact.${outcome.result}` as MessageKey, { diff: Math.abs(outcome.difference ?? 0) })
              : t(`outcome.heuristic.${outcome.result}` as MessageKey, { score: (outcome.score ?? 0).toFixed(2), depth: outcome.depth ?? 0 })}
          </span>
        ) : null}
        {robust ? <span className="badge" data-testid="robust-badge">{t('explain.robust', { scenarios: robust.scenarios, percent: robust.safety_percent })}</span> : null}
        {confidence ? (
          <span className="badge" data-testid="confidence-badge">
            {confidence.solved_to_end ? t('explain.solved') : t('explain.depth', { depth: confidence.depth_reached })}
          </span>
        ) : null}
      </div>
      {unknown && unknown.assumed.length > 0 ? (
        <div className="assumed" data-testid="assumed">
          <h4>{t('explain.assumed', { count: unknown.opponent_unknown })}</h4>
          <ul>
            {unknown.assumed.map((a) => {
              const info = cardInfo(a.card_id)
              return (
                <li key={`${a.slot}-${a.card_id}`}>
                  {t('explain.assumedCard', { id: a.card_id, star: a.star, edges: info ? info.edges.map(edgeText).join(' ') : '?' })}
                </li>
              )
            })}
          </ul>
        </div>
      ) : null}
      {text ? (
        <>
          <button type="button" className="link-button" onClick={() => setFull(!full)} aria-expanded={full}>
            {full ? t('explain.hideFull') : t('explain.showFull')}
          </button>
          {full ? (
            <div className="full" data-testid="explanation-full">
              {text.full.split('\n\n').map((p, i) => (
                <p key={i}>{p.split('\n').map((line, j) => (j === 0 ? line : [<br key={j} />, line]))}</p>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  )
}
