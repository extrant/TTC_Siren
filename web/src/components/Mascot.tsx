import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useSettings } from './context'
import { neutralMood, type Expression, type Mood } from '../state/mood'

// Seren, the mascot. The pictures are the originals of assets/mascot/raw, converted at build time to WebP (padded, never cropped).
const modules = import.meta.glob('../generated/mascot/*.512.webp', { eager: true, query: '?url', import: 'default' }) as Record<string, string>
const big = import.meta.glob('../generated/mascot/e01_neutral.1024.webp', { eager: true, query: '?url', import: 'default' }) as Record<string, string>

const FILE: Record<Expression, string> = {
  neutral: 'e01_neutral', thinking: 'e02_thinking', confident: 'e03_confident', surprised: 'e04_surprised',
  worried: 'e05_worried', happy: 'e06_happy', explaining: 'e07_explaining', disappointed: 'e08_disappointed',
}

export function mascotUrl(expression: Expression, size: 512 | 1024 = 512): string {
  const table = size === 512 ? modules : big
  const found = Object.entries(table).find(([k]) => k.includes(`/${FILE[expression]}.`))
  // a missing expression falls back to the neutral one
  return found ? found[1] : Object.entries(table).find(([k]) => k.includes(`/${FILE.neutral}.`))?.[1] ?? ''
}

interface MascotContextValue {
  mood: Mood
  setMood: (m: Mood) => void
}

const MascotContext = createContext<MascotContextValue | null>(null)

export function MascotProvider({ children }: { children: ReactNode }) {
  const [mood, setMood] = useState<Mood>(neutralMood())
  const value = useMemo(() => ({ mood, setMood }), [mood])
  return <MascotContext.Provider value={value}>{children}</MascotContext.Provider>
}

/** A page tells the mascot how to feel; it goes back to neutral when the page goes away. */
export function useMood(mood: Mood | null): void {
  const ctx = useContext(MascotContext)
  const key = mood ? `${mood.expression}|${mood.caption}|${JSON.stringify(mood.params)}` : ''
  useEffect(() => {
    if (!ctx || !mood) return
    ctx.setMood(mood)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  useEffect(() => () => ctx?.setMood(neutralMood()), []) // eslint-disable-line react-hooks/exhaustive-deps
}

/** The dock in the corner: the picture of the current expression, a crossfade of about 150 ms, a short neutral caption, a button to hide it. */
export function MascotDock() {
  const ctx = useContext(MascotContext)
  const { settings, update, t } = useSettings()
  const [shown, setShown] = useState<Expression>('neutral')
  const [previous, setPrevious] = useState<Expression | null>(null)
  const expression = ctx?.mood.expression ?? 'neutral'
  useEffect(() => {
    if (expression === shown) return
    setPrevious(shown)
    setShown(expression)
    const id = window.setTimeout(() => setPrevious(null), 200)
    return () => window.clearTimeout(id)
  }, [expression, shown])
  if (!settings.mascot || !ctx) return null
  return (
    <aside className="mascot-dock" aria-label={t('mascot.label')} data-testid="mascot" data-expression={expression}>
      <button type="button" className="mascot-hide" onClick={() => update({ mascot: false })} aria-label={t('mascot.hide')} title={t('mascot.hide')}>
        ×
      </button>
      <p className="mascot-caption" role="status" data-testid="mascot-caption">{t(ctx.mood.caption, ctx.mood.params)}</p>
      <div className="mascot-pic">
        {previous ? <img className="mascot-img fading" src={mascotUrl(previous)} alt="" /> : null}
        <img className="mascot-img" src={mascotUrl(shown)} alt={t('mascot.alt', { expression: t(`mascot.expression.${shown}` as never) })} />
      </div>
    </aside>
  )
}
