import { useEffect } from 'react'
import demo from '../demo/demo-game.json'
import { ReplayPlayer } from '../components/ReplayPlayer'
import { Link } from '../components/router'
import { useSession, useSettings } from '../components/context'
import { mascotUrl, useMood } from '../components/Mascot'
import type { LogRecord } from '../domain/types'

/** The page of a visitor who is not logged in: what this is, the demo replay of a sample game on the same table, and the way in. */
export function LandingPage() {
  const { t } = useSettings()
  const { status } = useSession()
  useMood({ expression: 'neutral', caption: 'mascot.landing', params: {} })
  useEffect(() => {
    document.title = 'TTC Siren'
  }, [])
  return (
    <div data-testid="landing-page">
      <section className="hero">
        <div>
          <h1>TTC Siren</h1>
          <p className="lead">{t('landing.lead')}</p>
          <p>{t('landing.body')}</p>
          <div className="row">
            {status === 'user' ? (
              <Link to="/live" className="button primary" data-testid="cta-live">{t('landing.openLive')}</Link>
            ) : (
              <Link to="/login" className="button primary" data-testid="cta-login">{t('landing.login')}</Link>
            )}
            <a className="button" href="#demo" data-testid="cta-demo">{t('landing.seeDemo')}</a>
          </div>
        </div>
        <img className="hero-mascot" src={mascotUrl('neutral', 1024)} alt={t('landing.mascotAlt')} width={420} />
      </section>

      <section className="features" aria-label={t('landing.featuresLabel')}>
        <article><h3>{t('landing.f1.title')}</h3><p>{t('landing.f1.body')}</p></article>
        <article><h3>{t('landing.f2.title')}</h3><p>{t('landing.f2.body')}</p></article>
        <article><h3>{t('landing.f3.title')}</h3><p>{t('landing.f3.body')}</p></article>
      </section>

      <section id="demo" className="demo" aria-labelledby="demo-title">
        <h2 id="demo-title">{t('landing.demoTitle')}</h2>
        <p className="muted">{t('landing.demoNote')}</p>
        <ReplayPlayer records={(demo as unknown as { records: LogRecord[] }).records} />
      </section>
    </div>
  )
}

export function NotFoundPage() {
  const { t } = useSettings()
  return (
    <section className="page narrow" data-testid="not-found">
      <h1>{t('notFound.title')}</h1>
      <p>{t('notFound.body')}</p>
      <Link to="/" className="button">{t('nav.home')}</Link>
    </section>
  )
}
