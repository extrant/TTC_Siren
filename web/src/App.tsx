import { MascotDock, MascotProvider } from './components/Mascot'
import { Link, RouterProvider, useRouter } from './components/router'
import { SessionProvider, SettingsProvider, useSession, useSettings } from './components/context'
import { GamesPage, GameViewPage } from './pages/Games'
import { LandingPage, NotFoundPage } from './pages/Landing'
import { LivePage } from './pages/Live'
import { LoginPage } from './pages/Login'
import { SettingsPage } from './pages/Settings'

function Header() {
  const { t, settings, update } = useSettings()
  const { status, me, logout } = useSession()
  const { route, navigate } = useRouter()
  const nav = (to: string, label: string, test: string) => (
    <Link to={to} aria-current={route.path === to || (to !== '/' && route.path.startsWith(to + '/')) ? 'page' : undefined} data-testid={test}>
      {label}
    </Link>
  )
  return (
    <header className="site-header">
      <Link to="/" className="brand" data-testid="brand">TTC Siren</Link>
      <nav aria-label={t('nav.main')}>
        {nav('/', t('nav.home'), 'nav-home')}
        {nav('/live', t('nav.live'), 'nav-live')}
        {nav('/games', t('nav.games'), 'nav-games')}
        {nav('/settings', t('nav.settings'), 'nav-settings')}
      </nav>
      <div className="header-actions">
        <button type="button" onClick={() => update({ lang: settings.lang === 'zh-CN' ? 'en' : 'zh-CN' })} data-testid="lang-toggle" aria-label={t('nav.language')} lang={settings.lang === 'zh-CN' ? 'en' : 'zh-CN'}>
          {settings.lang === 'zh-CN' ? 'English' : '中文'}
        </button>
        {status === 'user' ? (
          <>
            <span className="who" data-testid="who">{me?.key.name}</span>
            <button type="button" onClick={() => void logout().then(() => navigate('/'))} data-testid="logout">{t('nav.logout')}</button>
          </>
        ) : status === 'anonymous' ? (
          <Link to="/login" className="button" data-testid="nav-login">{t('nav.login')}</Link>
        ) : null}
      </div>
    </header>
  )
}

function Routes() {
  const { route } = useRouter()
  const [first, second] = route.segments
  if (route.path === '/') return <LandingPage />
  if (first === 'login' && !second) return <LoginPage />
  if (first === 'live' && !second) return <LivePage />
  if (first === 'games' && !second) return <GamesPage />
  if (first === 'games' && second && route.segments.length === 2) return <GameViewPage id={decodeURIComponent(second)} />
  if (first === 'settings' && !second) return <SettingsPage />
  return <NotFoundPage />
}

function Shell() {
  const { t } = useSettings()
  const { route } = useRouter()
  return (
    <>
      <a className="skip" href="#main">{t('nav.skip')}</a>
      <Header />
      <main id="main" tabIndex={-1}>
        <Routes />
      </main>
      <footer className="site-footer">
        <span>{t('footer.text')}</span>
      </footer>
      {route.path !== '/' && <MascotDock />}
    </>
  )
}

export function App() {
  return (
    <SettingsProvider>
      <RouterProvider>
        <SessionProvider>
          <MascotProvider>
            <Shell />
          </MascotProvider>
        </SessionProvider>
      </RouterProvider>
    </SettingsProvider>
  )
}
