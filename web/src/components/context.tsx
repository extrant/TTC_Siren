import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import type { Lang } from '../domain/types'
import { translatorFor, type Translate } from '../i18n'
import { fetchMe, login as apiLogin, logout as apiLogout, type Me } from '../api/client'
import { animationsOn, loadSettings, saveSettings, type Settings } from '../state/settings'

// ---------------------------------------------------------------- settings and language

interface SettingsValue {
  settings: Settings
  update: (patch: Partial<Settings>) => void
  lang: Lang
  t: Translate
  /** animations are played (the settings and the system preference agree) */
  animate: boolean
}

const SettingsContext = createContext<SettingsValue | null>(null)

function useSystemReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => (typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false))
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const q = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setReduced(q.matches)
    q.addEventListener('change', on)
    return () => q.removeEventListener('change', on)
  }, [])
  return reduced
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(() => loadSettings(safeStorage(), navigator.language))
  const systemReduced = useSystemReducedMotion()
  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((s) => {
      const next = { ...s, ...patch }
      saveSettings(safeStorage(), next)
      return next
    })
  }, [])
  useEffect(() => {
    document.documentElement.lang = settings.lang === 'zh-CN' ? 'zh-CN' : 'en'
  }, [settings.lang])
  const animate = animationsOn(settings, systemReduced)
  useEffect(() => {
    document.documentElement.dataset.motion = animate ? 'full' : 'reduced'
  }, [animate])
  const value = useMemo<SettingsValue>(() => ({ settings, update, lang: settings.lang, t: translatorFor(settings.lang), animate }), [settings, update, animate])
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

function safeStorage(): Storage | null {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

export function useSettings(): SettingsValue {
  const v = useContext(SettingsContext)
  if (!v) throw new Error('SettingsProvider is missing')
  return v
}

// ---------------------------------------------------------------- who is logged in

export type SessionStatus = 'unknown' | 'anonymous' | 'user'

interface SessionValue {
  status: SessionStatus
  me: Me | null
  /** exchanges the key for a session cookie and asks who we are; throws ApiError on failure. The key is not kept. */
  login: (key: string) => Promise<void>
  logout: () => Promise<void>
  refresh: () => Promise<void>
}

const SessionContext = createContext<SessionValue | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ status: SessionStatus; me: Me | null }>({ status: 'unknown', me: null })
  const refresh = useCallback(async () => {
    try {
      const me = await fetchMe()
      setState(me ? { status: 'user', me } : { status: 'anonymous', me: null })
    } catch {
      setState({ status: 'anonymous', me: null })
    }
  }, [])
  useEffect(() => {
    void refresh()
  }, [refresh])
  const login = useCallback(
    async (key: string) => {
      await apiLogin(key)
      await refresh()
    },
    [refresh],
  )
  const logout = useCallback(async () => {
    try {
      await apiLogout()
    } finally {
      setState({ status: 'anonymous', me: null })
    }
  }, [])
  const value = useMemo(() => ({ ...state, login, logout, refresh }), [state, login, logout, refresh])
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession(): SessionValue {
  const v = useContext(SessionContext)
  if (!v) throw new Error('SessionProvider is missing')
  return v
}
