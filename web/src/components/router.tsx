import { createContext, useCallback, useContext, useEffect, useMemo, useState, type AnchorHTMLAttributes, type ReactNode } from 'react'

// A router of about forty lines: the history API, no library. The server answers the page for any path of the UI, so a reload or a pasted link works.

export interface Route {
  path: string
  /** the segments of the path: '/games/g-1' -> ['games', 'g-1'] */
  segments: string[]
  search: URLSearchParams
}

export function parseLocation(pathname: string, search: string): Route {
  const clean = pathname.replace(/\/+$/, '') || '/'
  return { path: clean, segments: clean.split('/').filter(Boolean), search: new URLSearchParams(search) }
}

interface RouterValue {
  route: Route
  navigate: (to: string, options?: { replace?: boolean }) => void
}

const RouterContext = createContext<RouterValue | null>(null)

export function RouterProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(() => parseLocation(window.location.pathname, window.location.search))
  useEffect(() => {
    const onPop = () => setRoute(parseLocation(window.location.pathname, window.location.search))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])
  const navigate = useCallback((to: string, options?: { replace?: boolean }) => {
    const url = new URL(to, window.location.origin)
    if (url.origin !== window.location.origin) return
    if (options?.replace) window.history.replaceState(null, '', url.pathname + url.search)
    else window.history.pushState(null, '', url.pathname + url.search)
    setRoute(parseLocation(url.pathname, url.search))
    window.scrollTo(0, 0)
  }, [])
  const value = useMemo(() => ({ route, navigate }), [route, navigate])
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>
}

export function useRouter(): RouterValue {
  const v = useContext(RouterContext)
  if (!v) throw new Error('RouterProvider is missing')
  return v
}

export function Link({ to, onClick, ...rest }: { to: string } & AnchorHTMLAttributes<HTMLAnchorElement>) {
  const { navigate } = useRouter()
  return (
    <a
      href={to}
      {...rest}
      onClick={(e) => {
        onClick?.(e)
        if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
        e.preventDefault()
        navigate(to)
      }}
    />
  )
}
