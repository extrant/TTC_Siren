import { Component, lazy, Suspense, useEffect, useState, type ReactNode } from 'react'
import { Board2D } from './Board2D'
import { StateList } from './StateList'
import { useSettings } from './context'
import type { TableView } from '../table/boardModel'

// three.js is large: the 3D table is loaded only when it is going to be shown, so the first page (landing, login) does not pay for it.
const Board3D = lazy(() => import('../table/Board3D'))

export function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window.matchMedia === 'function' ? window.matchMedia(query).matches : false))
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const q = window.matchMedia(query)
    const on = () => setMatches(q.matches)
    q.addEventListener('change', on)
    return () => q.removeEventListener('change', on)
  }, [query])
  return matches
}

/** The 3D table needs WebGL and a screen that is neither narrow nor touch-first; everything else, and a failure of the 3D scene, gets the flat table with the same information. */
export function useTableMode(): '3d' | '2d' {
  const { settings } = useSettings()
  const narrow = useMedia('(max-width: 720px)')
  const coarse = useMedia('(pointer: coarse)')
  const [webgl] = useState(() => webglAvailable())
  return settings.use3d && webgl && !narrow && !coarse ? '3d' : '2d'
}

class Fallback extends Component<{ view: TableView; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? <Board2D view={this.props.view} /> : this.props.children
  }
}

export function BoardView({ view }: { view: TableView }) {
  const mode = useTableMode()
  return (
    <div className="table-wrap" data-table-mode={mode}>
      {mode === '3d' ? (
        <Fallback view={view}>
          <Suspense fallback={<Board2D view={view} />}>
            <Board3D view={view} />
          </Suspense>
        </Fallback>
      ) : (
        <Board2D view={view} />
      )}
      <StateList view={view} />
    </div>
  )
}
