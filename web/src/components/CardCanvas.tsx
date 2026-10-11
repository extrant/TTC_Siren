import { useEffect, useRef } from 'react'
import { drawCardFace, type CardArt } from '../table/cardArt'

/** One card face as a picture. The same drawing is the texture of the 3D cards; the accessible name is given by the caller (the picture itself is decoration). */
export function CardCanvas({ art, size = 192, className }: { art: CardArt; size?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const key = JSON.stringify(art)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    drawCardFace(ctx, size, art)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, size])
  return <canvas ref={ref} width={size} height={size} className={className} aria-hidden="true" />
}
