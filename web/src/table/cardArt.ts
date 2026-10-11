import { edgeText } from '../domain/board'

// The faces of the cards. ORIGINAL artwork drawn with canvas paths -- no picture of the game is used: a gradient by owner, the four edge numbers (A for 10),
// star pips, a small glyph for the card type, a badge for the bonus of a same-type rule. The same drawing is the texture of the 3D cards and the picture of the flat cards.

export type Side = 'mine' | 'theirs' | 'none'

export interface CardArt {
  /** the four values shown: up, right, down, left */
  edges: [number, number, number, number]
  side: Side
  star: number
  type: string | null
  /** bonus of a same-type rule (shown as a badge when it is not 0) */
  mod: number
  id: number | null
  /** drawn translucent: a card of a principal variation that has not been played */
  ghost?: boolean
  /** a number on the card (the ply of a ghost) */
  ply?: number | null
  /** the edges are all that is known (the card was identified by them, and other cards share them) */
  ambiguous?: boolean
}

export const PALETTE = {
  mineA: '#0d4a5c', mineB: '#23919c', mineEdge: '#7be0d6',
  theirsA: '#6b2236', theirsB: '#c0485f', theirsEdge: '#ffb3bd',
  noneA: '#27384d', noneB: '#4a6078', noneEdge: '#b9c7d6',
  gold: '#e1c27a', ink: '#06121c', paper: '#f4f1ea',
}

export function sideColors(side: Side): { a: string; b: string; edge: string } {
  return side === 'mine' ? { a: PALETTE.mineA, b: PALETTE.mineB, edge: PALETTE.mineEdge } : side === 'theirs' ? { a: PALETTE.theirsA, b: PALETTE.theirsB, edge: PALETTE.theirsEdge } : { a: PALETTE.noneA, b: PALETTE.noneB, edge: PALETTE.noneEdge }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function glyph(ctx: CanvasRenderingContext2D, type: string | null, cx: number, cy: number, r: number) {
  ctx.save()
  ctx.translate(cx, cy)
  ctx.lineWidth = r * 0.14
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.strokeStyle = 'rgba(255,255,255,0.34)'
  ctx.fillStyle = 'rgba(255,255,255,0.14)'
  switch (type) {
    case '兽人': // three claw marks
      for (const dx of [-0.5, 0, 0.5]) {
        ctx.beginPath()
        ctx.moveTo(dx * r - r * 0.25, -r * 0.8)
        ctx.lineTo(dx * r + r * 0.25, r * 0.8)
        ctx.stroke()
      }
      break
    case '拂晓': // a rising sun
      ctx.beginPath()
      ctx.arc(0, r * 0.35, r * 0.55, Math.PI, 0)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      for (let i = 0; i < 5; i++) {
        const a = Math.PI + (i + 0.5) * (Math.PI / 5)
        ctx.beginPath()
        ctx.moveTo(Math.cos(a) * r * 0.72, r * 0.35 + Math.sin(a) * r * 0.72)
        ctx.lineTo(Math.cos(a) * r * 1.0, r * 0.35 + Math.sin(a) * r * 1.0)
        ctx.stroke()
      }
      break
    case '帝国': // a hexagon with a triangle
      ctx.beginPath()
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i - Math.PI / 2
        ctx[i === 0 ? 'moveTo' : 'lineTo'](Math.cos(a) * r * 0.95, Math.sin(a) * r * 0.95)
      }
      ctx.closePath()
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, -r * 0.5)
      ctx.lineTo(r * 0.45, r * 0.35)
      ctx.lineTo(-r * 0.45, r * 0.35)
      ctx.closePath()
      ctx.fill()
      break
    case '蛮神': // an eight-pointed burst
      ctx.beginPath()
      for (let i = 0; i < 16; i++) {
        const a = (Math.PI / 8) * i
        const rr = i % 2 === 0 ? r : r * 0.45
        ctx[i === 0 ? 'moveTo' : 'lineTo'](Math.cos(a) * rr, Math.sin(a) * rr)
      }
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      break
    default: // no type: a ring and a wave
      ctx.beginPath()
      ctx.arc(0, 0, r * 0.7, 0, Math.PI * 2)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(-r * 0.5, 0)
      ctx.bezierCurveTo(-r * 0.25, -r * 0.3, r * 0.25, r * 0.3, r * 0.5, 0)
      ctx.stroke()
  }
  ctx.restore()
}

function star(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath()
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (Math.PI / 5) * i
    const rr = i % 2 === 0 ? r : r * 0.42
    ctx[i === 0 ? 'moveTo' : 'lineTo'](cx + Math.cos(a) * rr, cy + Math.sin(a) * rr)
  }
  ctx.closePath()
  ctx.fill()
}

/** Draws a card face into a square of `size` pixels. */
export function drawCardFace(ctx: CanvasRenderingContext2D, size: number, art: CardArt): void {
  const s = size
  const c = sideColors(art.side)
  ctx.save()
  ctx.clearRect(0, 0, s, s)
  if (art.ghost) ctx.globalAlpha = 0.62

  const pad = s * 0.04
  roundRect(ctx, pad, pad, s - pad * 2, s - pad * 2, s * 0.09)
  const g = ctx.createLinearGradient(0, 0, s, s)
  g.addColorStop(0, c.a)
  g.addColorStop(1, c.b)
  ctx.fillStyle = g
  ctx.fill()
  ctx.lineWidth = s * (art.star >= 5 ? 0.035 : 0.022)
  ctx.strokeStyle = art.star >= 5 ? PALETTE.gold : art.star === 4 ? '#cfd8e3' : c.edge
  ctx.stroke()

  // inner glow and wave lines
  ctx.save()
  roundRect(ctx, pad, pad, s - pad * 2, s - pad * 2, s * 0.09)
  ctx.clip()
  const glow = ctx.createRadialGradient(s / 2, s / 2, s * 0.05, s / 2, s / 2, s * 0.6)
  glow.addColorStop(0, 'rgba(255,255,255,0.22)')
  glow.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = glow
  ctx.fillRect(0, 0, s, s)
  ctx.strokeStyle = 'rgba(255,255,255,0.08)'
  ctx.lineWidth = s * 0.012
  for (let i = 0; i < 5; i++) {
    ctx.beginPath()
    for (let x = 0; x <= s; x += s / 16) {
      const y = s * (0.2 + i * 0.16) + Math.sin(x / s * Math.PI * 4 + i) * s * 0.025
      if (x === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  }
  ctx.restore()

  glyph(ctx, art.type, s / 2, s / 2, s * 0.17)

  // the four numbers, as the game shows them: up, right, down, left
  ctx.fillStyle = PALETTE.paper
  ctx.strokeStyle = 'rgba(2,10,16,0.85)'
  ctx.lineJoin = 'round'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const fontPx = s * 0.26
  ctx.font = `700 ${fontPx}px ui-sans-serif, "Segoe UI", system-ui, sans-serif`
  ctx.lineWidth = s * 0.05
  const place: [number, number][] = [[0.5, 0.2], [0.8, 0.5], [0.5, 0.8], [0.2, 0.5]]
  art.edges.forEach((v, i) => {
    const [px, py] = place[i]
    const text = edgeText(v)
    ctx.strokeText(text, s * px, s * py)
    ctx.fillText(text, s * px, s * py)
  })

  // stars
  ctx.fillStyle = PALETTE.gold
  const n = Math.max(0, Math.min(5, art.star))
  const sr = s * 0.032
  for (let i = 0; i < n; i++) star(ctx, s * 0.5 + (i - (n - 1) / 2) * s * 0.075, s * 0.935, sr)

  // id and bonus badge
  if (art.id !== null) {
    ctx.fillStyle = 'rgba(244,241,234,0.7)'
    ctx.font = `600 ${s * 0.075}px ui-sans-serif, system-ui, sans-serif`
    ctx.textAlign = 'left'
    ctx.fillText(`#${art.id}`, s * 0.1, s * 0.09)
  }
  if (art.mod !== 0) {
    ctx.beginPath()
    ctx.arc(s * 0.86, s * 0.14, s * 0.095, 0, Math.PI * 2)
    ctx.fillStyle = art.mod > 0 ? '#2fae6b' : '#d8534f'
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.font = `800 ${s * 0.1}px ui-sans-serif, system-ui, sans-serif`
    ctx.fillText(art.mod > 0 ? `+${art.mod}` : `${art.mod}`, s * 0.86, s * 0.145)
  }
  if (art.ambiguous) {
    ctx.fillStyle = PALETTE.gold
    ctx.textAlign = 'right'
    ctx.font = `800 ${s * 0.09}px ui-sans-serif, system-ui, sans-serif`
    ctx.fillText('?', s * 0.9, s * 0.09)
  }
  if (art.ply) {
    ctx.beginPath()
    ctx.arc(s * 0.5, s * 0.5, s * 0.11, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(6,18,28,0.82)'
    ctx.fill()
    ctx.fillStyle = PALETTE.paper
    ctx.textAlign = 'center'
    ctx.font = `800 ${s * 0.13}px ui-sans-serif, system-ui, sans-serif`
    ctx.fillText(String(art.ply), s * 0.5, s * 0.51)
  }
  ctx.restore()
}

/** The back of a card: waves and a star map. */
export function drawCardBack(ctx: CanvasRenderingContext2D, size: number): void {
  const s = size
  ctx.save()
  ctx.clearRect(0, 0, s, s)
  const pad = s * 0.04
  roundRect(ctx, pad, pad, s - pad * 2, s - pad * 2, s * 0.09)
  const g = ctx.createLinearGradient(0, 0, s, s)
  g.addColorStop(0, '#0b1d33')
  g.addColorStop(1, '#17436b')
  ctx.fillStyle = g
  ctx.fill()
  ctx.lineWidth = s * 0.022
  ctx.strokeStyle = PALETTE.gold
  ctx.stroke()
  ctx.strokeStyle = 'rgba(123,224,214,0.35)'
  ctx.lineWidth = s * 0.014
  for (let i = 0; i < 6; i++) {
    ctx.beginPath()
    for (let x = s * 0.08; x <= s * 0.92; x += s / 20) {
      const y = s * (0.25 + i * 0.1) + Math.sin(x / s * Math.PI * 5 + i) * s * 0.03
      if (x === s * 0.08) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  }
  ctx.fillStyle = 'rgba(244,241,234,0.8)'
  for (const [x, y] of [[0.3, 0.18], [0.5, 0.12], [0.7, 0.2], [0.22, 0.8], [0.6, 0.86], [0.8, 0.78]]) {
    ctx.beginPath()
    ctx.arc(s * x, s * y, s * 0.008, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}
