import { flipOrder, type FlipStep } from '../domain/timeline'
import type { Cause } from '../domain/types'

// One recognisable effect per cause of a flip (decision A-29). The same plan drives the 3D cards and the CSS classes of the flat table.

export interface EffectSpec {
  cause: Cause
  durationMs: number
  /** the axis the card turns about (3D), in the table's coordinates: x to the right, z towards the viewer */
  axis: [number, number, number]
  /** half turns: 1 = the card turns once and shows its other side */
  halfTurns: number
  /** how high the card rises while it turns (table units) */
  lift: number
  /** a ring that spreads from the card */
  ring: { color: string; scale: number } | null
  /** floating text that rises from the card: the sum of Plus, the arrow of Reverse */
  sprite: 'sum' | 'reverse' | null
  /** the table shakes when the card lands (Fallen Ace) */
  shake: boolean
  /** an arc from the card that set this flip off (Combo) */
  arc: boolean
  color: string
}

export const EFFECTS: Record<Cause, EffectSpec> = {
  basic: { cause: 'basic', durationMs: 620, axis: [0, 0, 1], halfTurns: 1, lift: 0.35, ring: null, sprite: null, shake: false, arc: false, color: '#9ad8ff' },
  same: { cause: 'same', durationMs: 780, axis: [1, 0, 0], halfTurns: 3, lift: 0.25, ring: { color: '#7be0d6', scale: 2.4 }, sprite: null, shake: false, arc: false, color: '#7be0d6' },
  plus: { cause: 'plus', durationMs: 780, axis: [0, 0, 1], halfTurns: 3, lift: 0.6, ring: null, sprite: 'sum', shake: false, arc: false, color: '#ffd37a' },
  combo: { cause: 'combo', durationMs: 880, axis: [1, 0, 1], halfTurns: 3, lift: 0.45, ring: null, sprite: null, shake: false, arc: true, color: '#c79bff' },
  reverse: { cause: 'reverse', durationMs: 820, axis: [1, 0, -1], halfTurns: 1, lift: 0.5, ring: { color: '#c79bff', scale: 1.8 }, sprite: 'reverse', shake: false, arc: false, color: '#b58cff' },
  fallen_ace: { cause: 'fallen_ace', durationMs: 640, axis: [0, 0, 1], halfTurns: 1, lift: 1.3, ring: { color: '#ff6b6b', scale: 3.2 }, sprite: null, shake: true, arc: false, color: '#ff6b6b' },
}

/** A flip whose cause the log does not give (an opponent move the rules did not account for) is shown as a plain flip. */
export const specOf = (cause: Cause | null): EffectSpec => EFFECTS[cause ?? 'basic']

export interface PlannedFlip {
  flip: FlipStep
  cell: number
  spec: EffectSpec
  /** when it starts, in ms from the beginning of the step (the placement comes first) */
  startMs: number
  /** the cell of the card that set it off (Combo) */
  viaCell: number | null
}

export const PLACE_MS = 420
const STAGGER_MS = 150
const COMBO_GAP_MS = 90

/** Orders the flips of a step in time: independent flips follow one another with a short stagger; a combo starts only when the flip that set it off has ended. */
export function planFlips(flips: readonly FlipStep[]): PlannedFlip[] {
  const order = flipOrder(flips)
  const planned: PlannedFlip[] = []
  let next = PLACE_MS
  const startOf: number[] = []
  const endOf: number[] = []
  order.forEach(({ flip, after }, i) => {
    const spec = specOf(flip.cause)
    let start = next
    if (after !== null && endOf[after] !== undefined) start = Math.max(start, endOf[after] + COMBO_GAP_MS)
    startOf[i] = start
    endOf[i] = start + spec.durationMs
    next = start + STAGGER_MS
    planned.push({
      flip, cell: flip.row * 3 + flip.col, spec, startMs: start,
      viaCell: flip.viaRow !== undefined && flip.viaCol !== undefined && flip.viaRow >= 0 ? flip.viaRow * 3 + flip.viaCol : null,
    })
  })
  return planned
}

export const totalMs = (plan: readonly PlannedFlip[]): number => plan.reduce((m, p) => Math.max(m, p.startMs + p.spec.durationMs), PLACE_MS)

/** The easing of the turn: slow start, quick middle, soft landing. */
export const ease = (x: number): number => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x))
