import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useSettings } from '../components/context'
import { artFor, ghostArt, type TableView } from './boardModel'
import { drawCardFace, type CardArt } from './cardArt'
import { ease, planFlips, PLACE_MS, totalMs, type PlannedFlip } from './flipEffects'

// The 3D table (deep-sea casino): a dark glass board of 3 x 3 plates on a round table; the cards are thin slabs whose faces are the drawing of cardArt.ts.
// What flips, in which order and for which reason comes from the plan of the facts (flipEffects.ts); nothing is worked out here.

const CELL = 1.12
const CARD = 0.92
const THICK = 0.045
const cellPos = (cell: number): [number, number] => [((cell % 3) - 1) * CELL, (Math.floor(cell / 3) - 1) * CELL]

const textures = new Map<string, THREE.CanvasTexture>()

function textureFor(key: string, draw: (ctx: CanvasRenderingContext2D, size: number) => void): THREE.CanvasTexture {
  let tex = textures.get(key)
  if (!tex) {
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 256
    const ctx = canvas.getContext('2d')!
    draw(ctx, 256)
    tex = new THREE.CanvasTexture(canvas)
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = 4
    textures.set(key, tex)
  }
  return tex
}

const faceTexture = (art: CardArt) => textureFor('f' + JSON.stringify(art), (ctx, s) => drawCardFace(ctx, s, art))

function labelTexture(text: string, color: string): THREE.CanvasTexture {
  return textureFor(`t${text}|${color}`, (ctx, s) => {
    ctx.clearRect(0, 0, s, s)
    ctx.font = `800 ${text.length > 4 ? 64 : 110}px ui-sans-serif, system-ui, sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.lineWidth = 14
    ctx.strokeStyle = 'rgba(2,10,16,0.9)'
    ctx.strokeText(text, s / 2, s / 2)
    ctx.fillStyle = color
    ctx.fillText(text, s / 2, s / 2)
  })
}

const swapSide = (art: CardArt): CardArt => ({ ...art, side: art.side === 'mine' ? 'theirs' : 'mine' })

interface CardProps {
  cell: number
  art: CardArt
  plan: PlannedFlip | undefined
  placed: boolean
  startedAt: number | null
  animate: boolean
}

function Card({ cell, art, plan, placed, startedAt, animate }: CardProps) {
  const root = useRef<THREE.Group>(null)
  const inner = useRef<THREE.Group>(null)
  const top = useRef<THREE.MeshBasicMaterial>(null)
  const bottom = useRef<THREE.MeshBasicMaterial>(null)
  const ring = useRef<THREE.Mesh>(null)
  const sprite = useRef<THREE.Sprite>(null)
  const [x, z] = cellPos(cell)
  const oldArt = useMemo(() => swapSide(art), [art])
  const finalTex = useMemo(() => faceTexture(art), [art])
  const oldTex = useMemo(() => faceTexture(oldArt), [oldArt])
  const label = useMemo(() => {
    if (!plan) return null
    if (plan.spec.sprite === 'sum') return labelTexture(plan.flip.sum !== undefined ? `${plan.flip.myValue}+${plan.flip.theirValue}=${plan.flip.sum}` : '+', '#ffd37a')
    if (plan.spec.sprite === 'reverse') return labelTexture('↺', '#c79bff')
    return null
  }, [plan])
  const axis = useMemo(() => new THREE.Vector3(...(plan?.spec.axis ?? [0, 0, 1])).normalize(), [plan])
  const quat = useMemo(() => new THREE.Quaternion(), [])

  useFrame(() => {
    const g = root.current
    const i = inner.current
    if (!g || !i || !top.current || !bottom.current) return
    const now = performance.now()
    const elapsed = startedAt === null || !animate ? Infinity : now - startedAt
    let lift = 0
    let theta = 0
    let showOld = false
    let ringP = -1
    // placement: the card falls onto the board
    if (placed && elapsed < PLACE_MS) {
      const p = ease(elapsed / PLACE_MS)
      lift = (1 - p) * 1.4
      g.scale.setScalar(0.85 + 0.15 * p)
    } else g.scale.setScalar(1)
    if (plan && elapsed !== Infinity) {
      const p = (elapsed - plan.startMs) / plan.spec.durationMs
      if (p < 0) showOld = true
      else if (p < 1) {
        theta = plan.spec.halfTurns * Math.PI * ease(p)
        lift = Math.max(lift, plan.spec.lift * Math.sin(Math.PI * p))
        showOld = Math.floor(theta / Math.PI) % 2 === 0
        ringP = p
        if (plan.spec.cause === 'fallen_ace') {
          // slam: rise fast, fall, the card turns at the moment of impact
          const q = p < 0.55 ? p / 0.55 : 1
          lift = plan.spec.lift * Math.sin((Math.PI / 2) * q) * (p < 0.55 ? 1 : 0)
          theta = p < 0.55 ? 0 : Math.PI * ease((p - 0.55) / 0.45)
          showOld = p < 0.78
        }
      }
    }
    quat.setFromAxisAngle(axis, theta)
    i.quaternion.copy(quat)
    g.position.y = lift
    // the face that points up is the old one before the turn is half done, the new one after; at the end the new face is on top with no turn left
    top.current.map = showOld && plan ? oldTex : finalTex
    bottom.current.map = showOld && plan ? finalTex : oldTex
    top.current.needsUpdate = false
    if (ring.current) {
      const m = ring.current.material as THREE.MeshBasicMaterial
      if (ringP >= 0 && plan?.spec.ring) {
        ring.current.visible = true
        const s = 0.4 + (plan.spec.ring.scale - 0.4) * ringP
        ring.current.scale.set(s, s, s)
        m.opacity = 0.85 * (1 - ringP)
      } else ring.current.visible = false
    }
    if (sprite.current) {
      if (ringP >= 0 && label) {
        sprite.current.visible = true
        sprite.current.position.y = 0.5 + 0.7 * ringP
        ;(sprite.current.material as THREE.SpriteMaterial).opacity = Math.min(1, 3 * (1 - ringP))
      } else sprite.current.visible = false
    }
  })

  return (
    <group position={[x, THICK / 2 + 0.02, z]}>
      <group ref={root}>
        <group ref={inner}>
          <mesh>
            <boxGeometry args={[CARD, THICK, CARD]} />
            <meshStandardMaterial color="#0a1b2b" roughness={0.5} metalness={0.2} />
          </mesh>
          <mesh position={[0, THICK / 2 + 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[CARD, CARD]} />
            <meshBasicMaterial ref={top} map={finalTex} transparent toneMapped={false} />
          </mesh>
          <mesh position={[0, -THICK / 2 - 0.002, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <planeGeometry args={[CARD, CARD]} />
            <meshBasicMaterial ref={bottom} map={oldTex} transparent toneMapped={false} />
          </mesh>
        </group>
        {plan?.spec.ring ? (
          <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]} visible={false}>
            <ringGeometry args={[0.45, 0.5, 48]} />
            <meshBasicMaterial color={plan.spec.ring.color} transparent opacity={0.8} side={THREE.DoubleSide} toneMapped={false} />
          </mesh>
        ) : null}
        {label ? (
          <sprite ref={sprite} scale={[1.1, 1.1, 1.1]} visible={false}>
            <spriteMaterial map={label} transparent toneMapped={false} depthTest={false} />
          </sprite>
        ) : null}
      </group>
    </group>
  )
}

function GhostCard({ cell, art }: { cell: number; art: CardArt }) {
  const [x, z] = cellPos(cell)
  const tex = useMemo(() => faceTexture(art), [art])
  return (
    <mesh position={[x, 0.03, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[CARD * 0.94, CARD * 0.94]} />
      <meshBasicMaterial map={tex} transparent opacity={0.7} toneMapped={false} depthWrite={false} />
    </mesh>
  )
}

function Combo({ plan, startedAt, animate }: { plan: PlannedFlip; startedAt: number | null; animate: boolean }) {
  const line = useMemo(() => {
    if (plan.viaCell === null) return null
    const [ax, az] = cellPos(plan.viaCell)
    const [bx, bz] = cellPos(plan.cell)
    const pts: THREE.Vector3[] = []
    for (let i = 0; i <= 24; i++) {
      const u = i / 24
      pts.push(new THREE.Vector3(ax + (bx - ax) * u, 0.12 + Math.sin(Math.PI * u) * 0.7, az + (bz - az) * u))
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(pts)
    const material = new THREE.LineBasicMaterial({ color: plan.spec.color, transparent: true, opacity: 0.9, toneMapped: false })
    return new THREE.Line(geometry, material)
  }, [plan])
  useFrame(() => {
    if (!line) return
    const elapsed = startedAt === null || !animate ? Infinity : performance.now() - startedAt
    const p = (elapsed - plan.startMs + 260) / (plan.spec.durationMs + 260)
    line.visible = elapsed !== Infinity && p > 0 && p < 1.05
    line.geometry.setDrawRange(0, Math.max(2, Math.floor(25 * Math.min(1, Math.max(0, p * 1.4)))))
    ;(line.material as THREE.LineBasicMaterial).opacity = p > 0.85 ? Math.max(0, (1.05 - p) / 0.2) : 0.9
  })
  return line ? <primitive object={line} /> : null
}

function Scene({ view, animate }: { view: TableView; animate: boolean }) {
  const plan = useMemo(() => planFlips(view.flips), [view.flips])
  const startedAt = useMemo<number | null>(() => (animate ? performance.now() : null), [view.animKey, animate]) // eslint-disable-line react-hooks/exhaustive-deps
  const table = useRef<THREE.Group>(null)
  const shake = plan.some((p) => p.spec.shake)
  useFrame(() => {
    const g = table.current
    if (!g) return
    g.position.set(0, 0, 0)
    if (!shake || startedAt === null) return
    const ace = plan.find((p) => p.spec.shake)!
    const q = (performance.now() - startedAt - ace.startMs - ace.spec.durationMs * 0.78) / 260
    if (q > 0 && q < 1) g.position.set(Math.sin(q * 40) * 0.05 * (1 - q), 0, Math.cos(q * 33) * 0.04 * (1 - q))
  })
  return (
    <>
      <ambientLight intensity={0.9} />
      <pointLight position={[3, 5, 2]} intensity={40} color="#7be0d6" />
      <pointLight position={[-4, 4, -2]} intensity={26} color="#ff9db0" />
      <group ref={table}>
        <mesh position={[0, -0.12, 0]} rotation={[0, 0, 0]}>
          <cylinderGeometry args={[3.6, 3.8, 0.2, 64]} />
          <meshStandardMaterial color="#071624" roughness={0.45} metalness={0.4} />
        </mesh>
        <mesh position={[0, -0.01, 0]}>
          <boxGeometry args={[CELL * 3 + 0.2, 0.05, CELL * 3 + 0.2]} />
          <meshStandardMaterial color="#12324a" roughness={0.2} metalness={0.5} transparent opacity={0.9} />
        </mesh>
        {Array.from({ length: 9 }, (_, cell) => {
          const [x, z] = cellPos(cell)
          return (
            <mesh key={'p' + cell} position={[x, 0.02, z]}>
              <boxGeometry args={[CELL - 0.08, 0.02, CELL - 0.08]} />
              <meshStandardMaterial color={view.marked === cell ? '#2d8f95' : '#2b6f93'} roughness={0.15} metalness={0.3} transparent opacity={view.marked === cell ? 0.95 : 0.8} emissive={view.marked === cell ? '#1c6b70' : '#12405c'} />
            </mesh>
          )
        })}
        {view.board.map((_, cell) => {
          const art = artFor(view.board, cell, view.mine, view.placed)
          if (!art) return null
          const p = plan.find((x) => x.cell === cell)
          const placed = view.placed !== null && view.placed.row * 3 + view.placed.col === cell
          return <Card key={cell} cell={cell} art={art} plan={p} placed={placed} startedAt={startedAt} animate={animate} />
        })}
        {view.ghosts.map((g) => (
          <GhostCard key={'g' + g.cell} cell={g.cell} art={ghostArt(g)} />
        ))}
        {plan.filter((p) => p.spec.arc).map((p) => (
          <Combo key={'c' + p.cell} plan={p} startedAt={startedAt} animate={animate} />
        ))}
      </group>
    </>
  )
}

/** Keeps the 3D context alive for a while after the last animation, then stops rendering (the table is still, the GPU rests). */
function Idle({ view, animate }: { view: TableView; animate: boolean }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    if (!animate) {
      invalidate()
      return
    }
    const plan = planFlips(view.flips)
    const until = performance.now() + totalMs(plan) + 300
    let raf = 0
    const tick = () => {
      invalidate()
      if (performance.now() < until) raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [view.animKey, view.board, animate, invalidate, view.flips])
  return null
}

export default function Board3D({ view }: { view: TableView }) {
  const { animate, t } = useSettings()
  const causes = view.flips.map((f) => f.cause ?? 'basic').join(',')
  return (
    <div className="board3d" data-testid="board3d" data-anim-key={view.animKey} data-flip-causes={causes} role="img" aria-label={t('table.label')}>
      <Canvas frameloop="demand" dpr={[1, 2]} camera={{ position: [0, 5.6, 3.9], fov: 36 }} gl={{ antialias: true, alpha: true }} onCreated={({ camera }) => camera.lookAt(0, 0, 0.1)}>
        <Scene view={view} animate={animate} />
        <Idle view={view} animate={animate} />
      </Canvas>
    </div>
  )
}
