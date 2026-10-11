// Turns the originals of the mascot (assets/mascot/raw/seren_e01_neutral.png ...) into WebP files of two sizes for the UI (src/generated/mascot/e01_neutral.512.webp, .1024.webp).
// The originals are never changed. The art reaches the edges of the picture, so a transparent margin is ADDED around it before the picture is scaled -- it is padded, never cropped.
import { mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import sharp from 'sharp'

const here = dirname(fileURLToPath(import.meta.url))
const rawDir = join(here, '..', '..', 'assets', 'mascot', 'raw')
const outDir = join(here, '..', 'src', 'generated', 'mascot')
mkdirSync(outDir, { recursive: true })

const PAD = 0.06 // of the width, on every side
const widths = [512, 1024]
const files = readdirSync(rawDir).filter((f) => /^seren_e\d\d_.+\.png$/.test(f)).sort()
if (files.length === 0) throw new Error(`no expression images in ${rawDir}`)

for (const file of files) {
  const name = file.replace(/^seren_/, '').replace(/\.png$/, '') // e01_neutral
  const image = sharp(join(rawDir, file))
  const { width, height } = await image.metadata()
  const pad = Math.round(width * PAD)
  const padded = await image
    .extend({ top: pad, bottom: pad, left: pad, right: pad, background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toBuffer()
  for (const w of widths) {
    await sharp(padded).resize({ width: w, fit: 'inside' }).webp({ quality: 84, alphaQuality: 92, effort: 5 }).toFile(join(outDir, `${name}.${w}.webp`))
  }
  console.log(`${name}: ${width}x${height} + ${pad}px margin -> ${widths.join(', ')}`)
}
