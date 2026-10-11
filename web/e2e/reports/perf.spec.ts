import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from '@playwright/test'

// Frame rate of the table while a replay plays with its animations. On demand: npm run report
// The headless browser draws WebGL in software (SwiftShader), so these numbers are a floor, not what a GPU gives.

const out = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'docs', 'reports', 'phase8b')

async function measure(page: import('@playwright/test').Page, ms: number) {
  return page.evaluate(async (duration) => {
    const stamps: number[] = []
    await new Promise<void>((resolve) => {
      const start = performance.now()
      const tick = (t: number) => {
        stamps.push(t)
        if (t - start < duration) requestAnimationFrame(tick)
        else resolve()
      }
      requestAnimationFrame(tick)
    })
    const deltas = stamps.slice(1).map((t, i) => t - stamps[i]).sort((a, b) => a - b)
    const total = stamps[stamps.length - 1] - stamps[0]
    return { frames: deltas.length, fps: Math.round((deltas.length / total) * 10000) / 10, p50_ms: deltas[Math.floor(deltas.length * 0.5)], p95_ms: deltas[Math.floor(deltas.length * 0.95)], max_ms: deltas[deltas.length - 1] }
  }, ms)
}

test('frame rate of the 3D and the flat table during a replay', async ({ context }) => {
  const result: Record<string, unknown> = { browser: 'Microsoft Edge (headless), WebGL in software (SwiftShader)', viewport: '1180x760', seconds: 8 }
  for (const mode of ['3d', '2d'] as const) {
    const page = await context.newPage() // a fresh page per mode: the setting is read when the page loads
    await page.addInitScript((m) => localStorage.setItem('ttc.settings', JSON.stringify({ lang: 'en', use3d: m === '3d', motion: 'full', mascot: true })), mode)
    await page.setViewportSize({ width: 1180, height: 760 })
    await page.goto('/#demo')
    await expect(page.getByTestId(mode === '3d' ? 'board3d' : 'board2d')).toBeVisible()
    await page.getByTestId('replay-speed').selectOption('1')
    await page.getByTestId('replay-play').click()
    await page.waitForTimeout(1000)
    result[mode] = await measure(page, 8000)
    await page.close()
  }
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, 'ui_frame_rate.json'), JSON.stringify(result, null, 2) + '\n')
  console.log(JSON.stringify(result, null, 2))
})
