import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import sharp from 'sharp'
import { authFile, freshGameId, postGame } from '../helpers'

// The screenshots of the report (docs/reports/phase8b/screenshots): every page, in both languages, with the 3D table and with the flat one. Run on demand:
//   npm run report
// Animations are off (reduced motion) so that every picture shows the settled table; the PNGs are palette-quantised to stay small.

const here = dirname(fileURLToPath(import.meta.url))
const out = join(here, '..', '..', '..', 'docs', 'reports', 'phase8b', 'screenshots')

test.use({ storageState: authFile, viewport: { width: 1180, height: 760 } })

const gameId = freshGameId('shots')
const liveGameId = freshGameId('shots-live') // a game still in progress: the live page follows it
const langs = ['en', 'zh-CN'] as const
const modes = ['3d', '2d'] as const

async function shot(page: Page, name: string, fullPage = false) {
  const png = await page.screenshot({ fullPage, type: 'png' })
  await sharp(png).png({ palette: true, quality: 80, compressionLevel: 9, effort: 10 }).toFile(join(out, `${name}.png`))
}

test.beforeAll(async ({ request }) => {
  mkdirSync(out, { recursive: true })
  await postGame(request, gameId)
  await postGame(request, liveGameId, 3)
})

for (const lang of langs) {
  for (const mode of modes) {
    test(`pages in ${lang}, ${mode}`, async ({ page }) => {
      await page.emulateMedia({ reducedMotion: 'reduce' })
      await page.addInitScript(([l, m]) => localStorage.setItem('ttc.settings', JSON.stringify({ lang: l, use3d: m === '3d', motion: 'reduce', mascot: true })), [lang, mode])
      const name = (p: string) => `${p}-${lang}-${mode}`
      const settled = async () => {
        if (mode === '3d') await expect(page.getByTestId('board3d')).toBeVisible()
        else await expect(page.getByTestId('board2d')).toBeVisible()
        await page.waitForTimeout(1200) // the first frame of the 3D scene and the mascot's image
      }

      await page.goto('/#demo')
      await page.getByTestId('timeline-3').click().catch(() => undefined)
      await settled()
      await shot(page, name('landing'), true)

      // the pages without a table look the same in both modes: one picture each, taken in the 3D run
      if (mode === '3d') {
        await page.goto('/login')
        await expect(page.getByTestId('login-done')).toBeVisible() // the test browser is logged in: this is the page of a logged-in user
        await shot(page, `login-loggedin-${lang}`)
      }

      await page.goto('/live')
      await expect(page.getByTestId('decision')).toBeVisible()
      await settled()
      await shot(page, name('live'), true)

      if (mode === '3d') {
        await page.goto('/games')
        await expect(page.getByTestId(`open-${gameId}`)).toBeVisible()
        await shot(page, `games-${lang}`)
      }

      await page.goto(`/games/${gameId}`)
      await expect(page.getByTestId('replay')).toBeVisible()
      await page.getByTestId('timeline-1').click()
      await expect(page.getByTestId('explanation-brief')).toBeVisible()
      await settled()
      await shot(page, name('replay-our-move'), true)
      await page.getByTestId('timeline-2').click()
      await settled()
      await shot(page, name('replay-opponent-move'), true)

      if (mode === '3d') {
        await page.goto('/settings')
        await shot(page, `settings-${lang}`)
        await page.goto('/no/such/page')
        await shot(page, `notfound-${lang}`)
      }
    })
  }
}

test('a phone: the flat table, both languages', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  for (const lang of langs) {
    await page.addInitScript((l) => localStorage.setItem('ttc.settings', JSON.stringify({ lang: l, use3d: true, motion: 'reduce', mascot: true })), lang)
    await page.goto(`/games/${gameId}`)
    await expect(page.getByTestId('replay')).toBeVisible()
    await page.getByTestId('timeline-1').click()
    await expect(page.getByTestId('board2d')).toBeVisible()
    await page.waitForTimeout(600)
    await shot(page, `phone-replay-${lang}`, true)
  }
})

test.describe('not logged in', () => {
  test.use({ storageState: { cookies: [], origins: [] } })
  for (const lang of langs) {
    test(`the login page and the pages that need it, ${lang}`, async ({ page }) => {
      await page.addInitScript((l) => localStorage.setItem('ttc.settings', JSON.stringify({ lang: l, use3d: true, motion: 'reduce', mascot: true })), lang)
      await page.goto('/login')
      await expect(page.getByTestId('login-page')).toBeVisible()
      await page.waitForTimeout(500)
      await shot(page, `login-${lang}`)
      await page.goto('/live')
      await expect(page.getByTestId('live-login-needed')).toBeVisible()
      await shot(page, `live-login-needed-${lang}`)
    })
  }
})
