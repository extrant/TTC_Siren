import { expect, test } from '@playwright/test'
import { authFile, cspViolations, freshGameId, loginWithUi, postGame, state, watchProblems } from './helpers'

// Browser tests against the real server (see global-setup.ts). They are in the order of the user's journey; the tests that follow a game share one game id (serial block).

test.describe('landing and demo (anonymous)', () => {
  test('the landing page explains the product and plays the demo replay', async ({ page }) => {
    const watch = watchProblems(page)
    await page.goto('/')
    await expect(page.getByTestId('landing-page')).toBeVisible()
    await expect(page.getByTestId('mascot')).toHaveCount(0) // the landing page has its own picture of the mascot
    await page.getByTestId('cta-demo').click()
    await expect(page.getByTestId('replay')).toBeVisible()
    await expect(page.getByTestId('replay-position')).toContainText('1')
    await page.getByTestId('replay-next').click()
    await expect(page.getByTestId('replay')).toHaveAttribute('data-step', '1')
    // play to the end: the steps advance by themselves
    await page.getByTestId('replay-speed').selectOption('4')
    await page.getByTestId('replay-last').click()
    await expect(page.getByTestId('replay')).toHaveAttribute('data-step-kind', 'end')
    expect(await cspViolations(page)).toEqual([])
    expect(watch.problems()).toEqual([])
  })

  test('the pages that need a login say so and offer the login', async ({ page }) => {
    for (const path of ['/live', '/games', '/games/anything']) {
      await page.goto(path)
      await expect(page.getByTestId(path === '/live' ? 'live-login-needed' : 'games-login-needed')).toBeVisible()
    }
  })

  test('an unknown address is a page, not a blank', async ({ page }) => {
    const response = await page.goto('/nothing/here')
    expect(response?.status()).toBe(200) // the single-page application answers the page; the page says it is not found
    await expect(page.getByTestId('not-found')).toBeVisible()
  })

  test('the page is served under a strict content security policy', async ({ request }) => {
    const r = await request.get('/')
    const csp = r.headers()['content-security-policy']
    expect(csp).toContain("script-src 'self'")
    expect(csp).not.toMatch(/script-src[^;]*unsafe-(inline|eval)/)
    expect(csp).toContain("frame-ancestors 'none'")
    // the API paths never answer with the page
    expect((await request.get('/v1/nothing-here')).headers()['content-type']).toContain('json')
  })
})

test.describe('login and logout', () => {
  test('a wrong key is refused with one message; the right key logs in; nothing of the key stays in the browser', async ({ page, context }) => {
    const s = state()
    const watch = watchProblems(page)
    const seen: string[] = []
    page.on('request', (r) => seen.push(`${r.method()} ${r.url()} ${JSON.stringify(r.headers())}`))
    await page.goto('/login')
    await page.getByTestId('login-key').fill('ttc_not-a-real-key')
    await page.getByTestId('login-submit').click()
    await expect(page.getByTestId('login-message')).toBeVisible()
    await expect(page.getByTestId('who')).toHaveCount(0)

    await page.getByTestId('login-key').fill(s.key)
    await page.getByTestId('login-submit').click()
    await expect(page.getByTestId('who')).toHaveText(s.keyName)
    await expect(page).toHaveURL(/\/live$/)

    // the cookie: HttpOnly, SameSite=Strict, and no script can read it
    const cookies = await context.cookies()
    const session = cookies.find((c) => /ttc_session/.test(c.name))!
    expect(session, JSON.stringify(cookies.map((c) => c.name))).toBeDefined()
    expect(session.httpOnly).toBe(true)
    expect(session.sameSite).toBe('Strict')
    expect(session.path).toBe('/')
    expect(await page.evaluate(() => document.cookie)).not.toContain('ttc_session')

    // the key is in no storage and no address, and the session value is in neither
    const dump = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage }, url: location.href, title: document.title }))
    expect(dump).not.toContain(s.key)
    expect(dump).not.toContain(session.value)
    const idb = await page.evaluate(async () => (await indexedDB.databases?.())?.length ?? 0)
    expect(idb).toBe(0)
    expect(page.url()).not.toContain(s.key)

    // a reload keeps the login (the cookie), and the live page opens
    await page.reload()
    await expect(page.getByTestId('who')).toHaveText(s.keyName)
    await expect(page.getByTestId('live-page')).toBeVisible()

    // the key itself was in the body of the one login call and nowhere else: not in a URL, not in a header
    expect(seen.filter((line) => line.includes(s.key))).toEqual([])

    // logout ends the session on the server too
    await page.getByTestId('logout').click()
    await expect(page.getByTestId('nav-login')).toBeVisible()
    const me = await page.evaluate(async () => (await fetch('/v1/me')).status)
    expect(me).toBe(401)
    await page.goto('/live')
    await expect(page.getByTestId('live-login-needed')).toBeVisible()
    expect(await cspViolations(page)).toEqual([])
    expect(watch.problems()).toEqual([])
  })

  test('text that the server returns is shown as text, never as markup', async ({ page }) => {
    const s = state()
    await loginWithUi(page, s.htmlKey)
    await expect(page.getByTestId('who')).toHaveText('<i>xss</i>')
    await expect(page.getByTestId('who').locator('i')).toHaveCount(0)
  })
})

test.describe.serial('a plugin game, live and afterwards', () => {
  test.use({ storageState: authFile })
  const gameId = freshGameId('live')

  test('requests posted to /ai_move appear in the live view, with depths, the decision and the explanation', async ({ page, request }) => {
    const watch = watchProblems(page)
    await page.goto('/live')
    await expect(page.getByTestId('connection-status')).toHaveText(/Connected/)

    await postGame(request, gameId, 1)
    await expect(page.getByTestId('live-game-id')).toContainText(gameId)
    await expect(page.getByTestId('decision')).toBeVisible()
    await expect(page.getByTestId('depth-table')).toBeVisible()
    await expect(page.getByTestId('explanation-brief')).toBeVisible()
    await expect(page.getByTestId('outcome-badge')).toBeVisible()
    // the second request replaces the first one's data
    await postGame(request, gameId, 2, 1)
    await expect(page.getByTestId('live-game-id')).toContainText(/request 2/)

    // the rest of the game, and the end
    await postGame(request, gameId, Infinity, 2)
    await expect(page.getByTestId('game-ended')).toBeVisible()
    expect(await cspViolations(page)).toEqual([])
    expect(watch.problems()).toEqual([])
  })

  test('the game is in the log viewer and its timeline is that of the log', async ({ page, request }) => {
    await page.goto('/games')
    await expect(page.getByTestId(`open-${gameId}`)).toBeVisible()
    await page.getByTestId(`open-${gameId}`).click()
    await expect(page.getByTestId('game-page')).toBeVisible()
    const s = state()
    type Rec = { type: string; state?: unknown; result?: { move?: unknown }; explanation?: { facts?: { type: string }[] } }
    const log = (await (await request.get(`${s.baseUrl}/v1/games/${gameId}`, { headers: { 'X-Api-Key': s.key } })).json()) as { records: Rec[] }
    // one step for each position, for each move of ours that the facts explain, for each opponent move, and the end
    const count = (pred: (r: Rec) => boolean) => log.records.filter(pred).length
    const expected =
      count((r) => r.type === 'request' && !!r.state) +
      count((r) => r.type === 'request' && !!r.result?.move && !!r.explanation?.facts?.some((f) => f.type === 'immediate_effect')) +
      count((r) => r.type === 'opponent_move') +
      count((r) => r.type === 'game_end')
    await expect(page.getByTestId('replay')).toBeVisible()
    const items = await page.getByTestId('timeline').locator('li').count()
    expect(items).toBe(expected)
    expect(log.records.filter((r) => r.type === 'game_end')).toHaveLength(1)
    await expect(page.getByTestId('replay-position')).toContainText(`of ${items}`)
  })
})

test.describe.serial('replay controls, download, delete', () => {
  test.use({ storageState: authFile })
  const gameId = freshGameId('replay')

  test('the controls move through the steps and the table follows', async ({ page, request }) => {
    await postGame(request, gameId)
    await page.goto(`/games/${gameId}`)
    const replay = page.getByTestId('replay')
    await expect(replay).toHaveAttribute('data-step', '0')
    await expect(page.getByTestId('replay-first')).toBeDisabled()
    await page.getByTestId('replay-next').click()
    await expect(replay).toHaveAttribute('data-step', '1')
    await page.getByTestId('replay-prev').click()
    await expect(replay).toHaveAttribute('data-step', '0')
    await page.getByTestId('replay-last').click()
    await expect(replay).toHaveAttribute('data-step-kind', 'end')
    await expect(page.getByTestId('replay-next')).toBeDisabled()
    await page.getByTestId('replay-first').click()
    // the timeline buttons and the scrubber jump
    await page.getByTestId('timeline-2').click()
    await expect(replay).toHaveAttribute('data-step', '2')
    await page.getByTestId('replay-scrubber').fill('1')
    await expect(replay).toHaveAttribute('data-step', '1')
    // play: the steps advance by themselves, and pause stops it
    await page.getByTestId('replay-speed').selectOption('4')
    await page.getByTestId('replay-first').click()
    await page.getByTestId('replay-play').click()
    await expect(replay).not.toHaveAttribute('data-step', '0', { timeout: 15_000 })
    await page.getByTestId('replay-play').click()
    const at = await replay.getAttribute('data-step')
    await page.waitForTimeout(1500)
    expect(await replay.getAttribute('data-step')).toBe(at)
  })

  test('a step that flips cards lists each flip with its reason', async ({ page }) => {
    await page.goto(`/games/${gameId}`)
    await expect(page.getByTestId('replay')).toBeVisible()
    const total = await page.getByTestId('timeline').locator('li').count()
    expect(total).toBeGreaterThan(5)
    const causes = new Set<string>()
    for (let i = 0; i < total; i++) {
      await page.getByTestId(`timeline-${i}`).click()
      await expect(page.getByTestId('replay')).toHaveAttribute('data-step', String(i))
      for (const li of await page.getByTestId('flip-list').locator('li').all()) causes.add((await li.getAttribute('data-cause')) ?? '')
    }
    // the sample game shows every cause but Reverse
    for (const c of ['basic', 'same', 'plus', 'fallen_ace', 'combo']) expect([...causes], c).toContain(c)
  })

  test('download gives the JSON Lines of the game', async ({ page }) => {
    await page.goto(`/games/${gameId}`)
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('game-download').click()])
    expect(download.suggestedFilename()).toBe(`${gameId}.jsonl`)
    const stream = await download.createReadStream()
    let text = ''
    for await (const chunk of stream) text += chunk.toString()
    const lines = text.trim().split('\n').map((l) => JSON.parse(l) as { type: string; seq: number })
    expect(lines[0].type).toBe('game_start')
    expect(lines.map((l) => l.seq)).toEqual(lines.map((_, i) => i))
  })

  test('delete asks first, and then removes the game for good', async ({ page }) => {
    await page.goto('/games')
    await page.getByTestId(`delete-${gameId}`).click()
    await expect(page.getByTestId('confirm-delete')).toBeVisible()
    await page.getByTestId('confirm-no').click()
    await expect(page.getByTestId(`open-${gameId}`)).toBeVisible() // cancelled: still there
    await page.getByTestId(`delete-${gameId}`).click()
    await page.getByTestId('confirm-yes').click()
    await expect(page.getByTestId(`open-${gameId}`)).toHaveCount(0)
    const status = await page.evaluate(async (id) => (await fetch(`/v1/games/${id}`)).status, gameId)
    expect(status).toBe(404)
    await page.goto(`/games/${gameId}`)
    await expect(page.getByTestId('game-error')).toBeVisible()
  })
})

test.describe('fallbacks and preferences', () => {
  test.use({ storageState: authFile })
  test('without WebGL the flat table is used, with the same information', async ({ page }) => {
    await page.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext
      HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, kind: string, ...rest: unknown[]) {
        if (/webgl/i.test(kind)) return null
        return (original as (...a: unknown[]) => unknown).call(this, kind, ...rest) as never
      } as typeof original
    })
    await page.goto('/#demo')
    await expect(page.getByTestId('replay')).toBeVisible()
    await expect(page.locator('[data-table-mode]')).toHaveAttribute('data-table-mode', '2d')
    await expect(page.getByTestId('board2d')).toBeVisible()
    await expect(page.getByTestId('board3d')).toHaveCount(0)
    await expect(page.getByTestId('state-list')).toBeAttached()
    await page.goto('/settings')
    await expect(page.getByTestId('settings-page')).toContainText('no WebGL')
  })

  test('with WebGL the 3D table is used, and the setting turns it off', async ({ page }) => {
    await page.goto('/#demo')
    await expect(page.locator('[data-table-mode]')).toHaveAttribute('data-table-mode', '3d')
    await expect(page.getByTestId('board3d')).toBeVisible()
    await page.goto('/settings')
    await page.getByTestId('setting-3d').uncheck()
    await page.goto('/#demo')
    await expect(page.locator('[data-table-mode]')).toHaveAttribute('data-table-mode', '2d')
  })

  test('a narrow screen gets the flat table, and the page does not scroll sideways', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 })
    for (const path of ['/', '/login', '/settings', '/#demo']) {
      await page.goto(path)
      await expect(page.locator('main')).toBeVisible()
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      expect(overflow, path).toBeLessThanOrEqual(0)
    }
    await expect(page.locator('[data-table-mode]')).toHaveAttribute('data-table-mode', '2d')
  })

  test('the language switch changes the interface and keeps the choice', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('html')).toHaveAttribute('lang', 'en')
    await expect(page.getByTestId('nav-home')).toHaveText('Home')
    await page.getByTestId('lang-toggle').click()
    await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN')
    await expect(page.getByTestId('nav-home')).toHaveText('首页')
    await page.reload()
    await expect(page.getByTestId('nav-home')).toHaveText('首页')
    // only the preferences are stored, and no key
    const stored = await page.evaluate(() => JSON.stringify({ ...localStorage }))
    expect(Object.keys(JSON.parse(stored) as object)).toEqual(['ttc.settings'])
    expect(JSON.parse((JSON.parse(stored) as Record<string, string>)['ttc.settings']) as object).toEqual({ lang: 'zh-CN', use3d: true, motion: 'auto', mascot: true })
  })

  test('the explanation of a move follows the language', async ({ page, request }) => {
    const gameId = freshGameId('lang')
    await postGame(request, gameId, 1)
    await page.goto(`/games/${gameId}`)
    await page.getByTestId('timeline-1').click()
    const english = await page.getByTestId('explanation-brief').innerText()
    expect(english).toMatch(/[A-Za-z]{4}/)
    expect(english).not.toMatch(/[一-鿿]/)
    await page.getByTestId('lang-toggle').click()
    await expect(page.getByTestId('explanation-brief')).toHaveText(/[一-鿿]/)
    const chinese = await page.getByTestId('explanation-brief').innerText()
    expect(chinese).not.toBe(english)
    // the numbers in the sentence are the same in both
    const digits = (t: string) => (t.match(/\d+(\.\d+)?/g) ?? []).join(',')
    expect(digits(chinese)).toBe(digits(english))
  })

  test('with "reduce motion" the page does not animate, and the setting can override the system', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.goto('/')
    await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduced')
    await page.goto('/settings')
    await page.getByTestId('setting-motion').selectOption('full')
    await expect(page.locator('html')).toHaveAttribute('data-motion', 'full')
    await page.getByTestId('setting-motion').selectOption('auto')
    await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduced')
    // a replay under reduced motion goes through its steps without waiting for animations
    await page.goto('/#demo')
    await page.getByTestId('replay-speed').selectOption('4')
    await page.getByTestId('replay-play').click()
    await expect(page.getByTestId('replay')).toHaveAttribute('data-step-kind', 'end', { timeout: 20_000 })
  })

  test('the mascot can be turned off and stays off', async ({ page }) => {
    await page.goto('/settings')
    await expect(page.getByTestId('mascot')).toBeVisible()
    await page.getByTestId('setting-mascot').uncheck()
    await expect(page.getByTestId('mascot')).toHaveCount(0)
    await page.reload()
    await expect(page.getByTestId('mascot')).toHaveCount(0)
  })
})
