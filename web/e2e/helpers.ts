import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, type APIRequestContext, type Page } from '@playwright/test'
import type { E2eState } from './global-setup'


const here = dirname(fileURLToPath(import.meta.url))

/** the storage state of a browser that is logged in (made once by the global setup) */
export const authFile = join(here, '.tmp', 'auth.json')

export const state = (): E2eState => JSON.parse(readFileSync(join(here, '.tmp', 'state.json'), 'utf8')) as E2eState

interface Fixture {
  game_id: string
  rules: string
  seed: number
  requests: string[]
}

export const fixture = (): Fixture => JSON.parse(readFileSync(join(here, 'fixtures', 'plugin-game.json'), 'utf8')) as Fixture

let counter = 0
/** a game id that no other test used (the server keeps every game it saw) */
export const freshGameId = (label: string): string => `e2e-${label}-${Date.now().toString(36)}-${counter++}`

/** Plays the sample game the way the plugin does: one POST /ai_move per request, with the key in a header. The seed header makes the server's sampling reproducible. */
export async function postGame(request: APIRequestContext, gameId: string, upTo = Infinity, from = 0): Promise<number> {
  const s = state()
  const fx = fixture()
  let posted = from
  for (const body of fx.requests.slice(from, upTo)) {
    const response = await request.post(`${s.baseUrl}/ai_move`, {
      data: body.replace(`"gameId":"${fx.game_id}"`, `"gameId":"${gameId}"`),
      headers: { 'Content-Type': 'application/json', 'X-Api-Key': s.key, 'X-Ttc-Test-Seed': String(9000 + posted) },
    })
    expect(response.status(), await response.text()).toBe(200)
    posted++
  }
  return posted
}

export async function loginWithUi(page: Page, key = state().key): Promise<void> {
  await page.goto('/login')
  await page.getByTestId('login-key').fill(key)
  await page.getByTestId('login-submit').click()
  await expect(page.getByTestId('who')).toBeVisible()
}

/** Collects what a page must never do: a policy violation, a script error. Call `.problems()` at the end of a test. */
export function watchProblems(page: Page): { problems: () => string[] } {
  const found: string[] = []
  page.on('pageerror', (e) => found.push(`pageerror: ${e.message}`))
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) found.push(`console.error: ${m.text()}`)
  })
  void page.addInitScript(() => {
    document.addEventListener('securitypolicyviolation', (e) => {
      ;(window as unknown as { __csp: string[] }).__csp = [...((window as unknown as { __csp?: string[] }).__csp ?? []), `${e.violatedDirective} ${e.blockedURI}`]
    })
  })
  return { problems: () => found }
}

export async function cspViolations(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __csp?: string[] }).__csp ?? [])
}
