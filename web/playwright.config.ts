import { defineConfig } from '@playwright/test'

// End-to-end tests run against a REAL TTC.Siren.Server (see e2e/global-setup.ts): a test key store, TestSeedHeader on loopback, a temporary game-log folder, and the built UI
// (dist/) served from the same origin. Nothing is mocked. The browser is the Microsoft Edge that Windows ships with (channel 'msedge'), so no browser is downloaded; set
// E2E_CHANNEL=chrome (or an empty value, for Playwright's own Chromium) to use another one.
const port = Number(process.env.E2E_PORT ?? 5431)
const channel = process.env.E2E_CHANNEL === undefined ? 'msedge' : process.env.E2E_CHANNEL || undefined

export default defineConfig({
  testDir: 'e2e',
  globalSetup: './e2e/global-setup.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  outputDir: 'e2e/.tmp/results',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel,
    headless: true,
    // software WebGL: the 3D table renders without a GPU
    launchOptions: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] },
    viewport: { width: 1280, height: 860 },
    locale: 'en-US',
    trace: 'off',
  },
})
