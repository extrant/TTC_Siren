import { defineConfig } from '@playwright/test'
import base from './playwright.config'

// The pictures and the frame rate of docs/reports/phase8b, made on demand: npm run report (same server and browser as the tests, nothing is asserted about the numbers).
export default defineConfig({ ...base, testDir: 'e2e/reports', testIgnore: undefined, timeout: 120_000 })
