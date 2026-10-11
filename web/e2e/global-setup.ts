import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// Starts the real server for the browser tests and stops it afterwards. Keys are created with the server's own command line; the secrets are handed to the tests through a
// file under e2e/.tmp (git-ignored) and never printed.

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '..', '..')
const dll = join(repo, 'csharp', 'src', 'TTC.Siren.Server', 'bin', 'Release', 'net10.0', 'TTC.Siren.Server.dll')
export const stateFile = join(here, '.tmp', 'state.json')
export const authFile = join(here, '.tmp', 'auth.json')

export interface E2eState {
  baseUrl: string
  /** the key the tests log in with */
  key: string
  keyName: string
  /** a key whose name is HTML: the page must show it as text */
  htmlKey: string
  htmlKeyName: string
  logDir: string
}

function addKey(store: string, name: string): string {
  const r = spawnSync('dotnet', [dll, 'keys', 'add', '--name', name, '--rate', '100000', '--burst', '1000', '--inflight', '32', '--store', store], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`keys add failed: ${r.stderr || r.stdout}`)
  const lines = r.stdout.trim().split(/\r?\n/)
  return lines[lines.length - 1].trim()
}

async function waitHealthy(baseUrl: string, child: ChildProcess, log: () => string): Promise<void> {
  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`the server exited early (code ${child.exitCode}):\n${log()}`)
    try {
      const r = await fetch(`${baseUrl}/health`)
      if (r.ok) return
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 200))
  }
  throw new Error(`the server did not become healthy in 60 s:\n${log()}`)
}

export default async function globalSetup(): Promise<() => Promise<void>> {
  if (!existsSync(dll)) throw new Error(`build the server first (dotnet build -c Release csharp/TTC.Siren.sln): ${dll} is missing`)
  const dist = join(repo, 'web', 'dist')
  if (!existsSync(join(dist, 'index.html'))) throw new Error('build the UI first (npm run build): web/dist/index.html is missing')
  const port = Number(process.env.E2E_PORT ?? 5431)
  const baseUrl = `http://127.0.0.1:${port}`

  mkdirSync(join(here, '.tmp'), { recursive: true })
  const work = mkdtempSync(join(tmpdir(), 'ttc-e2e-'))
  const keyStore = join(work, 'keys.json')
  const logDir = join(work, 'logs')
  const state: E2eState = { baseUrl, key: addKey(keyStore, 'e2e'), keyName: 'e2e', htmlKey: addKey(keyStore, '<i>xss</i>'), htmlKeyName: '<i>xss</i>', logDir }
  writeFileSync(stateFile, JSON.stringify(state))

  let output = ''
  const child = spawn('dotnet', [dll, '--host', '127.0.0.1', '--port', String(port)], {
    env: {
      ...process.env,
      TTC_Server__Workers: '2',
      TTC_Server__TtEntries: String(1 << 18),
      TTC_Server__WarmUp: 'false',
      TTC_Server__TestSeedHeader: 'true',
      TTC_Server__StaticDirectory: dist,
      TTC_Server__GameLog__Directory: logDir,
      TTC_Server__Security__KeyStoreFile: keyStore,
      TTC_Server__Security__Sessions__AuditLogFile: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  child.stdout?.on('data', (d: Buffer) => (output += d.toString()))
  child.stderr?.on('data', (d: Buffer) => (output += d.toString()))
  await waitHealthy(baseUrl, child, () => output)

  // one login for all the tests that only need to be logged in (the login is throttled per address, like in production, and is tested through the page separately)
  const login = await fetch(`${baseUrl}/v1/session`, { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: baseUrl }, body: JSON.stringify({ key: state.key }) })
  if (!login.ok) throw new Error(`the test login failed: HTTP ${login.status}`)
  const cookies = login.headers.getSetCookie().map((line) => {
    const [pair] = line.split(';')
    const eq = pair.indexOf('=')
    return { name: pair.slice(0, eq), value: pair.slice(eq + 1), domain: '127.0.0.1', path: '/', expires: Math.floor(Date.now() / 1000) + 3600, httpOnly: true, secure: false, sameSite: 'Strict' as const }
  })
  writeFileSync(authFile, JSON.stringify({ cookies, origins: [] }))

  return async () => {
    child.kill()
    await new Promise<void>((r) => {
      if (child.exitCode !== null) return r()
      child.once('exit', () => r())
      setTimeout(r, 5000)
    })
    try {
      rmSync(work, { recursive: true, force: true })
      rmSync(stateFile, { force: true })
      rmSync(authFile, { force: true })
    } catch {
      // a file still held by the stopping process: the temp folder is cleaned by the system
    }
  }
}
