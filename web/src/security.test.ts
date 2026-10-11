import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'

// Static rules of the UI that the content security policy and the design rely on (docs/design/WEB_UI.md section 6). They are checked on the source.

const root = join(process.cwd(), 'src')
const sources: { file: string; text: string }[] = []
const walk = (dir: string) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) {
      if (name !== 'generated' && name !== 'demo') walk(p)
    } else if (/\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name)) sources.push({ file: relative(root, p).split(/[\\/]/).join('/'), text: readFileSync(p, 'utf8') })
  }
}
walk(root)

const offenders = (re: RegExp, allowed: string[] = []) => sources.filter((s) => re.test(s.text) && !allowed.includes(s.file)).map((s) => s.file)

describe('source rules', () => {
  it('finds the sources', () => expect(sources.length).toBeGreaterThan(20))
  it('never inserts markup: text from the server is always rendered as text', () => {
    expect(offenders(/dangerouslySetInnerHTML|\.innerHTML|\.outerHTML|insertAdjacentHTML|document\.write/)).toEqual([])
  })
  it('never evaluates strings (the policy forbids it)', () => {
    expect(offenders(/\beval\s*\(|new Function\s*\(|setTimeout\s*\(\s*['"`]/)).toEqual([])
  })
  it('keeps nothing in the browser except the preferences, and only through settings.ts (context.tsx only hands it the storage object)', () => {
    expect(offenders(/localStorage|sessionStorage|indexedDB/, ['state/settings.ts', 'components/context.tsx'])).toEqual([])
    expect(offenders(/document\.cookie/)).toEqual([])
  })
  it('puts no key or session in a URL', () => {
    expect(offenders(/[?&](key|api_?key|token|session)=/i)).toEqual([])
  })
  it('talks to its own origin only', () => {
    expect(offenders(/https?:\/\/(?!www\.w3\.org)/)).toEqual([])
  })
})
