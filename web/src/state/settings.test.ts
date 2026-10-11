import { describe, expect, it } from 'vitest'
import { animationsOn, defaultLang, defaultSettings, loadSettings, parseSettings, saveSettings, serializeSettings, STORAGE_KEY } from './settings'

describe('settings', () => {
  it('chooses the language from the browser: Chinese for zh, English for anything else', () => {
    expect(defaultLang('zh-CN')).toBe('zh-CN')
    expect(defaultLang('zh-TW')).toBe('zh-CN')
    expect(defaultLang('en-US')).toBe('en')
    expect(defaultLang('de')).toBe('en')
    expect(defaultLang(undefined)).toBe('en')
  })

  it('falls back to the defaults for missing, damaged or foreign text', () => {
    const d = defaultSettings('en')
    expect(parseSettings(null, 'en')).toEqual(d)
    expect(parseSettings('', 'en')).toEqual(d)
    expect(parseSettings('{not json', 'en')).toEqual(d)
    expect(parseSettings('42', 'en')).toEqual(d)
    expect(parseSettings('null', 'en')).toEqual(d)
    expect(parseSettings('{"lang":"fr","use3d":"yes","motion":"sometimes","mascot":1}', 'en')).toEqual(d)
  })

  it('keeps the valid fields and replaces only the invalid ones', () => {
    expect(parseSettings('{"lang":"zh-CN","use3d":false,"motion":"reduce","mascot":"x"}', 'en')).toEqual({ lang: 'zh-CN', use3d: false, motion: 'reduce', mascot: true })
  })

  it('writes exactly the four preferences and nothing else: no key, no token, no session', () => {
    const polluted = { ...defaultSettings('en'), apiKey: 'secret', token: 'x' } as unknown as Parameters<typeof serializeSettings>[0]
    const written = JSON.parse(serializeSettings(polluted)) as Record<string, unknown>
    expect(Object.keys(written).sort()).toEqual(['lang', 'mascot', 'motion', 'use3d'])
    expect(serializeSettings(polluted)).not.toContain('secret')
  })

  it('round-trips through a storage', () => {
    const data = new Map<string, string>()
    const storage = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) }
    const s = { lang: 'zh-CN' as const, use3d: false, motion: 'full' as const, mascot: false }
    saveSettings(storage, s)
    expect([...data.keys()]).toEqual([STORAGE_KEY])
    expect(loadSettings(storage, 'en')).toEqual(s)
  })

  it('works when the storage is blocked or missing', () => {
    const throwing = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
    }
    expect(loadSettings(throwing, 'en')).toEqual(defaultSettings('en'))
    expect(() => saveSettings(throwing, defaultSettings('en'))).not.toThrow()
    expect(loadSettings(null, 'zh')).toEqual(defaultSettings('zh'))
    expect(() => saveSettings(null, defaultSettings('en'))).not.toThrow()
  })

  it('lets the setting win over the system preference for animations', () => {
    const s = defaultSettings('en')
    expect(animationsOn({ ...s, motion: 'auto' }, false)).toBe(true)
    expect(animationsOn({ ...s, motion: 'auto' }, true)).toBe(false)
    expect(animationsOn({ ...s, motion: 'reduce' }, false)).toBe(false)
    expect(animationsOn({ ...s, motion: 'full' }, true)).toBe(true)
  })
})
