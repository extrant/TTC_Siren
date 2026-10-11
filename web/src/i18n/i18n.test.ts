import { describe, expect, it } from 'vitest'
import { en } from './en'
import { translate, messages, type MessageKey } from './index'
import { zh } from './zh-CN'

const slots = (text: string): string[] => [...text.matchAll(/\{([a-zA-Z0-9_]+)\}/g)].map((m) => m[1]).sort()

describe('i18n completeness', () => {
  it('has exactly the same keys in both languages', () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(zh).sort())
  })

  it('has no empty text', () => {
    for (const lang of ['zh-CN', 'en'] as const) {
      for (const [key, text] of Object.entries(messages[lang])) expect(text.trim().length, `${lang} ${key}`).toBeGreaterThan(0)
    }
  })

  it('uses the same {placeholders} for a key in both languages', () => {
    for (const key of Object.keys(zh) as MessageKey[]) expect(slots(en[key]), key).toEqual(slots(zh[key]))
  })

  it('does not leave a text identical in both languages unless it is a proper noun or a symbol', () => {
    // a text copied into the other language untranslated is a missing translation; the few that are the same in both are listed
    const same = (Object.keys(zh) as MessageKey[]).filter((k) => zh[k] === en[k])
    expect(same).toEqual([])
  })
})

describe('translate', () => {
  it('fills the slots', () => {
    expect(translate('en', 'live.moveText', { id: 343, cell: 'centre' })).toBe('card #343 → centre')
    expect(translate('zh-CN', 'live.moveText', { id: 343, cell: '中心' })).toContain('343')
  })

  it('leaves a slot that was not given as it is, so the gap is noticed', () => {
    expect(translate('en', 'live.moveText', { id: 1 })).toBe('card #1 → {cell}')
  })

  it('does not interpret the values: text from the server stays text', () => {
    expect(translate('en', 'login.already', { name: '<img src=x onerror=alert(1)>' })).toBe('Logged in as “<img src=x onerror=alert(1)>”.')
  })
})
