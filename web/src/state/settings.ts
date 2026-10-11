import type { Lang } from '../domain/types'

/** Preferences of this browser. They are only preferences: the key of the API is never kept here (nor anywhere in the browser). */
export interface Settings {
  lang: Lang
  /** show the table in 3D (needs WebGL; a narrow or touch screen falls back to the flat table either way) */
  use3d: boolean
  /** 'auto' follows the system setting prefers-reduced-motion */
  motion: 'auto' | 'reduce' | 'full'
  mascot: boolean
}

export const STORAGE_KEY = 'ttc.settings'

export function defaultLang(navigatorLanguage: string | undefined): Lang {
  return navigatorLanguage && navigatorLanguage.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en'
}

export function defaultSettings(navigatorLanguage?: string): Settings {
  return { lang: defaultLang(navigatorLanguage), use3d: true, motion: 'auto', mascot: true }
}

/** Reads settings from the text in storage; anything that is not a valid value falls back to the default, so a damaged or foreign entry cannot break the page. */
export function parseSettings(text: string | null, navigatorLanguage?: string): Settings {
  const d = defaultSettings(navigatorLanguage)
  if (!text) return d
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return d
  }
  if (typeof raw !== 'object' || raw === null) return d
  const r = raw as Record<string, unknown>
  return {
    lang: r.lang === 'zh-CN' || r.lang === 'en' ? r.lang : d.lang,
    use3d: typeof r.use3d === 'boolean' ? r.use3d : d.use3d,
    motion: r.motion === 'auto' || r.motion === 'reduce' || r.motion === 'full' ? r.motion : d.motion,
    mascot: typeof r.mascot === 'boolean' ? r.mascot : d.mascot,
  }
}

export function serializeSettings(s: Settings): string {
  // exactly these four fields and nothing else
  return JSON.stringify({ lang: s.lang, use3d: s.use3d, motion: s.motion, mascot: s.mascot })
}

export function loadSettings(storage: Pick<Storage, 'getItem'> | null, navigatorLanguage?: string): Settings {
  let text: string | null = null
  try {
    text = storage?.getItem(STORAGE_KEY) ?? null
  } catch {
    text = null
  }
  return parseSettings(text, navigatorLanguage)
}

export function saveSettings(storage: Pick<Storage, 'setItem'> | null, s: Settings): void {
  try {
    storage?.setItem(STORAGE_KEY, serializeSettings(s))
  } catch {
    // private window or blocked storage: the page works without it
  }
}

/** Whether animations are played: the setting wins over the system preference. */
export function animationsOn(s: Settings, systemPrefersReduced: boolean): boolean {
  return s.motion === 'full' ? true : s.motion === 'reduce' ? false : !systemPrefersReduced
}
