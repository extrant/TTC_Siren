import type { Lang } from '../domain/types'
import { en } from './en'
import { zh } from './zh-CN'

export type MessageKey = keyof typeof zh
export const messages: Record<Lang, Record<MessageKey, string>> = { 'zh-CN': zh, en }

/** Looks a message up and fills {name} slots. A key that does not exist is a bug: it is shown as itself so that it is noticed. */
export function translate(lang: Lang, key: MessageKey, params?: Record<string, string | number>): string {
  const template = messages[lang][key] ?? String(key)
  if (!params) return template
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (m, name: string) => (name in params ? String(params[name]) : m))
}

export type Translate = (key: MessageKey, params?: Record<string, string | number>) => string

export const translatorFor = (lang: Lang): Translate => (key, params) => translate(lang, key, params)
