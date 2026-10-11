import { describe, expect, it } from 'vitest'
import { parseLocation } from './router'

describe('parseLocation', () => {
  it('splits a path into segments', () => {
    const r = parseLocation('/games/g-1', '')
    expect(r.path).toBe('/games/g-1')
    expect(r.segments).toEqual(['games', 'g-1'])
  })

  it('treats the root and a trailing slash alike', () => {
    expect(parseLocation('/', '').segments).toEqual([])
    expect(parseLocation('/', '').path).toBe('/')
    expect(parseLocation('/live/', '').path).toBe('/live')
    expect(parseLocation('///', '').path).toBe('/')
  })

  it('reads the query', () => {
    expect(parseLocation('/games', '?page=3').search.get('page')).toBe('3')
  })

  it('keeps an encoded segment encoded (the page decodes it where it is used)', () => {
    expect(parseLocation('/games/a%2Fb', '').segments).toEqual(['games', 'a%2Fb'])
  })
})
