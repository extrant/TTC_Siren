import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api, fetchMe, login, logout } from './client'

const reply = (status: number, body: unknown, headers: Record<string, string> = {}) => new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers })

afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('calls the same origin with the cookie, and never adds an Authorization or key header', async () => {
    const fetchMock = vi.fn(async () => reply(200, { ok: true }))
    vi.stubGlobal('fetch', fetchMock)
    await api.get('/v1/games')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/v1/games')
    expect(init.credentials).toBe('same-origin')
    expect(Object.keys(init.headers as Record<string, string>).map((h) => h.toLowerCase())).toEqual(['accept'])
  })

  it('sends the key only in the body of the login call, and not in the URL', async () => {
    const fetchMock = vi.fn(async () => reply(200, {}))
    vi.stubGlobal('fetch', fetchMock)
    await login('sk-test-123')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/v1/session')
    expect(url).not.toContain('sk-test-123')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual({ key: 'sk-test-123' })
  })

  it('logs out with DELETE', async () => {
    const fetchMock = vi.fn(async () => reply(200, {}))
    vi.stubGlobal('fetch', fetchMock)
    await logout()
    expect((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].method).toBe('DELETE')
  })

  it('turns an error answer into an ApiError with the server message and the Retry-After', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => reply(429, { error: 'too many attempts' }, { 'Retry-After': '17' })))
    const e = await login('x').catch((x: unknown) => x)
    expect(e).toBeInstanceOf(ApiError)
    expect(e).toMatchObject({ status: 429, message: 'too many attempts', retryAfterSeconds: 17 })
  })

  it('copes with an error answer that is not JSON (a proxy page)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => reply(502, '<html>bad gateway</html>')))
    const e = (await api.get('/v1/games').catch((x: unknown) => x)) as ApiError
    expect(e.status).toBe(502)
    expect(e.message).toBe('HTTP 502')
    expect(e.retryAfterSeconds).toBeNull()
  })

  it('says nobody is logged in on 401, and raises every other failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => reply(401, { error: 'no' })))
    expect(await fetchMe()).toBeNull()
    vi.stubGlobal('fetch', vi.fn(async () => reply(500, { error: 'boom' })))
    await expect(fetchMe()).rejects.toBeInstanceOf(ApiError)
  })
})
