// Calls to the server. Same origin only; the session cookie is HttpOnly and travels by itself; nothing here ever sees or stores the API key except the one login call.

export class ApiError extends Error {
  constructor(readonly status: number, message: string, readonly retryAfterSeconds: number | null = null) {
    super(message)
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } }
  if (body !== undefined) {
    init.body = JSON.stringify(body)
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  const response = await fetch(path, init)
  const text = await response.text()
  let data: unknown = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = null
  }
  if (!response.ok) {
    const retryHeader = Number(response.headers.get('Retry-After'))
    const message = typeof data === 'object' && data !== null && 'error' in data ? String((data as { error: unknown }).error) : `HTTP ${response.status}`
    throw new ApiError(response.status, message, Number.isFinite(retryHeader) && retryHeader > 0 ? retryHeader : null)
  }
  return data as T
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body: unknown) => request<T>('POST', path, body),
  delete: <T>(path: string) => request<T>('DELETE', path),
}

export interface Me {
  authenticated: boolean
  via: 'session' | 'api_key'
  class: string
  key: { id: string; name: string; rate_per_minute: number; burst: number; max_in_flight: number }
  session: { created: string; expires: string } | null
}

/** Asks who the browser is. 401 means nobody (not logged in, or the session ended). */
export async function fetchMe(): Promise<Me | null> {
  try {
    return await api.get<Me>('/v1/me')
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) return null
    throw e
  }
}

/** Exchanges the key for a session cookie. The key is passed in and not kept. */
export async function login(key: string): Promise<void> {
  await api.post('/v1/session', { key })
}

export async function logout(): Promise<void> {
  await api.delete('/v1/session')
}
