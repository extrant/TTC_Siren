import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError } from '../api/client'
import { useRouter } from '../components/router'
import { useSession, useSettings } from '../components/context'
import { useMood } from '../components/Mascot'

/**
 * The login: the key is typed once, sent once, and forgotten. It is held in the input element only (never in React state, storage, the address, or a log); the field is
 * emptied the moment the request is sent. Every failure says only "login failed"; being throttled says to try again later.
 */
export function LoginPage() {
  const { t } = useSettings()
  const { status, login, me, logout } = useSession()
  const { navigate } = useRouter()
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  useMood({ expression: 'neutral', caption: 'mascot.loginHello', params: {} })
  useEffect(() => {
    document.title = `${t('nav.login')} · TTC Siren`
  }, [t])

  async function submit(e: FormEvent) {
    e.preventDefault()
    const field = input.current
    const key = field?.value.trim() ?? ''
    if (!key) {
      setMessage(t('login.failed'))
      return
    }
    setBusy(true)
    setMessage(null)
    if (field) field.value = '' // forgotten at once
    try {
      await login(key)
      navigate('/live')
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) setMessage(t('login.throttled', { seconds: err.retryAfterSeconds ?? 10 }))
      else setMessage(t('login.failed'))
    } finally {
      setBusy(false)
    }
  }

  if (status === 'user' && me) {
    return (
      <section className="page narrow" data-testid="login-done">
        <h1>{t('login.title')}</h1>
        <p>{t('login.already', { name: me.key.name })}</p>
        <button type="button" onClick={() => void logout().then(() => navigate('/'))} data-testid="logout-button">{t('nav.logout')}</button>
      </section>
    )
  }

  return (
    <section className="page narrow" data-testid="login-page">
      <h1>{t('login.title')}</h1>
      <p className="muted">{t('login.explain')}</p>
      <form onSubmit={(e) => void submit(e)} autoComplete="off" noValidate>
        <label htmlFor="api-key">{t('login.key')}</label>
        <input
          id="api-key" ref={input} type="password" name="ttc-key" autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck={false} data-testid="login-key"
          aria-describedby={message ? 'login-message' : undefined} aria-invalid={message ? true : undefined}
        />
        <button type="submit" disabled={busy} data-testid="login-submit">{t('login.submit')}</button>
      </form>
      {message ? <p id="login-message" className="error" role="alert" data-testid="login-message">{message}</p> : null}
      <p className="muted">{t('login.note')}</p>
    </section>
  )
}
