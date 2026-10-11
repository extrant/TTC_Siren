import { useEffect } from 'react'
import { useSettings } from '../components/context'
import { webglAvailable } from '../components/BoardView'
import type { Settings } from '../state/settings'

export function SettingsPage() {
  const { settings, update, t } = useSettings()
  useEffect(() => {
    document.title = `${t('nav.settings')} · TTC Siren`
  }, [t])
  const webgl = webglAvailable()
  return (
    <section className="page narrow" data-testid="settings-page">
      <h1>{t('settings.title')}</h1>
      <p className="muted">{t('settings.note')}</p>
      <form className="settings" onSubmit={(e) => e.preventDefault()}>
        <label>
          {t('settings.language')}
          <select value={settings.lang} onChange={(e) => update({ lang: e.target.value as Settings['lang'] })} data-testid="setting-lang">
            <option value="zh-CN">中文</option>
            <option value="en">English</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.use3d} onChange={(e) => update({ use3d: e.target.checked })} data-testid="setting-3d" />
          {t('settings.use3d')}
        </label>
        <p className="muted">{webgl ? t('settings.webglYes') : t('settings.webglNo')}</p>
        <label>
          {t('settings.motion')}
          <select value={settings.motion} onChange={(e) => update({ motion: e.target.value as Settings['motion'] })} data-testid="setting-motion">
            <option value="auto">{t('settings.motionAuto')}</option>
            <option value="reduce">{t('settings.motionReduce')}</option>
            <option value="full">{t('settings.motionFull')}</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.mascot} onChange={(e) => update({ mascot: e.target.checked })} data-testid="setting-mascot" />
          {t('settings.mascot')}
        </label>
      </form>
    </section>
  )
}
