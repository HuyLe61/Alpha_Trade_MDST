const NY = 'America/New_York'

export const usd = (v, digits = 0) =>
  v == null ? '—' : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: digits, minimumFractionDigits: digits })

export const pct = (v, digits = 2, signed = true) =>
  v == null ? '—' : `${signed && v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`

export const num = (v, digits = 2) => (v == null ? '—' : v.toFixed(digits))

export const signed = (v, digits = 2) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}`)

export const timeET = (ts) =>
  new Date(ts).toLocaleTimeString('en-US', { timeZone: NY, hour: 'numeric', minute: '2-digit' })

export const dateTimeET = (ts) =>
  new Date(ts).toLocaleString('en-US', { timeZone: NY, weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

export const dayLabel = (day) =>
  new Date(`${day}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

export function ago(ts) {
  const s = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

// Is the regular NYSE session open right now? (Ignores holidays.)
export function sessionOpen(now = new Date()) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', { timeZone: NY, weekday: 'short', hour: 'numeric', minute: 'numeric', hourCycle: 'h23' })
      .formatToParts(now)
      .map((p) => [p.type, p.value]),
  )
  if (parts.weekday === 'Sat' || parts.weekday === 'Sun') return false
  const m = Number(parts.hour) * 60 + Number(parts.minute)
  return m >= 570 && m < 960
}
