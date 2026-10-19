// Small live-status pieces shared by the header and pages.
import { ago, dateTimeET, sessionOpen } from '../lib/format.js'
import { isDemo } from '../lib/data.js'

export function agentStatus(data) {
  if (!data) return { tone: 'idle', label: 'Connecting…' }
  const { state, cycles } = data
  const last = cycles[0]
  if (state?.halted) return { tone: 'alert', label: `Halted: ${state.halted_reason}` }
  if (state && !state.enabled) return { tone: 'idle', label: 'Paused by operator' }
  if (!last) return { tone: 'idle', label: 'Waiting for the first decision' }
  if (isDemo) return { tone: 'idle', label: 'Sample data · not the live account' }
  if (!sessionOpen()) return { tone: 'idle', label: `Market closed · last decision ${dateTimeET(last.ts)} ET` }
  const stale = Date.now() - new Date(last.ts).getTime() > 45 * 60e3
  if (stale) return { tone: 'alert', label: `No decision since ${dateTimeET(last.ts)} ET` }
  return { tone: 'live', label: `Live · last decision ${ago(last.ts)}` }
}

export function StatusPill({ data }) {
  const s = agentStatus(data)
  return (
    <span className={`pill pill--${s.tone}`}>
      <span className="pill__dot" aria-hidden="true" />
      {s.label}
    </span>
  )
}

export function DemoBanner() {
  if (!isDemo) return null
  return (
    <div className="demo-banner" role="note">
      Showing sample data. The live paper account isn’t connected yet.
    </div>
  )
}
