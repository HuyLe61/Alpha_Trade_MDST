import { useMemo, useState } from 'react'
import { Area, ComposedChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { compareSeries, intradaySeries } from '../lib/metrics.js'
import { dayLabel, pct, timeET, usd } from '../lib/format.js'

const RANGES = [
  { id: '1D', days: 1, intraday: true },
  { id: '1W', days: 7, intraday: true },
  { id: '1M', days: 31 },
  { id: '3M', days: 92 },
  { id: 'All', days: null },
]

export default function EquityChart({ data, defaultRange = 'All', height = 340 }) {
  const [range, setRange] = useState(defaultRange)
  const r = RANGES.find((x) => x.id === range)

  const series = useMemo(() => {
    if (r.intraday) {
      // 1D = the latest session only, not the last 24 hours.
      const last = data.intraday[data.intraday.length - 1]
      const from = r.days === 1 && last
        ? new Date(new Date(last.ts).getTime() - 7 * 3600e3).toISOString()
        : new Date(Date.now() - r.days * 864e5).toISOString()
      return intradaySeries(data.intraday, from)
    }
    const from = r.days ? new Date(Date.now() - r.days * 864e5).toISOString().slice(0, 10) : null
    return compareSeries(data.daily, data.benchmark, from)
  }, [data, r])

  const last = series[series.length - 1]
  const tickFmt = (t) => (r.intraday ? (r.days === 1 ? timeET(t) : new Date(t).toLocaleDateString('en-US', { weekday: 'short' })) : dayLabel(t).replace(/, \d{4}$/, ''))

  return (
    <div className="chart">
      <div className="chart__head">
        <div className="chart__legend mono">
          <span className="key key--agent">Agent {last ? pct(last.agent) : ''}</span>
          {!r.intraday && <span className="key key--spy">SPY {last?.spy != null ? pct(last.spy) : ''}</span>}
          {r.intraday && <span className="chart__note">SPY shown on daily ranges</span>}
        </div>
        <div className="segmented" role="tablist" aria-label="Chart range">
          {RANGES.map((x) => (
            <button key={x.id} role="tab" aria-selected={x.id === range} onClick={() => setRange(x.id)}>
              {x.id}
            </button>
          ))}
        </div>
      </div>
      {series.length < 2 ? (
        <p className="empty" style={{ height }}>Not enough history for this range yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={series} margin={{ top: 8, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="agentFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--buy)" stopOpacity={0.18} />
                <stop offset="100%" stopColor="var(--buy)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis dataKey="t" tickFormatter={tickFmt} minTickGap={48} tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} />
            <YAxis tickFormatter={(v) => pct(v, 0)} width={52} tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
            <Tooltip content={<ChartTip intraday={r.intraday} />} cursor={{ stroke: 'var(--muted)', strokeDasharray: '3 3' }} />
            <Area type="monotone" dataKey="agent" stroke="var(--buy)" strokeWidth={2} fill="url(#agentFill)" isAnimationActive={false} />
            {!r.intraday && <Line type="monotone" dataKey="spy" stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />}
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

function ChartTip({ active, payload, intraday }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="tip mono">
      <div className="tip__time">{intraday ? `${new Date(p.t).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${timeET(p.t)} ET` : dayLabel(p.t)}</div>
      <div>Equity {usd(p.equity)}</div>
      <div className="tip__agent">Agent {pct(p.agent)}</div>
      {p.spy != null && <div className="tip__spy">SPY {pct(p.spy)}</div>}
    </div>
  )
}
