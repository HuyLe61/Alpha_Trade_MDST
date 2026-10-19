import { dateTimeET, num, pct, usd, timeET } from '../lib/format.js'
import { summarize } from '../lib/metrics.js'

export function Readouts({ data }) {
  const s = summarize(data)
  const items = [
    ['Equity', usd(s.equity)],
    ['Today', pct(s.today), s.today],
    ['Since start', pct(s.total), s.total],
    ['SPY, same period', pct(s.spy), s.spy],
    ['Sharpe (daily, ann.)', num(s.sharpe)],
    ['Max drawdown', pct(s.maxDrawdown == null ? null : -s.maxDrawdown, 1, false)],
    ['Filled orders', data.filledCount.toLocaleString()],
  ]
  return (
    <dl className="readouts">
      {items.map(([label, value, sign]) => (
        <div key={label} className="readout">
          <dt>{label}</dt>
          <dd className={`mono ${sign > 0 ? 'up' : sign < 0 ? 'down' : ''}`}>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function PositionsTable({ positions }) {
  if (!positions.length) return <p className="empty">No open positions. The agent is fully in cash.</p>
  const total = positions.reduce((a, p) => a + (p.market_value || 0), 0)
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr><th>Ticker</th><th className="r">Shares</th><th className="r">Avg cost</th><th className="r">Price</th><th className="r">Value</th><th className="r">Weight</th><th className="r">Unrealized</th></tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.symbol}>
              <td className="mono strong">{p.symbol}</td>
              <td className="r mono">{p.qty}</td>
              <td className="r mono">{usd(p.avg_entry_price, 2)}</td>
              <td className="r mono">{usd(p.current_price, 2)}</td>
              <td className="r mono">{usd(p.market_value)}</td>
              <td className="r mono">{pct(p.market_value / total, 1, false)}</td>
              <td className={`r mono ${p.unrealized_pl >= 0 ? 'up' : 'down'}`}>{usd(p.unrealized_pl)} <span className="dim">{pct(p.unrealized_plpc, 1)}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function OrdersTable({ orders }) {
  if (!orders.length) return <p className="empty">No orders yet.</p>
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr><th>Submitted (ET)</th><th>Side</th><th>Ticker</th><th className="r">Qty</th><th className="r">Fill price</th><th>Status</th></tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.id}>
              <td className="mono dim">{o.submitted_at ? dateTimeET(o.submitted_at) : '—'}</td>
              <td><span className={`side side--${o.side}`}>{o.side}</span></td>
              <td className="mono strong">{o.symbol}</td>
              <td className="r mono">{o.filled_qty || o.qty}</td>
              <td className="r mono">{o.filled_avg_price ? usd(o.filled_avg_price, 2) : '—'}</td>
              <td className="dim">{o.status.replace(/_/g, ' ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const STATUS_TEXT = { traded: 'Traded', no_orders: 'No trade', skipped: 'Skipped', halted: 'Halted', error: 'Error', dry_run: 'Dry run' }

export function CycleLog({ cycles }) {
  if (!cycles.length) return <p className="empty">No agent runs recorded yet.</p>
  return (
    <ol className="log">
      {cycles.slice(0, 30).map((c) => (
        <li key={c.id} className={`log__row log__row--${c.status}`}>
          <span className="mono dim">{timeET(c.ts)}</span>
          <span className="log__status">{STATUS_TEXT[c.status] || c.status}</span>
          <span className="log__detail">{c.n_orders ? `${c.n_orders} orders placed` : c.reason || '—'}</span>
          <span className="mono dim r">{c.latency_ms ? `${(c.latency_ms / 1000).toFixed(1)}s` : ''}</span>
        </li>
      ))}
    </ol>
  )
}
