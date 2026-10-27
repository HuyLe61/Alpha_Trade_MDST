// The signature element: the policy's raw output. One value per ticker in
// [-1, 1] from tanh; positive = buy that fraction of buying power, negative =
// sell that fraction of the shares held.
import { TICKERS } from '../lib/demo.js'
import { signed, timeET, dateTimeET } from '../lib/format.js'

// Cell fill on the buy / sell diverging scale, via CSS color-mix so it follows the theme.
export function actionFill(a) {
  if (a == null) return 'var(--cell-empty)'
  const strength = Math.round(Math.min(Math.abs(a), 1) * 100)
  return `color-mix(in oklab, var(${a >= 0 ? '--buy' : '--sell'}) ${strength}%, var(--cell-neutral))`
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

export function DecisionGrid({ cycle }) {
  if (!cycle?.actions) return <p className="empty">No decisions yet. The first one appears 15 minutes after the market opens.</p>
  const placed = new Map((cycle.planned_orders || []).map((o) => [o.symbol, o]))
  const buys = TICKERS.filter((t) => placed.get(t)?.side === 'buy').length
  const sells = TICKERS.filter((t) => placed.get(t)?.side === 'sell').length
  return (
    <figure className="decision">
      <div className="decision__grid">
        {TICKERS.map((t) => {
          const a = cycle.actions[t]
          const order = placed.get(t)
          return (
            <div
              key={t}
              className={`tile ${a >= 0 ? 'tile--buy' : 'tile--sell'} ${Math.abs(a) > 0.5 ? 'tile--strong' : ''}`}
              style={{ '--fill': actionFill(a) }}
              title={order ? `${order.side} ${order.qty} ${t}` : `${t}: below trade threshold`}
            >
              <span className="tile__ticker">{t}</span>
              <span className="tile__value">{signed(a, 2)}</span>
              {order && <span className="tile__order">{order.side === 'buy' ? '▲' : '▼'} {order.qty}</span>}
            </div>
          )
        })}
      </div>
      <figcaption className="decision__caption mono">
        Decision at {timeET(cycle.ts)} ET · {plural(buys, 'buy')} · {plural(sells, 'sell')} · {20 - buys - sells} held
      </figcaption>
    </figure>
  )
}

export function DecisionHeatmap({ cycles, limit = 52 }) {
  const cols = cycles.filter((c) => c.actions).slice(0, limit).reverse()
  if (!cols.length) return <p className="empty">The heatmap fills in as the agent makes decisions.</p>
  return (
    <div className="heatmap" role="img" aria-label={`Policy actions for the last ${cols.length} decisions, by ticker`}>
      <div className="heatmap__scroll">
        <table className="heatmap__table">
          <tbody>
            {TICKERS.map((t) => (
              <tr key={t}>
                <th scope="row" className="mono">{t}</th>
                {cols.map((c) => (
                  <td key={c.id} style={{ background: actionFill(c.actions[t]) }} title={`${t} · ${dateTimeET(c.ts)} ET · ${signed(c.actions[t], 3)}`} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="heatmap__axis mono">
        <span>{dateTimeET(cols[0].ts)}</span>
        <span>{dateTimeET(cols[cols.length - 1].ts)} ET</span>
      </div>
      <ScaleLegend />
    </div>
  )
}

export function ScaleLegend() {
  return (
    <div className="legend mono" aria-hidden="true">
      <span>sell 100% of shares −1</span>
      <span className="legend__bar" />
      <span>+1 buy with all buying power</span>
    </div>
  )
}
