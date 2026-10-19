import EquityChart from '../components/EquityChart.jsx'
import { DecisionGrid, DecisionHeatmap } from '../components/Decisions.jsx'
import { CycleLog, OrdersTable, PositionsTable, Readouts } from '../components/Tables.jsx'
import { StatusPill } from '../components/Live.jsx'
import { dateTimeET, usd } from '../lib/format.js'

export default function Dashboard({ data, error }) {
  if (error) return <p className="error">{error}</p>
  if (!data) return <div className="skeleton" style={{ height: 600, marginTop: 32 }} />
  const { state } = data
  const latest = data.cycles.find((c) => c.actions)

  return (
    <div className="dash">
      <header className="dash__head">
        <div>
          <h1>Live dashboard</h1>
          <p className="dim">
            Alpaca paper account
            {state?.started_at && <> · agent running since {dateTimeET(state.started_at)} ET</>}
            {state?.trade_step ? <> · {state.trade_step.toLocaleString()} decisions</> : null}
          </p>
        </div>
        <StatusPill data={data} />
      </header>

      <Readouts data={data} />

      <section className="panel panel--wide">
        <h2>Equity vs SPY</h2>
        <EquityChart data={data} defaultRange="1M" height={380} />
      </section>

      <section className="panel">
        <h2>Latest decision</h2>
        <DecisionGrid cycle={latest} />
      </section>

      <section className="panel">
        <h2>Agent state</h2>
        <dl className="specs specs--tight">
          <div><dt>Trading</dt><dd>{state?.halted ? 'Halted' : state?.enabled === false ? 'Paused' : 'Enabled'}</dd></div>
          <div><dt>Starting equity</dt><dd className="mono">{usd(state?.initial_value)}</dd></div>
          <div><dt>Peak equity</dt><dd className="mono">{usd(state?.peak_value)}</dd></div>
          <div><dt>Model</dt><dd className="mono">{latest?.model_version || '—'}</dd></div>
          <div><dt>Risk limits</dt><dd>15% max per stock · 5% daily loss · 30% max drawdown</dd></div>
        </dl>
        <h3 className="panel__sub">Recent runs</h3>
        <CycleLog cycles={data.cycles} />
      </section>

      <section className="panel panel--wide">
        <h2>Every decision, by ticker</h2>
        <p className="dim panel__lede">Each column is one 15-minute decision, oldest on the left. Blue cells are buy signals and orange cells are sell signals.</p>
        <DecisionHeatmap cycles={data.cycles} />
      </section>

      <section className="panel panel--wide">
        <h2>Positions</h2>
        <PositionsTable positions={data.positions} />
      </section>

      <section className="panel panel--wide">
        <h2>Orders</h2>
        <OrdersTable orders={data.orders} />
      </section>
    </div>
  )
}
