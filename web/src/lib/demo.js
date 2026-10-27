// Deterministic fake data in the exact shape of the Supabase tables, so the
// site renders before the database exists (and for design work).
export const TICKERS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'UNH', 'JNJ',
  'V', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'DIS', 'BAC', 'XOM', 'CVX',
]

const PRICES = {
  AAPL: 247, MSFT: 512, GOOGL: 238, AMZN: 221, NVDA: 182, META: 712, TSLA: 431, NFLX: 1188, UNH: 342, JNJ: 189,
  V: 344, JPM: 301, WMT: 102, MA: 571, PG: 151, HD: 389, DIS: 113, BAC: 51, XOM: 114, CVX: 154,
}

function rng(seed) {
  return () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return seed / 4294967296
  }
}

function gauss(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r())
}

function tradingDays(n, end = new Date()) {
  const days = []
  const d = new Date(end)
  while (days.length < n) {
    if (d.getDay() !== 0 && d.getDay() !== 6) days.unshift(new Date(d))
    d.setDate(d.getDate() - 1)
  }
  return days
}

// 15-min decision times (UTC) for one New York session, 9:45 to 16:00 ET (EDT offset).
function sessionTimes(day) {
  const out = []
  for (let m = 13 * 60 + 45; m <= 20 * 60; m += 15) {
    const t = new Date(day)
    t.setUTCHours(Math.floor(m / 60), m % 60, 0, 0)
    out.push(t)
  }
  return out
}

export function demoData() {
  const r = rng(7)
  const days = tradingDays(126)
  let equity = 100000
  let spy = 571
  const daily = []
  const benchmark = []
  days.forEach((d, i) => {
    if (i > 0) {
      const mkt = 0.0004 + 0.009 * gauss(r)
      spy *= 1 + mkt
      equity *= 1 + 0.0002 + 1.1 * mkt + 0.006 * gauss(r)
    }
    const day = d.toISOString().slice(0, 10)
    daily.push({ day, ts: d.toISOString(), equity: +equity.toFixed(2) })
    benchmark.push({ day, close: +spy.toFixed(2) })
  })

  // Intraday snapshots for the last 5 sessions, ending on the last daily value.
  const intraday = []
  const recent = days.slice(-5)
  recent.forEach((d, di) => {
    const close = daily[daily.length - 5 + di].equity
    const open = daily[daily.length - 6 + di].equity
    const times = sessionTimes(d)
    times.forEach((t, ti) => {
      const frac = (ti + 1) / times.length
      const noise = ti === times.length - 1 ? 0 : 140 * gauss(r)
      intraday.push({ ts: t.toISOString(), equity: +(open + (close - open) * frac + noise).toFixed(2), last_equity: open })
    })
  })

  // Agent cycles for the same sessions, newest first. Actions drift slowly
  // per ticker, like a policy reacting to slowly changing features.
  const bias = Object.fromEntries(TICKERS.map((t) => [t, gauss(r) * 0.5]))
  const cycles = []
  const orders = []
  recent.forEach((d) =>
    sessionTimes(d).forEach((t) => {
      const actions = {}
      TICKERS.forEach((tk) => {
        bias[tk] = 0.85 * bias[tk] + 0.35 * gauss(r)
        actions[tk] = +Math.tanh(bias[tk]).toFixed(4)
      })
      const id = t.toISOString().replace(/[-:]/g, '').slice(0, 13) + '-' + Math.floor(r() * 1e6).toString(16)
      const planned = TICKERS.filter((tk) => Math.abs(actions[tk]) > 0.55).map((tk) => ({
        symbol: tk,
        side: actions[tk] > 0 ? 'buy' : 'sell',
        qty: Math.max(1, Math.round((Math.abs(actions[tk]) * 4000) / PRICES[tk])),
        price: PRICES[tk],
      }))
      planned.forEach((o, k) =>
        orders.push({
          id: `${id}-${k}`,
          cycle_id: id,
          symbol: o.symbol,
          side: o.side,
          qty: o.qty,
          filled_qty: o.qty,
          filled_avg_price: +(o.price * (1 + 0.002 * gauss(r))).toFixed(2),
          order_type: 'market',
          status: 'filled',
          submitted_at: new Date(t.getTime() + 4000).toISOString(),
          filled_at: new Date(t.getTime() + 5000).toISOString(),
        }),
      )
      cycles.push({
        id,
        ts: t.toISOString(),
        status: planned.length ? 'traded' : 'no_orders',
        reason: planned.length ? null : 'all actions below trade thresholds',
        model_version: 'ppo_trading-7cc517241cfe',
        portfolio_value: equity,
        actions,
        planned_orders: planned,
        n_orders: planned.length,
        latency_ms: 2400 + Math.floor(r() * 1800),
      })
    }),
  )
  cycles.reverse()
  orders.reverse()

  const held = TICKERS.filter(() => r() > 0.35).slice(0, 12)
  const positions = held
    .map((s) => {
      const qty = Math.round((4000 + 6000 * r()) / PRICES[s])
      const entry = PRICES[s] * (1 - 0.06 * gauss(r) * 0.5)
      const mv = qty * PRICES[s]
      return {
        symbol: s, qty, side: 'long', avg_entry_price: entry, current_price: PRICES[s], market_value: mv,
        cost_basis: qty * entry, unrealized_pl: mv - qty * entry, unrealized_plpc: PRICES[s] / entry - 1,
      }
    })
    .sort((a, b) => b.market_value - a.market_value)

  return {
    intraday,
    daily,
    benchmark,
    cycles,
    orders: orders.slice(0, 60),
    positions,
    state: {
      enabled: true, halted: false, halted_reason: null, initial_value: 100000,
      peak_value: Math.max(...daily.map((d) => d.equity)), trade_step: 4012,
      started_at: daily[0].ts, last_cycle_at: cycles[0].ts,
    },
    filledCount: 4280,
    loadedAt: new Date(),
  }
}
