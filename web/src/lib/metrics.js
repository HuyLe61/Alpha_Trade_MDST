// Performance math, computed in the browser from equity_daily + benchmark_daily.

export function dailyReturns(values) {
  const out = []
  for (let i = 1; i < values.length; i++) out.push(values[i] / values[i - 1] - 1)
  return out
}

export function sharpe(values) {
  const r = dailyReturns(values)
  if (r.length < 2) return null
  const mean = r.reduce((a, b) => a + b, 0) / r.length
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (r.length - 1))
  return sd ? (mean / sd) * Math.sqrt(252) : null
}

export function maxDrawdown(values) {
  let peak = -Infinity
  let worst = 0
  for (const v of values) {
    peak = Math.max(peak, v)
    worst = Math.max(worst, (peak - v) / peak)
  }
  return worst
}

export function totalReturn(values) {
  return values.length > 1 ? values[values.length - 1] / values[0] - 1 : null
}

// Agent and SPY on a shared "% since start of range" axis, joined by day.
export function compareSeries(daily, benchmark, fromDay) {
  const spyByDay = new Map(benchmark.map((b) => [b.day, b.close]))
  const rows = daily.filter((d) => !fromDay || d.day >= fromDay)
  if (!rows.length) return []
  const base = rows[0].equity
  const spyBase = rows.map((d) => spyByDay.get(d.day)).find(Boolean)
  return rows.map((d) => {
    const spy = spyByDay.get(d.day)
    return {
      t: d.day,
      equity: d.equity,
      agent: d.equity / base - 1,
      spy: spy && spyBase ? spy / spyBase - 1 : null,
    }
  })
}

export function intradaySeries(intraday, fromTs) {
  const rows = intraday.filter((d) => !fromTs || d.ts >= fromTs)
  if (!rows.length) return []
  const base = rows[0].equity
  return rows.map((d) => ({ t: d.ts, equity: d.equity, agent: d.equity / base - 1, spy: null }))
}

export function summarize(data) {
  const values = data.daily.map((d) => d.equity)
  const series = compareSeries(data.daily, data.benchmark)
  const last = series[series.length - 1]
  const latest = data.intraday[data.intraday.length - 1]
  return {
    equity: latest?.equity ?? values[values.length - 1] ?? null,
    today: latest?.last_equity ? latest.equity / latest.last_equity - 1 : null,
    total: totalReturn(values),
    spy: last?.spy ?? null,
    sharpe: sharpe(values),
    maxDrawdown: values.length ? maxDrawdown(values) : null,
    days: values.length,
    since: data.daily[0]?.day ?? null,
  }
}
