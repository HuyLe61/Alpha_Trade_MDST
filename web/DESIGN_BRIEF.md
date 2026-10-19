# Alpha Trade site: brief for Claude Design

## What it is
A public showcase for an MDST (University of Michigan) student project. A PPO
reinforcement-learning agent paper-trades 20 US large caps on Alpaca, making a
decision every 15 minutes. The site is the project's README and its live record.

**Audience:** recruiters, MDST members, and project-showcase visitors. Most are
technical, but few are traders.
**Job:** prove the agent is actually running, show how it is doing against SPY,
and explain how it works in under two minutes.

## Pages
**`/` Overview (the README)**
1. Hero: headline, short lede, two CTAs (dashboard, GitHub), live status pill, and the **decision grid** (see below).
2. Paper-trading record: equity vs SPY chart with range tabs (1D, 1W, 1M, 3M, All), then a row of readouts (equity, today, since start, SPY same period, Sharpe, max drawdown, filled orders).
3. How one decision is made: 8 ordered steps, each linking to its source file.
4. System: GitHub Actions → trader worker ⇄ Alpaca; worker → Supabase → site.
5. Model (training range, 2020 validation result, hyperparameters) and Tools (stack by layer).
6. Caveats ("Before you read the numbers").

**`/dashboard` Live dashboard**
Header with status, the readouts, the equity chart, latest decision grid, agent
state with a log of recent runs, a ticker × decision heatmap, positions, and orders.

## Current design decisions (keep or challenge)
- **Signature element:** the policy's raw output. There are 20 tiles, each with the action in [−1, 1]. Positive means buy that fraction of buying power, negative means sell that fraction of the shares held. The heatmap is the same data over time.
- **Color:** cobalt `#2F6FED` for buy and amber `#E0702B` for sell. The site deliberately uses no red or green: it is colorblind-safe and maps directly onto the diverging output. Navy ink `#14213D` on a cool graph-paper ground `#EEF1F4`. Every color is a token in `src/styles.css :root`, with a matching dark theme.
- **Type:** Bricolage Grotesque for display, IBM Plex Sans for body, and IBM Plex Mono for all tickers, numbers, and times.
- Numbered markers appear only in the decision pipeline, because it is a real sequence.

## What I want from the redesign
- A more memorable hero. The decision grid should feel like the live instrument it is.
- A stronger visual hierarchy on the dashboard. Right now the panels have equal weight.
- A better architecture diagram than the current boxes and arrows.
- Keep it honest. Paper money only, and the caveats stay visible.

## Data contract (what the components receive)
All data comes from `src/lib/data.js → loadAll()`. Without Supabase keys it
returns demo data in the same shape (`src/lib/demo.js`).

```
intraday:  [{ ts, equity, last_equity }]                 // 15-min snapshots, last 7 days
daily:     [{ day: 'YYYY-MM-DD', ts, equity }]           // last snapshot per trading day
benchmark: [{ day, close }]                              // SPY daily closes
cycles:    [{ id, ts, status, reason, model_version, portfolio_value,
              actions: { AAPL: 0.42, ... },              // 20 tickers, [-1, 1]
              planned_orders: [{ symbol, side, qty, price }],
              n_orders, latency_ms }]                    // newest first
orders:    [{ id, cycle_id, symbol, side, qty, filled_qty, filled_avg_price,
              status, submitted_at, filled_at }]         // newest first
positions: [{ symbol, qty, avg_entry_price, current_price, market_value,
              unrealized_pl, unrealized_plpc }]
state:     { enabled, halted, halted_reason, initial_value, peak_value,
              trade_step, started_at, last_cycle_at }
filledCount: number
```
`cycles[].status` is one of `traded`, `no_orders`, `skipped`, `halted`, `error`.

## Files
- `src/pages/Overview.jsx`, `src/pages/Dashboard.jsx`: page layout and copy
- `src/components/Decisions.jsx`: decision grid, heatmap, and scale legend
- `src/components/EquityChart.jsx`: Recharts equity vs SPY chart
- `src/components/Tables.jsx`: readouts, positions, orders, and run log
- `src/components/Live.jsx`: status pill and demo banner
- `src/styles.css`: every token and style
