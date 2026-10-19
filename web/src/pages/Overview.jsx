// The project as a page: the two agents and their results, the live record,
// how the system runs, the tools behind it, and what we learned.
import { Link } from 'react-router-dom'
import EquityChart from '../components/EquityChart.jsx'
import { DecisionGrid, ScaleLegend } from '../components/Decisions.jsx'
import { Readouts } from '../components/Tables.jsx'
import { StatusPill } from '../components/Live.jsx'

const PIPELINE = [
  ['Wake up', 'GitHub Actions starts the worker every 15 minutes on weekdays. It asks Alpaca whether the market is open and stops if it isn’t.'],
  ['Check the risk limits', 'Skip the cycle if equity is down 5% from yesterday’s close. Halt the agent for good if it’s 30% below its peak.'],
  ['Pull market data', 'Seven days of finished 15-minute IEX bars for each of the 20 tickers, over Alpaca’s REST API.'],
  ['Compute indicators', 'RSI-14, Bollinger bands, EMA-12/26, MACD and relative volume. These are the same formulas used in training.'],
  ['Build the observation', '263 numbers: 12 z-scored features × 20 tickers, plus shares held, net worth, peak net worth and the step count.'],
  ['Run the policy', 'The trained agent returns one value per ticker, from −1 (sell everything) to +1 (buy with all buying power).'],
  ['Size and send orders', 'Buy with a fraction of buying power, capped at 15% of the portfolio per stock. Sell a fraction of the shares held, never short. Ignore anything under $100.'],
  ['Record everything', 'The decision, the orders, fills, positions and equity are written to Supabase, and this site picks them up in real time.'],
]

const STACK = [
  ['Agent', 'PyTorch PPO written from scratch: actor-critic, GAE, clipped surrogate objective'],
  ['Environments', 'Two custom Gymnasium environments: one trades shares for a return reward, one sets portfolio weights for a Sharpe reward'],
  ['Data', 'Yahoo Finance daily bars for training, Alpaca IEX 15-minute bars live'],
  ['Broker', 'Alpaca paper trading API (alpaca-py)'],
  ['Scheduler', 'GitHub Actions cron, one run every 15 minutes'],
  ['Database', 'Supabase Postgres with row-level security and Realtime'],
  ['Site', 'React, Vite and Recharts, hosted on Vercel'],
]

// Held-out test, 2021-01-04 to 2024-12-31, $100k start. Sharpe is annualized from daily returns.
const MODELS = [
  {
    name: 'Return reward',
    live: true,
    how: 'Each step it decides how much of each stock to buy or sell. It is rewarded for the change in portfolio value, so it chases gains and ignores how bumpy the ride is.',
    habit: 'It concentrates its bets: by the end of 2020 it held 86% of its money in NVDA.',
    stats: [['Return', '+103.0%'], ['Sharpe', '0.83'], ['Max drawdown', '32.4%']],
  },
  {
    name: 'Sharpe reward',
    how: 'Each step it sets portfolio weights across the 20 stocks and cash. It is rewarded with the running Sharpe ratio, return divided by volatility, so big swings cost it. It also sees how the stocks move together, the same covariance a minimum-variance portfolio is built from, which pulls it toward low-variance mixes.',
    habit: 'It stays spread out: it never put more than 12% in one stock and leaned toward steadier names like JNJ and BAC.',
    stats: [['Return', '+96.2%'], ['Sharpe', '1.07'], ['Max drawdown', '29.7%']],
  },
]

const BACKTEST = [
  ['Return-reward agent', 103.0, '0.83', '32.4%', 'agent'],
  ['Sharpe-reward agent', 96.2, '1.07', '29.7%', 'agent'],
  ['Equal weight, buy and hold', 108.2, '1.12', '25.9%', 'base'],
  ['Minimum variance, rebalanced monthly', 55.9, '0.77', '23.6%', 'base'],
]
const BEST_RETURN = Math.max(...BACKTEST.map((r) => r[1]))

const TAKEAWAYS = [
  ['Holding everything beat both agents.', 'Owning all 20 stocks in equal amounts returned +108% from 2021 to 2024, with the best Sharpe (1.12) and a smaller drawdown than either agent.'],
  ['The reward sets the personality.', 'Rewarding raw returns produced a concentrated, high-swing trader. Rewarding the Sharpe ratio produced a diversified one with a better Sharpe (1.07 vs 0.83) and a smaller drawdown, for about 7 points less return.'],
  ['Lower risk has a price.', 'The minimum-variance portfolio had the smallest drawdown (23.6%) but returned about half as much as equal weight.'],
  ['One backtest proves little.', 'Retraining the return-reward agent with different random seeds moved its 2020 result anywhere from +31% to +110%.'],
  ['Live is not the backtest.', 'The live agent learned from daily bars but trades 15-minute bars, so its paper results may not match the test period. It trades paper money, and none of this is investment advice.'],
]

export default function Overview({ data }) {
  return (
    <>
      <section className="hero">
        <div className="hero__text">
          <h1>A reinforcement-learning agent trading 20 stocks every 15 minutes.</h1>
          <p className="lede">
            Alpha Trade trains reinforcement-learning agents with PPO on daily prices for 20 large S&amp;P 500 stocks
            and compares two reward designs. The return-reward agent now trades a $100k Alpaca paper account on its
            own. Each tile on the right is one of its outputs from the last decision.
          </p>
          <div className="hero__actions">
            <Link className="button" to="/dashboard">Open the live dashboard</Link>
          </div>
          <StatusPill data={data} />
        </div>
        <div className="hero__grid">
          {data ? <DecisionGrid cycle={data.cycles.find((c) => c.actions)} /> : <div className="skeleton" style={{ height: 320 }} />}
          <ScaleLegend />
        </div>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>Two agents, two rewards</h2>
          <p>
            Both agents use the same PPO algorithm and the same daily prices for the 20 stocks. They learned on 2014 to
            2019, were checked on 2020, and were tested on 2021 to 2024, which neither saw during training. The only real
            difference is what they are rewarded for.
          </p>
        </header>
        <div className="models">
          {MODELS.map((m) => (
            <article key={m.name} className="panel model">
              <div className="model__head">
                <h3>{m.name}</h3>
                {m.live && <span className="badge">Trading the paper account</span>}
              </div>
              <p>{m.how}</p>
              <p className="dim">{m.habit}</p>
              <dl className="model__stats">
                {m.stats.map(([k, v]) => (
                  <div key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>
                ))}
              </dl>
            </article>
          ))}
        </div>
        <div className="panel backtest">
          <h2>Test period, January 2021 to December 2024</h2>
          <p className="panel__lede dim">Each strategy starts with $100k. The agents are compared with two classic portfolios.</p>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Strategy</th><th>Return</th><th className="r">Sharpe</th><th className="r">Max drawdown</th></tr>
              </thead>
              <tbody>
                {BACKTEST.map(([name, ret, sharpe, dd, kind]) => (
                  <tr key={name}>
                    <td className={kind === 'agent' ? 'strong' : 'dim'}>{name}</td>
                    <td>
                      <span className="bar">
                        <span className={`bar__fill bar__fill--${kind}`} style={{ width: `${(ret / BEST_RETURN) * 100}%` }} />
                        <span className="mono">+{ret.toFixed(1)}%</span>
                      </span>
                    </td>
                    <td className="mono r">{sharpe}</td>
                    <td className="mono r">{dd}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="panel__lede dim backtest__note">
            Sharpe is annualized from daily returns. The return-reward environment charges 0.5% per trade and the
            Sharpe-reward one 0.1%.
          </p>
        </div>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>Paper-trading record</h2>
          <p>Equity of the Alpaca paper account against SPY, both shown as % change from the start of the selected range.</p>
        </header>
        {data ? (
          <>
            <EquityChart data={data} />
            <Readouts data={data} />
          </>
        ) : (
          <div className="skeleton" style={{ height: 400 }} />
        )}
      </section>

      <section className="section">
        <header className="section__head">
          <h2>Workflow</h2>
          <p>Every 15 minutes while the market is open, the live agent runs these steps in order.</p>
        </header>
        <ol className="pipeline">
          {PIPELINE.map(([title, body]) => (
            <li key={title} className="pipeline__step">
              <h3>{title}</h3>
              <p>{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>System</h2>
          <p>Nothing runs on a server we have to keep alive. A scheduled job does the trading, and the database and the site are both managed services.</p>
        </header>
        <div className="arch" role="img" aria-label="GitHub Actions runs the trader worker, which talks to the Alpaca API and writes to Supabase. The website reads from Supabase.">
          <div className="arch__node"><span className="mono dim">every 15 min</span><strong>GitHub Actions</strong><span>cron trigger</span></div>
          <div className="arch__edge">runs →</div>
          <div className="arch__node arch__node--core"><span className="mono dim">scheduled job</span><strong>Trader worker</strong><span>policy · risk · orders</span></div>
          <div className="arch__edge">⇄ REST</div>
          <div className="arch__node"><span className="mono dim">paper account</span><strong>Alpaca</strong><span>bars · quotes · orders</span></div>
          <div className="arch__edge arch__edge--down">writes ↓</div>
          <div className="arch__node"><span className="mono dim">Postgres + Realtime</span><strong>Supabase</strong><span>6 tables · public read-only</span></div>
          <div className="arch__edge">streams →</div>
          <div className="arch__node"><span className="mono dim">Vercel</span><strong>This site</strong><span>React · Recharts</span></div>
        </div>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>Tools</h2>
          <p>The libraries and services behind each layer.</p>
        </header>
        <dl className="stack stack--wide">
          {STACK.map(([k, v]) => (
            <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
          ))}
        </dl>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>Takeaways</h2>
        </header>
        <ol className="takeaways">
          {TAKEAWAYS.map(([title, body]) => (
            <li key={title}><strong>{title}</strong> {body}</li>
          ))}
        </ol>
      </section>
    </>
  )
}
