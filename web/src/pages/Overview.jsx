// The project README as a page: what the agent is, how one decision is made,
// how the system runs, and what the numbers do and don't mean.
import { Link } from 'react-router-dom'
import EquityChart from '../components/EquityChart.jsx'
import { DecisionGrid, ScaleLegend } from '../components/Decisions.jsx'
import { Readouts } from '../components/Tables.jsx'
import { StatusPill } from '../components/Live.jsx'
import { REPO } from '../lib/data.js'

const PIPELINE = [
  ['Wake up', 'GitHub Actions starts the worker every 15 minutes on weekdays. It asks Alpaca whether the market is open and stops if it isn’t.', 'trader/__main__.py'],
  ['Check the risk limits', 'Skip the cycle if equity is down 5% from yesterday’s close. Halt the agent for good if it’s 30% below its peak.', 'trader/cycle.py'],
  ['Pull market data', 'Seven days of finished 15-minute IEX bars for each of the 20 tickers, over Alpaca’s REST API.', 'trader/alpaca_io.py'],
  ['Compute indicators', 'RSI-14, Bollinger bands, EMA-12/26, MACD and relative volume. These are the same formulas used in training.', 'src/data_loader.py'],
  ['Build the observation', '263 numbers: 12 z-scored features × 20 tickers, plus shares held, net worth, peak net worth and the step count.', 'deploy/alpaca_utils.py'],
  ['Run the policy', 'The PPO actor network (263 → 256 → 20, tanh output) returns one value per ticker in [−1, 1].', 'trader/policy.py'],
  ['Size and send orders', 'Buy with a fraction of buying power, capped at 15% of the portfolio per stock. Sell a fraction of the shares held, never short. Ignore anything under $100.', 'trader/cycle.py'],
  ['Record everything', 'The decision, the orders, fills, positions and equity are written to Supabase, and this site picks them up in real time.', 'trader/sync.py'],
]

const STACK = [
  ['Agent', 'PyTorch PPO written from scratch: actor-critic, GAE, clipped surrogate objective'],
  ['Environment', 'Gymnasium AlphaTradeEnv, PnL reward, 0.5% transaction cost'],
  ['Data', 'yfinance daily bars for training, Alpaca IEX 15-minute bars live'],
  ['Broker', 'Alpaca paper trading API (alpaca-py)'],
  ['Scheduler', 'GitHub Actions cron, one run every 15 minutes'],
  ['Database', 'Supabase Postgres with row-level security and Realtime'],
  ['Site', 'React, Vite and Recharts, hosted on Vercel'],
]

const HYPERPARAMS = [
  ['Learning rate', '3e-4'], ['Discount γ', '0.99'], ['GAE λ', '0.95'], ['Clip ε', '0.2'],
  ['Epochs / update', '10'], ['Minibatch', '64'], ['Entropy coef.', '0.01'], ['Value coef.', '0.5'],
  ['Steps / iteration', '2,048'], ['Iterations', '200'],
]

// The yfinance CSVs store columns as Close, High, Low, Open, Volume. Training used to
// relabel them by position, which swapped Open and Close.
const COLUMN_ROWS = [
  ['Stored in the file', ['Close', 'High', 'Low', 'Open', 'Volume'], null],
  ['Read as, before the fix', ['Open', 'High', 'Low', 'Close', 'Volume'], 'wrong'],
  ['Read as, now', ['Close', 'High', 'Low', 'Open', 'Volume'], 'right'],
]
const STORED = COLUMN_ROWS[0][1]

const VALIDATION_RUNS = [
  ['Figure previously shown on this page', '+39.6%', '1.00', '34.3%'],
  ['Model live before the fix', '+57.0%', '1.32', '31.1%'],
  ['Model live now', '+110.0%', '1.64', '37.0%'],
]

export default function Overview({ data }) {
  return (
    <>
      <section className="hero">
        <div className="hero__text">
          <p className="eyebrow">MDST · University of Michigan</p>
          <h1>A reinforcement-learning agent trading 20 stocks every 15 minutes.</h1>
          <p className="lede">
            Alpha Trade is a PPO policy trained on ten years of S&amp;P 500 prices. It now runs on its own against a
            $100k Alpaca paper account. Each tile on the right is one of its outputs from the last decision.
          </p>
          <div className="hero__actions">
            <Link className="button" to="/dashboard">Open the live dashboard</Link>
            <a className="button button--quiet" href={REPO}>Read the code</a>
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
          <h2>How one decision is made</h2>
          <p>These steps run in order every 15 minutes while the market is open. Each one links to the file that does it.</p>
        </header>
        <ol className="pipeline">
          {PIPELINE.map(([title, body, file]) => (
            <li key={title} className="pipeline__step">
              <h3>{title}</h3>
              <p>{body}</p>
              <a className="mono file" href={`${REPO}/blob/main/${file}`}>{file}</a>
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
          <div className="arch__node arch__node--core"><span className="mono dim">python -m trader tick</span><strong>Trader worker</strong><span>policy · risk · orders</span></div>
          <div className="arch__edge">⇄ REST</div>
          <div className="arch__node"><span className="mono dim">paper account</span><strong>Alpaca</strong><span>bars · quotes · orders</span></div>
          <div className="arch__edge arch__edge--down">writes ↓</div>
          <div className="arch__node"><span className="mono dim">Postgres + Realtime</span><strong>Supabase</strong><span>6 tables · public read-only</span></div>
          <div className="arch__edge">streams →</div>
          <div className="arch__node"><span className="mono dim">Vercel</span><strong>This site</strong><span>React · Recharts</span></div>
        </div>
      </section>

      <section className="section split">
        <div>
          <header className="section__head">
            <h2>Model</h2>
            <p>
              Trained on daily bars from 2014 to 2019 and tuned on 2020. On that 2020 validation year it returned{' '}
              <strong className="mono">+110.0%</strong> with a Sharpe ratio of <strong className="mono">1.64</strong> and a{' '}
              <strong className="mono">37.0%</strong> max drawdown.
            </p>
          </header>
          <dl className="specs">
            {HYPERPARAMS.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>
            ))}
          </dl>
        </div>
        <div>
          <header className="section__head">
            <h2>Tools</h2>
            <p>The libraries and services behind each layer.</p>
          </header>
          <dl className="stack">
            {STACK.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
        </div>
      </section>

      <section className="section">
        <header className="section__head">
          <h2>The column mix-up we fixed</h2>
          <p>
            The price files from yfinance list their columns as Close, High, Low, Open, Volume. Training renamed them by
            position as Open, High, Low, Close, Volume, so the model learned with the open and close prices swapped. RSI,
            Bollinger bands, the EMAs and MACD all read the close column, so they were built from opening prices too.
            Live bars from Alpaca arrive in the right order, which meant the trader fed the model data that didn’t match
            its training.
          </p>
        </header>
        <div className="fix">
          <div className="panel">
            <h2>Column by column</h2>
            <div className="table-wrap">
              <table className="colmap">
                <thead>
                  <tr>
                    <td />
                    {STORED.map((_, i) => <th key={i} scope="col" className="mono">{i + 1}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {COLUMN_ROWS.map(([label, cols, kind]) => (
                    <tr key={label}>
                      <th scope="row">{label}</th>
                      {cols.map((c, i) => {
                        const state = !kind ? '' : c === STORED[i] ? (kind === 'right' ? ' colmap__cell--ok' : '') : ' colmap__cell--bad'
                        return <td key={i} className={`mono colmap__cell${state}`}>{c}</td>
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="panel__lede dim">
              The loaders now read the header row and pick each column by name, so its position in the file no longer
              matters. The fix is in <span className="mono">train.ipynb</span>, <span className="mono">load_data.ipynb</span>{' '}
              and <span className="mono">trader/export_model.py</span>.
            </p>
            <pre className="snippet"><code>{`df = pd.read_csv(path, header=0,
                 skiprows=[1, 2], index_col=0)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]`}</code></pre>
          </div>
          <div className="panel">
            <h2>2020 validation year</h2>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>Model</th><th className="r">Return</th><th className="r">Sharpe</th><th className="r">Max drawdown</th></tr>
                </thead>
                <tbody>
                  {VALIDATION_RUNS.map(([label, ret, sharpe, dd], i) => (
                    <tr key={label} className={i === VALIDATION_RUNS.length - 1 ? 'strong' : ''}>
                      <td>{label}</td>
                      <td className="mono r">{ret}</td>
                      <td className="mono r">{sharpe}</td>
                      <td className="mono r">{dd}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="panel__lede dim fix__note">
              Don’t read the jump as the fix making the model better. Three more training runs on the corrected data,
              with different random seeds, returned <strong className="mono">+31.4%</strong> to{' '}
              <strong className="mono">+32.2%</strong> (Sharpe 0.83 to 0.92). The live model ended 2020 with 86% of its
              money in NVDA, which is most of its lead. The real gain is that it now trains on the same kind of data it
              trades on.
            </p>
            <p className="panel__lede dim">
              The old +39.6% came from a training run that was never saved. The model that was actually live scored +57.0%.
            </p>
          </div>
        </div>
      </section>

      <section className="section caveats">
        <header className="section__head">
          <h2>Before you read the numbers</h2>
        </header>
        <ul>
          <li>This is paper money. Alpaca simulates the fills, and there is no slippage beyond the quoted spread.</li>
          <li>The policy was trained on daily bars but trades on 15-minute bars. Its features are z-scored with the daily training statistics.</li>
          <li>Prices come from IEX only, which is a small share of US volume, so quotes can be thin early in the session.</li>
          <li>GitHub can start scheduled runs a few minutes late, so decisions don’t always land exactly on the quarter hour.</li>
          <li>None of this is investment advice.</li>
        </ul>
      </section>
    </>
  )
}
