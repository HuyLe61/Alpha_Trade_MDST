# Live paper-trading system

How two strategies trade their own Alpaca paper accounts on a schedule, with no
laptop involved, and how the public site shows the PPO agent.

```
GitHub Actions (cron, every 15 min, weekdays)                       intraday account
   └─ python -m trader tick
        ├─ cycle: slot check → risk checks → 15-min SIP bars → indicators → 263-dim obs → policy → orders
        └─ sync:  equity, positions, orders, SPY  ──►  Supabase Postgres  ──►  web/ on Vercel
                                                     (RLS: public read-only)   (Realtime updates)

GitHub Actions (cron, daily ~10:35 ET, weekdays)                    min-variance account
   └─ python deploy/min_variance_live.py --once
        └─ 60 days of 15-min returns → long-only min-variance weights → sells, then buys with cash
```

| Piece | Where | What it does |
|---|---|---|
| Worker | `trader/` | Stateless Python job. Each run makes at most one decision and records why. |
| Policy | `trader/model/policy.npz` | The PPO actor exported to numpy (270 KB), so the worker doesn't need PyTorch. |
| Database | `supabase/migrations/` | Tables `market_bars`, `account_snapshots`, `positions`, `agent_cycles`, `orders`, `agent_state` and views `equity_daily`, `benchmark_daily`. |
| Scheduler | `.github/workflows/trader.yml` | Runs every 15 min, 13:00–21:59 UTC on weekdays. The worker checks Alpaca's clock itself and trades each 9:45–15:45 ET quarter-hour slot at most once (orders tagged `at-<slot>-<symbol>`), so late or repeated runs are harmless. |
| Min variance | `.github/workflows/min-variance.yml` | Runs `deploy/min_variance_live.py --once` at 14:35, 15:35 and 16:35 UTC. It rebalances once a day after open + 30 min; orders tagged `mv-<date>-` mark the day as done. See `research/README.md` for why this strategy. |
| Site | `web/` | README-style overview (`/`) and live dashboard (`/dashboard`). |

`deploy/alpaca_live.py` and `deploy/min_variance_live.py` (without `--once`) still work
as long-running local loops, but only while the machine stays awake. The worker reuses
the live trader's feature and observation code (`deploy/alpaca_utils.py`) and the same
sizing rules (cash only, never margin) and risk limits, so both paths give the policy
identical inputs. Never run a local loop and the schedule against the same account.

## Setup

### 1. Alpaca paper keys
Each strategy has its own paper account. Generate keys under **Paper Trading → API Keys**
at https://app.alpaca.markets and put them in `deploy/.env`:

| Variables | Account |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | minimum variance |
| `ALPACA_INTRADAY_API_KEY`, `ALPACA_INTRADAY_SECRET_KEY` | intraday PPO agent |

### 2. Supabase
1. Create a project at https://supabase.com (the free tier is enough).
2. Open **SQL Editor**, paste `supabase/migrations/20261019000000_alpha_trade.sql`, and run it.
   (Or with the CLI: `supabase link` then `supabase db push`.)
3. From **Project Settings → API**, copy the project URL, the `anon` key (goes to the site), and the `service_role` key (goes to the worker only).

### 3. Try the worker locally
```bash
pip install -r trader/requirements.txt
cp trader/.env.example trader/.env     # fill in the Supabase URL + service_role key
python -m trader tick --dry-run        # computes actions, sends no orders, writes nothing
python -m trader backfill              # loads past equity + a year of SPY into Supabase
python -m trader tick                  # one real cycle (paper money)
python deploy/min_variance_live.py --dry-run   # today's min-variance target and orders, none sent
```

### 4. Schedule it
In GitHub, open **Settings → Secrets and variables → Actions** and add
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_INTRADAY_API_KEY` and `ALPACA_INTRADAY_SECRET_KEY`
(plus `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` for the site). From a terminal,
`gh secret set NAME` prompts for each value. The `trader` and `min-variance` workflows
then run on their own. You can also run either by hand from the **Actions** tab,
including a dry run.

GitHub often starts scheduled runs several minutes late and occasionally skips
one; both jobs are written for that. Logs of a public repo's runs are public
(secrets are masked), so they show the paper accounts' values and orders.

Scheduled workflows only run from the default branch. On a free account a
private repo uses about 600 of its 2,000 Actions minutes a month. A public repo
doesn't count against minutes.

### 5. Deploy the site to Vercel
1. Import the GitHub repo in Vercel and set **Root Directory** to `web`. The framework (Vite) is detected from `web/vercel.json`.
2. Add environment variables `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`.
3. Deploy. Without those variables the site runs on generated demo data and says so in a banner.

Local preview: `cd web && npm install && npm run dev`.

## Operating it
- **Pause:** disable the workflow under **Actions** (or, with Supabase, set `agent_state.enabled = false`).
  GitHub also disables scheduled workflows after 60 days without a commit to the repo.
- **After a drawdown halt** (equity more than 30% below peak): fix the cause, then set `halted = false`. Set `peak_value` to null as well if you want to reset the peak.
- **New model:** retrain, then run `python -m trader.export_model --checkpoint models/<file>.pt`
  (this needs torch). The model version shows up on every `agent_cycles` row.
- Every run writes an `agent_cycles` row with its status (`traded`, `no_orders`, `skipped`, `halted`, `error`) and a reason, so you can debug from the dashboard.

## Known gaps
- The deployed policy is `models/ppo_trading_intraday.pt` (15-minute SIP bars, 2020–2024). `research/README.md`
  found no edge for it after costs; it runs for the showcase, and the min-variance account is the benchmark.
- Without Supabase there is no stored peak equity, so the worker takes the starting and peak
  equity from the account's daily portfolio history.
