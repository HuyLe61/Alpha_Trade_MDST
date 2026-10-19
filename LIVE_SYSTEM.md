# Live paper-trading system

How the trained PPO agent trades the Alpaca paper account on its own, and how
the public site shows it.

```
GitHub Actions (cron, every 15 min, weekdays)
   └─ python -m trader tick
        ├─ cycle: risk checks → 15-min IEX bars → indicators → 263-dim obs → policy → orders
        └─ sync:  equity, positions, orders, SPY  ──►  Supabase Postgres  ──►  web/ on Vercel
                                                     (RLS: public read-only)   (Realtime updates)
```

| Piece | Where | What it does |
|---|---|---|
| Worker | `trader/` | Stateless Python job. Each run makes at most one decision and records why. |
| Policy | `trader/model/policy.npz` | The PPO actor exported to numpy (270 KB), so the worker doesn't need PyTorch. |
| Database | `supabase/migrations/` | Tables `market_bars`, `account_snapshots`, `positions`, `agent_cycles`, `orders`, `agent_state` and views `equity_daily`, `benchmark_daily`. |
| Scheduler | `.github/workflows/trader.yml` | Runs every 15 min, 13:00–21:59 UTC on weekdays. The worker checks Alpaca's clock itself. |
| Site | `web/` | README-style overview (`/`) and live dashboard (`/dashboard`). |

`deploy/alpaca_live.py` still works as the long-running WebSocket version. The
worker reuses its feature and observation code (`deploy/alpaca_utils.py`) and
the same sizing rules and risk limits, so both paths give the policy identical inputs.

## Setup

### 1. Alpaca paper keys
The keys in `deploy/.env` are rejected by Alpaca (HTTP 401), so generate new
ones under **Paper Trading → API Keys** at https://app.alpaca.markets and put
them in `deploy/.env`.

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
```

### 4. Schedule it
In GitHub, open **Settings → Secrets and variables → Actions** and add
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
The `trader` workflow then runs on its own. You can also run it by hand from
the **Actions** tab with any command, including a dry run.

Scheduled workflows only run from the default branch. On a free account a
private repo uses about 600 of its 2,000 Actions minutes a month. A public repo
doesn't count against minutes.

### 5. Deploy the site to Vercel
1. Import the GitHub repo in Vercel and set **Root Directory** to `web`. The framework (Vite) is detected from `web/vercel.json`.
2. Add environment variables `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`.
3. Deploy. Without those variables the site runs on generated demo data and says so in a banner.

Local preview: `cd web && npm install && npm run dev`.

## Operating it
- **Pause:** in Supabase's table editor, set `agent_state.enabled = false`.
- **After a drawdown halt** (equity more than 30% below peak): fix the cause, then set `halted = false`. Set `peak_value` to null as well if you want to reset the peak.
- **New model:** retrain, then run `python -m trader.export_model --checkpoint models/<file>.pt`
  (this needs torch). The model version shows up on every `agent_cycles` row.
- Every run writes an `agent_cycles` row with its status (`traded`, `no_orders`, `skipped`, `halted`, `error`) and a reason, so you can debug from the dashboard.

## Known gaps
- `models/ppo_trading.pt` has no `feat_mean` / `feat_std`, so `deploy/alpaca_live.py` refuses to load it. The export rebuilds them from `data/*.csv` exactly the way `AlphaTradeEnv` computes them.
- The CSVs in `data/` are stored by yfinance as Close, High, Low, Open, Volume, but `train.ipynb` relabels them as Open, High, Low, Close, Volume. The policy was trained with that column mix-up, while live bars arrive in the true order.
- The policy was trained on daily bars and trades 15-minute bars normalized with daily statistics.
