"""
Long-only minimum-variance portfolio, rebalanced once a day on Alpaca paper trading.

This is the strategy that held up in research/README.md: no return forecast, weights come only from
risk, using exactly the weighting code that was backtested (research.long_only.min_variance):
  - covariance of the last 60 trading days of 15-minute returns, shrunk halfway to its diagonal
  - long-only, at most 15% per stock
Every trading day at 10:00:30 ET, when the first 15-minute bar of the day is available, the trader
computes target weights and trades the account to them: sells first, then buys with the cash those
sells free up. It never uses margin and keeps 2% in cash for price moves between quote and fill.

State (the last rebalance date) and every order are written to deploy/logs/, so a restart neither
trades twice in a day nor loses the history.

    python deploy/min_variance_live.py             # run forever, rebalance daily
    python deploy/min_variance_live.py --dry-run   # compute today's target and orders, submit nothing
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from alpaca_utils import TRAINING_TICKERS
from research.long_only import LOOKBACK_DAYS, min_variance
from research.panel import BARS_PER_DAY
from src.alpaca_intraday import fetch_recent_universe, make_client

LOG_DIR = Path(__file__).parent / 'logs'
STATE_FILE = LOG_DIR / 'min_variance_state.json'
TRADE_LOG = LOG_DIR / 'min_variance_orders.csv'

CASH_BUFFER = 0.02          # keep 2% of the portfolio in cash
MIN_TRADE_VALUE = 100       # skip trades smaller than $100
REBALANCE_AFTER_OPEN = timedelta(minutes=30, seconds=30)   # 10:00:30 on a normal day
FILL_TIMEOUT_SECONDS = 90


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    LOG_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_order(row):
    LOG_DIR.mkdir(exist_ok=True)
    new = not TRADE_LOG.exists()
    with TRADE_LOG.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class MinVarianceTrader:
    def __init__(self, api, data_client, dry_run=False):
        self.api = api
        self.data = data_client
        self.dry_run = dry_run

    def target_weights(self):
        """Same computation as the backtest: last 60 trading days of 15-minute returns."""
        frames = fetch_recent_universe(self.data, TRAINING_TICKERS, calendar_days=LOOKBACK_DAYS * 7 // 5 + 10)
        if frames is None:
            return None, None
        close = pd.DataFrame({t: frames[t]['Close'] for t in TRAINING_TICKERS})
        ret = close.pct_change().dropna().values[-LOOKBACK_DAYS * BARS_PER_DAY:]
        if len(ret) < 20 * BARS_PER_DAY:
            print(f"Only {len(ret)} bars of history; need at least {20 * BARS_PER_DAY}")
            return None, None
        return dict(zip(TRAINING_TICKERS, min_variance(ret))), close.index[-1]

    def prices(self):
        out = {}
        for t in TRAINING_TICKERS:
            q = self.api.get_latest_quote(t, feed='iex')
            ask, bid = float(q.ask_price or 0), float(q.bid_price or 0)
            out[t] = (ask + bid) / 2 if ask > 0 and bid > 0 else max(ask, bid, float(self.api.get_latest_trade(t, feed='iex').price))
        return out

    def submit(self, ticker, side, qty, price, weight):
        row = {'time': datetime.now().isoformat(timespec='seconds'), 'ticker': ticker, 'side': side, 'qty': qty,
               'est_price': round(price, 2), 'est_value': round(qty * price, 2), 'target_weight': round(weight, 4),
               'order_id': '', 'status': 'dry-run' if self.dry_run else 'submitted'}
        print(f"  {side.upper():4s} {qty:>5} {ticker:<5} ~${qty * price:>10,.2f}  (target {weight:6.1%})")
        if not self.dry_run:
            order = self.api.submit_order(symbol=ticker, qty=qty, side=side, type='market', time_in_force='day')
            row['order_id'] = order.id
        log_order(row)
        return row['order_id']

    def wait_for_fills(self, order_ids):
        deadline = time.time() + FILL_TIMEOUT_SECONDS
        pending = set(order_ids)
        while pending and time.time() < deadline:
            for oid in list(pending):
                if self.api.get_order(oid).status in ('filled', 'canceled', 'rejected', 'expired'):
                    pending.discard(oid)
            time.sleep(2)
        if pending:
            print(f"  {len(pending)} sell order(s) still open after {FILL_TIMEOUT_SECONDS}s")

    def rebalance(self):
        print(f"\n{'=' * 70}\nRebalance  |  {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 70}")
        weights, last_bar = self.target_weights()
        if weights is None:
            print("Could not compute target weights; will retry")
            return False
        print(f"Data through the bar starting {last_bar}")

        self.api.cancel_all_orders()
        account = self.api.get_account()
        value = float(account.portfolio_value)
        held = {p.symbol: int(float(p.qty)) for p in self.api.list_positions()}
        px = self.prices()
        investable = value * (1 - CASH_BUFFER)
        target_qty = {t: int(investable * weights[t] / px[t]) for t in TRAINING_TICKERS}
        print(f"Portfolio ${value:,.2f}  |  cash ${float(account.cash):,.2f}")

        # Anything held outside the universe is sold too
        trades = {t: target_qty.get(t, 0) - held.get(t, 0) for t in set(target_qty) | set(held)}
        sells = {t: -d for t, d in trades.items() if d < 0 and -d * px.get(t, 0) >= MIN_TRADE_VALUE}
        buys = {t: d for t, d in trades.items() if d > 0 and d * px[t] >= MIN_TRADE_VALUE}

        sell_ids = [self.submit(t, 'sell', q, px.get(t, 0), weights.get(t, 0)) for t, q in sorted(sells.items())]
        if sell_ids and not self.dry_run:
            self.wait_for_fills(sell_ids)

        # Buy with cash only, biggest shortfall first, never below the buffer
        cash = float(self.api.get_account().cash) if not self.dry_run else \
            float(account.cash) + sum(q * px.get(t, 0) for t, q in sells.items())
        budget = cash - value * CASH_BUFFER
        for t, q in sorted(buys.items(), key=lambda kv: -kv[1] * px[kv[0]]):
            q = min(q, int(max(budget, 0) / px[t]))
            if q * px[t] < MIN_TRADE_VALUE:
                continue
            self.submit(t, 'buy', q, px[t], weights[t])
            budget -= q * px[t]

        print("Target weights: " + ", ".join(f"{t} {w:.1%}" for t, w in sorted(weights.items(), key=lambda kv: -kv[1]) if w > 0.001))
        return True

    def run(self):
        state = load_state()
        print(f"Daily minimum-variance rebalance at open + {REBALANCE_AFTER_OPEN}. Last rebalance: {state.get('last_rebalance', 'never')}")
        while True:
            try:
                clock = self.api.get_clock()
                now = clock.timestamp
                if not clock.is_open:
                    wait = (clock.next_open - now + REBALANCE_AFTER_OPEN).total_seconds()
                    print(f"Market closed. Next rebalance in {wait / 3600:.1f} h")
                    time.sleep(min(max(wait, 60), 3600))
                    continue
                today = str(now.date())
                session = self.api.get_calendar(start=today, end=today)[0]
                due = datetime.combine(now.date(), session.open, tzinfo=now.tzinfo) + REBALANCE_AFTER_OPEN
                if state.get('last_rebalance') == today:
                    time.sleep(1800)
                    continue
                if now < due:
                    time.sleep(max(1, (due - now).total_seconds()))
                    continue
                if self.rebalance():
                    state['last_rebalance'] = today
                    save_state(state)
                    print(f"Done. Portfolio ${float(self.api.get_account().portfolio_value):,.2f}")
                else:
                    time.sleep(120)
            except KeyboardInterrupt:
                print("\nStopped by user")
                return
            except Exception as e:
                print(f"Error: {e}. Retrying in 60 s")
                time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='compute and print one rebalance, submit nothing')
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / '.env')
    key, secret = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    if base.endswith('/v2'):
        base = base[:-3]
    if 'paper' not in base:
        sys.exit(f"Refusing to run against {base}: this strategy is for paper trading only")
    api = tradeapi.REST(key, secret, base, api_version='v2')
    acct = api.get_account()
    print(f"Connected — status: {acct.status}  |  portfolio ${float(acct.portfolio_value):,.2f}")

    trader = MinVarianceTrader(api, make_client(key, secret), dry_run=args.dry_run)
    if args.dry_run:
        trader.rebalance()
    else:
        trader.run()


if __name__ == '__main__':
    main()
