"""
Mirror the Alpaca paper account into Supabase: equity, positions, orders, and
the SPY benchmark. The website only ever reads these tables.
"""

from datetime import datetime, timedelta, timezone

from trader import config
from trader.alpaca_io import Alpaca
from trader.db import Database


def _cycle_id(client_order_id: str):
    # Agent orders are tagged "at-<cycle id>-<symbol>" in cycle.py
    if client_order_id and client_order_id.startswith("at-"):
        return client_order_id[3:].rsplit("-", 1)[0]
    return None


def sync_account(alpaca: Alpaca, db: Database, order_days: int = 7, benchmark_days: int = 10):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    acct = alpaca.account()
    db.upsert("account_snapshots", [{
        "ts": now, "equity": acct["equity"], "cash": acct["cash"], "buying_power": acct["buying_power"],
        "long_market_value": acct["long_market_value"], "last_equity": acct["last_equity"], "source": "cycle",
    }], on_conflict="ts")

    positions = alpaca.positions()
    db.upsert("positions", [{**p, "updated_at": now} for p in positions], on_conflict="symbol")
    held = ",".join(f'"{p["symbol"]}"' for p in positions)
    db.delete("positions", symbol=f"not.in.({held})" if held else "not.is.null")

    orders = alpaca.orders_since(now - timedelta(days=order_days))
    db.upsert("orders", [{**o, "cycle_id": _cycle_id(o["client_order_id"])} for o in orders], on_conflict="id")

    sync_benchmark(alpaca, db, days=benchmark_days)
    print(f"Synced: equity ${acct['equity']:,.2f}, {len(positions)} positions, {len(orders)} recent orders")


def sync_benchmark(alpaca: Alpaca, db: Database, days: int):
    df = alpaca.bars(config.BENCHMARK, minutes=None, days=days)
    db.upsert("market_bars", [
        {"symbol": config.BENCHMARK, "timeframe": "1Day", "ts": ts.tz_localize("UTC").to_pydatetime(),
         "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
        for ts, r in df.iterrows()
    ], on_conflict="symbol,timeframe,ts")


def backfill(alpaca: Alpaca, db: Database, period: str = "1A"):
    """One-off: load the account's daily equity history and a year of SPY closes."""
    history = alpaca.portfolio_history(period=period, timeframe="1D")
    db.upsert("account_snapshots", [{**h, "source": "history"} for h in history],
              on_conflict="ts", ignore_duplicates=True)
    sync_benchmark(alpaca, db, days=400)
    print(f"Backfilled {len(history)} daily equity points and SPY benchmark")
