"""
Alpha Trade live worker.

    python -m trader tick        # what the scheduler runs: trade (if market open), then sync
    python -m trader cycle       # one agent decision only
    python -m trader sync        # mirror account / positions / orders / SPY into Supabase
    python -m trader backfill    # one-off: load past equity history + a year of SPY
    python -m trader tick --dry-run   # compute actions, send no orders, write nothing
"""

import argparse
import sys
import time

from trader import config
from trader.alpaca_io import Alpaca
from trader.cycle import run_cycle
from trader.db import Database
from trader.policy import Policy
from trader.sync import backfill, sync_account


def main():
    parser = argparse.ArgumentParser(prog="python -m trader")
    parser.add_argument("command", choices=["tick", "cycle", "sync", "backfill"])
    parser.add_argument("--dry-run", action="store_true", help="no orders, no database writes")
    args = parser.parse_args()

    if not config.ALPACA_PAPER:
        sys.exit("Refusing to run: ALPACA_PAPER=false. This worker is for paper trading only.")

    if config.ALPACA_API_KEY and config.ALPACA_API_KEY == config.MIN_VARIANCE_API_KEY:
        sys.exit("Refusing to run: ALPACA_INTRADAY_API_KEY is the minimum-variance account's key.")

    alpaca = Alpaca(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    db = Database(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY, write=not args.dry_run)
    if not db.online:
        print("Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY); running offline")

    if args.command in ("tick", "cycle"):
        result = run_cycle(alpaca, db, Policy(), dry_run=args.dry_run)
        if result.get("n_orders"):
            time.sleep(5)  # let market orders fill so the sync below records them
    if args.command in ("tick", "sync"):
        sync_account(alpaca, db)
    if args.command == "backfill":
        backfill(alpaca, db)


if __name__ == "__main__":
    main()
