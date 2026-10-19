"""
One agent decision: risk checks -> market data -> observation -> action -> orders.

This is deploy/alpaca_live.AlpacaPPOTrader.execute_trade() made stateless.
Everything the old trader kept in memory between cycles (initial value, peak
value, step counter) lives in the agent_state table, so any machine can run
the next cycle. Every run -- traded, skipped, or failed -- leaves an
agent_cycles row explaining what happened.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from deploy.alpaca_utils import build_observation, prepare_features
from trader import config
from trader.alpaca_io import Alpaca
from trader.db import Database
from trader.policy import Policy

BARS_STORED_PER_CYCLE = 8  # recent bars written to market_bars each run (2 hours)


def plan_orders(actions: np.ndarray, quotes: Dict[str, dict], positions: Dict[str, int],
                buying_power: float, portfolio_value: float, risk: dict) -> List[dict]:
    """
    Turn policy actions into order sizes, same rules as
    deploy/alpaca_utils.place_orders_from_actions (and naive_env.step):

    buy  (a > 0): spend a * remaining buying power, capped at max_position_size of the portfolio
    sell (a < 0): sell |a| of the shares held, never short
    """
    planned = []
    remaining = buying_power
    for ticker, a in zip(config.TICKERS, actions):
        a = float(a)
        q = quotes.get(ticker) or {}
        ask, bid = q.get("ask", 0), q.get("bid", 0)
        if ask <= 0 and bid <= 0:
            continue
        ask, bid = ask or bid, bid or ask
        mid = (ask + bid) / 2

        if a > 0:
            qty = int(remaining * a / ask)
            room = max(portfolio_value * risk["max_position_size"] - positions.get(ticker, 0) * ask, 0)
            qty = min(qty, int(room / ask))
            if qty <= 0 or qty * ask < risk["min_trade_value"]:
                continue
            remaining -= qty * ask
            planned.append({"symbol": ticker, "side": "buy", "qty": qty, "price": round(ask, 4)})
        elif a < 0:
            held = positions.get(ticker, 0)
            qty = min(int(held * abs(a)), held)
            if qty <= 0 or qty * mid < risk["min_trade_value"]:
                continue
            # Sell proceeds don't free buying power until the order fills, so
            # they're not added back to `remaining` (see alpaca_utils for details).
            planned.append({"symbol": ticker, "side": "sell", "qty": qty, "price": round(mid, 4)})
    return planned


def run_cycle(alpaca: Alpaca, db: Database, policy: Policy, dry_run: bool = False) -> dict:
    started = time.monotonic()
    now = datetime.now(timezone.utc)
    cycle = {
        "id": now.strftime("%Y%m%dT%H%M") + "-" + uuid.uuid4().hex[:6],
        "ts": now,
        "model_version": policy.version,
    }

    def finish(status, reason=None, **fields):
        cycle.update(status=status, reason=reason, latency_ms=int((time.monotonic() - started) * 1000), **fields)
        print(f"Cycle {cycle['id']}: {status}" + (f" ({reason})" if reason else ""))
        if not dry_run:
            db.upsert("agent_cycles", [cycle])
            db.update_state(last_cycle_at=now)
        return cycle

    state = db.get_state()
    if not state["enabled"]:
        return finish("skipped", "agent disabled (agent_state.enabled = false)")
    if state["halted"]:
        return finish("halted", state.get("halted_reason") or "halted")

    try:
        clock = alpaca.clock()
        if not clock.is_open:
            # Not recorded: the scheduler fires outside market hours too.
            print(f"Market closed. Next open: {clock.next_open}")
            return {"status": "market_closed"}

        acct = alpaca.account()
        value = acct["equity"]
        cycle.update(portfolio_value=value, cash=acct["cash"], buying_power=acct["buying_power"])

        # Risk limits (same thresholds as deploy/alpaca_live.py)
        initial = state.get("initial_value") or value
        peak = max(state.get("peak_value") or value, value)
        if not dry_run:
            db.update_state(initial_value=initial, peak_value=peak,
                            started_at=state.get("started_at") or now)

        last_close = acct["last_equity"] or value
        daily_loss = (last_close - value) / last_close
        if daily_loss > config.RISK["daily_loss_limit"]:
            return finish("skipped", f"daily loss {daily_loss:.2%} exceeds {config.RISK['daily_loss_limit']:.0%} limit")

        drawdown = (peak - value) / peak
        if drawdown > config.RISK["max_drawdown"]:
            reason = f"drawdown {drawdown:.2%} from peak exceeds {config.RISK['max_drawdown']:.0%} limit"
            if not dry_run:
                db.update_state(halted=True, halted_reason=reason)
            return finish("halted", reason)

        # Market data: completed 15-min bars per ticker
        bars = {t: alpaca.bars(t, minutes=config.BAR_MINUTES, days=config.BAR_LOOKBACK_DAYS) for t in config.TICKERS}
        missing = [t for t, df in bars.items() if df.empty]
        if missing:
            return finish("skipped", f"no IEX bars for {', '.join(missing)}")
        if not dry_run:
            db.upsert("market_bars", [
                {"symbol": t, "timeframe": f"{config.BAR_MINUTES}Min", "ts": ts.tz_localize("UTC").to_pydatetime(),
                 "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
                for t, df in bars.items() for ts, r in df.tail(BARS_STORED_PER_CYCLE).iterrows()
            ], on_conflict="symbol,timeframe,ts")

        features = prepare_features(bars)
        if features is None:
            return finish("skipped", "indicator warm-up: fewer than 20 bars for some tickers")

        # Pending orders reserve buying power; clear the previous cycle's leftovers first.
        if not dry_run:
            n = alpaca.cancel_open_orders()
            if n:
                print(f"Cancelled {n} open order(s) from the previous cycle")
            acct = alpaca.account()

        positions = {p["symbol"]: int(p["qty"]) for p in alpaca.positions()}
        obs = build_observation(
            features,
            balance=acct["buying_power"] / acct["multiplier"],  # undo 2x margin, as in alpaca_live.py
            shares_held=positions,
            net_worth=value, max_net_worth=peak,
            current_step=state.get("trade_step") or 0, max_steps=config.MAX_STEPS,
            initial_balance=initial,
            feat_mean=policy.feat_mean, feat_std=policy.feat_std,
        )
        actions = policy.act(obs)
        cycle["actions"] = {t: round(float(a), 4) for t, a in zip(config.TICKERS, actions)}

        quotes = alpaca.latest_quotes(config.TICKERS)
        planned = plan_orders(actions, quotes, positions, acct["buying_power"], value, config.RISK)
        cycle["planned_orders"] = planned

        if dry_run:
            for o in planned:
                print(f"  [dry run] {o['side'].upper():4} {o['qty']:>5} {o['symbol']:<5} @ ~${o['price']:.2f}")
            return finish("dry_run", f"{len(planned)} orders planned, none sent")

        placed = 0
        for o in planned:
            try:
                alpaca.submit_market_order(o["symbol"], o["qty"], o["side"], f"at-{cycle['id']}-{o['symbol']}")
                placed += 1
                print(f"  {o['side'].upper():4} {o['qty']:>5} {o['symbol']:<5} @ ~${o['price']:.2f}")
            except Exception as e:
                o["error"] = str(e)[:200]
                print(f"  Order failed for {o['symbol']}: {e}")

        db.update_state(trade_step=(state.get("trade_step") or 0) + 1)
        return finish("traded" if placed else "no_orders", None if placed else "all actions below trade thresholds",
                      n_orders=placed)

    except Exception as e:
        finish("error", f"{type(e).__name__}: {e}"[:500])
        raise
