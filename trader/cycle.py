"""
One agent decision: risk checks -> market data -> observation -> action -> orders.

This is deploy/alpaca_live.AlpacaPPOTrader.execute_trade() made stateless.
Everything the old trader kept in memory between cycles (initial value, peak
value, step counter) lives in the agent_state table, so any machine can run
the next cycle. Every run -- traded, skipped, or failed -- leaves an
agent_cycles row explaining what happened.

A scheduler may fire late or twice, so each run belongs to a quarter-hour slot
(9:45, 10:00, ... 15:45 ET, as in alpaca_live.py) and the slot is the cycle id.
Orders are tagged "at-<slot>-<symbol>"; if the account already has orders for
the slot, the run does nothing.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from deploy.alpaca_utils import build_observation, prepare_features
from src.alpaca_intraday import EASTERN, SESSION_FIRST_BAR, SESSION_LAST_BAR, fetch_recent_universe
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


def current_slot(alpaca: Alpaca, clock):
    """The quarter hour this run trades for, or None outside 9:45-15:45 ET (holidays and early closes included)."""
    now = clock.timestamp.astimezone(EASTERN).replace(microsecond=0) - timedelta(seconds=config.CYCLE_OFFSET_SECONDS)
    slot = now.replace(minute=(now.minute // config.BAR_MINUTES) * config.BAR_MINUTES, second=0)
    session = alpaca.calendar(slot.date())
    if session is None:
        return None
    # alpaca-py gives the session's open and close as naive ET datetimes
    first = session.open.replace(tzinfo=EASTERN) + timedelta(minutes=config.BAR_MINUTES)
    last = session.close.replace(tzinfo=EASTERN) - timedelta(minutes=config.BAR_MINUTES)
    return slot if first <= slot <= last else None


def latest_features(alpaca: Alpaca):
    """Completed SIP 15-minute bars with indicators, exactly as deploy/alpaca_live.fetch_latest_data. (raw, features)."""
    raw = fetch_recent_universe(alpaca.data, config.TICKERS, calendar_days=config.BAR_LOOKBACK_DAYS)
    if raw is None:
        return None, "15-minute bars incomplete"
    # Once today's first bar is visible, the newest bar must be recent. (At the 9:45 cycle the
    # newest bar is yesterday's last one, which is expected.)
    last = next(iter(raw.values())).index[-1]
    expected = datetime.now(EASTERN).replace(tzinfo=None) - timedelta(minutes=2 * config.BAR_MINUTES)
    expected = expected.replace(minute=(expected.minute // 15) * 15, second=0, microsecond=0)
    if SESSION_FIRST_BAR <= expected.time() <= SESSION_LAST_BAR and last < expected:
        return None, f"stale data: newest bar {last}, expected {expected} or later"
    features = prepare_features(raw)
    if features is None:
        return None, "indicator warm-up: fewer than 20 bars for some tickers"
    return (raw, features), None


def run_cycle(alpaca: Alpaca, db: Database, policy: Policy, dry_run: bool = False) -> dict:
    started = time.monotonic()
    now = datetime.now(timezone.utc)

    if policy.bar_minutes != config.BAR_MINUTES:
        raise SystemExit(f"Policy {policy.version} was trained on {policy.bar_minutes}-minute bars, "
                         f"not {config.BAR_MINUTES}. Re-export it with trader/export_model.py")

    clock = alpaca.clock()
    if not clock.is_open:
        # Not recorded: the scheduler fires outside market hours too.
        print(f"Market closed. Next open: {clock.next_open}")
        return {"status": "market_closed"}
    slot = current_slot(alpaca, clock)
    if slot is None:
        print("Outside the 9:45-15:45 ET trading slots")
        return {"status": "outside_slots"}

    cycle = {"id": f"{slot:%Y%m%dT%H%M}", "ts": now, "model_version": policy.version}
    tag = f"at-{cycle['id']}-"
    if any(o["client_order_id"].startswith(tag) for o in alpaca.orders_since(slot.astimezone(timezone.utc))):
        print(f"Slot {slot:%H:%M} ET already traded; nothing to do")
        return {"status": "already_traded"}
    print(f"Slot {slot:%Y-%m-%d %H:%M} ET")

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
        acct = alpaca.account()
        value = acct["equity"]
        cycle.update(portfolio_value=value, cash=acct["cash"], buying_power=acct["buying_power"])

        # Risk limits (same thresholds as deploy/alpaca_live.py). Without a database there is no
        # stored state, so the starting and peak equity come from the account's own history.
        if state.get("initial_value") is None:
            history = [h["equity"] for h in alpaca.portfolio_history(period="1A", timeframe="1D")]
            state["initial_value"] = history[0] if history else value
            state["peak_value"] = max(history + [value])
        initial = state["initial_value"]
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

        # Market data: completed SIP 15-min bars on one grid for all tickers
        data, problem = latest_features(alpaca)
        if data is None:
            return finish("skipped", problem)
        raw, features = data
        if not dry_run:
            db.upsert("market_bars", [
                {"symbol": t, "timeframe": f"{config.BAR_MINUTES}Min", "ts": ts.tz_localize(EASTERN).to_pydatetime(),
                 "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
                for t, df in raw.items() for ts, r in df.tail(BARS_STORED_PER_CYCLE).iterrows()
            ], on_conflict="symbol,timeframe,ts")

        # Pending orders reserve buying power; clear the previous cycle's leftovers first.
        if not dry_run:
            n = alpaca.cancel_open_orders()
            if n:
                print(f"Cancelled {n} open order(s) from the previous cycle")
            acct = alpaca.account()

        # Balance and order sizing use cash, never margin: in training the agent could only spend
        # the cash it had, and a paper account's 4x buying power would let it lever up.
        cash = max(acct["cash"], 0.0)
        positions = {p["symbol"]: int(p["qty"]) for p in alpaca.positions()}
        obs = build_observation(
            features,
            balance=cash,
            shares_held=positions,
            net_worth=value, max_net_worth=peak,
            # In training the step feature is the position inside a random 1024-bar window,
            # which has no live meaning, so feed the window's midpoint (as alpaca_live.py does).
            current_step=1, max_steps=2,
            initial_balance=initial,
            feat_mean=policy.feat_mean, feat_std=policy.feat_std,
        )
        actions = policy.act(obs)
        cycle["actions"] = {t: round(float(a), 4) for t, a in zip(config.TICKERS, actions)}

        quotes = alpaca.latest_quotes(config.TICKERS)
        planned = plan_orders(actions, quotes, positions, cash, value, config.RISK)
        cycle["planned_orders"] = planned
        print(f"Equity ${value:,.2f}  |  cash ${cash:,.2f}  |  {len(planned)} orders planned")

        if dry_run:
            for o in planned:
                print(f"  [dry run] {o['side'].upper():4} {o['qty']:>5} {o['symbol']:<5} @ ~${o['price']:.2f}")
            return finish("dry_run", f"{len(planned)} orders planned, none sent")

        placed = 0
        for o in planned:
            try:
                alpaca.submit_market_order(o["symbol"], o["qty"], o["side"], tag + o["symbol"])
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
