"""
Every call the worker makes to Alpaca, in one place (alpaca-py SDK).

Unlike deploy/alpaca_live.py, which keeps a WebSocket open all day, the worker
is stateless: each run pulls the last few days of 15-min bars over REST (the
same request alpaca_websocket._seed_history uses), acts once, and exits. That
makes it cheap to run on a schedule.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest, MarketOrderRequest


def _f(v):
    return float(v) if v not in (None, "") else None


def _enum(v):
    return getattr(v, "value", v)


class Alpaca:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    # Account
    def clock(self):
        return self.trading.get_clock()

    def account(self) -> dict:
        a = self.trading.get_account()
        return {
            "equity": _f(a.equity),
            "cash": _f(a.cash),
            "buying_power": _f(a.buying_power),
            "long_market_value": _f(a.long_market_value),
            "last_equity": _f(a.last_equity),
            "multiplier": _f(a.multiplier) or 1.0,
        }

    def positions(self) -> List[dict]:
        return [{
            "symbol": p.symbol,
            "qty": _f(p.qty),
            "side": _enum(p.side),
            "avg_entry_price": _f(p.avg_entry_price),
            "current_price": _f(p.current_price),
            "market_value": _f(p.market_value),
            "cost_basis": _f(p.cost_basis),
            "unrealized_pl": _f(p.unrealized_pl),
            "unrealized_plpc": _f(p.unrealized_plpc),
        } for p in self.trading.get_all_positions()]

    def portfolio_history(self, period="1A", timeframe="1D") -> List[dict]:
        h = self.trading.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
        return [
            {"ts": datetime.fromtimestamp(t, tz=timezone.utc), "equity": e}
            for t, e in zip(h.timestamp, h.equity) if e  # zeros = before the account was funded
        ]

    # Market data
    def bars(self, symbol: str, minutes: int = None, days: int = 7) -> pd.DataFrame:
        """
        Completed bars as a DataFrame[Open, High, Low, Close, Volume] with a
        tz-naive UTC index -- the same shape IEXStream15MinFetcher produces, so
        deploy/alpaca_utils.prepare_features works unchanged. minutes=None means daily.
        """
        tf = TimeFrame(minutes, TimeFrameUnit.Minute) if minutes else TimeFrame.Day
        now = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=now - timedelta(days=days), feed=DataFeed.IEX)
        rows = self.data.get_stock_bars(req).data.get(symbol, [])
        if minutes:
            # Drop the bar that's still forming; the policy only ever saw finished bars.
            rows = [b for b in rows if b.timestamp + timedelta(minutes=minutes) <= now]
        df = pd.DataFrame(
            [(b.timestamp, b.open, b.high, b.low, b.close, b.volume) for b in rows],
            columns=["Date", "Open", "High", "Low", "Close", "Volume"],
        ).set_index("Date")
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df

    def latest_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        quotes = self.data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX))
        return {s: {"bid": float(q.bid_price or 0), "ask": float(q.ask_price or 0)} for s, q in quotes.items()}

    # Orders
    def cancel_open_orders(self) -> int:
        return len(self.trading.cancel_orders() or [])

    def submit_market_order(self, symbol: str, qty: int, side: str, client_order_id: str):
        return self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        ))

    def orders_since(self, after: datetime) -> List[dict]:
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=after, limit=500))
        return [{
            "id": str(o.id),
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "side": _enum(o.side),
            "qty": _f(o.qty),
            "filled_qty": _f(o.filled_qty),
            "filled_avg_price": _f(o.filled_avg_price),
            "order_type": _enum(o.order_type or o.type),
            "status": _enum(o.status),
            "submitted_at": o.submitted_at,
            "filled_at": o.filled_at,
            "updated_at": o.updated_at,
        } for o in orders]
