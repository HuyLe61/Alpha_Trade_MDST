"""
15-minute bars from Alpaca, shared by training and the live trader.

Both sides build their bars with the functions in this file, so the model sees the
same kind of input in training and in live trading:

  - feed=SIP: consolidated volume (the IEX feed carries only a few percent of it).
    The free plan allows SIP for historical bars, up to 15 minutes behind real time.
  - adjustment=ALL: prices adjusted for splits and dividends.
  - Regular session only: bars that start 09:30 to 15:45 Eastern (26 per full day).
  - Index = bar START time in Eastern, as a naive timestamp.
  - All tickers share one time grid. A ticker with no trades in a bar is filled with
    the previous close and zero volume, because the trading environment indexes every
    ticker by position and needs the rows to line up.
"""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

EASTERN = ZoneInfo("America/New_York")
BAR_MINUTES = 15
SESSION_FIRST_BAR = time(9, 30)
SESSION_LAST_BAR = time(15, 45)
# The free plan's SIP data is 15 minutes behind: ask for nothing newer than this.
SIP_DELAY = timedelta(minutes=16)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def make_client(api_key: str, secret_key: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_bars(client: StockHistoricalDataClient, symbol: str, start: datetime,
               end: Optional[datetime] = None, feed: DataFeed = DataFeed.SIP) -> pd.DataFrame:
    """Regular-session 15-minute bars for one symbol. Empty DataFrame if there are none."""
    latest = datetime.now(timezone.utc) - SIP_DELAY
    end = min(end, latest) if end else latest
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
        adjustment=Adjustment.ALL,
    )
    raw = client.get_stock_bars(request).df
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = raw.droplevel("symbol") if "symbol" in raw.index.names else raw
    df = df.rename(columns=str.capitalize)[COLUMNS]
    df.index = pd.DatetimeIndex(df.index).tz_convert(EASTERN).tz_localize(None)
    df.index.name = "Timestamp"

    # Keep only bars that have finished AND are old enough for the delayed feed to be complete:
    # one that started within the last 30 minutes (15-minute bar + 15-minute delay) may be partial.
    now_et = datetime.now(EASTERN).replace(tzinfo=None)
    df = df[df.index <= now_et - timedelta(minutes=BAR_MINUTES + 15)]
    return regular_session_only(df)


def regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    t = df.index.time
    keep = (t >= SESSION_FIRST_BAR) & (t <= SESSION_LAST_BAR)
    return df[keep & (df.index.dayofweek < 5)]


def align_universe(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Put every ticker on the union of all tickers' timestamps (see module docstring)."""
    grid = None
    for df in frames.values():
        grid = df.index if grid is None else grid.union(df.index)
    out = {}
    for symbol, df in frames.items():
        df = df.reindex(grid)
        df["Close"] = df["Close"].ffill()
        for col in ("Open", "High", "Low"):
            df[col] = df[col].fillna(df["Close"])
        df["Volume"] = df["Volume"].fillna(0.0)
        out[symbol] = df.dropna()
    return out


def cache_path(cache_dir: Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{symbol}_15min.csv"


def download_universe(client: StockHistoricalDataClient, tickers: Iterable[str], start: datetime,
                      cache_dir: Path, end: Optional[datetime] = None) -> None:
    """Download every ticker's history into cache_dir, one CSV per ticker."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for symbol in tickers:
        df = fetch_bars(client, symbol, start, end)
        df.to_csv(cache_path(cache_dir, symbol))
        print(f"  {symbol}: {len(df):,} bars, {df.index.min()} to {df.index.max()}", flush=True)


def load_universe(tickers: Iterable[str], cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read the cached CSVs back and align them on one time grid."""
    frames = {}
    for symbol in tickers:
        frames[symbol] = pd.read_csv(cache_path(cache_dir, symbol), index_col="Timestamp", parse_dates=True)
    return align_universe(frames)


def fetch_recent_universe(client: StockHistoricalDataClient, tickers: Iterable[str],
                          calendar_days: int = 20) -> Optional[Dict[str, pd.DataFrame]]:
    """Recent bars for every ticker on one grid, for the live trader. None if any ticker is empty."""
    start = datetime.now(timezone.utc) - timedelta(days=calendar_days)
    frames = {}
    for symbol in tickers:
        try:
            df = fetch_bars(client, symbol, start)
        except Exception as e:
            print(f"  {symbol}: bar request failed: {e}")
            return None
        if df.empty:
            print(f"  {symbol}: no bars returned")
            return None
        frames[symbol] = df
    return align_universe(frames)
