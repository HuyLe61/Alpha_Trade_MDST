"""
Feature panel for the 15-minute research: every array is [time, stock] or [time, stock, feature].

All features are computed from bars up to and including bar t, and are stationary: returns,
ratios and ranks, never raw price or volume levels (those drift with the market and had no
predictive power, see research/README.md). Stock features are z-scored across the 20 stocks
at each bar, so the model compares stocks with each other at that moment.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

from src.alpaca_intraday import load_universe  # noqa: E402
from alpaca_utils import TRAINING_TICKERS      # noqa: E402

DATA_DIR = ROOT / 'data' / 'intraday'
BARS_PER_DAY = 26

TRAIN_END = '2024-12-31 23:59'
VAL = ('2025-01-01', '2025-12-31 23:59')
TEST_START = '2026-01-01'

STOCK_FEATURES = ['ret_1bar', 'ret_1h', 'ret_1d', 'ret_5d', 'bollinger_b', 'dev_ema26',
                  'rsi', 'macd', 'rel_volume', 'realized_vol']
MARKET_FEATURES = ['mkt_ret_1bar', 'mkt_ret_1h', 'mkt_ret_1d', 'time_of_day', 'first_bar']


@dataclass
class Panel:
    index: pd.DatetimeIndex
    tickers: list
    close: np.ndarray        # [T, N]
    ret: np.ndarray          # [T, N] return of bar t: close[t] / close[t-1] - 1
    stock_x: np.ndarray      # [T, N, F] cross-sectionally z-scored stock features
    market_x: np.ndarray     # [T, M] market-wide features
    valid: np.ndarray        # [T] bool: all features defined

    def slice(self, start=None, end=None):
        mask = np.ones(len(self.index), bool)
        if start is not None:
            mask &= self.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.index <= pd.Timestamp(end)
        idx = np.where(mask & self.valid)[0]
        return Panel(self.index[idx], self.tickers, self.close[idx], self.ret[idx],
                     self.stock_x[idx], self.market_x[idx], self.valid[idx])

    def splits(self):
        return {'train': self.slice(None, TRAIN_END), 'val': self.slice(*VAL), 'test': self.slice(TEST_START)}


def _xs_z(df: pd.DataFrame, clip=3.0) -> pd.DataFrame:
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0)
    return z.clip(-clip, clip)


def build_panel(tickers=TRAINING_TICKERS, data_dir=DATA_DIR, frames=None, market_scale=None) -> Panel:
    """frames: optional dict of already-aligned OHLCV DataFrames (the live trader passes recent bars)."""
    frames = frames or load_universe(tickers, data_dir)
    close = pd.DataFrame({t: frames[t]['Close'] for t in tickers})
    volume = pd.DataFrame({t: frames[t]['Volume'] for t in tickers})
    ret = close.pct_change()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    up, down = ret.clip(lower=0), (-ret).clip(lower=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-12)

    raw = {
        'ret_1bar': ret,
        'ret_1h': close.pct_change(4),
        'ret_1d': close.pct_change(BARS_PER_DAY),
        'ret_5d': close.pct_change(5 * BARS_PER_DAY),
        'bollinger_b': (close - (sma20 - 2 * sd20)) / (4 * sd20 + 1e-12),
        'dev_ema26': close / ema26 - 1,
        'rsi': 1 - 1 / (1 + rs),
        'macd': (ema12 - ema26) / close,
        'rel_volume': np.log1p(volume) - np.log1p(volume.rolling(20).mean()),
        'realized_vol': ret.rolling(BARS_PER_DAY).std(),
    }
    stock_x = np.stack([_xs_z(raw[f]).values for f in STOCK_FEATURES], axis=-1)

    mkt = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt.fillna(0)).cumprod()
    idx = close.index
    minutes = (idx.hour * 60 + idx.minute - 570) / (375.0)  # 0 at 09:30, 1 at 15:45
    first_bar = pd.Series(idx.date, index=idx) != pd.Series(idx.date, index=idx).shift(1)
    market = pd.DataFrame({
        'mkt_ret_1bar': mkt,
        'mkt_ret_1h': mkt_close.pct_change(4),
        'mkt_ret_1d': mkt_close.pct_change(BARS_PER_DAY),
        'time_of_day': minutes,
        'first_bar': first_bar.astype(float).values,
    }, index=idx)
    # Scale market returns by their volatility in the TRAINING period only (no look-ahead). The live
    # trader passes the stored constants back in via market_scale.
    if market_scale is None:
        market_scale = {c: float(market[c].loc[:TRAIN_END].std()) for c in ('mkt_ret_1bar', 'mkt_ret_1h', 'mkt_ret_1d')}
    for c, s in market_scale.items():
        market[c] = (market[c] / (s + 1e-12)).clip(-5, 5)

    valid = np.isfinite(stock_x).all(axis=(1, 2)) & np.isfinite(market.values).all(axis=1) & np.isfinite(ret.values).all(axis=1)
    panel = Panel(idx, list(tickers), close.values, ret.fillna(0).values,
                  np.nan_to_num(stock_x), np.nan_to_num(market.values), valid)
    panel.market_scale = market_scale
    return panel
