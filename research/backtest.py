"""
One backtester for every approach (signal rules, ridge, RL), so the numbers are comparable.

Timing convention:
  target[t]  weights decided with information up to the close of bar t
  lag        bars between that decision and the trade. lag=0: we trade at the close of bar t
             (needs real-time data). lag=1: we trade at the close of bar t+1, which is what the
             live trader does with Alpaca's free SIP feed (15 minutes delayed).
  held[u]    weights held from the close of bar u to the close of bar u+1
  pnl[u+1] = held[u] . ret[u+1]  -  cost * sum|held[u+1] - held[u]|  (cost charged when we trade)

Costs are per unit of notional traded (5 bps = 0.0005), covering half-spread, impact and fees.
Weight drift between rebalances is ignored: at a 15-minute horizon it is second order.
"""

import numpy as np
import pandas as pd

BARS_PER_DAY = 26


def execute(target, lag):
    held = np.zeros_like(target)
    held[lag:] = target[:len(target) - lag] if lag else target
    return held


def run(target, ret, dates, cost=0.0005, lag=1, mkt=None):
    """target, ret: [T, N]; dates: [T] trading date of each bar. Returns metrics plus the net per-bar P&L."""
    held = execute(target, lag)
    gross = np.zeros(len(ret))
    gross[1:] = np.einsum('tn,tn->t', held[:-1], ret[1:])
    trades = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1]))))
    traded = trades.sum(axis=1)
    net = gross - (trades * cost).sum(axis=1)  # cost: scalar or one value per stock
    return summarize(net, gross, traded, held, dates, mkt)


def summarize(net, gross, traded, held, dates, mkt=None):
    days = pd.Series(net).groupby(np.asarray(dates)).sum().values
    gross_days = pd.Series(gross).groupby(np.asarray(dates)).sum().values
    n_days = len(days)
    equity = np.cumsum(net)
    out = {
        'ann_return': days.mean() * 252,
        'ann_vol': days.std() * np.sqrt(252),
        'sharpe': days.mean() / (days.std() + 1e-12) * np.sqrt(252),
        'gross_sharpe': gross_days.mean() / (gross_days.std() + 1e-12) * np.sqrt(252),
        'max_dd': float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) else 0.0,
        'turnover_per_day': traded.sum() / n_days,
        'cost_drag': (gross.sum() - net.sum()) / n_days * 252,
        'gross_exposure': np.abs(held).sum(axis=1).mean(),
        'net_exposure': held.sum(axis=1).mean(),
        'hit_rate_days': float((days > 0).mean()),
        'net': net,
    }
    if mkt is not None:
        b = np.polyfit(mkt[1:], gross[1:], 1)[0] if len(mkt) > 2 else 0.0
        out['beta'] = b
    return out


def xs_weights(score):
    """Dollar-neutral weights from a cross-sectional score: demean, scale to gross exposure 1."""
    w = score - score.mean(axis=1, keepdims=True)
    return w / (np.abs(w).sum(axis=1, keepdims=True) + 1e-12)


def smooth(target, halflife):
    """EWMA of target weights (halflife in bars), re-scaled to gross 1: trades less, keeps the signal."""
    if not halflife:
        return target
    a = 1 - 0.5 ** (1 / halflife)
    out = np.empty_like(target)
    out[0] = target[0]
    for t in range(1, len(target)):
        out[t] = (1 - a) * out[t - 1] + a * target[t]
    return out / (np.abs(out).sum(axis=1, keepdims=True) + 1e-12)


def band_rebalance(target, band):
    """
    Cost-aware trading with a no-trade band: keep the current weight while it is within `band` of the
    target, otherwise trade only to the edge of the band. With linear costs this is the optimal policy
    shape: small forecast changes are not worth paying the spread for.
    """
    if not band:
        return target
    out = np.empty_like(target)
    w = np.zeros(target.shape[1])
    for t in range(len(target)):
        w = np.clip(w, target[t] - band, target[t] + band)
        w = w - w.mean()  # stay dollar-neutral
        out[t] = w
    return out


def fmt(m):
    return (f"Sharpe {m['sharpe']:+5.2f} (gross {m['gross_sharpe']:+5.2f})  ret {100*m['ann_return']:+6.1f}%/yr  "
            f"vol {100*m['ann_vol']:4.1f}%  maxDD {100*m['max_dd']:4.1f}%  turnover {m['turnover_per_day']:5.1f}x/day  "
            f"cost {100*m['cost_drag']:5.1f}%/yr")
