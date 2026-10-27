"""
Approach 4: risk-based long-only allocation, the classical alternative to forecasting returns.

    python research/long_only.py

No return forecast at all: weights come from risk only, rebalanced once a day at the close of the
first bar after the data arrives. Compared with equal weight on the same costs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import backtest as bt
from research.costs import cost_vector
from research.panel import BARS_PER_DAY, build_panel

LAG = 1
LOOKBACK_DAYS = 60


def daily_rebalance(weights_fn, p):
    """Recompute weights on the first bar of each day from bars up to then; hold them all day."""
    T, N = p.ret.shape
    target = np.zeros((T, N))
    dates = p.index.date
    current = np.full(N, 1.0 / N)
    for t in range(T):
        if t == 0 or dates[t] != dates[t - 1]:
            lo = max(0, t - LOOKBACK_DAYS * BARS_PER_DAY)
            if t - lo > 5 * BARS_PER_DAY:
                current = weights_fn(p.ret[lo:t])
        target[t] = current
    return target


def equal_weight(r):
    return np.full(r.shape[1], 1.0 / r.shape[1])


def inverse_vol(r):
    w = 1.0 / (r.std(axis=0) + 1e-12)
    return w / w.sum()


def min_variance(r):
    """Long-only minimum variance from daily-scaled covariance, shrunk toward its diagonal."""
    cov = np.cov(r.T)
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    w = np.linalg.solve(cov, np.ones(len(cov)))
    w = np.clip(w, 0, None)
    w = np.minimum(w / w.sum(), 0.15)  # same 15% cap as the live risk rules
    return w / w.sum()


def main():
    panel = build_panel()
    cost = cost_vector(panel.tickers)
    sp = panel.splits()
    for name in ('train', 'val', 'test'):
        p = sp[name]
        print(f"\n{name}")
        for label, fn in (('equal weight', equal_weight), ('inverse vol', inverse_vol), ('min variance', min_variance)):
            m = bt.run(daily_rebalance(fn, p), p.ret, p.index.date, cost=cost, lag=LAG, mkt=p.ret.mean(1))
            print(f"  {label:13s} {bt.fmt(m)}  beta {m['beta']:+.2f}")


if __name__ == '__main__':
    main()
