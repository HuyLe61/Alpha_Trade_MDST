"""
Approach 2: a supervised forecaster. Ridge regression of each stock's return relative to the other 19
over the next H bars on all 10 stationary stock features, traded dollar-neutral with a no-trade band.

    python research/ridge_forecast.py

Fit on 2021-2024. Ridge strength, horizon and band are picked on 2025 (validation). 2026 (test) is
reported once for the picked settings.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import backtest as bt
from research.costs import cost_vector
from research.panel import STOCK_FEATURES, build_panel

LAG = 1                       # the live setup: free SIP data is one bar behind
HORIZONS = [1, 4, 26, 130]    # 15 min, 1 hour, 1 day, 1 week
LAMBDAS = [1e0, 1e2, 1e4]
BANDS = [0.0, 0.01, 0.02, 0.04, 0.08]


def target(p, horizon, lag=LAG):
    """Sum of the next `horizon` bar returns after the trade, minus the cross-sectional mean."""
    r = p.ret
    csum = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    T = len(r)
    y = np.full_like(r, np.nan)
    start, end = lag + 1, lag + horizon  # bars t+lag+1 .. t+lag+horizon
    valid = T - end
    if valid > 0:
        y[:valid] = csum[end + 1:end + 1 + valid] - csum[start:start + valid]
    return y - np.nanmean(y, axis=1, keepdims=True)


def fit_ridge(X, y, lam):
    m = np.isfinite(y)
    X, y = X[m], y[m]
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def design(p):
    T, N, F = p.stock_x.shape
    return p.stock_x.reshape(T * N, F)


def ic(pred, y):
    m = np.isfinite(y).all(axis=1)
    a = np.argsort(np.argsort(pred[m], 1), 1).astype(float)
    b = np.argsort(np.argsort(y[m], 1), 1).astype(float)
    a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    v = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return v.mean()


def main():
    panel = build_panel()
    cost = cost_vector(panel.tickers)
    sp = panel.splits()
    T, N, F = sp['train'].stock_x.shape

    rows = []
    for h, lam in itertools.product(HORIZONS, LAMBDAS):
        beta = fit_ridge(design(sp['train']), target(sp['train'], h).reshape(-1), lam)
        preds = {k: (design(v) @ beta).reshape(len(v.index), N) for k, v in sp.items()}
        ics = {k: ic(preds[k], target(v, h)) for k, v in sp.items()}
        for band in BANDS:
            w = {k: bt.band_rebalance(bt.xs_weights(preds[k]), band) for k in ('val',)}
            m = bt.run(w['val'], sp['val'].ret, sp['val'].index.date, cost=cost, lag=LAG)
            rows.append((m['sharpe'], h, lam, band, ics, beta, m))
        print(f"horizon {h:3d} bars  lambda {lam:7.0e}  IC train {ics['train']:+.4f}  val {ics['val']:+.4f}  test {ics['test']:+.4f}")

    rows.sort(key=lambda r: -r[0])
    print("\nBest 5 settings on VALIDATION (net Sharpe, measured costs, 15-min data delay):")
    for s, h, lam, band, ics, beta, m in rows[:5]:
        print(f"  horizon {h:3d}  lambda {lam:7.0e}  band {band:.2f}  {bt.fmt(m)}")

    s, h, lam, band, ics, beta, m = rows[0]
    print(f"\nChosen: horizon {h} bars, lambda {lam:g}, band {band}")
    print("  Feature weights (per 1 s.d. of cross-sectional z-score, in bps of relative return):")
    for name, b in sorted(zip(STOCK_FEATURES, beta), key=lambda x: -abs(x[1])):
        print(f"    {name:14s} {b*1e4:+.2f}")
    for name in ('train', 'val', 'test'):
        p = sp[name]
        w = bt.band_rebalance(bt.xs_weights((design(p) @ beta).reshape(len(p.index), N)), band)
        r = bt.run(w, p.ret, p.index.date, cost=cost, lag=LAG, mkt=p.ret.mean(1))
        r2 = bt.run(w, p.ret, p.index.date, cost=cost * 2, lag=LAG)
        print(f"  {name:5s} {bt.fmt(r)}  beta {r['beta']:+.2f}  | at 2x costs: Sharpe {r2['sharpe']:+.2f}")


if __name__ == '__main__':
    main()
