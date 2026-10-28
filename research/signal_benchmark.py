"""
Approach 1: simple rule-based signals, traded dollar-neutral (long the cheap, short the rich).

    python research/signal_benchmark.py

Every parameter is chosen on the 2021-2024 training period only. 2025 (validation) and 2026
(test) are reported for the chosen settings and nothing is tuned on them.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import backtest as bt
from research.costs import cost_vector
from research.panel import STOCK_FEATURES, build_panel

COST = None  # set in main(): measured per-stock spread + impact, see research/costs.py
HALFLIVES = [0, 2, 4, 8, 16, 32]
LAGS = {0: 'real-time data (trade at the bar close)', 1: 'free SIP feed (15 min delayed, current live setup)'}


def f(p, name):
    return p.stock_x[:, :, STOCK_FEATURES.index(name)]


SIGNALS = {
    'reversal 1 bar':  lambda p: -f(p, 'ret_1bar'),
    'reversal 1 hour': lambda p: -f(p, 'ret_1h'),
    'Bollinger fade':  lambda p: -f(p, 'bollinger_b'),
    'EMA26 fade':      lambda p: -f(p, 'dev_ema26'),
    'RSI fade':        lambda p: -f(p, 'rsi'),
    'reversal combo':  lambda p: -(f(p, 'ret_1bar') + f(p, 'ret_1h') + f(p, 'bollinger_b')) / 3,
    'momentum 5 days': lambda p: f(p, 'ret_5d'),
}


def ic_decay(p, max_lag=4):
    """Rank IC of 1-bar reversal against the return `lag` bars after the decision bar's next bar."""
    s = -f(p, 'ret_1bar')
    out = []
    for lag in range(max_lag + 1):
        fwd = np.roll(p.ret, -(1 + lag), axis=0)[:-(1 + lag)]
        a = np.argsort(np.argsort(s[:-(1 + lag)], 1), 1).astype(float)
        b = np.argsort(np.argsort(fwd, 1), 1).astype(float)
        a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
        ic = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
        out.append((ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic))))
    return out


def evaluate(p, sig, hl, lag, cost=None):
    cost = COST if cost is None else cost
    target = bt.smooth(bt.xs_weights(SIGNALS[sig](p)), hl)
    mkt = p.ret.mean(axis=1)
    return bt.run(target, p.ret, p.index.date, cost=cost, lag=lag, mkt=mkt)


def main():
    global COST
    panel = build_panel()
    COST = cost_vector(panel.tickers)
    print(f"cost per unit traded: mean {COST.mean()*1e4:.2f} bps (measured half-spread + 0.25 bp)")
    sp = panel.splits()
    print({k: len(v.index) for k, v in sp.items()})

    print("\nHow fast does the 1-bar reversal decay? Rank IC vs the return k bars later (t-stat):")
    for name, p in sp.items():
        print(f"  {name:5s} " + "  ".join(f"k={k}: {m:+.4f} [{t:+4.1f}]" for k, (m, t) in enumerate(ic_decay(p))))

    for name in ('val', 'test'):
        p = sp[name]
        m = bt.run(np.full_like(p.ret, 1 / len(panel.tickers)), p.ret, p.index.date, cost=COST, lag=0)
        print(f"\nEqual-weight long-only hold, {name}: {bt.fmt(m)}")

    results = {}
    for lag in LAGS:
        print(f"\n=== Lag {lag}: {LAGS[lag]}  |  cost {COST.mean()*1e4:.2f} bps avg per unit traded ===")
        rows = []
        for sig, hl in itertools.product(SIGNALS, HALFLIVES):
            m = evaluate(sp['train'], sig, hl, lag)
            rows.append((m['sharpe'], sig, hl, m))
        rows.sort(key=lambda r: -r[0])
        print("  Best 6 on TRAIN (net Sharpe):")
        for s, sig, hl, m in rows[:6]:
            print(f"    {sig:16s} halflife {hl:2d}  {bt.fmt(m)}")
        _, sig, hl, _ = rows[0]
        print(f"  Chosen on train: {sig}, halflife {hl}. Out of sample:")
        for name in ('val', 'test'):
            m = evaluate(sp[name], sig, hl, lag)
            print(f"    {name:5s} {bt.fmt(m)}  beta {m['beta']:+.2f}")
            results[(lag, name)] = m
        print("  Cost sensitivity, net Sharpe val / test:  " + "  ".join(
            f"x{k:g}: {evaluate(sp['val'], sig, hl, lag, cost=COST*k)['sharpe']:+.2f} / {evaluate(sp['test'], sig, hl, lag, cost=COST*k)['sharpe']:+.2f}" for k in (0, 1, 2, 4)))
        print("  Every reversal rule out of sample at its train-best half-life (net Sharpe val / test):")
        for s in [k for k in SIGNALS if k != 'momentum 5 days']:
            best = max(HALFLIVES, key=lambda h: evaluate(sp['train'], s, h, lag)['sharpe'])
            print(f"    {s:16s} hl {best:2d}: train {evaluate(sp['train'], s, best, lag)['sharpe']:+.2f}  "
                  f"val {evaluate(sp['val'], s, best, lag)['sharpe']:+.2f}  test {evaluate(sp['test'], s, best, lag)['sharpe']:+.2f}")
    return results


if __name__ == '__main__':
    main()
