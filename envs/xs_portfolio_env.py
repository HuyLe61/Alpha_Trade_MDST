"""
Cross-sectional portfolio environment for the 15-minute research (research/train_rl.py).

Differences from AlphaTradeEnv, each fixing a problem found in the first intraday model:
  - Observation: stationary features z-scored across the 20 stocks at each bar, the current weights
    and market-wide features. No raw prices or volumes.
  - Action: one score per stock, turned into dollar-neutral target weights (sum 0, gross <= 1). The
    agent can be flat by outputting equal scores. No ordering effect between tickers.
  - Reward: this decision's own P&L net of costs, in basis points. Not the cumulative return level.
  - Execution delay: weights decided at bar t start earning at bar t+lag+1, like the live trader.
  - Decisions every `decision_every` bars, so slower policies can be tested.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class XSPortfolioEnv(gym.Env):
    def __init__(self, panel, cost, lag=1, decision_every=1, episode_length=1024, random_start=True):
        super().__init__()
        self.x = panel.stock_x.astype(np.float32)           # [T, N, F]
        self.m = panel.market_x.astype(np.float32)          # [T, M]
        self.ret = panel.ret.astype(np.float64)             # [T, N]
        self.cost = np.asarray(cost, dtype=np.float64)       # per stock
        self.lag, self.k = lag, decision_every
        T, self.N, F = self.x.shape
        self.last_t = T - 1 - lag - decision_every          # last t whose P&L window exists
        self.episode_length = episode_length
        self.random_start = random_start
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.N,), dtype=np.float32)
        dim = self.N * F + self.N + self.m.shape[1]
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(dim,), dtype=np.float32)

    def _obs(self):
        return np.concatenate([self.x[self.t].ravel(), self.w.astype(np.float32) * 10, self.m[self.t]])

    @staticmethod
    def to_weights(action):
        w = np.asarray(action, dtype=np.float64)
        w = w - w.mean()
        gross = np.abs(w).sum()
        return w / gross if gross > 1 else w

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n_decisions = self.last_t // self.k
        length = min(self.episode_length, n_decisions) if self.episode_length else n_decisions
        start = int(self.np_random.integers(0, n_decisions - length + 1)) if self.random_start else 0
        self.t = start * self.k
        self.end_t = (start + length) * self.k
        self.w = np.zeros(self.N)
        return self._obs(), {}

    def step(self, action):
        w = self.to_weights(action)
        cost = float((np.abs(w - self.w) * self.cost).sum())
        window = self.ret[self.t + self.lag + 1: self.t + self.lag + 1 + self.k].sum(axis=0)
        pnl = float(w @ window) - cost
        self.w = w
        self.t += self.k
        done = self.t >= self.end_t or self.t > self.last_t
        info = {'pnl': pnl, 'cost': cost}
        return self._obs(), pnl * 1e4, done, False, info
