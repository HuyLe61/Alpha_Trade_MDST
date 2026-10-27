"""
Train the PPO trading agent on 15-minute bars from Alpaca (the intraday counterpart of train.ipynb).

    python download_intraday.py      # once: caches 15-minute bars into data/intraday/
    python train_intraday.py         # trains, evaluates, saves models/ppo_trading_intraday.pt

Same environment, agent and hyperparameters as train.ipynb, with three changes that intraday data needs:
  - Episodes are random 1024-step windows (about 39 trading days). A year of 15-minute bars is 6,500
    steps, far more than one PPO rollout, so replaying from step 0 each iteration would train on the
    first two weeks of the data only.
  - Transaction cost is 0.05% per trade, not 0.5%. A 0.5% fee on every 15-minute trade would swamp any
    signal; 0.05% is a rough allowance for spread and slippage on large caps (Alpaca charges no commission).
  - Splits are by date: train 2021-2024, validation 2025, test 2026 to date.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

from agent.ppo import PPO
from envs.naive_env import AlphaTradeEnv
from src.alpaca_intraday import load_universe
from src.data_loader import process_data_with_indicators
from alpaca_utils import TRAINING_TICKERS

DATA_DIR = ROOT / 'data' / 'intraday'
SAVE_PATH = ROOT / 'models' / 'ppo_trading_intraday.pt'

TRAIN_END = '2024-12-31 23:59'
VAL_RANGE = ('2025-01-01', '2025-12-31 23:59')
TEST_START = '2026-01-01'

TRANSACTION_COST = 0.0005
EPISODE_LENGTH = 1024
N_ITERATIONS = 300
STEPS_PER_ITER = 2048
BARS_PER_YEAR = 252 * 26
SEED = 0


def add_indicators_aligned(raw):
    """
    Add the indicators, then keep only timestamps every ticker still has. The indicator step drops
    rows where RSI is undefined (a ticker flat for 14 bars) and that happens on different bars for
    different tickers, but the environment pairs rows across tickers by position.
    """
    stock_data = {}
    for ticker, df in raw.items():
        df = process_data_with_indicators({ticker: df})[ticker]
        stock_data[ticker] = df.replace([np.inf, -np.inf], np.nan).dropna()
    common = None
    for df in stock_data.values():
        common = df.index if common is None else common.intersection(df.index)
    return {ticker: df.loc[common] for ticker, df in stock_data.items()}


def split(stock_data):
    train = {t: df.loc[:TRAIN_END] for t, df in stock_data.items()}
    val = {t: df.loc[VAL_RANGE[0]:VAL_RANGE[1]] for t, df in stock_data.items()}
    test = {t: df.loc[TEST_START:] for t, df in stock_data.items()}
    return train, val, test


def evaluate(agent, stock_data, label):
    env = AlphaTradeEnv(stock_data, transaction_cost_percent=TRANSACTION_COST)
    state, _ = env.reset()
    net_worths = [env.initial_balance]
    for _ in range(env.max_steps):
        state, _, done, truncated, _ = env.step(agent.predict(state))
        net_worths.append(env.net_worth)
        if done or truncated:
            break
    nw = np.array(net_worths)
    returns = np.diff(nw) / nw[:-1]
    total = (nw[-1] / nw[0] - 1) * 100
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(BARS_PER_YEAR)
    max_dd = np.max((np.maximum.accumulate(nw) - nw) / np.maximum.accumulate(nw)) * 100

    # Reference: split the cash equally across all tickers on the first bar and hold
    first = np.array([df['Close'].iloc[0] for df in stock_data.values()])
    last = np.array([df['Close'].iloc[env.current_step] for df in stock_data.values()])
    hold = (last / first - 1).mean() * 100

    days = len(nw) / 26
    print(f"\n{label}: {len(nw) - 1} bars (~{days:.0f} trading days)")
    print(f"  PPO agent           return {total:+7.2f}%   Sharpe {sharpe:6.2f}   max drawdown {max_dd:5.1f}%")
    print(f"  Equal-weight hold   return {hold:+7.2f}%")
    return {'total_return': float(total), 'sharpe': float(sharpe), 'max_dd': float(max_dd), 'hold_return': float(hold)}


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading cached 15-minute bars...")
    stock_data = add_indicators_aligned(load_universe(TRAINING_TICKERS, DATA_DIR))
    train, val, test = split(stock_data)
    n = len(next(iter(train.values())))
    print(f"train {n:,} bars | val {len(next(iter(val.values()))):,} | test {len(next(iter(test.values()))):,}")

    train_env = AlphaTradeEnv(train, transaction_cost_percent=TRANSACTION_COST,
                              episode_length=EPISODE_LENGTH, random_start=True)
    train_env.reset(seed=SEED)
    print(f"Observation space {train_env.observation_space.shape} | action space {train_env.action_space.shape}")

    ppo = PPO(env=train_env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              n_epochs=10, batch_size=64, ent_coef=0.01, vf_coef=0.5)
    stats = ppo.train(n_iterations=N_ITERATIONS, steps_per_iter=STEPS_PER_ITER)

    results = {'validation': evaluate(ppo, val, 'Validation (2025)'),
               'test': evaluate(ppo, test, 'Test (2026 to date)')}

    SAVE_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        'policy_state_dict': ppo.policy.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'training_stats': stats,
        'obs_shape': train_env.observation_space.shape[0],
        'action_shape': train_env.action_space.shape[0],
        'feat_mean': train_env._feat_mean,
        'feat_std': train_env._feat_std,
        'bar_minutes': 15,
        'tickers': TRAINING_TICKERS,
        'transaction_cost': TRANSACTION_COST,
        'eval': results,
    }, SAVE_PATH)
    print(f"\nSaved {SAVE_PATH}")


if __name__ == '__main__':
    main()
