"""
Training orchestration: SB3 agents with AlphaTradeEnv, MVP with custom PPO + PortfolioEnvWithBaselines.
"""
import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/project") if Path("/project/envs").exists() else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import load_stock_data
from src.data_loader import add_technical_indicators


def _prepare_data(tickers, start, end):
    raw = load_stock_data(tickers, start, end)
    processed = {}
    for ticker, df in raw.items():
        if df.empty:
            continue
        processed[ticker] = add_technical_indicators(df)
    return processed


def run_training_job(
    tickers, algorithm, train_start, train_end,
    test_start, test_end, n_iterations, steps_per_iter,
    progress_callback,
):
    train_data = _prepare_data(tickers, train_start, train_end)
    test_data = _prepare_data(tickers, test_start, test_end)

    if not train_data or not test_data:
        raise ValueError("No data available for the selected tickers/dates.")

    if algorithm == "mvp":
        return _train_mvp(train_data, test_data, n_iterations, steps_per_iter, progress_callback)
    else:
        return _train_sb3(algorithm, train_data, test_data, n_iterations, steps_per_iter, progress_callback)


# ---------------------------------------------------------------------------
# MVP: custom PPO + PortfolioEnvWithBaselines
# ---------------------------------------------------------------------------

def _train_mvp(train_data, test_data, n_iterations, steps_per_iter, callback):
    from envs.portfolio_env import PortfolioEnvWithBaselines
    from agent.ppo import PPO

    env = PortfolioEnvWithBaselines(train_data)
    ppo = PPO(env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              n_epochs=10, batch_size=64, ent_coef=0.01, vf_coef=0.5)

    training_stats = {
        "iterations": [], "mean_rewards": [],
        "policy_losses": [], "value_losses": [], "entropies": [],
    }

    for iteration in range(n_iterations):
        trajectories = ppo.collect_trajectories(steps_per_iter)
        advantages, returns = ppo.compute_gae(
            trajectories["rewards"], trajectories["values"], trajectories["dones"],
        )
        update_stats = ppo.update_policy(
            trajectories["states"], trajectories["actions"],
            trajectories["log_probs"], advantages, returns,
        )

        mean_reward = float(np.mean(trajectories["rewards"]))
        training_stats["iterations"].append(iteration)
        training_stats["mean_rewards"].append(mean_reward)
        training_stats["policy_losses"].append(update_stats["policy_loss"])
        training_stats["value_losses"].append(update_stats["value_loss"])
        training_stats["entropies"].append(update_stats["entropy"])

        callback({
            "status": "training",
            "progress": (iteration + 1) / n_iterations,
            "current_iteration": iteration + 1,
            "total_iterations": n_iterations,
            "mean_reward": mean_reward,
            "policy_loss": update_stats["policy_loss"],
            "value_loss": update_stats["value_loss"],
            "entropy": update_stats["entropy"],
        })

    # Backtest on test data
    callback({"status": "backtesting", "progress": 0.95})
    test_env = PortfolioEnvWithBaselines(test_data)
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [float(test_env._value)]
    baseline_values = {name: [float(test_env.initial_value)] for name in test_env.baselines}
    steps_list = [0]

    while not done:
        action = ppo.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        portfolio_values.append(float(info.get("portfolio_value", test_env._value)))
        for name in test_env.baselines:
            baseline_values[name].append(float(info.get(f"baseline_{name}", test_env.initial_value)))
        steps_list.append(len(portfolio_values) - 1)

    comparison = test_env.get_comparison()
    metrics = {
        "total_return": float(test_env.total_return()),
        "sharpe_ratio": float(test_env.sharpe_ratio()),
        "max_drawdown": float(test_env.max_drawdown() * 100),
        "final_value": float(test_env._value),
    }

    return {
        "algorithm": "mvp",
        "training_stats": training_stats,
        "backtest": {
            "portfolio_values": portfolio_values,
            "baseline_values": baseline_values,
            "steps": steps_list,
            "metrics": metrics,
            "comparison": _serialize_comparison(comparison),
        },
    }


# ---------------------------------------------------------------------------
# SB3 agents with AlphaTradeEnv
# ---------------------------------------------------------------------------

def _train_sb3(algorithm, train_data, test_data, n_iterations, steps_per_iter, callback):
    from stable_baselines3 import PPO, DDPG, SAC, A2C, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from envs.naive_env import AlphaTradeEnv

    algo_map = {"ppo": PPO, "ddpg": DDPG, "sac": SAC, "a2c": A2C, "td3": TD3}
    AlgoClass = algo_map[algorithm]

    train_env = AlphaTradeEnv(train_data)
    total_timesteps = n_iterations * steps_per_iter

    POLICY_LOSS_KEYS = ("train/policy_gradient_loss", "train/policy_loss", "train/actor_loss")
    VALUE_LOSS_KEYS = ("train/value_loss", "train/critic_loss")
    ENTROPY_KEYS = ("train/entropy_loss", "train/ent_coef_loss", "train/ent_coef")

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.iteration = 0
            self.rewards = []
            self.policy_losses = []
            self.value_losses = []
            self.entropies = []
            self._last_p_loss = 0.0
            self._last_v_loss = 0.0
            self._last_ent = 0.0

        def _capture_losses(self):
            """Read training metrics from the SB3 logger's name_to_value dict.

            Different algorithms use different key names (e.g. actor_loss vs
            policy_gradient_loss), so we check all known variants."""
            nv = getattr(self.model.logger, "name_to_value", None)
            if not nv:
                return
            for k in POLICY_LOSS_KEYS:
                if k in nv:
                    self._last_p_loss = float(nv[k])
                    break
            for k in VALUE_LOSS_KEYS:
                if k in nv:
                    self._last_v_loss = float(nv[k])
                    break
            for k in ENTROPY_KEYS:
                if k in nv:
                    self._last_ent = float(nv[k])
                    break

        def _on_step(self):
            self._capture_losses()

            if self.n_calls % steps_per_iter == 0:
                self.iteration += 1
                mean_reward = float(np.mean(self.rewards[-steps_per_iter:])) if self.rewards else 0.0

                self.policy_losses.append(self._last_p_loss)
                self.value_losses.append(self._last_v_loss)
                self.entropies.append(self._last_ent)

                callback({
                    "status": "training",
                    "progress": self.iteration / n_iterations,
                    "current_iteration": self.iteration,
                    "total_iterations": n_iterations,
                    "mean_reward": mean_reward,
                    "policy_loss": self._last_p_loss,
                    "value_loss": self._last_v_loss,
                    "entropy": self._last_ent,
                })

            infos = self.locals.get("infos", [])
            if infos:
                for info in infos:
                    if "episode" in info:
                        self.rewards.append(float(info["episode"]["r"]))
            if "rewards" in self.locals:
                r = self.locals["rewards"]
                self.rewards.append(float(r[0]) if hasattr(r, "__getitem__") else float(r))
            return True

    cb = ProgressCallback()

    on_policy = algorithm in ("ppo", "a2c")
    kwargs = {"policy": "MlpPolicy", "env": train_env, "verbose": 0}
    if on_policy:
        kwargs["n_steps"] = min(steps_per_iter, train_env.max_steps - 1) if train_env.max_steps > 1 else steps_per_iter

    model = AlgoClass(**kwargs)
    model.learn(total_timesteps=total_timesteps, callback=cb)

    training_stats = {
        "iterations": list(range(1, len(cb.policy_losses) + 1)),
        "mean_rewards": [float(r) for r in cb.rewards[-len(cb.policy_losses):]],
        "policy_losses": cb.policy_losses,
        "value_losses": cb.value_losses,
        "entropies": cb.entropies,
    }
    # Pad if needed
    while len(training_stats["mean_rewards"]) < len(training_stats["iterations"]):
        training_stats["mean_rewards"].append(0.0)

    # Backtest
    callback({"status": "backtesting", "progress": 0.95})
    test_env = AlphaTradeEnv(test_data)
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [float(test_env.net_worth)]
    steps_list = [0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        portfolio_values.append(float(test_env.net_worth))
        steps_list.append(len(portfolio_values) - 1)

    total_ret = (test_env.net_worth / test_env.initial_balance - 1) * 100
    returns_arr = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-8)
    sharpe = float(np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)) if len(returns_arr) > 1 else 0.0
    peak = np.maximum.accumulate(portfolio_values)
    max_dd = float(np.max((peak - portfolio_values) / (peak + 1e-8)) * 100)

    metrics = {
        "total_return": float(total_ret),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": float(test_env.net_worth),
    }

    # Buy-and-hold benchmark
    bnh_values = _buy_and_hold_benchmark(test_data, test_env.initial_balance)

    return {
        "algorithm": algorithm,
        "training_stats": training_stats,
        "backtest": {
            "portfolio_values": portfolio_values,
            "baseline_values": {"buy_hold": bnh_values},
            "steps": steps_list,
            "metrics": metrics,
            "comparison": None,
        },
    }


def _buy_and_hold_benchmark(stock_data, initial_value):
    tickers = list(stock_data.keys())
    n = len(tickers)
    if n == 0:
        return [initial_value]

    alloc = initial_value / n
    shares = {}
    first_prices = {}
    for ticker in tickers:
        df = stock_data[ticker]
        if len(df) > 0:
            price = float(df["Close"].iloc[0])
            shares[ticker] = alloc / (price + 1e-8)
            first_prices[ticker] = price

    min_len = min(len(df) for df in stock_data.values())
    values = []
    for step in range(min_len):
        total = 0.0
        for ticker in tickers:
            df = stock_data[ticker]
            price = float(df["Close"].iloc[step])
            total += shares.get(ticker, 0) * price
        values.append(total)

    return values


def _serialize_comparison(comparison):
    out = {}
    for key, val in comparison.items():
        out[key] = {k: float(v) for k, v in val.items()}
    return out
