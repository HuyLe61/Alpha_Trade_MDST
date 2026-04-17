# Alpha Trade

A deep reinforcement learning trading bot and backtesting framework. We train a custom PPO (Proximal Policy Optimization) agent from scratch to trade a portfolio of 20 S&P 500 stocks, then deploy it to paper trade via Alpaca.

## Getting Started

```bash
# 1) Clone and enter the repo
git clone https://github.com/your-org/Alpha_Trade_MDST.git
cd Alpha_Trade_MDST

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Choose your workflow
#    - Classic single-asset training: load_data.ipynb -> train.ipynb
#    - Portfolio training/backtesting: train_portfolio.ipynb
#    - Hyperparameter tuning (Optuna): scripts/tune_ppo_portfolio.ipynb
```

## Project Structure

```
Alpha_Trade_MDST/
├── agent/                              # RL model + optimization code
│   ├── actor_critic.py                 # Policy/value network architecture
│   ├── ppo.py                          # Core PPO training loop (main agent logic)
│   ├── actor_critic_todo.py
│   └── ppo_todo.py
├── envs/                               # Gymnasium environments
│   ├── naive_env.py                    # Earlier single-env prototype
│   ├── portfolio_env.py                # Main portfolio environment used in portfolio training
│   └── portfolio_env_todo.py
├── src/                                # Data + portfolio core utilities
│   ├── data_loader.py                  # CSV ingestion, indicators, time-based splits
│   ├── alpha_portfolio.py              # Portfolio accounting + rebalancing logic
│   └── alpha_portfolio_todo.py
├── deploy/                             # Alpaca live/paper trading integration
│   ├── alpaca_live.py                  # Paper/live trading execution entrypoint
│   ├── alpaca_websocket.py             # Streaming market/order events
│   ├── alpaca_utils.py                 # Order/account helper functions
│   ├── alpaca_live_todo.py
│   └── alpaca_utils_todo.py
├── load_data.ipynb                     # Data download + preprocessing notebook
├── train.ipynb                         # Baseline PPO training notebook
├── train_portfolio.ipynb               # Portfolio PPO training + backtesting notebook
├── tune_ppo_portfolio.ipynb            # Optuna tuning notebook (portfolio PPO)
├── main.ipynb                          # Combined experimentation notebook
├── requirements.txt
└── README.md
```

### Schedule

| Week | Topic | Slides | Files |
| :--- | :--- | :--- | :--- |
| 1 | Data collection, technical indicators, EDA | [Slides](https://docs.google.com/presentation/d/1MB4WktJqUcoYgy9vfiVS6eWCyiRjWCXWML_GCmgXu08/edit?slide=id.p#slide=id.p) | `load_data.ipynb`, `src/data_loader.py` |
| 2 | Trading environment, observation/action spaces | [Slides](https://docs.google.com/presentation/d/1Fz_zIOxDV1Mw2xFspRWnuhvosbqI8gxhD8HC0k2AinY/edit?slide=id.p#slide=id.p) | `envs/naive_env.py`, `train.ipynb` |
| 3 | Actor-Critic architecture, PPO implementation | [Slides](https://docs.google.com/presentation/d/1dXnemZr9gQ1cQiRfPlmK6yuyFUM6mT_9ZZIjgj29_BQ/edit?slide=id.p#slide=id.p) | `agent/actor_critic.py`, `agent/ppo.py` |
| 4 | Deployment & Alpaca integration | [Slides](https://docs.google.com/presentation/d/1ayk8Eq6rLssic1OnMZ1bfabPMiaxi5ssq6gawcAFkos/edit) | `deploy/alpaca_live.py`, `deploy/alpaca_utils.py`, `deploy/alpaca_websocket.py` |
| 5 | Buffer / integration week | - | `main.ipynb`, `train.ipynb` |
| 6 | Modern Portfolio Theory & CAPM | [Slides](https://docs.google.com/presentation/d/1sOPQYjc88CTajQBT4rOWRIFPB_9X5aQTcbGegdYXhrA/edit?slide=id.p#slide=id.p) | `src/alpha_portfolio.py` |
| 7 | Portfolio environment & rebalancing | [Slides](https://docs.google.com/presentation/d/1lLT1wrkYv2Z5tAGAy60cV2HvwfpboZnmONB-KaWMJ9s/edit?slide=id.p#slide=id.p) | `envs/portfolio_env.py`, `src/alpha_portfolio.py`, `train_portfolio.ipynb` |
| 8 | Portfolio code review & reward shaping | [Slides](https://docs.google.com/presentation/d/1aevGAH0bvsGimvKnqZ6ulbYZ2rq4klJ1fBM-u8aXBrU/edit?slide=id.p#slide=id.p) | `envs/portfolio_env.py` |
| 9 | Hyperparameter tuning (Optuna) | [Slides](https://docs.google.com/presentation/d/1NSjxNF2Ke38pNrPKff4qzM69HZlSS2LVxc5WqTkgDD4/edit?slide=id.p#slide=id.p) | `tune_ppo_portfolio.ipynb` |
| 10 | Web interface / delivery prep | - | `deploy/alpaca_live.py` |

## Technical Details

**Data and features (Weeks 1-2):** Historical OHLCV data is ingested from per-ticker CSV files, enriched with technical indicators (RSI, Bollinger Bands, EMA-12/26, MACD, Relative Volume), and split into train/validation/test windows to support proper out-of-sample evaluation.

**RL environment design (Weeks 2, 7-8):** The project includes both a simpler environment (`envs/naive_env.py`) and a portfolio environment (`envs/portfolio_env.py`) that handles portfolio constraints, transaction costs, state construction, and reward computation (including risk-aware reward shaping).

**Portfolio mechanics and finance foundations (Weeks 6-7):** `src/alpha_portfolio.py` encapsulates portfolio accounting and rebalancing logic informed by MPT/CAPM concepts, enabling allocation-level decisions rather than single-asset trade signals.

**Agent architecture and optimization (Weeks 3, 9):** The policy/value model is an Actor-Critic network with a tanh-squashed Gaussian action distribution. Training uses PPO with GAE, clipping, entropy regularization, and gradient clipping, and can be tuned via Optuna in `scripts/tune_ppo_portfolio.ipynb`.

**Backtesting and deployment workflow (Weeks 4, 10):** After offline training/backtesting (`train.ipynb`, `train_portfolio.ipynb`), deployment utilities in `deploy/` support Alpaca paper/live execution and streaming-based integration tests.

## Acknowledgements

**Project Leads**
- Huy Le
- Muhammad (Abubakar) Siddiq

Built as part of the Michigan Data Science Team (MDST).

## License

MIT License

Copyright (c) 2026 Alpha Trade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
