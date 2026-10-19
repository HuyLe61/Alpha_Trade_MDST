"""Settings for the live trading worker. Secrets come from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Single source of truth for the ticker order the policy was trained on.
from deploy.alpaca_utils import TRAINING_TICKERS as TICKERS

ROOT = Path(__file__).resolve().parent.parent

# Local runs read deploy/.env (Alpaca keys) and trader/.env (Supabase keys).
# In GitHub Actions these are injected as repository secrets instead.
load_dotenv(ROOT / "deploy" / ".env")
load_dotenv(Path(__file__).parent / ".env")

BENCHMARK = "SPY"

# Same limits as deploy/alpaca_live.py
RISK = {
    "max_position_size": 0.15,  # max 15% of portfolio per stock
    "daily_loss_limit": 0.05,   # skip trading for the day if down 5% vs. prior close
    "max_drawdown": 0.30,       # halt the agent if down 30% from peak equity
    "min_trade_value": 100,     # ignore orders smaller than $100
}

BAR_MINUTES = 15
BAR_LOOKBACK_DAYS = 7  # enough 15-min bars for the 26-period EMA / 20-period bands
MAX_STEPS = 252        # step normalizer used in the observation, as in alpaca_live.py

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() != "false"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
