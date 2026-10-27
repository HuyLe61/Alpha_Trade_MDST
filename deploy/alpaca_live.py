"""
Alpaca Paper Trading

Workflow: fetches the latest completed 15-minute bars (src/alpaca_intraday.py)
-> builds an observation
-> gets an action from the policy
-> places orders via the Alpaca REST API

The policy is trained on 15-minute bars (train_intraday.py). Every 15 minutes the
engine reads the last bar that has finished, decides, and trades. In training the
agent sees bar t and trades at the close of bar t+1; live is the same: the free
plan's SIP data is 15 minutes behind, so at HH:MM:30 on a quarter hour the newest
bar we can see is the one that closed 15 minutes ago, and the order goes in now,
at the close of the bar after it.

Everything goes through the REST API (alpaca_trade_api for account, positions and
orders; alpaca-py for market data). These are "request-response" calls: we send a
request, we get one response.
API Documentation: https://github.com/alpacahq/alpaca-trade-api-python?tab=readme-ov-file

Every 15 minutes, from 9:45 to 15:45 Eastern, the engine runs:
(1) Read the latest completed 15-minute bars.
(2) Add technical indicators.
(3) Build observation.
(4) Policy predicts action.
(5) Place orders via REST.

Run: python deploy/alpaca_live.py
Config: paper trading by default (ALPACA_BASE_URL in deploy/.env).
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch

# Add parent directory so we can import agent/ and src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from agent.actor_critic import ActorCritic
from alpaca_utils import (
    TRAINING_TICKERS,
    prepare_features,
    build_observation,
    place_orders_from_actions,
    get_current_positions,
)

# Make sure to install alpaca-py package: pip install alpaca-py
from src.alpaca_intraday import (
    make_client, fetch_recent_universe, BAR_MINUTES, EASTERN, SESSION_FIRST_BAR, SESSION_LAST_BAR,
)

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

# Risk limits: stop trading if we hit these limits
RISK_PARAMS = {
    'max_position_size': 0.15,   # Max 15 % per stock
    'daily_loss_limit': 0.05,    # Stop if down 2 % in a day
    'max_drawdown': 0.30,        # Stop if down 15 % from peak
    'min_trade_value': 100,      # Minimum $100 per trade
}

# Trained PPO policy model 
DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / 'models' / 'ppo_trading_intraday.pt')

# Seconds past each quarter hour before we ask for data, so the newest visible bar is in.
CYCLE_OFFSET_SECONDS = 30
# Skip a cycle if the newest bar is older than this: a bar is missing and the policy would be acting on old data.
MAX_BAR_AGE_MINUTES = 2 * BAR_MINUTES + 5

class AlpacaPPOTrader:
    """Live trading manager"""
    def __init__(self, api, model_path: str, api_key: str = None, secret_key: str = None):
        self.api = api
        # Include your API keys from Alpaca to .env
        self._data_client = make_client(api_key, secret_key)
        self.running = False
        self._last_slot = None      # quarter hour of the last finished cycle
        self._session_open = {}     # date -> market open time

        # Performance tracking
        self.start_time = datetime.now()
        self.initial_value = None
        self.peak_value = None
        self.daily_start_value = None
        self.portfolio_history = []
        self.trade_step = 0

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained PPO policy from .pt checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if checkpoint.get('bar_minutes') != BAR_MINUTES:
            raise RuntimeError(
                f"{model_path} was not trained on {BAR_MINUTES}-minute bars "
                f"(bar_minutes={checkpoint.get('bar_minutes')}). Train with train_intraday.py."
            )

        obs_dim = checkpoint['obs_shape']
        act_dim = checkpoint['action_shape']

        self.policy = ActorCritic(obs_dim, act_dim)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        self.feat_mean = checkpoint.get('feat_mean')
        self.feat_std = checkpoint.get('feat_std')
        if self.feat_mean is None or self.feat_std is None:
            raise RuntimeError(
                "Checkpoint missing feat_mean/feat_std. "
                "Re-run the training save cell to include z-score stats."
            )

        print(f"Policy loaded (obs={obs_dim}, act={act_dim})\n")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the policy"""
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy.forward(state)
            action = torch.tanh(dist.mean)
        return action.cpu().numpy()[0]

    def get_portfolio_value(self):
        """Get total portfolio value """
        try:
            return float(self.api.get_account().portfolio_value)
        except Exception as e:
            print(f"Error getting portfolio value: {e}")
            return None

    def is_market_open(self):
        """Check if the US equity market is currently open"""
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            print(f"  Error checking market status: {e}")
            return False

    def get_market_hours(self):
        """Get next market open and close times (for display when closed)"""
        try:
            clock = self.api.get_clock()
            return clock.next_open, clock.next_close
        except Exception:
            return None, None

    # validate risk limits
    def check_risk_limits(self):
        value = self.get_portfolio_value()
        if value is None:
            return True

        if self.initial_value is None:
            self.initial_value = value
            self.peak_value = value
            self.daily_start_value = value

        if value > self.peak_value:
            self.peak_value = value

        daily_loss = (self.daily_start_value - value) / self.daily_start_value
        if daily_loss > RISK_PARAMS['daily_loss_limit']:
            print(f"\n RISK LIMIT BREACHED: Daily loss {daily_loss*100:.2f}% > limit {RISK_PARAMS['daily_loss_limit']*100:.1f}%")
            return False

        drawdown = (self.peak_value - value) / self.peak_value
        if drawdown > RISK_PARAMS['max_drawdown']:
            print(f"\n RISK LIMIT BREACHED: Drawdown {drawdown*100:.2f}% > limit {RISK_PARAMS['max_drawdown']*100:.1f}%")
            return False

        return True

    # Trading cycle: data -> observation -> action -> orders
    def fetch_latest_data(self):
        """
        Read the latest completed 15-minute bars for every training ticker
        and add the technical indicators
        """
        raw = fetch_recent_universe(self._data_client, TRAINING_TICKERS)
        if raw is None:
            print("15-minute bars incomplete")
            return None

        # Once today's first bar is visible, the newest bar must be recent. (Before that, at the
        # 9:45 cycle, the newest bar is yesterday's last one, which is expected.)
        last = next(iter(raw.values())).index[-1]
        expected = datetime.now(EASTERN).replace(tzinfo=None) - timedelta(minutes=2 * BAR_MINUTES)
        expected = expected.replace(minute=(expected.minute // 15) * 15, second=0, microsecond=0)
        if SESSION_FIRST_BAR <= expected.time() <= SESSION_LAST_BAR and last < expected:
            print(f"Newest bar is {last}, expected {expected} or later. Data is stale")
            return None
        return prepare_features(raw)

    def execute_trade(self):
        """
        One cycle: get 15-minute data, add features, build obs, get action, place orders via REST.
        Returns 'halt' (risk limit hit, stop the engine), 'retry' (could not run, try again soon)
        or 'done' (cycle finished).
        """
        print(f"\n{'='*70}")
        print(f"Trade {self.trade_step}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        if not self.check_risk_limits():
            return 'halt'

        portfolio_value = self.get_portfolio_value()
        if portfolio_value is None:
            print("Can't read portfolio value. Will retry")
            return 'retry'

        print(f"Portfolio value: ${portfolio_value:,.2f}")

        # Fetch data
        stock_data = self.fetch_latest_data()
        if stock_data is None:
            print("Data fetch failed. Will retry")
            return 'retry'

        # Cancel any open orders from the previous cycle before placing new
        # ones. Pending orders reserve buying power; leaving them open across
        # cycles drains buying power to $0 even if no trades actually fill.
        try:
            cancelled = self.api.cancel_all_orders()
            if cancelled:
                print(f"Cancelled {len(cancelled)} open order(s) from previous cycle")
        except Exception as e:
            print(f"  Warning: could not cancel open orders: {e}")

        # Balance and order sizing use cash, never margin: in training the agent could only
        # spend the cash it had, and a paper account's 4x buying power would let it lever up.
        positions = get_current_positions(self.api)
        account = self.api.get_account()
        cash = max(float(account.cash), 0.0)
        initial = self.initial_value or portfolio_value
        max_nw = self.peak_value or portfolio_value

        print(f"Cash: ${cash:,.2f}  |  Buying power: ${float(account.buying_power):,.2f} (not used for sizing)")

        # Step feature: in training it is the position inside a random 1024-bar window (0 to 1),
        # which has no live meaning, so feed the window's midpoint.
        obs = build_observation(
            stock_data, balance=cash, shares_held=positions,
            net_worth=portfolio_value, max_net_worth=max_nw,
            current_step=1, max_steps=2,
            initial_balance=initial,
            feat_mean=self.feat_mean, feat_std=self.feat_std,
        )

        # Policy outputs one action per ticker in [-1, 1]
        action = self.predict(obs)

        # Output model decisions
        # buys = [(t, f"{a:+.3f}") for t, a in zip(TRAINING_TICKERS, action) if a > 0]
        # sells = [(t, f"{a:+.3f}") for t, a in zip(TRAINING_TICKERS, action) if a < 0]
        # holds = [t for t, a in zip(TRAINING_TICKERS, action) if a == 0]
        # print(f"Actions → Buy: {buys} \nSell: {sells}  \nHold: {len(holds)}")

        orders = place_orders_from_actions(
            self.api, action, TRAINING_TICKERS,
            positions,
            buying_power=cash,
            portfolio_value=portfolio_value,
            max_position_size=RISK_PARAMS['max_position_size'],
            min_trade_value=RISK_PARAMS['min_trade_value'],
        )
        print(f"Placed {len(orders)} orders")

        self.portfolio_history.append({'time': datetime.now(), 'value': portfolio_value})
        self.trade_step += 1
        return 'done'

    def session_open_time(self, now):
        """Today's market open (from the broker calendar, so early closes and holidays are right)."""
        today = now.date()
        if today not in self._session_open:
            session = self.api.get_calendar(start=str(today), end=str(today))[0]
            self._session_open[today] = datetime.combine(today, session.open, tzinfo=now.tzinfo)
        return self._session_open[today]

    def run(self):
        """
        Every 15 minutes while the market is open, run execute_trade(): first cycle 15 minutes
        after the open (9:45), last one 15 minutes before the close (15:45).
        Sleeps through nights, weekends and holidays.
        """
        step = timedelta(minutes=BAR_MINUTES)
        offset = timedelta(seconds=CYCLE_OFFSET_SECONDS)
        print(f"{'='*70}")
        print(f"Trade interval: every {BAR_MINUTES} min, {CYCLE_OFFSET_SECONDS}s past the quarter hour")
        print(f"Data source: Alpaca {BAR_MINUTES}-minute bars (SIP, completed bars only)")
        pv = self.get_portfolio_value()
        print(f"Starting portfolio: ${pv:,.2f}\n")

        self.running = True
        last_date = None

        # Don't trade the quarter hour we start in (it may already have been traded before a restart):
        # the first cycle is the next one.
        start = self.api.get_clock().timestamp.floor('s').to_pydatetime() - offset
        self._last_slot = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)

        try:
            while self.running:
                now = datetime.now()

                # Reset daily-start value at the beginning of each calendar day.
                if last_date is None or now.date() != last_date:
                    self.daily_start_value = self.get_portfolio_value()
                    print(f"\n New day - Current portfolio value: ${self.daily_start_value:,.2f}")
                    last_date = now.date()

                try:
                    clock = self.api.get_clock()
                    # The broker clock is a pandas Timestamp with nanoseconds, which .replace(microsecond=0)
                    # keeps; drop them so the same quarter hour always gives the same slot.
                    broker_now = clock.timestamp.floor('s').to_pydatetime()
                    if not clock.is_open:
                        wait = (clock.next_open - broker_now).total_seconds() + BAR_MINUTES * 60 + CYCLE_OFFSET_SECONDS
                        print(f"Market closed. Next cycle in {wait/3600:.1f} h")
                        time.sleep(min(max(wait, 60), 3600))
                        continue
                    first_slot = self.session_open_time(broker_now) + step
                except Exception as e:
                    print(f"  Could not read the market clock: {e}")
                    time.sleep(60)
                    continue

                # The cycle we are in: the latest quarter hour whose offset has passed.
                shifted = broker_now - offset
                slot = shifted.replace(minute=(shifted.minute // 15) * 15, second=0, microsecond=0)

                if slot < first_slot or slot == self._last_slot:
                    next_run = max(slot + step, first_slot) + offset
                    time.sleep(max(1, (next_run - broker_now).total_seconds()))
                    continue

                result = self.execute_trade()
                if result == 'halt':
                    print("\n Risk limit breached. Stopping trade cycle")
                    break
                if result == 'retry':
                    time.sleep(60)  # same slot again until it works or the next quarter hour arrives
                    continue
                self._last_slot = slot

                cv = self.get_portfolio_value()
                if self.initial_value and cv:
                    tot = (cv - self.initial_value) / self.initial_value * 100
                    day = (cv - self.daily_start_value) / self.daily_start_value * 100
                    print(f"Total return: {tot:+.2f}%  |  Today's return: {day:+.2f}%")

        # Handle keyboard interrupt and exceptions
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user")
        except Exception as e:
            print(f"\n\n  Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.running = False
            self._print_summary()

    def _print_summary(self):
        print(f"\nFINAL SUMMARY")
        fv = self.get_portfolio_value()
        if self.initial_value and fv:
            ret = (fv - self.initial_value) / self.initial_value * 100
            print(f"  Initial : ${self.initial_value:,.2f}")
            print(f"  Final   : ${fv:,.2f}")
            print(f"  Return  : {ret:+.2f}%")
            if self.peak_value:
                dd = (self.peak_value - fv) / self.peak_value * 100
                print(f"  Max DD  : {dd:.2f}%")
        print(f"  Runtime : {datetime.now() - self.start_time}")
        print(f"{'='*70}\n")


def main():
    """Load env, connect to Alpaca REST, then run the trader."""
    # The intraday agent trades its own paper account. ALPACA_API_KEY / ALPACA_SECRET_KEY belong to the
    # minimum-variance trader (min_variance_live.py), and two strategies must never share an account.
    api_key = os.getenv('ALPACA_INTRADAY_API_KEY')
    secret = os.getenv('ALPACA_INTRADAY_SECRET_KEY')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    # The REST client appends /v2 itself; a base URL that already ends in /v2 gives /v2/v2 and a 404.
    if base.endswith('/v2'):
        base = base[:-3]
    if 'paper' not in base:
        print(f"  Error: refusing to run against {base}: this agent is for paper trading only")
        return

    if not api_key or not secret:
        print("  Error: Alpaca API keys not found!")
        print("  Add ALPACA_INTRADAY_API_KEY and ALPACA_INTRADAY_SECRET_KEY to deploy/.env")
        return

    # Initialize Alpaca REST API
    api = tradeapi.REST(api_key, secret, base, api_version='v2')

    # Check connection
    try:
        acct = api.get_account()
        print(f"Connected — status: {acct.status}  |  account ...{acct.account_number[-4:]}")
        print(f"Portfolio : ${float(acct.portfolio_value):,.2f}")
        print(f"Buying pwr: ${float(acct.buying_power):,.2f}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    if os.getenv('ALPACA_API_KEY'):
        try:
            other = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), base,
                                  api_version='v2').get_account()
            if other.account_number == acct.account_number:
                print("  Error: the intraday keys point at the minimum-variance account. Use the other paper account's keys.")
                return
        except Exception:
            pass  # the other account being unreachable doesn't stop this one

    trader = AlpacaPPOTrader(
        api, DEFAULT_MODEL_PATH,
        api_key=api_key,
        secret_key=secret,
    )
    trader.run()


if __name__ == '__main__':
    main()
