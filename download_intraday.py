"""
Download 15-minute bars for the 20 training tickers from Alpaca into data/intraday/.

    python download_intraday.py

Needs ALPACA_API_KEY and ALPACA_SECRET_KEY in deploy/.env. Takes a few minutes.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

from src.alpaca_intraday import download_universe, make_client
from alpaca_utils import TRAINING_TICKERS

# Starts a month before the training split so the indicators are warmed up by 2021-01-01.
START = datetime(2020, 12, 1, tzinfo=timezone.utc)

if __name__ == '__main__':
    load_dotenv(ROOT / 'deploy' / '.env')
    client = make_client(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'))
    download_universe(client, TRAINING_TICKERS, START, ROOT / 'data' / 'intraday')
    print("Done")
