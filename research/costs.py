"""
Trading cost per unit of notional traded, per stock.

Half-spreads are medians of Alpaca SIP quotes sampled at 10:30, 12:30 and 15:00 ET on 2026-10-22.
On top we add 0.25 bp for market impact and regulatory fees: at a $100k book each trade is a few
thousand dollars in stocks that trade billions a day, so impact is small. Spreads were wider in
2021-2022, so results are also shown at 2x and 4x these costs.
"""

import numpy as np

HALF_SPREAD_BPS = {
    'AAPL': 0.43, 'MSFT': 0.86, 'GOOGL': 0.61, 'AMZN': 0.79, 'NVDA': 0.47,
    'META': 1.23, 'TSLA': 1.32, 'NFLX': 0.75, 'UNH': 2.00, 'JNJ': 2.28,
    'V': 1.69, 'JPM': 1.64, 'WMT': 0.83, 'MA': 2.21, 'PG': 1.50,
    'HD': 1.70, 'DIS': 1.78, 'BAC': 0.98, 'XOM': 1.98, 'CVX': 1.33,
}
IMPACT_AND_FEES_BPS = 0.25


def cost_vector(tickers, multiplier=1.0):
    return np.array([(HALF_SPREAD_BPS[t] + IMPACT_AND_FEES_BPS) * 1e-4 * multiplier for t in tickers])
