# Intraday research: does any approach beat holding the stocks?

October 2026. Universe: the 20 large-cap stocks the live trader uses. Data: Alpaca 15-minute bars
(SIP feed, split- and dividend-adjusted), regular session only, December 2020 to October 2026.

## Short answer

No market-neutral strategy we tried earns money after costs out of sample. That covers rule-based
signals, a ridge forecaster and reinforcement learning (PPO), 15 trained agents in all. The edge
that exists at 15-minute horizons is real but about a tenth of what it costs to trade. The only
approaches that hold up are long-only and risk-based, and their value is lower volatility and
drawdown, not extra return.

## Setup (identical for every approach)

| Item | Choice |
|---|---|
| Splits | Train 2021–2024 · validation 2025 · test 2026 to Oct 22 (203 trading days), test never used to choose anything |
| Features | 10 per stock: returns over 1 bar, 1 hour, 1 day, 5 days; Bollinger %b; distance from EMA26; RSI; MACD/price; relative volume; realized volatility. All z-scored across the 20 stocks at each bar. No raw prices. |
| Timing | Decide on bar *t*, trade at the close of bar *t+1*, the live setup with the free 15-minute-delayed feed. Real-time (trade at *t*) also tested for the reversal signals. |
| Costs | Measured: median half-spread per stock from Alpaca SIP quotes (0.43–2.28 bp, mean 1.33 bp) plus 0.25 bp for impact and fees. Also run at 2× and 4×. |
| Metric | Sharpe ratio from daily P&L, net of costs. |

Code: `panel.py` (features), `backtest.py` (one backtester for everything), `costs.py`,
`signal_benchmark.py`, `ridge_forecast.py`, `train_rl.py` with `envs/xs_portfolio_env.py`,
`long_only.py`.

## Results: net Sharpe, test period (2026)

| Approach | Train | Validation 2025 | Test 2026 | Notes |
|---|---|---|---|---|
| Equal-weight hold (benchmark) | +1.14 | +0.94 | **+1.51** | volatility 11.2%, max drawdown 8.0% |
| Inverse-volatility, long-only | +1.10 | +0.90 | +1.60 | volatility 10.6% |
| Minimum variance, long-only, 15% cap | +1.03 | +0.49 | +2.07 | volatility 9.2%, max drawdown 5.9%, beta 0.76 |
| 1-bar reversal, market-neutral | −10.6 | — | −9.4 | gross +1.5 but trades 37× capital a day: the cost is 140%/yr |
| Best rule on train (5-day momentum) | +0.78 | −0.05 | −0.47 | |
| Ridge, all features, tuned on validation | +0.81 | +0.59 | +0.13 | −0.14 at 2× costs |
| PPO, decide every bar (3 seeds) | −7.0 | −9.7 | −10.0 | |
| PPO, every hour (3 seeds) | +5.1 | −2.9 | −3.0 | |
| PPO, once a day (3 seeds) | +8.3 | −0.1 | +0.5 (−0.8 to +2.2) | train +8 vs validation 0: memorized the training years |
| PPO, once a day, small net + early stop | +5.7 | −0.1 | −0.6 | the seed with the best validation (+1.96) scored −1.19 on test |
| Original intraday PPO (before this research) | — | −0.31 | −0.25 | invested like equal weight, then lost ~14%/yr to trading costs |

How to read the test column: 203 trading days gives a Sharpe standard error of about 1.1, so a
single test Sharpe under about 2.2 is not distinguishable from zero. We evaluated about 160
configurations. By chance alone the best of them would show a Sharpe near 2 somewhere.

## What we learned

1. **The signal exists but cannot pay for itself.** 15-minute reversal has a stable rank IC of
   +0.012 to +0.022 (t ≈ 4–6) in every period. But a reversal book has to trade almost all of its
   capital every bar. Its gross edge is about 0.06–0.2 bp per unit traded, against 1.6 bp of cost.
   You only capture this edge as the liquidity provider (limit orders, rebates), not as a taker
   sending market orders.
2. **The free feed's 15-minute delay erases the short-term signal.** At one bar of delay the
   reversal IC drops to +0.004 in 2025 and +0.002 in 2026, statistically zero. Anything faster than
   about an hour needs real-time data.
3. **RL is not the bottleneck; the signal is.** With per-step rewards, stationary inputs and no
   entropy bonus, PPO now learns. But it learns the training years: train Sharpe +5 to +8,
   validation about 0. Seed-to-seed spread on test (−0.8 to +2.2) is as large as any effect.
   Early stopping on validation did not help.
4. **20 mega caps is the hardest possible universe.** They are the most efficiently priced stocks
   there are, and cross-sectional strategies need breadth: the information ratio grows with
   IC × √(number of independent bets). Twenty names leaves almost nothing to diversify.
5. **What does work is risk control, not prediction.** Minimum variance cut volatility by 18–31%
   and drawdown by 26–39% in every period. Its return edge over equal weight flipped
   sign between 2025 and 2026.

## Recommendations

**Now:** stop the intraday PPO trader. It is evaluated as a negative-edge, high-turnover policy.
If the project needs something live, run long-only minimum variance (or equal weight) rebalanced
daily. It is honest, cheap to trade and survived every period.

**Next research, in the order a desk would fund it:**

1. *Breadth:* move to the S&P 500 (Alpaca has the daily bars for free).
2. *Slower signals that survive costs:* 1-week reversal, 12-1 month momentum, low volatility and
   earnings drift, on daily data, rebalanced weekly with a no-trade band.
3. *Execution:* if intraday reversal stays interesting, simulate passive limit orders
   (earn the spread instead of paying it) with a real-time feed. That is a market-making
   problem, a different project.
4. *RL where it fits:* sizing and allocation on top of a validated signal, or execution, trained
   with walk-forward folds, five or more seeds and a fixed budget of configurations, so we know
   how much luck to expect.
5. *Release gate:* a strategy goes live only if its validation and test Sharpe both clear about
   2 standard errors after 2× costs, across seeds, and it beats equal weight on risk-adjusted terms.
