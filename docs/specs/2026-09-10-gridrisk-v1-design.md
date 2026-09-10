# gridrisk v1 design

Date: 2026-09-10

## Problem

Forecast day-ahead volatility in the ERCOT electricity market. Rising
renewable penetration and more frequent weather extremes have made ERCOT
prices larger and less predictable, which makes volatility forecasting
directly useful to trading desks, utilities, and risk teams operating in
that market.

## Decisions

- **Target**: next-day realized volatility for reporting (proxied by
  `|log_return|`), not price direction — volatility is materially more
  forecastable than returns, and is the quantity risk and trading desks
  actually need for sizing and hedging decisions. Models are *trained* on
  realized *variance* (`log_return**2`) rather than the volatility proxy
  directly, and expose volatility as their output — see `gridrisk.features`
  module docstring for why: training on `|log_return|` calibrates a model
  to `E[|r|] ≈ 0.798σ` rather than `σ`, which silently biases any
  comparison against GARCH, whose native output is `σ`.
- **Baseline**: GARCH(1,1) via the `arch` package — the standard
  econometric baseline any volatility model needs to beat to be worth
  deploying.
- **Challenger model**: gradient-boosted trees, implemented with
  scikit-learn's `HistGradientBoostingRegressor` rather than LightGBM, to
  keep the project installable without a native build toolchain. Revisit
  if categorical feature handling or training speed on larger datasets
  becomes a constraint.
- **Evaluation**: walk-forward (expanding window) backtesting, refitting
  both models every day by default (`step=1`) — an earlier version allowed
  a larger step for speed, but that held GARCH's forecast fixed across each
  block while the tree model's lagged features kept advancing within it,
  an asymmetric handicap rather than a neutral speed/accuracy tradeoff.
  Scored with QLIKE and MSE — the standard loss functions for evaluating
  variance/volatility forecasts. QLIKE is reported as both mean and
  median: a single day where realized volatility was a genuine surprise can
  dominate the mean, a property of the loss itself rather than necessarily
  a specific bug, so both summaries are reported rather than picking one
  (see `gridrisk.evaluation.qlike`'s docstring).
- **Decision layer**: inverse-volatility position sizing, backtested with
  Sharpe ratio, max drawdown, and a vol-target-tracking ratio (achieved
  volatility ÷ target volatility) against a constant-exposure baseline.
  Sharpe alone isn't a meaningful metric on the zero-drift synthetic
  returns this project defaults to; the vol-target ratio is what actually
  demonstrates the position sizing did what it claims. Returns are
  converted from log to simple space before exposure is applied and
  compounded — leverage composes correctly only in simple-return space.
- **Dashboard**: a standalone Streamlit + Plotly application, independently
  deployable, rather than embedding results in a hosted notebook or
  third-party canvas.
- **Data**: the default path is a synthetic generator (a simulated
  GARCH(1,1) process with volatility driven partly by temperature
  extremity, mirroring the real relationship between weather, demand, and
  ERCOT price volatility). Weather and demand are given independent
  structure (a large day-to-day noise term on temperature, an independent
  weekday/weekend demand effect) specifically so a calendar feature alone
  can't stand in for them — otherwise the weather-driven features add
  little the model couldn't already get from `month`/`day_of_week`. Real
  ERCOT and EIA data fetchers are implemented and documented
  (`gridrisk/data/ercot.py`, `gridrisk/data/eia.py`) but require API
  credentials that aren't configured in this repository, so they raise
  `NotImplementedError` rather than silently substituting synthetic data.

## v1 scope

Package skeleton, synthetic data generator, GARCH baseline, GBM model,
causal feature engineering (with a lookahead-bias regression test),
walk-forward evaluation harness, strategy/backtest layer, Streamlit
dashboard, CI (lint, format check, test), a scheduled daily-retrain
workflow definition, MIT license.

## Explicitly deferred (not v1)

Other ISOs beyond ERCOT, a deep-learning model, an options-aware strategy,
a public API, live ERCOT/EIA credential wiring, production deployment of
the dashboard. See the README's Roadmap section for the sequenced plan and
rationale for each of these.
