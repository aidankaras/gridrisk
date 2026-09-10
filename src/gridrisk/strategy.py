"""Turns a volatility forecast into a decision, and backtests that decision.

A forecast is only useful if it changes what you'd do. This module
implements the simplest decision a volatility forecast is directly useful
for — inverse-volatility position sizing (scale exposure down when
predicted volatility is high, up when it's low, targeting constant risk) —
and backtests it with the metrics a risk desk actually reports: Sharpe
ratio, max drawdown, and how closely the strategy's realized volatility
tracked its stated target.

This is deliberately not a full trading strategy; it exists to demonstrate
that the forecast has decision-relevant value, not to be deployed as-is.
On the zero-drift synthetic data this project defaults to, Sharpe ratio is
close to pure noise (there's no expected return to capture, by
construction) — the metric that's actually meaningful on that data is
whether the position sizing achieved its stated risk target, which is why
that's reported explicitly rather than left implicit in the Sharpe number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_scaled_exposure(
    vol_forecast: pd.Series, target_vol: float = 0.01, max_exposure: float = 2.0
) -> pd.Series:
    """Inverse-vol position sizing: exposure = target_vol / forecast, capped at max_exposure."""
    exposure = target_vol / vol_forecast.clip(lower=1e-6)
    return exposure.clip(upper=max_exposure)


def _annualized_sharpe(returns: pd.Series) -> float:
    """Sharpe ratio of a daily simple-return series, annualized over 252 trading days.

    Returns 0.0 for a degenerate series (zero or undefined dispersion)
    rather than dividing by zero.
    """
    std = returns.std()
    if std > 0:
        return float(returns.mean() / std * np.sqrt(252))
    return 0.0


def _max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve implied by daily simple returns."""
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def backtest_exposure(
    log_returns: pd.Series, exposure: pd.Series, target_vol: float
) -> dict[str, float]:
    """Backtest a per-day exposure series against realized returns.

    ``exposure`` for date ``t`` is assumed to already be decided using only
    information available before ``t`` (in this project, exposure is
    derived from a lagged/forecast volatility, never from the realized
    return itself). ``target_vol`` should be the same value passed to
    ``volatility_scaled_exposure`` when building ``exposure`` — it's used
    to report how closely the strategy tracked its own risk target.

    ``log_returns`` are converted to simple (arithmetic) returns internally:
    a leveraged position's return is ``exposure * simple_return``, which is
    only true for simple returns — leverage does not compose that way in
    log-return space, so applying ``exposure`` directly to log returns would
    silently misstate both the equity curve and the risk metrics below.
    """
    simple_returns = np.expm1(log_returns)
    strat_returns = simple_returns * exposure
    strat_std = strat_returns.std()

    return {
        "sharpe": _annualized_sharpe(strat_returns),
        "max_drawdown": _max_drawdown(strat_returns),
        # The ``naive_*`` baseline is constant full exposure to the same returns.
        "naive_sharpe": _annualized_sharpe(simple_returns),
        "naive_max_drawdown": _max_drawdown(simple_returns),
        "vol_target_ratio": float(strat_std / target_vol) if target_vol > 0 else float("nan"),
    }
