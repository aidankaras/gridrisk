"""Feature engineering, and the two target definitions used elsewhere.

The invariant that matters most in this whole project: every feature for
date ``t`` must be computable using only information available at the close
of date ``t - 1``. Breaking this (lookahead bias) is the single most common
way a backtest quietly lies to you. ``tests/test_features.py`` enforces it
two ways: a mutation test on future rows (catches a feature reaching
forward in time) and a mutation test on the target date itself (catches a
feature leaking same-day information, which a future-only mutation can't
detect).

The two target definitions must not be confused:

- ``build_target`` returns realized *volatility* (``|log_return|``) — the
  ground truth reported and plotted, directly comparable to GARCH's
  ``forecast_next_sigma``.
- ``build_variance_target`` returns realized *variance* (``log_return**2``)
  — what models should be trained to predict. Training on volatility
  directly would calibrate a model to ``E[|r|] ≈ 0.798σ`` rather than
  ``σ``, silently biasing any comparison against a model (like GARCH) that
  outputs ``σ`` directly. See ``gridrisk.models.gbm.GbmVolForecaster``.
"""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a causal feature matrix from a raw price/weather DataFrame.

    ``df`` must have ``log_return``, ``temp_f``, ``demand_mw`` columns
    indexed by a ``DatetimeIndex`` (see
    ``gridrisk.data.synthetic.generate_synthetic_dataset`` for the expected
    shape).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"build_features requires a DatetimeIndex, got {type(df.index).__name__}")

    feat = pd.DataFrame(index=df.index)
    feat["lag1_abs_return"] = df["log_return"].shift(1).abs()
    feat["lag2_abs_return"] = df["log_return"].shift(2).abs()
    feat["lag5_realized_vol"] = df["log_return"].shift(1).rolling(5).std()
    feat["lag10_realized_vol"] = df["log_return"].shift(1).rolling(10).std()
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month
    feat["temp_f_lag1"] = df["temp_f"].shift(1)
    feat["demand_mw_lag1"] = df["demand_mw"].shift(1)
    return feat


def build_target(df: pd.DataFrame) -> pd.Series:
    """Ground-truth realized volatility for reporting/evaluation, proxied by |log_return|."""
    return df["log_return"].abs().rename("realized_vol")


def build_variance_target(df: pd.DataFrame) -> pd.Series:
    """Realized-variance training target (``log_return**2``) — see module docstring."""
    return (df["log_return"] ** 2).rename("realized_variance")
