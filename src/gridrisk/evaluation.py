"""Walk-forward backtesting harness comparing the GARCH baseline vs. the GBM model.

Uses an expanding-window walk-forward split (never a random train/test
split — that would leak future information into the past for time-series
data) and scores forecasts with QLIKE and MSE against realized volatility,
the standard loss functions for evaluating volatility forecasts in
quantitative finance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridrisk.features import build_features, build_target, build_variance_target
from gridrisk.models.garch import GarchVolForecaster
from gridrisk.models.gbm import GbmVolForecaster


def walk_forward_splits(
    n: int, min_train: int, step: int = 1
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window (train_idx, test_idx) index arrays.

    Strictly causal by construction: every train index is less than every
    test index in the same split. The final block may be shorter than
    ``step`` if ``n - min_train`` isn't an exact multiple of it — every
    index from ``min_train`` to ``n`` is covered, none silently dropped.

    Raises ``ValueError`` if ``min_train >= n`` (no test data available at
    all), rather than silently yielding zero splits.
    """
    if min_train >= n:
        raise ValueError(f"min_train ({min_train}) must be less than n ({n}); no data to test on")

    start = min_train
    while start < n:
        test_end = min(start + step, n)
        yield np.arange(0, start), np.arange(start, test_end)
        start = test_end


def qlike(true_variance: np.ndarray, pred_variance: np.ndarray, reduction: str = "mean") -> float:
    """QLIKE loss for variance forecasts (lower is better). Requires variances, not vols.

    QLIKE's per-point loss is unbounded above as a forecast underpredicts
    (the ratio true/pred blows up), so its *mean* can be dominated by a
    single bad day — a real property of this loss, not necessarily a sign
    of a systematic problem. ``reduction="median"`` reports the more
    outlier-robust summary; report both rather than picking one, since they
    answer different questions ("how bad is the worst case" vs. "how good
    is the typical forecast").
    """
    if reduction not in ("mean", "median"):
        raise ValueError(f"reduction must be 'mean' or 'median', got {reduction!r}")

    ratio = true_variance / np.clip(pred_variance, 1e-12, None)
    losses = ratio - np.log(ratio) - 1
    if reduction == "median":
        return float(np.median(losses))
    return float(np.mean(losses))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    realized_vol: np.ndarray
    garch_forecast: np.ndarray
    gbm_forecast: np.ndarray

    @property
    def garch_mse(self) -> float:
        return mse(self.realized_vol, self.garch_forecast)

    @property
    def gbm_mse(self) -> float:
        return mse(self.realized_vol, self.gbm_forecast)

    @property
    def garch_qlike(self) -> float:
        return qlike(self.realized_vol**2, self.garch_forecast**2)

    @property
    def gbm_qlike(self) -> float:
        return qlike(self.realized_vol**2, self.gbm_forecast**2)

    @property
    def garch_qlike_median(self) -> float:
        return qlike(self.realized_vol**2, self.garch_forecast**2, reduction="median")

    @property
    def gbm_qlike_median(self) -> float:
        return qlike(self.realized_vol**2, self.gbm_forecast**2, reduction="median")


def run_backtest(df: pd.DataFrame, min_train: int = 250, step: int = 1) -> BacktestResult:
    """Run the walk-forward backtest comparing GARCH vs. GBM on ``df``.

    ``df`` must be shaped like ``gridrisk.data.synthetic.generate_synthetic_dataset``
    (a ``log_return`` column plus whatever ``gridrisk.features`` consumes).

    ``step`` controls how many days are held out and forecast before both
    models are refit. The default, ``1``, refits both models every day — the
    correct backtest, since it's what a live deployment would actually do.
    Raising ``step`` trades that fidelity for speed, but asymmetrically
    handicaps GARCH: its 1-step-ahead forecast gets held fixed and reused
    for every day in the block, while GBM's features (which include
    yesterday's return) keep advancing within the same block. Use ``step>1``
    only for fast iteration, never for numbers you intend to report.
    """
    features = build_features(df)
    vol_target = build_target(df)
    variance_target = build_variance_target(df)

    blocks = []
    for train_idx, test_idx in walk_forward_splits(len(df), min_train, step):
        garch = GarchVolForecaster().fit(df["log_return"].iloc[train_idx])
        # One 1-step-ahead sigma, reused for every day in the block (see docstring).
        garch_sigma = garch.forecast_next_sigma()

        train_features = features.iloc[train_idx].dropna()
        gbm = GbmVolForecaster().fit(train_features, variance_target.loc[train_features.index])

        # Feature warm-up (rolling windows) leaves the first few rows NaN, so
        # an early test block can legitimately have nothing to score yet.
        test_features = features.iloc[test_idx].dropna()
        if test_features.empty:
            continue

        blocks.append(
            pd.DataFrame(
                {
                    "realized_vol": vol_target.loc[test_features.index],
                    "garch_forecast": garch_sigma,
                    "gbm_forecast": gbm.predict(test_features),
                }
            )
        )

    if not blocks:
        raise ValueError(
            f"no test rows survived feature warm-up: min_train={min_train} is too small "
            f"for the {len(df)}-row input to produce any scoreable forecast"
        )

    scored = pd.concat(blocks)
    return BacktestResult(
        dates=pd.DatetimeIndex(scored.index),
        realized_vol=scored["realized_vol"].to_numpy(),
        garch_forecast=scored["garch_forecast"].to_numpy(),
        gbm_forecast=scored["gbm_forecast"].to_numpy(),
    )
