"""Gradient-boosted-trees volatility forecaster.

Uses scikit-learn's ``HistGradientBoostingRegressor`` (a LightGBM-style
histogram-based booster that ships without a native-build dependency),
trained on realized *variance* (see ``gridrisk.features`` for why variance
rather than volatility is the correct training target) and exposing
volatility as its public output, so it's directly comparable to
``gridrisk.models.garch.GarchVolForecaster``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class GbmVolForecaster:
    """Gradient-boosted-trees volatility forecaster.

    Trained on realized variance, with predictions floored before taking a
    square root to return volatility. ``HistGradientBoostingRegressor`` has
    no output constraint, so without a floor it can occasionally predict a
    slightly negative or near-zero variance — nonsensical, and numerically
    dangerous downstream (a near-zero denominator inside QLIKE blows up).
    The floor is a fraction of the median *training* variance rather than a
    fixed constant, because a fixed one either does nothing (set far below
    the data's scale) or distorts every low-volatility day (set near it).
    """

    def __init__(
        self,
        max_depth: int = 4,
        random_state: int = 0,
        min_variance_fraction: float = 0.05,
    ) -> None:
        self.model = HistGradientBoostingRegressor(max_depth=max_depth, random_state=random_state)
        self.min_variance_fraction = min_variance_fraction
        self._min_variance: float | None = None

    def fit(self, features: pd.DataFrame, variance_target: pd.Series) -> GbmVolForecaster:
        self.model.fit(features, variance_target)
        self._min_variance = float(variance_target.median()) * self.min_variance_fraction
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self._min_variance is None:
            raise RuntimeError("Call fit() before predicting.")
        variance_pred = np.clip(self.model.predict(features), self._min_variance, None)
        return pd.Series(np.sqrt(variance_pred), index=features.index, name="predicted_vol")
