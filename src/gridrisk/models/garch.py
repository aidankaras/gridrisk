"""GARCH(1,1) baseline volatility forecaster.

This is the industry-standard baseline any volatility model has to beat to
be worth using — see Bollerslev (1986). Implemented via the ``arch``
package rather than from scratch, since the point of this project is the
forecasting/evaluation pipeline and the ML comparison, not re-deriving a
well-established econometric model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


class GarchVolForecaster:
    """Wraps ``arch_model`` to forecast next-period conditional volatility.

    Returns are rescaled internally (arch recommends returns be on the order
    of 1-10 in magnitude for numerical stability, so log returns — typically
    ~0.01 — are converted to percentage points and converted back on output).
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q
        self._fitted = None

    def fit(self, returns: pd.Series) -> GarchVolForecaster:
        am = arch_model(returns * 100, vol="Garch", p=self.p, q=self.q, rescale=False)
        self._fitted = am.fit(disp="off")
        return self

    def forecast_next_sigma(self) -> float:
        """Forecast next-period volatility (standard deviation, return-scale)."""
        if self._fitted is None:
            raise RuntimeError("Call fit() before forecasting.")
        fc = self._fitted.forecast(horizon=1, reindex=False)
        variance_pct2 = float(fc.variance.iloc[-1, 0])
        return float(np.sqrt(variance_pct2) / 100)
