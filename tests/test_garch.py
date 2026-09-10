import pytest

from gridrisk.data.synthetic import generate_synthetic_dataset
from gridrisk.models.garch import GarchVolForecaster


def test_garch_forecast_is_positive():
    df = generate_synthetic_dataset(n_days=300, seed=0)
    forecaster = GarchVolForecaster().fit(df["log_return"])
    sigma = forecaster.forecast_next_sigma()
    assert sigma > 0


def test_garch_forecast_before_fit_raises():
    forecaster = GarchVolForecaster()
    with pytest.raises(RuntimeError):
        forecaster.forecast_next_sigma()


def test_garch_sensitive_to_volatility_regime():
    """A GARCH model fit on a high-vol regime should forecast higher vol
    than one fit on a low-vol regime — the basic sanity check that the
    model is actually responding to the data, not returning a constant."""
    df = generate_synthetic_dataset(n_days=600, seed=3, regime_shift_at=300, regime_multiplier=6.0)

    low_vol_forecaster = GarchVolForecaster().fit(df["log_return"].iloc[:300])
    high_vol_forecaster = GarchVolForecaster().fit(df["log_return"].iloc[300:600])

    assert high_vol_forecaster.forecast_next_sigma() > low_vol_forecaster.forecast_next_sigma()
