import numpy as np

from gridrisk.data.synthetic import generate_synthetic_dataset, simulate_garch_returns


def test_simulate_garch_returns_shape_and_positivity():
    returns, sigma = simulate_garch_returns(n=200, seed=1)
    assert returns.shape == (200,)
    assert sigma.shape == (200,)
    assert np.all(sigma > 0)


def test_regime_shift_increases_volatility():
    _, sigma = simulate_garch_returns(n=400, seed=1, regime_shift_at=200, regime_multiplier=5.0)
    pre_regime = sigma[:200].mean()
    post_regime = sigma[250:].mean()
    assert post_regime > pre_regime


def test_temperature_extremity_drives_volatility():
    """This is the test that protects the project's actual differentiator:
    weather features must carry real signal about volatility, not just be
    decorative columns a pure GARCH model can never be beaten using."""
    df = generate_synthetic_dataset(n_days=2000, seed=0)
    extreme_days = df[np.abs(df["temp_f"] - 65) > 25]
    mild_days = df[np.abs(df["temp_f"] - 65) < 5]
    assert len(extreme_days) > 20 and len(mild_days) > 20
    assert extreme_days["true_sigma"].mean() > mild_days["true_sigma"].mean()


def test_generate_synthetic_dataset_columns():
    df = generate_synthetic_dataset(n_days=100, seed=0)
    assert len(df) == 100
    for col in ["log_return", "true_sigma", "price", "temp_f", "demand_mw"]:
        assert col in df.columns
    assert df["price"].iloc[0] > 0
    assert df.index.is_monotonic_increasing
