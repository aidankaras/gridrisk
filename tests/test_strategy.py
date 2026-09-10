import numpy as np
import pandas as pd

from gridrisk.strategy import backtest_exposure, volatility_scaled_exposure


def test_volatility_scaled_exposure_inverse_relationship():
    vol = pd.Series([0.005, 0.01, 0.02, 0.04])
    exposure = volatility_scaled_exposure(vol, target_vol=0.01, max_exposure=5.0)
    # exposure should decrease as forecast vol increases
    assert list(exposure) == sorted(exposure, reverse=True)


def test_volatility_scaled_exposure_respects_cap():
    vol = pd.Series([0.0001, 0.0002])  # would imply huge exposure without a cap
    exposure = volatility_scaled_exposure(vol, target_vol=0.01, max_exposure=2.0)
    assert (exposure <= 2.0).all()


def test_backtest_exposure_metrics_are_finite():
    rng = np.random.default_rng(0)
    log_returns = pd.Series(rng.normal(0, 0.01, 200))
    exposure = pd.Series(rng.uniform(0.5, 1.5, 200))
    metrics = backtest_exposure(log_returns, exposure, target_vol=0.01)
    for key in ["sharpe", "max_drawdown", "naive_sharpe", "naive_max_drawdown", "vol_target_ratio"]:
        assert np.isfinite(metrics[key])
    assert -1.0 <= metrics["max_drawdown"] <= 0.0
    assert -1.0 <= metrics["naive_max_drawdown"] <= 0.0


def test_backtest_exposure_converts_log_returns_correctly():
    """A leveraged position's return is exposure * simple_return, not
    exposure * log_return — this is a hand-computed regression test for
    that conversion, since the bug (multiplying log returns by exposure
    directly) produced numbers close enough to correct at small return
    sizes that a loose tolerance test wouldn't have caught it.
    """
    log_returns = pd.Series([np.log(1.05), np.log(0.90)])  # +5%, then -10%
    exposure = pd.Series([2.0, 2.0])  # constant 2x leverage

    metrics = backtest_exposure(log_returns, exposure, target_vol=0.01)

    simple_returns = np.array([0.05, -0.10])
    strat_returns = simple_returns * 2.0
    expected_equity = np.cumprod(1 + strat_returns)
    expected_running_max = np.maximum.accumulate(expected_equity)
    expected_max_drawdown = float((expected_equity / expected_running_max - 1).min())

    assert abs(metrics["max_drawdown"] - expected_max_drawdown) < 1e-9


def test_vol_target_ratio_reflects_achieved_vs_target_vol():
    rng = np.random.default_rng(0)
    log_returns = pd.Series(rng.normal(0, 0.02, 500))
    exposure = pd.Series(np.ones(500))  # no scaling — achieved vol ≈ underlying vol
    metrics = backtest_exposure(log_returns, exposure, target_vol=0.02)
    assert 0.8 < metrics["vol_target_ratio"] < 1.2
