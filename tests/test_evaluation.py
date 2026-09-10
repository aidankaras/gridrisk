import numpy as np
import pytest

from gridrisk.data.synthetic import generate_synthetic_dataset
from gridrisk.evaluation import mse, qlike, run_backtest, walk_forward_splits


def test_walk_forward_splits_are_strictly_causal():
    splits = list(walk_forward_splits(n=50, min_train=20, step=5))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_walk_forward_splits_expanding_window():
    splits = list(walk_forward_splits(n=50, min_train=20, step=5))
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes == sorted(train_sizes)  # non-decreasing


def test_walk_forward_splits_cover_every_index_including_partial_tail():
    splits = list(walk_forward_splits(n=53, min_train=20, step=5))  # 33 not a multiple of 5
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    assert list(all_test_idx) == list(range(20, 53))  # nothing dropped


def test_walk_forward_splits_raises_when_no_test_data_possible():
    with pytest.raises(ValueError):
        list(walk_forward_splits(n=20, min_train=20, step=5))


def test_mse_and_qlike_zero_for_perfect_forecast():
    y = np.array([0.01, 0.02, 0.015])
    assert mse(y, y) == 0.0
    assert abs(qlike(y**2, y**2)) < 1e-9


def test_qlike_rejects_invalid_reduction():
    y = np.array([0.01, 0.02])
    with pytest.raises(ValueError):
        qlike(y, y, reduction="mediam")  # typo — must raise, not silently mean


def test_qlike_median_is_robust_to_single_outlier():
    """A single badly-underpredicted point should blow up the mean QLIKE
    but leave the median largely unaffected — that's the whole reason the
    median variant exists (see qlike's docstring)."""
    true_var = np.full(21, 1e-4)
    pred_var = np.full(21, 1e-4)
    pred_var[0] = 1e-10  # one severe underprediction

    mean_loss = qlike(true_var, pred_var, reduction="mean")
    median_loss = qlike(true_var, pred_var, reduction="median")

    assert mean_loss > 10 * median_loss
    assert median_loss < 1e-6


def test_run_backtest_produces_results_for_all_test_points():
    df = generate_synthetic_dataset(n_days=350, seed=0, regime_shift_at=200)
    result = run_backtest(df, min_train=250, step=10)

    assert len(result.dates) > 0
    assert len(result.realized_vol) == len(result.dates)
    assert len(result.garch_forecast) == len(result.dates)
    assert len(result.gbm_forecast) == len(result.dates)
    assert result.garch_mse > 0
    assert result.gbm_mse > 0


def test_garch_and_gbm_forecasts_are_calibrated_to_the_same_scale():
    """GARCH forecasts sigma directly; GBM is trained on realized variance
    and takes a sqrt to also report sigma. If GBM were still trained on
    |log_return| directly (E[|r|] = sigma * sqrt(2/pi) ~ 0.8*sigma), its
    mean forecast would sit noticeably below GARCH's and below realized
    vol — this test would have caught that regression.
    """
    df = generate_synthetic_dataset(n_days=750, seed=7, regime_shift_at=500)
    result = run_backtest(df, min_train=250, step=5)

    realized_mean = result.realized_vol.mean()
    garch_mean = result.garch_forecast.mean()
    gbm_mean = result.gbm_forecast.mean()

    assert 0.7 < garch_mean / realized_mean < 1.4
    assert 0.7 < gbm_mean / realized_mean < 1.4
