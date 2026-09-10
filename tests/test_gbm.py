import numpy as np
import pytest

from gridrisk.data.synthetic import generate_synthetic_dataset
from gridrisk.features import build_features, build_variance_target
from gridrisk.models.gbm import GbmVolForecaster


def _train_test_split(seed=0, regime_shift_at=300, n_train=400):
    """Chronological split into (train_X, train_y, test_X, test_y), warm-up NaNs dropped."""
    df = generate_synthetic_dataset(n_days=500, seed=seed, regime_shift_at=regime_shift_at)
    features = build_features(df).dropna()
    target = build_variance_target(df).loc[features.index]
    return (
        features.iloc[:n_train],
        target.iloc[:n_train],
        features.iloc[n_train:],
        target.iloc[n_train:],
    )


def test_gbm_fits_and_predicts_reasonable_values():
    train_X, train_y, test_X, _ = _train_test_split()
    preds = GbmVolForecaster().fit(train_X, train_y).predict(test_X)

    assert len(preds) == len(test_X)
    assert np.all(np.isfinite(preds))


def test_gbm_predictions_never_negative_or_below_floor():
    """HistGradientBoostingRegressor can predict slightly negative values
    near a non-negative target's boundary; GbmVolForecaster must floor
    these, since negative volatility is nonsensical and breaks downstream
    metrics like QLIKE (squaring a near-zero/negative value causes a
    division blowup)."""
    train_X, train_y, test_X, _ = _train_test_split()
    model = GbmVolForecaster(min_variance_fraction=0.05).fit(train_X, train_y)
    preds = model.predict(test_X)

    expected_floor_vol = np.sqrt(model._min_variance)
    assert (preds >= expected_floor_vol - 1e-12).all()


def test_gbm_predict_before_fit_raises():
    _, _, test_X, _ = _train_test_split()
    with pytest.raises(RuntimeError):
        GbmVolForecaster().predict(test_X)


def test_gbm_performs_comparably_to_naive_mean_baseline():
    """On this synthetic setup with a small hand-engineered feature set,
    the model is not expected to decisively beat predicting the training
    mean — measured MSE ratio is ~1.01 (statistically a tie). The point of
    this test is to catch a *regression* (the model getting much worse,
    e.g. >20% worse), not to claim a win that the data doesn't support.
    """
    train_X, train_y, test_X, test_y = _train_test_split()
    preds_vol = GbmVolForecaster().fit(train_X, train_y).predict(test_X)
    realized_vol = np.sqrt(test_y.to_numpy())

    model_mse = float(np.mean((realized_vol - preds_vol.to_numpy()) ** 2))
    naive_pred = np.sqrt(train_y).mean()
    naive_mse = float(np.mean((realized_vol - naive_pred) ** 2))

    assert model_mse <= naive_mse * 1.2
