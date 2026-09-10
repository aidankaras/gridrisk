import pandas as pd
import pytest

from gridrisk.data.synthetic import generate_synthetic_dataset
from gridrisk.features import build_features


def test_no_forward_lookahead_bias():
    """Mutating future rows must not change any earlier feature row.

    This catches a feature that reaches *forward* in time (e.g. a
    shift(-1) or an unshifted rolling window that includes future data).
    It cannot catch a feature that leaks *same-day* information — that
    needs a different mutation, see test_no_same_day_leak below.
    """
    df = generate_synthetic_dataset(n_days=100, seed=0)
    cutoff = 60

    features_before = build_features(df)

    mutated = df.copy()
    mutated.loc[mutated.index[cutoff:], "log_return"] = 999.0
    mutated.loc[mutated.index[cutoff:], "temp_f"] = -999.0
    mutated.loc[mutated.index[cutoff:], "demand_mw"] = -999.0
    features_after = build_features(mutated)

    # Every feature column is built from shift(>=1)/rolling-on-shift(1), so
    # row `cutoff - 1` (the last row before the mutation) only reads data
    # from positions < cutoff and must be unaffected — include it explicitly
    # rather than stopping one row short of what's actually guaranteed safe.
    safe_rows = features_before.index[:cutoff]
    pd.testing.assert_frame_equal(features_before.loc[safe_rows], features_after.loc[safe_rows])


def test_no_same_day_leak():
    """Mutating a single day's own raw data must not change that day's
    features. A future-row mutation (test_no_forward_lookahead_bias) can't
    catch this: a feature that reads df['log_return'] at row t unshifted
    (e.g. accidentally equal to the target) only shows up if row t itself
    is perturbed and row t's own features are checked — exactly what this
    test does.
    """
    df = generate_synthetic_dataset(n_days=100, seed=0)
    target_row = 60

    features_before = build_features(df)

    mutated = df.copy()
    mutated.iloc[target_row] = mutated.iloc[target_row] * 0 + 12345.0
    features_after = build_features(mutated)

    row_label = features_before.index[target_row]
    pd.testing.assert_series_equal(features_before.loc[row_label], features_after.loc[row_label])


def test_build_features_requires_datetime_index():
    df = generate_synthetic_dataset(n_days=20, seed=0).reset_index(drop=True)
    with pytest.raises(TypeError):
        build_features(df)


def test_warmup_nans_are_confined_to_the_head():
    """Lags/rolling windows make the first rows unusable, but only those.

    Callers train on ``build_features(df).dropna()``; if a NaN appeared
    mid-series that silently drops interior days from the backtest instead
    of just the warm-up period.
    """
    df = generate_synthetic_dataset(n_days=50, seed=0)
    features = build_features(df)

    warmup = 10  # longest window: shift(1) then rolling(10)
    assert features.iloc[:warmup].isna().any(axis=1).all()
    assert not features.iloc[warmup:].isna().any().any()
