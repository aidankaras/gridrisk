"""gridrisk Streamlit dashboard.

Shows the latest volatility forecast vs. realized volatility, rolling
forecast accuracy over time, and the backtested strategy equity curve.
Runs entirely on synthetic data for now — see README "Status / Roadmap"
for wiring up real ERCOT/EIA data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridrisk.data.synthetic import generate_synthetic_dataset
from gridrisk.evaluation import run_backtest
from gridrisk.strategy import backtest_exposure, volatility_scaled_exposure


@st.cache_data
def load_backtest():
    df = generate_synthetic_dataset(n_days=750, seed=7, regime_shift_at=500)
    result = run_backtest(df, min_train=250, step=1)
    return df, result


def line_chart(title: str, frame: pd.DataFrame, height: int) -> None:
    """Render one trace per column of ``frame``, against its shared index."""
    fig = go.Figure()
    for name in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[name], name=name))
    fig.update_layout(title=title, height=height)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="gridrisk — ERCOT volatility forecasting", layout="wide")
    st.title("gridrisk: ERCOT day-ahead volatility forecasting")
    st.caption(
        "Synthetic data for now (see README) — GARCH(1,1) baseline vs. a "
        "gradient-boosted-trees model, evaluated with a walk-forward backtest."
    )

    df, result = load_backtest()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GARCH MSE", f"{result.garch_mse:.6f}")
    mse_delta = result.gbm_mse - result.garch_mse
    col2.metric("GBM MSE", f"{result.gbm_mse:.6f}", delta=f"{mse_delta:+.6f}")
    qlike_med_delta = result.gbm_qlike_median - result.garch_qlike_median
    col3.metric(
        "GBM QLIKE (median)",
        f"{result.gbm_qlike_median:.4f}",
        delta=f"{qlike_med_delta:+.4f}",
        help="Median QLIKE, robust to the occasional bad outlier day.",
    )
    col4.metric(
        "GBM QLIKE (mean)",
        f"{result.gbm_qlike:.2f}",
        delta=f"{result.gbm_qlike - result.garch_qlike:+.2f}",
        help="Mean QLIKE can be dominated by a single severely underpredicted day; "
        "the median above is the more robust summary.",
    )

    forecasts = pd.DataFrame(
        {
            "Realized vol": result.realized_vol,
            "GARCH forecast": result.garch_forecast,
            "GBM forecast": result.gbm_forecast,
        },
        index=result.dates,
    )
    line_chart("Forecast vs. realized volatility", forecasts, height=400)

    window = 30
    errors = forecasts[["GARCH forecast", "GBM forecast"]].sub(forecasts["Realized vol"], axis=0)
    rolling_mae = errors.abs().rolling(window).mean()
    rolling_mae.columns = ["GARCH rolling MAE", "GBM rolling MAE"]
    line_chart(f"{window}-day rolling forecast error", rolling_mae, height=350)

    target_vol = 0.01
    exposure = volatility_scaled_exposure(forecasts["GBM forecast"], target_vol=target_vol)
    log_returns = df["log_return"].reindex(result.dates)
    metrics = backtest_exposure(log_returns, exposure, target_vol=target_vol)

    st.subheader("Vol-scaled strategy backtest (vs. constant-exposure baseline)")
    st.caption(
        "Synthetic returns here are zero-drift by construction, so Sharpe is close to pure "
        "noise — the metric that's actually meaningful on this data is the vol-target ratio "
        "below (achieved volatility ÷ target volatility; 1.0 is perfect tracking)."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe", f"{metrics['sharpe']:.2f}", help="Noisy on zero-drift synthetic data.")
    m2.metric("Baseline Sharpe", f"{metrics['naive_sharpe']:.2f}")
    m3.metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")
    m4.metric(
        "Vol-target ratio",
        f"{metrics['vol_target_ratio']:.2f}",
        help="Achieved daily volatility ÷ target volatility. 1.0 means the position sizing "
        "hit its risk target exactly.",
    )

    simple_returns = np.expm1(log_returns)
    equity = pd.DataFrame(
        {
            "Vol-scaled equity": (1 + simple_returns * exposure).cumprod(),
            "Constant-exposure equity": (1 + simple_returns).cumprod(),
        }
    )
    line_chart("Strategy equity curve", equity, height=350)


if __name__ == "__main__":
    main()
