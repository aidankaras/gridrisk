# gridrisk

ERCOT day-ahead electricity volatility forecasting: GARCH baseline vs. gradient-boosted-trees, walk-forward backtested, with a backtested vol-scaled strategy and a Streamlit dashboard.

## Commands

| Command | Description |
|---|---|
| `uv sync` | Install/update dependencies |
| `uv run pytest` | Run the test suite |
| `uv run ruff check .` | Lint |
| `uv run ruff format .` | Format (CI runs `ruff format --check .` — run this first or CI will fail on unformatted code) |
| `uv run streamlit run app.py` | Run the dashboard locally |

## Architecture

- `src/gridrisk/data/` — `synthetic.py` (default, no credentials needed) vs. `ercot.py`/`eia.py` (real sources, documented but not wired up — see their module docstrings before touching them)
- `src/gridrisk/features.py` — causal feature engineering, and the two target definitions (volatility vs. variance) — see the Gotchas below before changing anything here
- `src/gridrisk/models/` — `garch.py` (baseline, outputs volatility directly) and `gbm.py` (challenger, trained on variance, outputs volatility)
- `src/gridrisk/evaluation.py` — walk-forward backtest harness + QLIKE/MSE scoring
- `src/gridrisk/strategy.py` — turns a vol forecast into a position size, backtests it
- `app.py` — Streamlit dashboard, reads from the synthetic path

## Gotchas

- **No-lookahead is load-bearing, and has two failure modes.** Every feature in `features.py` must be computable using only data strictly before the target date. `tests/test_features.py::test_no_forward_lookahead_bias` catches a feature reaching *forward* in time; `test_no_same_day_leak` separately catches a feature reading the *current* day's raw data unshifted (a forward-only mutation test can't detect that case — it needs a same-day mutation). If you touch `build_features`, both must still pass; if you add a feature, extend both.
- **Walk-forward, never random split.** `evaluation.run_backtest` refits both models at each step on an expanding window (`walk_forward_splits`). Don't "simplify" this to a single random train/test split — it would silently leak future information into the backtest. `walk_forward_splits` raises `ValueError` if there isn't enough data for even one split, rather than silently yielding nothing.
- **`step > 1` in `run_backtest` asymmetrically handicaps GARCH, not GBM.** GARCH's forecast is 1-step-ahead and gets held fixed for the whole block when `step > 1`, while GBM's features keep advancing within the block (they include yesterday's return). The default is `step=1` (refit both every day) specifically to avoid this; only raise `step` for fast local iteration, never for numbers you intend to report.
- **GARCH and GBM must be trained/evaluated on the same estimand.** GARCH forecasts volatility (σ) directly. GBM is trained on realized *variance* (`features.build_variance_target`, i.e. `log_return**2`) and takes a square root before returning volatility — training it on `|log_return|` directly instead would calibrate it to `E[|r|] ≈ 0.798σ`, silently biasing MSE and QLIKE in opposite directions relative to GARCH. `tests/test_evaluation.py::test_garch_and_gbm_forecasts_are_calibrated_to_the_same_scale` guards this.
- **`GbmVolForecaster`'s prediction floor is relative, not a fixed constant.** It floors variance predictions at a fraction of the *training* target's median (`min_variance_fraction`, default 5%) before taking the square root. A fixed absolute floor either does nothing (set far below the data's scale) or distorts every low-volatility day (set near it) — it has to move with the data.
- **QLIKE's mean is inherently outlier-sensitive — and the cause isn't always the same.** A single day where realized volatility was a genuine surprise (large or, on the GARCH side, anomalously near-zero) can dominate the mean, because QLIKE's per-point loss is unbounded as a forecast underpredicts. This is a property of using a single day's squared/absolute return as a noisy proxy for latent volatility, not automatically evidence of a specific bug — check `qlike`'s median variant and look at the actual worst-day contributors before assuming a fix is needed. `BacktestResult` reports both `*_qlike` (mean) and `*_qlike_median`; read both.
- **`data/ercot.py` and `data/eia.py` intentionally raise `NotImplementedError`** rather than returning fake data — this is deliberate, not a bug. Don't quietly make them return synthetic data instead; if real API access gets configured, implement the actual HTTP calls per the module docstrings.
- **`strategy.backtest_exposure` takes log returns and converts them internally.** A leveraged position's return is `exposure * simple_return`; that's only true in simple-return space, so the conversion (`np.expm1`) has to happen before exposure is applied, not skipped because daily returns are small enough that the difference looks negligible.
- **Inverse-vol position sizing (`volatility_scaled_exposure`) systematically overshoots its volatility target — this is expected, not a miscalibration.** `exposure = target_vol / forecast` is convex in the forecast, so by Jensen's inequality its expectation exceeds `target_vol / E[forecast]` whenever the forecast has any dispersion — a real, known property of naive inverse-vol sizing, independent of forecast accuracy. `vol_target_ratio` in `backtest_exposure`'s output reports the resulting overshoot; don't "fix" it by re-tuning the forecast.

## Testing

- `uv run pytest` runs everything, including a Streamlit-import smoke test (`test_app_smoke.py`) — Streamlit apps can't be meaningfully unit-tested beyond that, but an import-time crash is still worth catching in CI.
- New models/features should get: (1) a basic sanity test (correct shape/positivity), (2) where applicable, a test that the model actually responds to signal (e.g. `test_garch_sensitive_to_volatility_regime`) rather than just not-crashing, and (3) for anything touching causality or scale (features, targets, evaluation), a test that would actually fail if the invariant were violated — not just a docstring promising it.
