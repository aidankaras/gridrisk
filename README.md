# gridrisk

**Day-ahead volatility forecasting for the ERCOT (Texas) electricity market** — a GARCH(1,1) baseline vs. a gradient-boosted-trees model, evaluated with a walk-forward backtest and turned into a backtested position-sizing strategy.

## Why this, why ERCOT

Electricity price volatility in ERCOT has become a genuinely hard, genuinely important forecasting problem: rising renewable penetration and more frequent weather extremes (Winter Storm Uri, 2021) have made price spikes larger and less predictable, and that volatility is exactly what energy traders, utilities, and risk desks are paid to anticipate. It's a market where a real forecasting edge has real value — not just a "predict the stock market" toy problem.

This project treats it as a quantitative finance problem, not a data-science demo:
- A **GARCH(1,1) baseline** (via the `arch` package) — the model any volatility forecast has to beat to be worth using.
- A **gradient-boosted-trees model** (scikit-learn's `HistGradientBoostingRegressor`), trained on realized variance so it's calibrated to the same quantity GARCH forecasts, using lagged volatility, calendar effects, and weather/demand features.
- **Walk-forward backtesting** (never a random train/test split on time series, and refit daily by default — see `CLAUDE.md` for why) scored with **QLIKE and MSE** (both mean and median), the standard losses for evaluating volatility forecasts.
- A **backtested position-sizing rule** (inverse-volatility exposure) scored with **Sharpe ratio, max drawdown, and a vol-target-tracking ratio** against a constant-exposure baseline — the vol-target ratio is the metric that actually demonstrates decision value on zero-drift synthetic returns, since Sharpe alone is close to pure noise without a real return premium in the data.
- A **live dashboard** (Streamlit) showing the forecast vs. realized volatility, rolling accuracy, and the strategy's equity curve.

## Status

**Working now, on synthetic data:** the full pipeline — data generation, both models, the walk-forward backtest, the strategy layer, and the dashboard — runs end to end and is covered by tests (30 passing). See `gridrisk.data.synthetic` for how the synthetic series is generated (a simulated GARCH process with volatility driven by temperature extremity and an optional discrete regime shift, plus weather/demand features with their own independent structure).

**Not yet wired up** (needs real ERCOT/EIA API credentials, not included in this repository):
- Real ERCOT day-ahead settlement point prices — see `gridrisk/data/ercot.py` for the documented integration path.
- Real EIA weather/demand data — see `gridrisk/data/eia.py`.
- Deploying the dashboard somewhere with a public URL (Streamlit Community Cloud or Render).
- The scheduled retrain workflow (`.github/workflows/daily-retrain.yml`) doing real work — it's defined and runs on schedule, but its underlying script is a placeholder until real data access is configured.

## Roadmap

Rough priority order, each building on the synthetic-data pipeline that already works end to end:

1. **Real data.** Wire up `gridrisk/data/ercot.py` and `gridrisk/data/eia.py`, deploy the dashboard publicly, and let the scheduled workflow run against live data. Nothing below this matters until forecasts are being made against real prices.
2. **Forward (live) evaluation, not just backtesting.** Everything in this repo today is a backtest — evaluated against history the model (and its author) already knows the outcome of. The stronger test is a live forecast log: each day, before the next day's price is known, log the model's forecast; once the actual price lands, pair it with the realized outcome and accumulate that into a running evaluation table, scored with the same `qlike`/`mse` functions already in `evaluation.py`. This is what `daily-retrain.yml` is ultimately for. A few months of accumulated live points is the real test of whether GBM's edge over GARCH (if any) survives outside the backtest sample.
3. **Regime detection (unsupervised).** Fit a Hidden Markov Model or Gaussian Mixture Model on rolling return statistics to let the model discover volatility regimes from data, rather than relying on hand-lagged features to catch up after the fact. This is a real, literature-grounded extension (Markov-switching GARCH is a published family of models built on exactly this idea) — feed the resulting regime label/probability into the GBM feature set.
4. **A PyTorch model, evaluated as a genuine third contender, not just added for its own sake.** Two variants, since they test different things: (a) an MLP on the same engineered feature set as the GBM model, for a fair apples-to-apples comparison; (b) a small LSTM/GRU consuming a raw window of past returns and weather directly, without hand-engineered features — deep learning's actual strength is learning its own temporal representations. Track all three models (GARCH, GBM, PyTorch) side by side in the dashboard. Note: daily-frequency, single-asset volatility data is not abundant, and deep learning models frequently don't beat GARCH-family or tree-ensemble baselines at this data scale — that's a legitimate, explainable outcome if it happens here too, not a failure to hide.
5. **Continuous improvement loop.** Once step 2's live evaluation log exists, the scheduled workflow should do more than just forecast: periodically retrain on the growing dataset, compare the new model's live-evaluation performance against the currently deployed one, and only promote it if it's genuinely better — a small, real version of a model-promotion gate, not a full MLOps platform.
6. **Dimensionality reduction**, once there are enough real, correlated features to justify it (multiple weather stations, multiple grid nodes) — PCA or similar on a small hand-built feature set like today's isn't worth the complexity yet.
7. Longer-term/lower-priority: other ISOs beyond ERCOT, an options-aware strategy (trading the gap between this model's forecast and market-implied volatility), a public API.

## Architecture

```
gridrisk/
  src/gridrisk/
    data/
      synthetic.py   # GARCH-simulated dev/test data (no credentials needed)
      ercot.py       # real ERCOT price fetcher (documented, not yet wired up)
      eia.py         # real EIA weather/demand fetcher (documented, not yet wired up)
    models/
      garch.py       # GARCH(1,1) baseline (wraps the `arch` package)
      gbm.py         # gradient-boosted-trees model (scikit-learn), trained on variance
    features.py      # causal (no-lookahead) feature engineering + target definitions
    evaluation.py    # walk-forward backtest harness, QLIKE/MSE
    strategy.py      # vol-scaled position sizing + Sharpe/drawdown/vol-target backtest
  app.py             # Streamlit dashboard
  tests/             # pytest suite, including lookahead-bias mutation tests
  .github/workflows/ # CI (lint, format check, test) and a scheduled retrain workflow
```

## Quickstart

```bash
git clone https://github.com/aidankaras/gridrisk.git
cd gridrisk
uv sync
uv run pytest
uv run ruff check .
uv run streamlit run app.py
```

No API keys or network access are required to run any of the above — everything defaults to the synthetic data path.

## Testing philosophy

The tests most worth reading are in `tests/test_features.py`: `test_no_forward_lookahead_bias` mutates future rows and checks past feature rows are byte-for-byte unaffected, and `test_no_same_day_leak` separately mutates a single day's own data and checks that day's own features are unaffected — the two failure modes of lookahead bias, and neither is caught by the other. Lookahead bias is the single most common way a backtest quietly lies to you, so it gets real regression tests, not comments promising it won't happen.

## License

MIT — see [LICENSE](LICENSE).
