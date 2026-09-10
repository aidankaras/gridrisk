"""Synthetic ERCOT-like price/weather data.

This is the dev/test data path: it needs no credentials or network access.
It produces a GARCH(1,1) return series whose conditional volatility is
driven partly by ordinary return clustering (the standard GARCH mechanism)
and partly by **temperature extremity** — mirroring the real ERCOT story
where heat waves and cold snaps push demand (and therefore price
volatility) up. That causal link matters: it's what gives the weather/
demand features in ``gridrisk.features`` genuine predictive signal, rather
than being noise a pure GARCH baseline can never lose to by construction.

For real historical data, see ``gridrisk.data.ercot`` and
``gridrisk.data.eia`` — those require external API credentials not
included in this repository, so they are documented but not wired into
the default pipeline yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_garch_returns(
    n: int,
    omega: float = 2e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: int = 0,
    regime_shift_at: int | None = None,
    regime_multiplier: float = 4.0,
    omega_multiplier: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a GARCH(1,1) return series with a time-varying omega.

    ``omega`` at each step ``t`` is scaled by ``omega_multiplier[t]`` (if
    given) and, on top of that, by ``regime_multiplier`` from
    ``regime_shift_at`` onward (a simple discrete step, independent of
    ``omega_multiplier`` — the two compose multiplicatively so callers can
    use either or both).

    Returns ``(returns, sigma)`` where ``sigma`` is the true conditional
    volatility used to generate each return — useful in tests, since real
    market data never gives you the ground-truth volatility to check against.
    """
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for a stationary GARCH process")

    if omega_multiplier is None:
        multiplier = np.ones(n)
    else:
        multiplier = np.asarray(omega_multiplier, dtype=float)
        if multiplier.shape != (n,):
            raise ValueError(f"omega_multiplier must have shape ({n},), got {multiplier.shape}")
        multiplier = multiplier.copy()

    if regime_shift_at is not None:
        multiplier[regime_shift_at:] *= regime_multiplier

    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = (omega * multiplier[0]) / (1 - alpha - beta)
    returns[0] = rng.standard_normal() * np.sqrt(sigma2[0])

    for t in range(1, n):
        this_omega = omega * multiplier[t]
        sigma2[t] = this_omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.standard_normal() * np.sqrt(sigma2[t])

    return returns, np.sqrt(sigma2)


def _temperature_driven_multiplier(temp_f: np.ndarray, sensitivity: float = 8.0) -> np.ndarray:
    """Map temperature extremity (distance from a mild 65F) to a vol multiplier.

    Mild days (near 65F) get a multiplier near 1; extreme heat/cold days get
    a multiplier well above 1 — this is the mechanism that ties weather
    features to actual predictive signal for volatility.
    """
    deviation = np.abs(temp_f - 65) / 20.0  # ~0 on mild days, ~1+ on extreme days
    return 1.0 + sensitivity * deviation**2


def generate_synthetic_dataset(
    n_days: int = 750,
    seed: int = 0,
    regime_shift_at: int | None = None,
    regime_multiplier: float = 4.0,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a daily synthetic ERCOT-like dataset: price, returns, weather, demand.

    Volatility is driven by both ordinary GARCH clustering and temperature
    extremity (see ``_temperature_driven_multiplier``); ``regime_shift_at``/
    ``regime_multiplier`` layer an additional discrete step-change on top,
    useful for regime-sensitivity tests independent of the weather link.

    Columns:
        log_return: simulated daily log return
        true_sigma: the ground-truth conditional volatility (test-only signal,
            not available for real data — do not use as a model feature)
        price: a price level derived by cumulating log_return
        temp_f: a synthetic daily temperature series (deg F)
        demand_mw: a synthetic demand series, elevated on temperature extremes
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    is_weekday = dates.dayofweek.to_numpy() < 5

    weather_rng = np.random.default_rng(seed + 1)
    # Day-to-day noise is set large relative to the seasonal amplitude on
    # purpose: if the noise term were small, `month` alone would predict
    # temperature extremity almost as well as the actual temperature
    # reading, and the weather features below would add little the model
    # couldn't already get from a calendar feature.
    temp_f = (
        65 + 20 * np.sin(2 * np.pi * (day_of_year - 30) / 365) + weather_rng.normal(0, 9, n_days)
    )
    # Demand has its own independent driver (weekday/weekend load pattern)
    # on top of the temperature-linked component, so it isn't a near-copy
    # of temp_f — real grid demand has structure beyond weather too.
    weekday_multiplier = np.where(is_weekday, 1.0, 0.85)
    demand_mw = weekday_multiplier * (40_000 + 350 * np.abs(temp_f - 65)) + weather_rng.normal(
        0, 2000, n_days
    )

    weather_multiplier = _temperature_driven_multiplier(temp_f)
    returns, true_sigma = simulate_garch_returns(
        n_days,
        seed=seed,
        regime_shift_at=regime_shift_at,
        regime_multiplier=regime_multiplier,
        omega_multiplier=weather_multiplier,
    )

    price = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "log_return": returns,
            "true_sigma": true_sigma,
            "price": price,
            "temp_f": temp_f,
            "demand_mw": demand_mw,
        },
        index=dates,
    )
