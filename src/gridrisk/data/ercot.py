"""Real ERCOT day-ahead settlement point price data — NOT YET WIRED UP.

ERCOT publishes day-ahead market (DAM) settlement point prices via its
Market Information System (MIS) public reports, e.g. the "DAM Settlement
Point Prices" report:
https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD

This module is a documented placeholder: no ERCOT API credentials are
configured in this repository, so ``fetch_dam_settlement_prices`` raises
``NotImplementedError`` rather than silently returning fake data. Wire it up
by:

1. Registering for ERCOT MIS API access (no cost, but requires an account).
2. Setting ``ERCOT_API_USERNAME`` / ``ERCOT_API_PASSWORD`` (or a token, per
   ERCOT's current auth scheme) as environment variables.
3. Implementing the actual HTTP calls below using ``requests`` and parsing
   the returned CSV/JSON into the same shape as
   ``gridrisk.data.synthetic.generate_synthetic_dataset`` (a DatetimeIndex
   with at least a ``price`` column), so the rest of the pipeline is a
   drop-in swap.

Until then, use ``gridrisk.data.synthetic`` for development and testing.
"""

from __future__ import annotations

import os

import pandas as pd


def fetch_dam_settlement_prices(
    settlement_point: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch ERCOT day-ahead settlement point prices for a date range.

    Not yet implemented — see module docstring for setup steps.
    """
    if not os.environ.get("ERCOT_API_USERNAME"):
        raise NotImplementedError(
            "ERCOT API credentials not configured (ERCOT_API_USERNAME/"
            "ERCOT_API_PASSWORD). See gridrisk.data.ercot module docstring "
            "for setup steps. Use gridrisk.data.synthetic for development."
        )
    raise NotImplementedError("ERCOT API client not yet implemented.")
