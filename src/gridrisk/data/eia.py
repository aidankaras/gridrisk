"""Real EIA weather/demand data — NOT YET WIRED UP.

The U.S. Energy Information Administration exposes a free, keyed REST API
(https://www.eia.gov/opendata/) with hourly electricity demand by balancing
authority (ERCOT's BA code is "ERCO"), e.g.:
https://api.eia.gov/v2/electricity/rto/region-data/data/

This module is a documented placeholder, same pattern as
``gridrisk.data.ercot``: no ``EIA_API_KEY`` is configured in this
repository, so ``fetch_demand`` raises ``NotImplementedError`` instead of
faking a response. Wire it up by:

1. Requesting a free API key at https://www.eia.gov/opendata/register.php.
2. Setting ``EIA_API_KEY`` as an environment variable.
3. Implementing the HTTP call below and parsing the response into a
   DataFrame shaped like ``gridrisk.data.synthetic.generate_synthetic_dataset``
   (DatetimeIndex, a ``demand_mw`` column).

Until then, use ``gridrisk.data.synthetic`` for development and testing.
"""

from __future__ import annotations

import os

import pandas as pd


def fetch_demand(balancing_authority: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly/daily demand data for a balancing authority from EIA.

    Not yet implemented — see module docstring for setup steps.
    """
    if not os.environ.get("EIA_API_KEY"):
        raise NotImplementedError(
            "EIA_API_KEY not configured. See gridrisk.data.eia module "
            "docstring for setup steps. Use gridrisk.data.synthetic for now."
        )
    raise NotImplementedError("EIA API client not yet implemented.")
