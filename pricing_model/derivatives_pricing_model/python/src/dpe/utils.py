"""Small numerical utilities shared by the demo and tests."""

from __future__ import annotations

from collections.abc import Sequence

import math

import numpy as np

from ._validation import require_integer

__all__ = ["historical_vol"]


def historical_vol(prices: Sequence[float] | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised close-to-close historical volatility from a price series.

    Computes the sample standard deviation (ddof=1) of the log returns
    ``ln(P_i / P_{i-1})`` and scales by ``sqrt(periods_per_year)`` — the
    standard realised-vol estimator under the GBM assumption that log
    returns are i.i.d. normal.

    Raises ``ValueError`` on fewer than 3 prices (no meaningful ddof=1
    standard deviation from fewer than 2 returns), on any non-positive or
    non-finite price (log returns would be undefined), or on a
    ``periods_per_year`` that is not an integer >= 1.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 3:
        raise ValueError(f"prices must be a 1-D series with >= 3 points, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("prices must all be finite and > 0")
    periods_per_year = require_integer("periods_per_year", periods_per_year, 1)
    log_returns = np.diff(np.log(p))
    return float(np.std(log_returns, ddof=1) * math.sqrt(periods_per_year))
