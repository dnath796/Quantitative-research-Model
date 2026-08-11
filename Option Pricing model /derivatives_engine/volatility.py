"""Implied-volatility quote inversion and surface construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .pricing import implied_volatility


REQUIRED_QUOTE_COLUMNS = {"strike", "maturity", "market_price"}


def implied_volatility_surface(
    quotes: pd.DataFrame | Iterable[Mapping],
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    default_option_type: str = "call",
) -> pd.DataFrame:
    """Invert an option quote grid into a tidy implied-volatility surface.

    Input rows require ``strike``, ``maturity``, and ``market_price``. An
    optional ``option_type`` column may mix calls and puts. Invalid quotes are
    retained with an ``error`` message and a missing implied volatility.
    """

    frame = quotes.copy() if isinstance(quotes, pd.DataFrame) else pd.DataFrame(quotes)
    missing = REQUIRED_QUOTE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing quote columns: {sorted(missing)}")
    if "option_type" not in frame:
        frame["option_type"] = default_option_type
    records = []
    for row in frame.to_dict("records"):
        try:
            iv = implied_volatility(
                float(row["market_price"]),
                spot,
                float(row["strike"]),
                float(row["maturity"]),
                rate,
                str(row["option_type"]),
                dividend_yield,
            )
            error = None
        except (ValueError, TypeError) as exc:
            iv, error = np.nan, str(exc)
        records.append({**row, "implied_volatility": iv, "error": error})
    return pd.DataFrame(records).sort_values(["maturity", "strike"]).reset_index(drop=True)


def surface_matrix(surface: pd.DataFrame) -> pd.DataFrame:
    """Pivot a tidy surface into maturity rows and strike columns."""

    required = {"strike", "maturity", "implied_volatility"}
    missing = required - set(surface.columns)
    if missing:
        raise ValueError(f"missing surface columns: {sorted(missing)}")
    return surface.pivot_table(
        index="maturity", columns="strike", values="implied_volatility", aggfunc="mean"
    ).sort_index().sort_index(axis=1)


def synthetic_smile_surface(
    spot: float,
    strikes,
    maturities,
    atm_volatility: float,
    skew: float = -0.12,
    curvature: float = 0.30,
    term_slope: float = 0.015,
) -> pd.DataFrame:
    """Create a transparent teaching surface from log-moneyness parameters.

    This is a scenario generator, not a calibration to market quotes.
    """

    if spot <= 0 or atm_volatility <= 0:
        raise ValueError("spot and atm_volatility must be positive")
    rows = []
    for maturity in np.asarray(maturities, dtype=float):
        if maturity <= 0:
            raise ValueError("maturities must be positive")
        for strike in np.asarray(strikes, dtype=float):
            if strike <= 0:
                raise ValueError("strikes must be positive")
            log_moneyness = np.log(strike / spot)
            vol = atm_volatility + skew * log_moneyness + curvature * log_moneyness**2 + term_slope * np.sqrt(maturity)
            rows.append(
                {
                    "strike": strike,
                    "maturity": maturity,
                    "log_moneyness": log_moneyness,
                    "implied_volatility": max(float(vol), 0.0001),
                }
            )
    return pd.DataFrame(rows)

