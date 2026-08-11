"""Revaluation-based sensitivity and scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import MarketData, Option
from .pricing import black_scholes


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    spot_pct: float = 0.0
    volatility_shift: float = 0.0
    rate_shift: float = 0.0
    days_elapsed: int = 0


DEFAULT_SCENARIOS = (
    Scenario("Base"),
    Scenario("Stock +10%", spot_pct=0.10),
    Scenario("Stock -10%", spot_pct=-0.10),
    Scenario("Vol +5 points", volatility_shift=0.05),
    Scenario("Vol -5 points", volatility_shift=-0.05),
    Scenario("30 days later", days_elapsed=30),
    Scenario("Rates +1 point", rate_shift=0.01),
)


def option_scenarios(
    option: Option,
    market: MarketData,
    scenarios: tuple[Scenario, ...] | list[Scenario] = DEFAULT_SCENARIOS,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
) -> pd.DataFrame:
    if option.style != "european":
        raise ValueError("scenario engine currently uses BSM European revaluation")
    base = black_scholes(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, option.option_type, market.dividend_yield,
    )
    rows = []
    for scenario in scenarios:
        shocked = market.shocked(
            spot_pct=scenario.spot_pct,
            volatility_shift=scenario.volatility_shift,
            rate_shift=scenario.rate_shift,
        )
        maturity = max(0.0, option.maturity - scenario.days_elapsed / 365.0)
        value = black_scholes(
            shocked.spot, option.strike, maturity, shocked.rate,
            shocked.volatility, option.option_type, shocked.dividend_yield,
        )
        factor = quantity * contract_multiplier
        rows.append(
            {
                "scenario": scenario.name,
                "spot": shocked.spot,
                "volatility": shocked.volatility,
                "rate": shocked.rate,
                "days_to_expiry": round(maturity * 365),
                "option_value": value,
                "position_pnl": (value - base) * factor,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_curve(
    option: Option,
    market: MarketData,
    variable: str,
    values,
) -> pd.DataFrame:
    allowed = {"spot", "volatility", "rate", "maturity", "strike"}
    if variable not in allowed:
        raise ValueError(f"variable must be one of {sorted(allowed)}")
    records = []
    for raw_value in np.asarray(values, dtype=float):
        params = {
            "S": market.spot,
            "K": option.strike,
            "T": option.maturity,
            "r": market.rate,
            "sigma": market.volatility,
        }
        key = {"spot": "S", "strike": "K", "maturity": "T", "rate": "r", "volatility": "sigma"}[variable]
        params[key] = float(raw_value)
        price = black_scholes(**params, option_type=option.option_type, q=market.dividend_yield)
        records.append({variable: raw_value, "option_price": price})
    return pd.DataFrame(records)
