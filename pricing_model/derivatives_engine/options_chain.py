"""Model-generated options chains with quotes, liquidity, IV, and Greeks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .models import OptionType
from .pricing import black_scholes, implied_volatility
from .risk import black_scholes_greeks


CHAIN_COLUMNS = [
    "expiration_days", "maturity", "option_type", "strike", "moneyness",
    "bid", "ask", "mid", "last", "spread", "volume", "open_interest",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
    "intrinsic_value", "extrinsic_value", "theoretical_price", "market_model_difference",
]


def option_moneyness(spot: float, strike: float, option_type: OptionType | str, atm_band: float = 0.025) -> str:
    """Classify a contract as ITM, ATM, or OTM using a spot-relative ATM band."""

    relative = strike / spot - 1.0
    if abs(relative) <= atm_band:
        return "ATM"
    kind = OptionType(option_type)
    is_itm = strike < spot if kind is OptionType.CALL else strike > spot
    return "ITM" if is_itm else "OTM"


def generate_options_chain(
    spot: float,
    rate: float,
    base_volatility: float,
    dividend_yield: float = 0.0,
    expiration_days: Iterable[int] = (30, 60, 90, 180, 365),
    strikes: Iterable[float] | None = None,
    strike_count: int = 13,
    strike_width: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible educational options chain.

    Quotes are synthetic: each contract receives a transparent skew/smile IV,
    a liquidity-sensitive bid/ask spread, and seeded volume/open interest. This
    function is useful for coursework and UI development, not live trading.
    """

    if spot <= 0 or base_volatility <= 0:
        raise ValueError("spot and base_volatility must be positive")
    if strike_count < 3:
        raise ValueError("strike_count must be at least 3")
    expiries = tuple(int(days) for days in expiration_days)
    if not expiries or any(days <= 0 for days in expiries):
        raise ValueError("expiration_days must contain positive integers")
    if strikes is None:
        raw = np.linspace(spot * (1.0 - strike_width), spot * (1.0 + strike_width), strike_count)
        step = max(1.0, round(spot * strike_width * 2 / (strike_count - 1) / 2.5) * 2.5)
        strike_values = np.unique(np.round(raw / step) * step)
    else:
        strike_values = np.asarray(tuple(strikes), dtype=float)
    if np.any(strike_values <= 0):
        raise ValueError("strikes must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for days in expiries:
        maturity = days / 365.0
        for strike in strike_values:
            log_moneyness = np.log(strike / spot)
            surface_vol = max(
                0.0001,
                base_volatility - 0.12 * log_moneyness + 0.30 * log_moneyness**2 + 0.015 * np.sqrt(maturity),
            )
            for kind in (OptionType.CALL, OptionType.PUT):
                fair = black_scholes(spot, strike, maturity, rate, surface_vol, kind, dividend_yield)
                base_fair = black_scholes(spot, strike, maturity, rate, base_volatility, kind, dividend_yield)
                distance = abs(log_moneyness)
                half_spread = max(0.01, 0.008 * fair + 0.015 + 0.12 * distance)
                mid = max(fair + rng.normal(0.0, half_spread * 0.12), 0.005)
                bid = max(mid - half_spread, 0.0)
                ask = mid + half_spread
                last = max(mid + rng.normal(0.0, half_spread * 0.35), 0.0)
                liquidity = np.exp(-7.5 * distance) * np.exp(-0.45 * maturity)
                volume = int(max(0, rng.poisson(15 + 900 * liquidity)))
                open_interest = int(max(volume, rng.poisson(80 + 4_500 * liquidity)))
                try:
                    quote_iv = implied_volatility(mid, spot, strike, maturity, rate, kind, dividend_yield)
                except ValueError:
                    quote_iv = np.nan
                risk = black_scholes_greeks(spot, strike, maturity, rate, surface_vol, kind, dividend_yield)
                intrinsic = max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)
                rows.append(
                    {
                        "expiration_days": days,
                        "maturity": maturity,
                        "option_type": kind.value,
                        "strike": float(strike),
                        "moneyness": option_moneyness(spot, strike, kind),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": last,
                        "spread": ask - bid,
                        "volume": volume,
                        "open_interest": open_interest,
                        "implied_volatility": quote_iv,
                        "delta": risk.delta,
                        "gamma": risk.gamma,
                        "theta": risk.theta / 365.0,
                        "vega": risk.vega / 100.0,
                        "rho": risk.rho / 100.0,
                        "intrinsic_value": intrinsic,
                        "extrinsic_value": max(mid - intrinsic, 0.0),
                        "theoretical_price": base_fair,
                        "market_model_difference": mid - base_fair,
                    }
                )
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS).sort_values(
        ["expiration_days", "strike", "option_type"]
    ).reset_index(drop=True)


def filter_options_chain(
    chain: pd.DataFrame,
    expiration_days: int | None = None,
    option_type: str | None = None,
    moneyness: str | None = None,
) -> pd.DataFrame:
    """Filter an options chain without mutating the source table."""

    result = chain
    if expiration_days is not None:
        result = result[result["expiration_days"] == expiration_days]
    if option_type and option_type.lower() != "all":
        result = result[result["option_type"] == OptionType(option_type).value]
    if moneyness and moneyness.upper() != "ALL":
        label = moneyness.upper()
        if label not in {"ITM", "ATM", "OTM"}:
            raise ValueError("moneyness must be ITM, ATM, OTM, or all")
        result = result[result["moneyness"] == label]
    return result.reset_index(drop=True).copy()

