"""Analytical option Greeks, portfolio aggregation, and delta hedging."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt

from scipy.stats import norm

from .models import MarketData, Option, OptionType


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def scaled(self, quantity: float, multiplier: float = 1.0) -> "Greeks":
        factor = quantity * multiplier
        return Greeks(*(value * factor for value in self.as_tuple()))

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.delta, self.gamma, self.theta, self.vega, self.rho

    def as_dict(self) -> dict[str, float]:
        return dict(zip(("delta", "gamma", "theta", "vega", "rho"), self.as_tuple()))


ZERO_GREEKS = Greeks(0.0, 0.0, 0.0, 0.0, 0.0)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
) -> Greeks:
    """Return BSM Greeks per one-unit changes in time, volatility, and rate.

    Theta is annual calendar-time decay. Vega and rho are per 1.00 change;
    divide by 100 for a one-percentage-point sensitivity.
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive for analytical Greeks")
    kind = OptionType(option_type)
    root_t = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    pdf = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
    spot_discount = exp(-q * T)
    strike_discount = exp(-r * T)

    gamma = spot_discount * pdf / (S * sigma * root_t)
    vega = S * spot_discount * pdf * root_t
    common_theta = -(S * spot_discount * pdf * sigma) / (2.0 * root_t)
    if kind is OptionType.CALL:
        delta = spot_discount * norm.cdf(d1)
        theta = common_theta - r * K * strike_discount * norm.cdf(d2) + q * S * spot_discount * norm.cdf(d1)
        rho = K * T * strike_discount * norm.cdf(d2)
    else:
        delta = spot_discount * (norm.cdf(d1) - 1.0)
        theta = common_theta + r * K * strike_discount * norm.cdf(-d2) - q * S * spot_discount * norm.cdf(-d1)
        rho = -K * T * strike_discount * norm.cdf(-d2)
    return Greeks(float(delta), float(gamma), float(theta), float(vega), float(rho))


@dataclass(frozen=True, slots=True)
class OptionHolding:
    option: Option
    quantity: float = 1.0
    contract_multiplier: float = 1.0

    def greeks(self, market: MarketData) -> Greeks:
        if self.option.style != "european":
            raise ValueError("analytical Greeks are available only for European options")
        return black_scholes_greeks(
            market.spot,
            self.option.strike,
            self.option.maturity,
            market.rate,
            market.volatility,
            self.option.option_type,
            market.dividend_yield,
        ).scaled(self.quantity, self.contract_multiplier)


@dataclass(frozen=True, slots=True)
class StockHolding:
    quantity: float

    def greeks(self) -> Greeks:
        return Greeks(self.quantity, 0.0, 0.0, 0.0, 0.0)


class Portfolio:
    def __init__(self) -> None:
        self.options: list[OptionHolding] = []
        self.stocks: list[StockHolding] = []

    def add_option(self, holding: OptionHolding) -> "Portfolio":
        self.options.append(holding)
        return self

    def add_stock(self, holding: StockHolding) -> "Portfolio":
        self.stocks.append(holding)
        return self

    def greeks(self, market: MarketData) -> Greeks:
        total = ZERO_GREEKS
        for holding in self.options:
            total = total + holding.greeks(market)
        for holding in self.stocks:
            total = total + holding.greeks()
        return total

    def delta_hedge(self, market: MarketData, round_shares: bool = False) -> float:
        """Shares to trade now to make portfolio delta approximately zero."""

        shares = -self.greeks(market).delta
        return float(round(shares)) if round_shares else shares


def greek_approximation(greeks: Greeks, dS: float = 0.0, dt: float = 0.0, dvol: float = 0.0, dr: float = 0.0) -> float:
    """Second-order delta-gamma P&L approximation with other first-order risks."""

    return (
        greeks.delta * dS
        + 0.5 * greeks.gamma * dS**2
        + greeks.theta * dt
        + greeks.vega * dvol
        + greeks.rho * dr
    )

