"""No-arbitrage, closed-form, lattice, and implied-volatility pricing."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .models import OptionType


def _validate_inputs(S: float, K: float, T: float, sigma: float | None = None) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma is not None and sigma < 0:
        raise ValueError("sigma must be non-negative")


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton value for a European option.

    Rates, yield, volatility, and maturity are expressed in decimal/year units.
    """

    _validate_inputs(S, K, T, sigma)
    kind = OptionType(option_type)
    if T == 0:
        return max(S - K, 0.0) if kind is OptionType.CALL else max(K - S, 0.0)
    if sigma == 0:
        forward_pv = S * exp(-q * T) - K * exp(-r * T)
        return max(forward_pv, 0.0) if kind is OptionType.CALL else max(-forward_pv, 0.0)

    root_t = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    if kind is OptionType.CALL:
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


@dataclass(slots=True)
class BinomialResult:
    price: float
    stock_tree: list[np.ndarray]
    option_tree: list[np.ndarray]
    risk_neutral_probability: float


def binomial_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int,
    option_type: OptionType | str = "call",
    q: float = 0.0,
    american: bool = False,
    return_tree: bool = False,
) -> float | BinomialResult:
    """Price an option with a Cox-Ross-Rubinstein recombining tree."""

    _validate_inputs(S, K, T, sigma)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    kind = OptionType(option_type)
    if T == 0:
        price = max(S - K, 0.0) if kind is OptionType.CALL else max(K - S, 0.0)
        result = BinomialResult(price, [np.array([S])], [np.array([price])], float("nan"))
        return result if return_tree else price
    if sigma == 0:
        price = black_scholes(S, K, T, r, sigma, kind, q)
        result = BinomialResult(price, [np.array([S])], [np.array([price])], float("nan"))
        return result if return_tree else price

    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    growth = exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("tree violates no-arbitrage: require d <= exp((r-q)dt) <= u")

    indices = np.arange(steps + 1)
    terminal_stock = S * u ** (steps - indices) * d**indices
    if kind is OptionType.CALL:
        values = np.maximum(terminal_stock - K, 0.0)
    else:
        values = np.maximum(K - terminal_stock, 0.0)

    stock_tree: list[np.ndarray] = []
    option_tree: list[np.ndarray] = []
    if return_tree:
        stock_tree = [
            S * u ** (level - np.arange(level + 1)) * d ** np.arange(level + 1)
            for level in range(steps + 1)
        ]
        option_tree = [np.empty(level + 1) for level in range(steps + 1)]
        option_tree[-1] = values.copy()

    discount = exp(-r * dt)
    for level in range(steps - 1, -1, -1):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            level_stock = S * u ** (level - np.arange(level + 1)) * d ** np.arange(level + 1)
            intrinsic = (
                np.maximum(level_stock - K, 0.0)
                if kind is OptionType.CALL
                else np.maximum(K - level_stock, 0.0)
            )
            values = np.maximum(values, intrinsic)
        if return_tree:
            option_tree[level] = values.copy()

    result = BinomialResult(float(values[0]), stock_tree, option_tree, p)
    return result if return_tree else result.price


def futures_price(S: float, r: float, T: float, q: float = 0.0) -> float:
    _validate_inputs(S, 1.0, T)
    return S * exp((r - q) * T)


def forward_value(
    S: float, delivery_price: float, T: float, r: float, q: float = 0.0
) -> float:
    """Current value of a long forward with delivery price K."""

    _validate_inputs(S, delivery_price, T)
    return S * exp(-q * T) - delivery_price * exp(-r * T)


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Observed (C-P) less theoretical (C-P); zero means parity holds."""

    return call_price - put_price - (S * exp(-q * T) - K * exp(-r * T))


def parity_implied_price(
    observed_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    target: OptionType | str,
    q: float = 0.0,
) -> float:
    """Infer a call from an observed put or a put from an observed call."""

    carry = S * exp(-q * T) - K * exp(-r * T)
    return observed_price + carry if OptionType(target) is OptionType.CALL else observed_price - carry


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
    lower: float = 1e-8,
    upper: float = 5.0,
) -> float:
    """Recover BSM implied volatility using a robust bracketed root solver."""

    _validate_inputs(S, K, T)
    if T == 0:
        raise ValueError("implied volatility is undefined at expiration")
    if market_price < 0:
        raise ValueError("market_price must be non-negative")
    kind = OptionType(option_type)
    discounted_spot = S * exp(-q * T)
    discounted_strike = K * exp(-r * T)
    if kind is OptionType.CALL:
        minimum, maximum = max(0.0, discounted_spot - discounted_strike), discounted_spot
    else:
        minimum, maximum = max(0.0, discounted_strike - discounted_spot), discounted_strike
    tolerance = 1e-12
    if market_price < minimum - tolerance or market_price >= maximum:
        raise ValueError(f"market price violates no-arbitrage bounds [{minimum:.6g}, {maximum:.6g})")

    objective = lambda vol: black_scholes(S, K, T, r, vol, kind, q) - market_price
    low_value, high_value = objective(lower), objective(upper)
    if abs(low_value) < tolerance:
        return 0.0
    if low_value * high_value > 0:
        raise ValueError("could not bracket implied volatility; increase upper bound")
    return float(brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12))


def binomial_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: list[int] | tuple[int, ...] = (5, 10, 25, 50, 100, 250, 500),
    option_type: OptionType | str = "call",
    q: float = 0.0,
) -> list[dict[str, float]]:
    benchmark = black_scholes(S, K, T, r, sigma, option_type, q)
    return [
        {
            "steps": int(n),
            "binomial": float(binomial_option_price(S, K, T, r, sigma, n, option_type, q)),
            "black_scholes": benchmark,
        }
        for n in steps
    ]
