"""Monte Carlo option pricing and analytical-model validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd

from .models import OptionType
from .pricing import black_scholes


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    simulations: int
    seed: int | None
    antithetic: bool


def monte_carlo_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
    simulations: int = 100_000,
    seed: int | None = 42,
    antithetic: bool = True,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Price a European option from exact GBM terminal draws.

    Antithetic sampling is enabled by default to reduce estimator variance.
    """

    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("S and K must be positive; T and sigma must be non-negative")
    if simulations < 2:
        raise ValueError("simulations must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1")
    kind = OptionType(option_type)
    if T == 0:
        payoff = max(S - K, 0.0) if kind is OptionType.CALL else max(K - S, 0.0)
        return MonteCarloResult(payoff, 0.0, payoff, payoff, simulations, seed, antithetic)

    rng = np.random.default_rng(seed)
    if antithetic:
        half = (simulations + 1) // 2
        draws = rng.standard_normal(half)
        z = np.concatenate((draws, -draws))[:simulations]
    else:
        z = rng.standard_normal(simulations)
    terminal = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)
    payoffs = np.maximum(terminal - K, 0.0) if kind is OptionType.CALL else np.maximum(K - terminal, 0.0)
    discounted = exp(-r * T) * payoffs
    price = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / sqrt(simulations))

    # Inverse-normal approximation keeps this module independent of scipy.stats.
    from scipy.stats import norm

    critical = float(norm.ppf(0.5 + confidence / 2.0))
    return MonteCarloResult(
        price,
        standard_error,
        price - critical * standard_error,
        price + critical * standard_error,
        simulations,
        seed,
        antithetic,
    )


def analytical_vs_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
    simulations: int = 100_000,
    seed: int | None = 42,
) -> dict[str, float | bool | int]:
    """Compare BSM with Monte Carlo and test whether BSM lies in the MC CI."""

    analytical = black_scholes(S, K, T, r, sigma, option_type, q)
    mc = monte_carlo_european(
        S, K, T, r, sigma, option_type, q, simulations, seed, antithetic=True
    )
    z_score = (mc.price - analytical) / mc.standard_error if mc.standard_error else 0.0
    return {
        "black_scholes": analytical,
        "monte_carlo": mc.price,
        "standard_error": mc.standard_error,
        "confidence_low": mc.confidence_low,
        "confidence_high": mc.confidence_high,
        "absolute_error": abs(mc.price - analytical),
        "z_score": z_score,
        "analytical_inside_confidence_interval": mc.confidence_low <= analytical <= mc.confidence_high,
        "simulations": simulations,
    }


def monte_carlo_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    sample_sizes=(1_000, 5_000, 10_000, 50_000, 100_000),
    option_type: OptionType | str = "call",
    q: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    benchmark = black_scholes(S, K, T, r, sigma, option_type, q)
    rows = []
    for index, size in enumerate(sample_sizes):
        result = monte_carlo_european(
            S, K, T, r, sigma, option_type, q, int(size), seed + index, True
        )
        rows.append(
            {
                "simulations": int(size),
                "monte_carlo": result.price,
                "standard_error": result.standard_error,
                "black_scholes": benchmark,
                "absolute_error": abs(result.price - benchmark),
            }
        )
    return pd.DataFrame(rows)

