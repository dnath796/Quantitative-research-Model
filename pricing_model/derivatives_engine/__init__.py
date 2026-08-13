"""Derivatives pricing, strategy, and risk analytics engine."""

from .models import Forward, Future, MarketData, Option, OptionType, PositionSide
from .pricing import (
    BinomialResult,
    black_scholes,
    binomial_option_price,
    futures_price,
    implied_volatility,
)
from .risk import Greeks, black_scholes_greeks
from .hedging import HedgingSimulationResult, simulate_delta_hedge
from .monte_carlo import MonteCarloResult, analytical_vs_monte_carlo, monte_carlo_european
from .volatility import implied_volatility_surface, synthetic_smile_surface
from .options_chain import filter_options_chain, generate_options_chain, option_moneyness

__all__ = [
    "BinomialResult",
    "Forward",
    "Future",
    "Greeks",
    "MarketData",
    "MonteCarloResult",
    "Option",
    "OptionType",
    "PositionSide",
    "black_scholes",
    "black_scholes_greeks",
    "monte_carlo_european",
    "analytical_vs_monte_carlo",
    "binomial_option_price",
    "futures_price",
    "implied_volatility",
    "implied_volatility_surface",
    "synthetic_smile_surface",
    "HedgingSimulationResult",
    "simulate_delta_hedge",
    "generate_options_chain",
    "filter_options_chain",
    "option_moneyness",
]

__version__ = "1.0.0"
