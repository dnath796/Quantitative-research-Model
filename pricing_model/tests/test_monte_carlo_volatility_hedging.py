import numpy as np
import pandas as pd
import pytest

from derivatives_engine.hedging import simulate_delta_hedge
from derivatives_engine.monte_carlo import analytical_vs_monte_carlo, monte_carlo_european
from derivatives_engine.pricing import black_scholes
from derivatives_engine.volatility import implied_volatility_surface, surface_matrix, synthetic_smile_surface


def test_monte_carlo_matches_black_scholes_with_confidence_interval():
    analytical = black_scholes(100, 105, 0.75, 0.04, 0.24, "call", 0.01)
    result = monte_carlo_european(100, 105, 0.75, 0.04, 0.24, "call", 0.01, simulations=120_000, seed=7)
    assert result.price == pytest.approx(analytical, abs=3.0 * result.standard_error)
    assert result.confidence_low < result.confidence_high
    comparison = analytical_vs_monte_carlo(100, 105, 0.75, 0.04, 0.24, "call", 0.01, 120_000, 7)
    assert comparison["analytical_inside_confidence_interval"]


def test_antithetic_monte_carlo_is_reproducible():
    first = monte_carlo_european(100, 100, 1, 0.05, 0.2, simulations=10_000, seed=42)
    second = monte_carlo_european(100, 100, 1, 0.05, 0.2, simulations=10_000, seed=42)
    assert first == second


def test_implied_volatility_surface_recovers_quote_grid():
    strikes = [90, 100, 110]
    maturities = [0.25, 1.0]
    rows = [
        {
            "strike": strike,
            "maturity": maturity,
            "market_price": black_scholes(100, strike, maturity, 0.03, 0.22, "call", 0.01),
            "option_type": "call",
        }
        for maturity in maturities
        for strike in strikes
    ]
    surface = implied_volatility_surface(pd.DataFrame(rows), 100, 0.03, 0.01)
    assert np.allclose(surface["implied_volatility"], 0.22, atol=1e-9)
    matrix = surface_matrix(surface)
    assert matrix.shape == (2, 3)


def test_synthetic_surface_has_smile_and_term_structure():
    surface = synthetic_smile_surface(100, [80, 100, 120], [0.25, 1.0], 0.2)
    short = surface[surface["maturity"] == 0.25]
    assert short.loc[short["strike"] == 80, "implied_volatility"].iloc[0] > short.loc[short["strike"] == 100, "implied_volatility"].iloc[0]
    atm = surface[surface["strike"] == 100].sort_values("maturity")
    assert atm["implied_volatility"].is_monotonic_increasing


def test_delta_hedging_pnl_is_centered_near_zero_without_costs():
    result = simulate_delta_hedge(100, 100, 0.5, 0.03, 0.2, "call", hedge_steps=32, paths=2_000, seed=12)
    sampling_error = result.std_pnl / np.sqrt(result.paths)
    assert abs(result.mean_pnl) < 4 * sampling_error
    assert len(result.sample_path) == 33
    assert result.percentile_05 < result.percentile_95


def test_transaction_costs_reduce_hedging_pnl():
    no_cost = simulate_delta_hedge(100, 100, 0.25, 0.03, 0.2, hedge_steps=16, paths=1_000, seed=9)
    with_cost = simulate_delta_hedge(100, 100, 0.25, 0.03, 0.2, hedge_steps=16, paths=1_000, seed=9, transaction_cost_bps=10)
    assert with_cost.mean_pnl < no_cost.mean_pnl
