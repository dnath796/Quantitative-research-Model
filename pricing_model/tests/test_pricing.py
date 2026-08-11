from math import exp

import pytest

from derivatives_engine.pricing import (
    black_scholes,
    binomial_option_price,
    forward_value,
    futures_price,
    implied_volatility,
    parity_implied_price,
    put_call_parity_gap,
)


def test_black_scholes_textbook_values():
    assert black_scholes(100, 100, 1, 0.05, 0.20, "call") == pytest.approx(10.45058357, rel=1e-8)
    assert black_scholes(100, 100, 1, 0.05, 0.20, "put") == pytest.approx(5.57352602, rel=1e-8)


def test_put_call_parity_and_implied_price():
    call = black_scholes(100, 100, 1, 0.05, 0.20, "call", 0.02)
    put = black_scholes(100, 100, 1, 0.05, 0.20, "put", 0.02)
    assert put_call_parity_gap(call, put, 100, 100, 1, 0.05, 0.02) == pytest.approx(0, abs=1e-12)
    assert parity_implied_price(put, 100, 100, 1, 0.05, "call", 0.02) == pytest.approx(call)


def test_binomial_converges_and_american_put_is_not_cheaper():
    bs = black_scholes(100, 100, 1, 0.05, 0.20, "call")
    tree = binomial_option_price(100, 100, 1, 0.05, 0.20, 1000, "call")
    assert tree == pytest.approx(bs, abs=0.003)
    european_put = binomial_option_price(100, 100, 1, 0.05, 0.20, 500, "put")
    american_put = binomial_option_price(100, 100, 1, 0.05, 0.20, 500, "put", american=True)
    assert american_put >= european_put


def test_small_tree_can_be_inspected():
    result = binomial_option_price(100, 100, 1, 0.05, 0.20, 2, return_tree=True)
    assert len(result.stock_tree) == 3
    assert len(result.option_tree) == 3
    assert 0 <= result.risk_neutral_probability <= 1
    assert result.price > 0


def test_implied_volatility_round_trip_and_bounds():
    price = black_scholes(100, 110, 0.75, 0.03, 0.31, "put", 0.01)
    assert implied_volatility(price, 100, 110, 0.75, 0.03, "put", 0.01) == pytest.approx(0.31, abs=1e-9)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(200, 100, 100, 1, 0.05, "call")


def test_futures_and_forward_cost_of_carry():
    fair = futures_price(100, 0.05, 1, 0.02)
    assert fair == pytest.approx(100 * exp(0.03))
    assert forward_value(100, fair, 1, 0.05, 0.02) == pytest.approx(0, abs=1e-12)


def test_expiry_and_zero_volatility_edges():
    assert black_scholes(120, 100, 0, 0.05, 0.2, "call") == 20
    assert black_scholes(100, 100, 1, 0.05, 0, "call") == pytest.approx(100 - 100 * exp(-0.05))

