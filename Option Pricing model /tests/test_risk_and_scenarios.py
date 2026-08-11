import pytest

from derivatives_engine.models import MarketData, Option
from derivatives_engine.pricing import black_scholes
from derivatives_engine.risk import OptionHolding, Portfolio, StockHolding, black_scholes_greeks
from derivatives_engine.scenario import Scenario, option_scenarios, sensitivity_curve


def test_greeks_against_central_differences():
    S, K, T, r, sigma, q = 100, 105, 0.75, 0.04, 0.24, 0.01
    greeks = black_scholes_greeks(S, K, T, r, sigma, "call", q)
    h = 1e-3
    up = black_scholes(S + h, K, T, r, sigma, "call", q)
    down = black_scholes(S - h, K, T, r, sigma, "call", q)
    base = black_scholes(S, K, T, r, sigma, "call", q)
    assert greeks.delta == pytest.approx((up - down) / (2 * h), rel=1e-6)
    assert greeks.gamma == pytest.approx((up - 2 * base + down) / h**2, rel=2e-4)

    vol_h = 1e-5
    vol_fd = (
        black_scholes(S, K, T, r, sigma + vol_h, "call", q)
        - black_scholes(S, K, T, r, sigma - vol_h, "call", q)
    ) / (2 * vol_h)
    assert greeks.vega == pytest.approx(vol_fd, rel=1e-7)


def test_portfolio_aggregation_and_delta_hedge():
    market = MarketData(100, 0.05, 0.2)
    option = Option(100, 1, "call")
    portfolio = (
        Portfolio()
        .add_option(OptionHolding(option, quantity=10, contract_multiplier=100))
        .add_stock(StockHolding(-100))
    )
    risk = portfolio.greeks(market)
    expected_option_delta = black_scholes_greeks(100, 100, 1, 0.05, 0.2).delta * 1000
    assert risk.delta == pytest.approx(expected_option_delta - 100)
    assert portfolio.delta_hedge(market) == pytest.approx(-risk.delta)


def test_scenarios_are_full_revaluations():
    market = MarketData(100, 0.05, 0.2)
    option = Option(100, 90 / 365, "call")
    scenarios = [Scenario("Base"), Scenario("Up", spot_pct=0.1), Scenario("Later", days_elapsed=30)]
    frame = option_scenarios(option, market, scenarios)
    assert frame.loc[0, "position_pnl"] == pytest.approx(0)
    assert frame.loc[1, "option_value"] > frame.loc[0, "option_value"]
    assert frame.loc[2, "days_to_expiry"] == 60


def test_sensitivity_curve_direction():
    frame = sensitivity_curve(
        Option(100, 1, "call"),
        MarketData(100, 0.05, 0.2),
        "volatility",
        [0.1, 0.2, 0.3],
    )
    assert frame["option_price"].is_monotonic_increasing

