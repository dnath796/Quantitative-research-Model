import numpy as np
import pytest

from derivatives_engine.models import Future, MarketData, Option
from derivatives_engine.payoffs import call_payoff, futures_payoff, option_profit, pnl_table, put_payoff


def test_option_payoffs_scalar_and_vector():
    assert call_payoff(120, 100) == 20
    assert put_payoff(80, 100) == 20
    np.testing.assert_array_equal(call_payoff(np.array([80, 100, 120]), 100), [0, 0, 20])
    np.testing.assert_array_equal(Option(100, 1, "put").payoff([80, 100, 120]), [20, 0, 0])


def test_long_and_short_profit_are_opposites():
    prices = np.array([80, 100, 120])
    long = option_profit(prices, 100, 5, "call", "long")
    short = option_profit(prices, 100, 5, "call", "short")
    np.testing.assert_array_equal(long, -short)
    assert pnl_table(prices, 100, 5).iloc[-1]["profit_loss"] == 15


def test_futures_contract_and_direction():
    future = Future(delivery_price=100, maturity=0.5, contract_size=50)
    assert future.payoff(104) == 200
    assert futures_payoff(104, 100, "short") == -4


def test_market_data_validation_and_shocks():
    market = MarketData(100, 0.05, 0.2, 0.01)
    shocked = market.shocked(spot_pct=0.1, volatility_shift=0.05, rate_shift=0.01)
    assert shocked.spot == pytest.approx(110)
    assert shocked.volatility == pytest.approx(0.25)
    assert shocked.rate == pytest.approx(0.06)
    with pytest.raises(ValueError):
        MarketData(0, 0.05, 0.2)

