from pathlib import Path

import numpy as np
import pytest

from derivatives_engine.models import MarketData, Option
from derivatives_engine.report import AnalyticsRequest, create_report
from derivatives_engine.strategies import (
    OptionPosition,
    OptionStrategy,
    bull_call_spread,
    butterfly,
    covered_call,
    iron_condor,
    protective_put,
    straddle,
)


def test_bull_call_spread_risk_and_break_even():
    strategy = bull_call_spread(100, 110, 7, 3)
    risk = strategy.risk_summary()
    assert risk["maximum_profit"] == pytest.approx(6)
    assert risk["maximum_loss"] == pytest.approx(4)
    assert risk["break_even_points"] == pytest.approx([104], abs=0.03)


def test_covered_call_and_protective_put_shapes():
    covered = covered_call(100, 110, 4)
    assert covered.profit(150) == covered.profit(110)
    assert covered.risk_summary()["maximum_profit"] == pytest.approx(14)
    protected = protective_put(100, 90, 2)
    assert protected.profit(0) == protected.profit(90)
    assert protected.risk_summary()["maximum_profit"] == float("inf")


def test_straddle_has_two_break_evens():
    risk = straddle(100, 6, 5).risk_summary()
    assert risk["break_even_points"] == pytest.approx([89, 111], abs=0.03)
    assert risk["maximum_profit"] == float("inf")
    assert risk["maximum_loss"] == pytest.approx(11)


def test_butterfly_and_iron_condor_validation():
    strategy = butterfly(90, 100, 110, (12, 7, 3))
    assert strategy.profit(100) > strategy.profit(90)
    with pytest.raises(ValueError):
        butterfly(90, 101, 110, (12, 7, 3))
    assert iron_condor((90, 95, 105, 110), (1, 2, 2, 1)).profit(100) == pytest.approx(2)


def test_generic_strategy_table():
    strategy = OptionStrategy("Call", [OptionPosition("call", 100, 5)])
    table = strategy.table(np.array([90, 100, 110]))
    assert list(table.columns) == ["underlying_price", "payoff", "profit_loss"]
    assert table.iloc[-1]["profit_loss"] == 5


def test_integrated_report_smoke(tmp_path: Path):
    request = AnalyticsRequest(Option(105, 0.5, "call"), MarketData(100, 0.05, 0.25, 0.02), quantity=2)
    report = create_report(request, tmp_path)
    assert report.exists()
    assert "Black-Scholes" in report.read_text()
    for filename in ("scenarios.csv", "volatility_surface.csv", "options_chain.csv", "hedging_sample_path.csv", "payoff_pnl.png", "greeks.png", "convergence.png"):
        assert (tmp_path / filename).stat().st_size > 0
