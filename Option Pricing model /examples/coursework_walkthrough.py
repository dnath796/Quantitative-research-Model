"""Small end-to-end example matching the coursework learning sequence."""

from pathlib import Path

import numpy as np

from derivatives_engine.models import MarketData, Option
from derivatives_engine.payoffs import pnl_table
from derivatives_engine.pricing import (
    black_scholes,
    binomial_convergence,
    futures_price,
    implied_volatility,
    put_call_parity_gap,
)
from derivatives_engine.report import AnalyticsRequest, create_report
from derivatives_engine.risk import OptionHolding, Portfolio
from derivatives_engine.scenario import option_scenarios
from derivatives_engine.strategies import covered_call


def main() -> None:
    market = MarketData(spot=100, rate=0.05, volatility=0.20, dividend_yield=0.01)
    call = Option(strike=100, maturity=1.0, option_type="call")
    put = Option(strike=100, maturity=1.0, option_type="put")

    call_value = black_scholes(100, 100, 1, 0.05, 0.20, "call", 0.01)
    put_value = black_scholes(100, 100, 1, 0.05, 0.20, "put", 0.01)
    print("Basic long-call P&L")
    print(pnl_table([80, 90, 100, 110, 120], 100, call_value).round(2).to_string(index=False))

    strategy = covered_call(100, 105, 3.0)
    print("\nCovered call risk:", strategy.risk_summary())
    print("Futures fair price:", round(futures_price(100, 0.05, 1, 0.01), 4))
    print("Put-call parity gap:", round(put_call_parity_gap(call_value, put_value, 100, 100, 1, 0.05, 0.01), 10))
    print("Recovered volatility:", round(implied_volatility(call_value, 100, 100, 1, 0.05, "call", 0.01), 6))
    print("\nConvergence")
    for row in binomial_convergence(100, 100, 1, 0.05, 0.20, (5, 25, 100, 500), "call", 0.01):
        print(row)

    portfolio = Portfolio().add_option(OptionHolding(call, quantity=10, contract_multiplier=100))
    print("\nPortfolio Greeks:", portfolio.greeks(market).as_dict())
    print("Delta-neutral stock trade:", round(portfolio.delta_hedge(market), 2), "shares")
    print("\nScenarios")
    print(option_scenarios(put, market).round(4).to_string(index=False))

    output = Path("output/walkthrough")
    report = create_report(AnalyticsRequest(call, market, quantity=10), output)
    print("\nReport created:", report)


if __name__ == "__main__":
    main()
