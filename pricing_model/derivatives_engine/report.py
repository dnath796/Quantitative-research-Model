"""Integrated option analytics report and chart generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import MarketData, Option, PositionSide
from .payoffs import option_profit
from .pricing import black_scholes, binomial_option_price
from .monte_carlo import monte_carlo_european
from .hedging import simulate_delta_hedge
from .volatility import synthetic_smile_surface
from .risk import black_scholes_greeks
from .scenario import option_scenarios
from .strategies import OptionPosition, OptionStrategy
from .visualization import close_figure, plot_convergence, plot_greeks, plot_strategy


@dataclass(frozen=True, slots=True)
class AnalyticsRequest:
    option: Option
    market: MarketData
    position: PositionSide | str = PositionSide.LONG
    quantity: float = 1.0
    contract_multiplier: float = 100.0
    premium: float | None = None


def analytics_summary(request: AnalyticsRequest) -> dict:
    option, market = request.option, request.market
    side = PositionSide(request.position)
    bs = black_scholes(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, option.option_type, market.dividend_yield,
    )
    b100 = binomial_option_price(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, 100, option.option_type, market.dividend_yield,
        american=option.style == "american",
    )
    b500 = binomial_option_price(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, 500, option.option_type, market.dividend_yield,
        american=option.style == "american",
    )
    mc = monte_carlo_european(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, option.option_type, market.dividend_yield,
        simulations=20_000, seed=42,
    )
    greek = black_scholes_greeks(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, option.option_type, market.dividend_yield,
    )
    signed_quantity = side.sign * request.quantity
    multiplier = request.contract_multiplier
    position_greeks = greek.scaled(signed_quantity, multiplier)
    premium = bs if request.premium is None else request.premium
    strategy = OptionStrategy(
        f"{side.value.title()} {option.option_type.value.title()}",
        [OptionPosition(option.option_type, option.strike, premium, signed_quantity * multiplier)],
    )
    return {
        "black_scholes": bs,
        "binomial_100": b100,
        "binomial_500": b500,
        "monte_carlo": mc.price,
        "monte_carlo_standard_error": mc.standard_error,
        "monte_carlo_confidence_interval": (mc.confidence_low, mc.confidence_high),
        "greeks_per_unit": greek.as_dict(),
        "position_greeks": position_greeks.as_dict(),
        "premium": premium,
        "strategy_risk": strategy.risk_summary(),
        "current_position_value": signed_quantity * multiplier * bs,
    }


def create_report(request: AnalyticsRequest, output_dir: str | Path) -> Path:
    """Create a Markdown report, CSV scenario table, and three PNG charts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = analytics_summary(request)
    option, market = request.option, request.market
    side = PositionSide(request.position)
    signed_quantity = side.sign * request.quantity
    premium = summary["premium"]
    strategy = OptionStrategy(
        f"{side.value.title()} {option.option_type.value.title()}",
        [OptionPosition(option.option_type, option.strike, premium, signed_quantity * request.contract_multiplier)],
    )

    scenario_frame = option_scenarios(
        option, market, quantity=signed_quantity, contract_multiplier=request.contract_multiplier
    )
    scenario_frame.to_csv(output / "scenarios.csv", index=False)
    surface = synthetic_smile_surface(
        market.spot,
        [market.spot * ratio for ratio in (0.8, 0.9, 1.0, 1.1, 1.2)],
        (0.1, 0.25, 0.5, 1.0, 2.0),
        market.volatility,
    )
    surface.to_csv(output / "volatility_surface.csv", index=False)
    hedge = simulate_delta_hedge(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, option.option_type, market.dividend_yield,
        hedge_steps=26, paths=500, seed=42, transaction_cost_bps=1,
    )
    hedge.sample_path.to_csv(output / "hedging_sample_path.csv", index=False)
    upper = max(market.spot, option.strike) * 1.6
    import numpy as np
    prices = np.linspace(0.35 * min(market.spot, option.strike), upper, 400)
    figures = (
        plot_strategy(strategy, prices, output / "payoff_pnl.png"),
        plot_greeks(option, market, output_path=output / "greeks.png"),
        plot_convergence(option, market, output_path=output / "convergence.png"),
    )
    for figure in figures:
        close_figure(figure)

    g = summary["greeks_per_unit"]
    pg = summary["position_greeks"]
    risk = summary["strategy_risk"]
    break_evens = ", ".join(f"${x:,.2f}" for x in risk["break_even_points"]) or "None"
    max_profit = "Unlimited" if risk["maximum_profit"] == float("inf") else f"${risk['maximum_profit']:,.2f}"
    max_loss = "Unlimited" if risk["maximum_loss"] == float("inf") else f"${risk['maximum_loss']:,.2f}"
    markdown = f"""# Derivatives Analytics Report

## Contract

| Field | Value |
|---|---:|
| Contract | {option.style.title()} {option.option_type.value.title()} |
| Position | {side.value.title()} {request.quantity:g} contract(s) |
| Contract multiplier | {request.contract_multiplier:g} |
| Spot | ${market.spot:,.2f} |
| Strike | ${option.strike:,.2f} |
| Maturity | {option.maturity:.4f} years |
| Volatility | {market.volatility:.2%} |
| Risk-free rate | {market.rate:.2%} |
| Dividend yield | {market.dividend_yield:.2%} |

## Pricing

| Model | Value per unit |
|---|---:|
| Black-Scholes | ${summary['black_scholes']:,.4f} |
| CRR binomial (100 steps) | ${summary['binomial_100']:,.4f} |
| CRR binomial (500 steps) | ${summary['binomial_500']:,.4f} |
| Monte Carlo (20,000 paths) | ${summary['monte_carlo']:,.4f} ± ${1.96 * summary['monte_carlo_standard_error']:,.4f} |

## Greeks

Vega and rho are shown per 1.00 change in volatility/rate; divide by 100 for a one-point move.

| Greek | Per unit | Position |
|---|---:|---:|
| Delta | {g['delta']:.6f} | {pg['delta']:.4f} |
| Gamma | {g['gamma']:.6f} | {pg['gamma']:.4f} |
| Theta (annual) | {g['theta']:.6f} | {pg['theta']:.4f} |
| Vega | {g['vega']:.6f} | {pg['vega']:.4f} |
| Rho | {g['rho']:.6f} | {pg['rho']:.4f} |

## Expiration Risk

| Metric | Value |
|---|---:|
| Premium used | ${premium:,.4f} per unit |
| Maximum profit | {max_profit} |
| Maximum loss | {max_loss} |
| Break-even | {break_evens} |

## Charts and Scenarios

![Payoff and P&L](payoff_pnl.png)

![Greeks](greeks.png)

![Model convergence](convergence.png)

Detailed scenario results are in `scenarios.csv`.

## Volatility and Hedging

The illustrative strike/maturity surface is in `volatility_surface.csv`.

| Delta-hedging statistic | P&L per option unit |
|---|---:|
| Mean | ${hedge.mean_pnl:,.4f} |
| Standard deviation | ${hedge.std_pnl:,.4f} |
| 5th percentile | ${hedge.percentile_05:,.4f} |
| Median | ${hedge.median_pnl:,.4f} |
| 95th percentile | ${hedge.percentile_95:,.4f} |

One complete rebalance history is in `hedging_sample_path.csv`.
"""
    report_path = output / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
