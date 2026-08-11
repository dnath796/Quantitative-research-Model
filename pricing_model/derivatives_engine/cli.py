"""Command-line interface for the integrated analytics report."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import MarketData, Option
from .report import AnalyticsRequest, analytics_summary, create_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derivatives pricing and risk analytics engine")
    parser.add_argument("--spot", type=float, default=200.0)
    parser.add_argument("--strike", type=float, default=105.0)
    parser.add_argument("--maturity", type=float, default=0.5, help="Years to expiration")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.25)
    parser.add_argument("--dividend-yield", type=float, default=0.02)
    parser.add_argument("--option-type", choices=("call", "put"), default="call")
    parser.add_argument("--style", choices=("european", "american"), default="european")
    parser.add_argument("--position", choices=("long", "short"), default="long")
    parser.add_argument("--quantity", type=float, default=10.0)
    parser.add_argument("--multiplier", type=float, default=100.0)
    parser.add_argument("--premium", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    request = AnalyticsRequest(
        option=Option(args.strike, args.maturity, args.option_type, args.style),
        market=MarketData(args.spot, args.rate, args.volatility, args.dividend_yield),
        position=args.position,
        quantity=args.quantity,
        contract_multiplier=args.multiplier,
        premium=args.premium,
    )
    summary = analytics_summary(request)
    report = create_report(request, args.output)
    print("DERIVATIVES ANALYTICS ENGINE")
    print("=" * 36)
    print(f"Black-Scholes:     ${summary['black_scholes']:,.4f}")
    print(f"Binomial 100-step: ${summary['binomial_100']:,.4f}")
    print(f"Binomial 500-step: ${summary['binomial_500']:,.4f}")
    print(f"Monte Carlo:       ${summary['monte_carlo']:,.4f}")
    print(f"Position delta:     {summary['position_greeks']['delta']:,.4f}")
    print(f"Report:             {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
