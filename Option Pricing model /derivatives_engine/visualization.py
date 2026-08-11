"""Reusable Matplotlib charts for payoffs, sensitivities, Greeks, and convergence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import MarketData, Option
from .pricing import binomial_convergence
from .risk import black_scholes_greeks
from .strategies import OptionStrategy


def _finish(fig, output_path: str | Path | None):
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_strategy(
    strategy: OptionStrategy,
    prices,
    output_path: str | Path | None = None,
):
    prices = np.asarray(prices, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    axes[0].plot(prices, strategy.payoff(prices), color="#3569b0", linewidth=2)
    axes[0].set_title(f"{strategy.name}: Payoff")
    axes[0].set_ylabel("Payoff")
    axes[1].plot(prices, strategy.profit(prices), color="#d05a47", linewidth=2)
    axes[1].set_title(f"{strategy.name}: Profit / Loss")
    axes[1].set_ylabel("Profit / Loss")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Underlying price at expiration")
    return _finish(fig, output_path)


def plot_greeks(
    option: Option,
    market: MarketData,
    spot_prices=None,
    output_path: str | Path | None = None,
):
    spots = np.asarray(
        spot_prices if spot_prices is not None else np.linspace(0.5 * market.spot, 1.5 * market.spot, 200),
        dtype=float,
    )
    curves = [
        black_scholes_greeks(
            spot, option.strike, option.maturity, market.rate,
            market.volatility, option.option_type, market.dividend_yield,
        ).as_dict()
        for spot in spots
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    colors = ["#3569b0", "#d05a47", "#6a9f58", "#8b62a8", "#c88a27"]
    for axis, greek, color in zip(axes.flat, ("delta", "gamma", "theta", "vega", "rho"), colors):
        axis.plot(spots, [row[greek] for row in curves], color=color, linewidth=2)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(option.strike, color="gray", linestyle="--", linewidth=0.8)
        axis.set(title=greek.title(), xlabel="Spot")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{option.option_type.value.title()} Greeks vs Spot")
    return _finish(fig, output_path)


def plot_convergence(
    option: Option,
    market: MarketData,
    steps=(5, 10, 25, 50, 100, 250, 500),
    output_path: str | Path | None = None,
):
    rows = binomial_convergence(
        market.spot, option.strike, option.maturity, market.rate,
        market.volatility, steps, option.option_type, market.dividend_yield,
    )
    x = [row["steps"] for row in rows]
    y = [row["binomial"] for row in rows]
    benchmark = rows[0]["black_scholes"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(x, y, marker="o", label="CRR binomial", color="#3569b0")
    axis.axhline(benchmark, linestyle="--", color="#d05a47", label=f"Black-Scholes ({benchmark:.4f})")
    axis.set(xlabel="Tree steps", ylabel="Option value", title="Binomial Convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    return _finish(fig, output_path)


def close_figure(fig) -> None:
    plt.close(fig)

