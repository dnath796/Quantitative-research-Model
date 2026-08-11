"""Discrete delta-hedging simulation and hedge P&L attribution."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd

from .models import OptionType
from .pricing import black_scholes
from .risk import black_scholes_greeks


@dataclass(frozen=True, slots=True)
class HedgingSimulationResult:
    pnl: np.ndarray
    mean_pnl: float
    std_pnl: float
    percentile_05: float
    median_pnl: float
    percentile_95: float
    sample_path: pd.DataFrame
    paths: int
    hedge_steps: int

    def summary(self) -> dict[str, float | int]:
        return {
            "paths": self.paths,
            "hedge_steps": self.hedge_steps,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl,
            "percentile_05": self.percentile_05,
            "median_pnl": self.median_pnl,
            "percentile_95": self.percentile_95,
        }


def simulate_delta_hedge(
    S: float,
    K: float,
    T: float,
    r: float,
    implied_volatility: float,
    option_type: OptionType | str = "call",
    q: float = 0.0,
    realized_volatility: float | None = None,
    hedge_steps: int = 63,
    paths: int = 2_000,
    seed: int | None = 42,
    transaction_cost_bps: float = 0.0,
) -> HedgingSimulationResult:
    """Simulate a discretely delta-hedged short European option.

    The writer receives the BSM premium, buys delta shares, finances the cash
    account, receives continuous dividend yield on stock inventory, rebalances,
    and settles intrinsic value at expiry. P&L is reported per option unit.
    """

    if min(S, K, T, implied_volatility) <= 0:
        raise ValueError("S, K, T, and implied_volatility must be positive")
    if hedge_steps < 1 or paths < 1:
        raise ValueError("hedge_steps and paths must be positive integers")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    kind = OptionType(option_type)
    realized = implied_volatility if realized_volatility is None else realized_volatility
    if realized < 0:
        raise ValueError("realized_volatility must be non-negative")
    dt = T / hedge_steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths, hedge_steps))
    stock = np.empty((paths, hedge_steps + 1), dtype=float)
    stock[:, 0] = S
    for step in range(1, hedge_steps + 1):
        stock[:, step] = stock[:, step - 1] * np.exp(
            (r - q - 0.5 * realized**2) * dt + realized * sqrt(dt) * shocks[:, step - 1]
        )

    premium = black_scholes(S, K, T, r, implied_volatility, kind, q)
    initial_delta = black_scholes_greeks(S, K, T, r, implied_volatility, kind, q).delta
    shares = np.full(paths, initial_delta)
    cash = np.full(paths, premium - initial_delta * S)
    cost_rate = transaction_cost_bps / 10_000.0

    sample_rows = [
        {"step": 0, "time": 0.0, "spot": S, "delta": initial_delta, "shares_traded": initial_delta, "cash": cash[0], "hedge_value": cash[0] + initial_delta * S}
    ]
    for step in range(1, hedge_steps + 1):
        previous_spot = stock[:, step - 1]
        current_spot = stock[:, step]
        cash *= exp(r * dt)
        cash += shares * previous_spot * (exp(q * dt) - 1.0)
        if step < hedge_steps:
            remaining = T - step * dt
            new_delta = np.array(
                [black_scholes_greeks(s, K, remaining, r, implied_volatility, kind, q).delta for s in current_spot]
            )
            trade = new_delta - shares
            costs = np.abs(trade) * current_spot * cost_rate
            cash -= trade * current_spot + costs
            shares = new_delta
        else:
            trade = np.zeros(paths)
        sample_rows.append(
            {
                "step": step,
                "time": step * dt,
                "spot": current_spot[0],
                "delta": shares[0],
                "shares_traded": trade[0],
                "cash": cash[0],
                "hedge_value": cash[0] + shares[0] * current_spot[0],
            }
        )

    terminal_payoff = (
        np.maximum(stock[:, -1] - K, 0.0)
        if kind is OptionType.CALL
        else np.maximum(K - stock[:, -1], 0.0)
    )
    pnl = cash + shares * stock[:, -1] - terminal_payoff
    quantiles = np.quantile(pnl, [0.05, 0.5, 0.95])
    return HedgingSimulationResult(
        pnl=pnl,
        mean_pnl=float(pnl.mean()),
        std_pnl=float(pnl.std(ddof=1)) if paths > 1 else 0.0,
        percentile_05=float(quantiles[0]),
        median_pnl=float(quantiles[1]),
        percentile_95=float(quantiles[2]),
        sample_path=pd.DataFrame(sample_rows),
        paths=paths,
        hedge_steps=hedge_steps,
    )
