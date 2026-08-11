"""Composable expiration-profit strategies and common strategy factories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import OptionType
from .payoffs import option_payoff


@dataclass(frozen=True, slots=True)
class OptionPosition:
    option_type: OptionType | str
    strike: float
    premium: float
    quantity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        if self.strike <= 0 or self.premium < 0:
            raise ValueError("strike must be positive and premium non-negative")

    def payoff(self, terminal_price):
        return self.quantity * option_payoff(terminal_price, self.strike, self.option_type)

    def profit(self, terminal_price):
        return self.payoff(terminal_price) - self.quantity * self.premium


@dataclass(frozen=True, slots=True)
class UnderlyingPosition:
    initial_price: float
    quantity: float = 1.0

    def payoff(self, terminal_price):
        return self.quantity * np.asarray(terminal_price, dtype=float)

    def profit(self, terminal_price):
        return self.quantity * (np.asarray(terminal_price, dtype=float) - self.initial_price)


class OptionStrategy:
    def __init__(self, name: str, positions=None) -> None:
        self.name = name
        self.positions: list[OptionPosition | UnderlyingPosition] = list(positions or [])

    def add(self, position: OptionPosition | UnderlyingPosition) -> "OptionStrategy":
        self.positions.append(position)
        return self

    def payoff(self, terminal_price):
        return sum((position.payoff(terminal_price) for position in self.positions), start=0.0)

    def profit(self, terminal_price):
        return sum((position.profit(terminal_price) for position in self.positions), start=0.0)

    def table(self, prices) -> pd.DataFrame:
        prices = np.asarray(prices, dtype=float)
        return pd.DataFrame(
            {
                "underlying_price": prices,
                "payoff": self.payoff(prices),
                "profit_loss": self.profit(prices),
            }
        )

    def break_even_points(self, upper: float | None = None, points: int = 20_001) -> list[float]:
        strikes = [p.strike for p in self.positions if isinstance(p, OptionPosition)]
        anchors = strikes + [p.initial_price for p in self.positions if isinstance(p, UnderlyingPosition)]
        scale = max(anchors, default=100.0)
        grid = np.linspace(0.0, upper or 4.0 * scale, points)
        pnl = np.asarray(self.profit(grid))
        roots: list[float] = []
        exact = np.flatnonzero(np.isclose(pnl, 0.0, atol=1e-10))
        for index in exact:
            roots.append(float(grid[index]))
        crossings = np.flatnonzero(pnl[:-1] * pnl[1:] < 0)
        for index in crossings:
            x0, x1 = grid[index], grid[index + 1]
            y0, y1 = pnl[index], pnl[index + 1]
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        # Collapse many grid points on a zero plateau and numerical duplicates.
        unique: list[float] = []
        for root in sorted(roots):
            if not unique or root - unique[-1] > 2.5 * grid[1]:
                unique.append(root)
        return unique

    def risk_summary(self) -> dict[str, float | list[float]]:
        strikes = [p.strike for p in self.positions if isinstance(p, OptionPosition)]
        anchors = strikes + [p.initial_price for p in self.positions if isinstance(p, UnderlyingPosition)]
        scale = max(anchors, default=100.0)
        probe = np.array([0.0, *strikes, 4.0 * scale, 4.0 * scale + 1.0])
        values = np.asarray(self.profit(probe), dtype=float)
        right_slope = values[-1] - values[-2]
        maximum_profit = float("inf") if right_slope > 1e-9 else float(np.max(values[:-1]))
        maximum_loss = float("inf") if right_slope < -1e-9 else float(max(0.0, -np.min(values[:-1])))
        return {
            "maximum_profit": maximum_profit,
            "maximum_loss": maximum_loss,
            "break_even_points": self.break_even_points(),
        }


def covered_call(spot: float, strike: float, call_premium: float) -> OptionStrategy:
    return OptionStrategy("Covered Call", [UnderlyingPosition(spot), OptionPosition("call", strike, call_premium, -1)])


def protective_put(spot: float, strike: float, put_premium: float) -> OptionStrategy:
    return OptionStrategy("Protective Put", [UnderlyingPosition(spot), OptionPosition("put", strike, put_premium, 1)])


def bull_call_spread(low_strike: float, high_strike: float, low_premium: float, high_premium: float) -> OptionStrategy:
    return OptionStrategy("Bull Call Spread", [OptionPosition("call", low_strike, low_premium, 1), OptionPosition("call", high_strike, high_premium, -1)])


def bear_put_spread(low_strike: float, high_strike: float, low_premium: float, high_premium: float) -> OptionStrategy:
    return OptionStrategy("Bear Put Spread", [OptionPosition("put", high_strike, high_premium, 1), OptionPosition("put", low_strike, low_premium, -1)])


def bull_put_spread(low_strike: float, high_strike: float, low_premium: float, high_premium: float) -> OptionStrategy:
    return OptionStrategy("Bull Put Spread", [OptionPosition("put", high_strike, high_premium, -1), OptionPosition("put", low_strike, low_premium, 1)])


def bear_call_spread(low_strike: float, high_strike: float, low_premium: float, high_premium: float) -> OptionStrategy:
    return OptionStrategy("Bear Call Spread", [OptionPosition("call", low_strike, low_premium, -1), OptionPosition("call", high_strike, high_premium, 1)])


def straddle(strike: float, call_premium: float, put_premium: float, quantity: float = 1.0) -> OptionStrategy:
    return OptionStrategy("Long Straddle", [OptionPosition("call", strike, call_premium, quantity), OptionPosition("put", strike, put_premium, quantity)])


def strangle(put_strike: float, call_strike: float, put_premium: float, call_premium: float) -> OptionStrategy:
    return OptionStrategy("Long Strangle", [OptionPosition("put", put_strike, put_premium), OptionPosition("call", call_strike, call_premium)])


def collar(spot: float, put_strike: float, call_strike: float, put_premium: float, call_premium: float) -> OptionStrategy:
    return OptionStrategy("Collar", [UnderlyingPosition(spot), OptionPosition("put", put_strike, put_premium), OptionPosition("call", call_strike, call_premium, -1)])


def butterfly(low: float, middle: float, high: float, premiums: tuple[float, float, float]) -> OptionStrategy:
    if not np.isclose(middle - low, high - middle):
        raise ValueError("a standard butterfly requires equally spaced strikes")
    return OptionStrategy("Long Call Butterfly", [OptionPosition("call", low, premiums[0]), OptionPosition("call", middle, premiums[1], -2), OptionPosition("call", high, premiums[2])])


def iron_condor(strikes: tuple[float, float, float, float], premiums: tuple[float, float, float, float]) -> OptionStrategy:
    k1, k2, k3, k4 = strikes
    if not k1 < k2 < k3 < k4:
        raise ValueError("iron condor strikes must be strictly increasing")
    return OptionStrategy("Iron Condor", [OptionPosition("put", k1, premiums[0]), OptionPosition("put", k2, premiums[1], -1), OptionPosition("call", k3, premiums[2], -1), OptionPosition("call", k4, premiums[3])])

