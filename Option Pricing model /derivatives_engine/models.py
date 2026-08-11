"""Financial contracts and market data objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionSide.LONG else -1


def _positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0 if allow_zero else value > 0
    if not valid or not np.isfinite(value):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {comparator}")
    return value


@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot", _positive("spot", self.spot))
        object.__setattr__(self, "volatility", _positive("volatility", self.volatility, allow_zero=True))
        for name in ("rate", "dividend_yield"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def shocked(
        self,
        *,
        spot_pct: float = 0.0,
        volatility_shift: float = 0.0,
        rate_shift: float = 0.0,
    ) -> "MarketData":
        return MarketData(
            spot=self.spot * (1.0 + spot_pct),
            rate=self.rate + rate_shift,
            volatility=max(0.0, self.volatility + volatility_shift),
            dividend_yield=self.dividend_yield,
        )


class Derivative(ABC):
    @abstractmethod
    def payoff(self, underlying_price):
        """Return terminal cash flow for one long contract."""


@dataclass(frozen=True, slots=True)
class Option(Derivative):
    strike: float
    maturity: float
    option_type: OptionType | str = OptionType.CALL
    style: str = "european"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", _positive("strike", self.strike))
        object.__setattr__(self, "maturity", _positive("maturity", self.maturity, allow_zero=True))
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        style = self.style.lower()
        if style not in {"european", "american"}:
            raise ValueError("style must be 'european' or 'american'")
        object.__setattr__(self, "style", style)

    def payoff(self, underlying_price):
        prices = np.asarray(underlying_price, dtype=float)
        if self.option_type is OptionType.CALL:
            result = np.maximum(prices - self.strike, 0.0)
        else:
            result = np.maximum(self.strike - prices, 0.0)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True, slots=True)
class Forward(Derivative):
    delivery_price: float
    maturity: float

    def payoff(self, underlying_price):
        result = np.asarray(underlying_price, dtype=float) - self.delivery_price
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True, slots=True)
class Future(Forward):
    contract_size: float = 1.0

    def payoff(self, underlying_price):
        return super().payoff(underlying_price) * self.contract_size


@dataclass(frozen=True, slots=True)
class FixedForFloatingSwap(Derivative):
    """Simple one-payment interest-rate swap representation."""

    notional: float
    fixed_rate: float
    accrual: float = 1.0

    def payoff(self, floating_rate):
        result = self.notional * self.accrual * (
            np.asarray(floating_rate, dtype=float) - self.fixed_rate
        )
        return float(result) if result.ndim == 0 else result

