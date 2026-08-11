"""Payoff and profit/loss helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import OptionType, PositionSide


def option_payoff(terminal_price, strike: float, option_type: OptionType | str = "call"):
    prices = np.asarray(terminal_price, dtype=float)
    kind = OptionType(option_type)
    result = (
        np.maximum(prices - strike, 0.0)
        if kind is OptionType.CALL
        else np.maximum(strike - prices, 0.0)
    )
    return float(result) if result.ndim == 0 else result


def call_payoff(terminal_price, strike: float):
    return option_payoff(terminal_price, strike, OptionType.CALL)


def put_payoff(terminal_price, strike: float):
    return option_payoff(terminal_price, strike, OptionType.PUT)


def option_profit(
    terminal_price,
    strike: float,
    premium: float,
    option_type: OptionType | str = "call",
    side: PositionSide | str = "long",
    quantity: float = 1.0,
):
    direction = PositionSide(side).sign
    return direction * quantity * (option_payoff(terminal_price, strike, option_type) - premium)


def futures_payoff(terminal_price, delivery_price: float, side: PositionSide | str = "long"):
    result = PositionSide(side).sign * (
        np.asarray(terminal_price, dtype=float) - delivery_price
    )
    return float(result) if result.ndim == 0 else result


def pnl_table(
    prices,
    strike: float,
    premium: float,
    option_type: OptionType | str = "call",
    side: PositionSide | str = "long",
    quantity: float = 1.0,
) -> pd.DataFrame:
    values = np.asarray(prices, dtype=float)
    raw_payoff = option_payoff(values, strike, option_type)
    signed_payoff = PositionSide(side).sign * quantity * raw_payoff
    profit = option_profit(values, strike, premium, option_type, side, quantity)
    return pd.DataFrame(
        {
            "underlying_price": values,
            "payoff": signed_payoff,
            "premium_cash_flow": -PositionSide(side).sign * quantity * premium,
            "profit_loss": profit,
        }
    )
