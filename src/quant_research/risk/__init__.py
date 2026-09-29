"""Optional illustrative sign strategy, with explicit turnover costs."""

import numpy as np
import pandas as pd

from quant_research.errors import ResearchError


def strategy_returns(forecasts: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """Hold sign(predicted mean) during each target period; charge entry, flips, and final exit.

    Cash earns zero. No leverage, financing, futures roll, slippage, or margin model.
    This is a return-series illustration, not an executable WTI futures simulation.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ResearchError("Transaction cost must be finite and nonnegative")
    if forecasts.empty or forecasts.duplicated(["model", "target_date"]).any():
        raise ResearchError("Strategy requires unique, nonempty forecasts")
    frames = []
    for _, group in forecasts.groupby("model"):
        group = group.sort_values("target_date").copy()
        if not np.isfinite(group[["actual", "mean"]].to_numpy()).all():
            raise ResearchError("Strategy inputs must be finite")
        position = np.sign(group["mean"].to_numpy())
        turnover = np.abs(np.diff(position, prepend=0))
        turnover[-1] += abs(position[-1])  # Close the final position.
        group["position"] = position
        group["turnover"] = turnover
        group["cost"] = turnover * cost_bps / 10000
        group["gross_return"] = position * group.actual.to_numpy()
        group["net_return"] = group.gross_return - group.cost
        if (group.net_return <= -1).any():
            raise ResearchError("Strategy wealth exhausted; this simple model cannot continue")
        group["wealth"] = (1 + group.net_return).cumprod()
        group["drawdown"] = group.wealth / group.wealth.cummax().clip(lower=1) - 1
        frames.append(group)
    return pd.concat(frames, ignore_index=True)
