"""Forecast losses and two-sided Diebold–Mariano comparisons with Bartlett HAC."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from quant_research.errors import ResearchError


def qlike(actual_squared: np.ndarray, second_moment: np.ndarray) -> np.ndarray:
    """log(m) + y²/m, omitting a model-independent term; valid even when y=0."""
    actual_squared, second_moment = np.broadcast_arrays(actual_squared, second_moment)
    if not np.isfinite(actual_squared).all() or not np.isfinite(second_moment).all():
        raise ResearchError("QLIKE inputs must be finite")
    if (actual_squared < 0).any() or (second_moment <= 0).any():
        raise ResearchError("QLIKE needs nonnegative squared returns and positive second moments")
    return np.log(second_moment) + actual_squared / second_moment


@dataclass(frozen=True)
class DMResult:
    statistic: float | None
    p_value: float | None
    mean_loss_difference: float
    observations: int
    hac_lags: int
    status: str


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 3) -> DMResult:
    """Negative statistic favors A. Asymptotic normal inference, no small-sample correction."""
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ResearchError("DM losses must be nonempty, aligned one-dimensional arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ResearchError("DM losses must be finite")
    if type(lags) is not int or lags < 0:
        raise ResearchError("DM lags must be a nonnegative integer")
    d, n = a - b, len(a)
    mean = float(d.mean())
    if n <= lags + 1:
        return DMResult(None, None, mean, n, lags, "insufficient_observations")
    if np.array_equal(a, b):
        return DMResult(0.0, 1.0, 0.0, n, lags, "identical_losses")
    centered = d - mean
    lrv = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        lrv += 2 * weight * float(centered[lag:] @ centered[:-lag] / n)
    # A constant nonzero loss gap has no estimable sampling variance.
    if lrv <= np.finfo(float).eps * float(np.mean(d**2)):
        return DMResult(None, None, mean, n, lags, "degenerate_loss_variance")
    statistic = mean / np.sqrt(lrv / n)
    return DMResult(float(statistic), float(2 * norm.sf(abs(statistic))), mean, n, lags, "ok")


def evaluate(forecasts: pd.DataFrame, dm_lags: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a shared test calendar; comparisons always use the historical-mean benchmark."""
    required = {"target_date", "model", "actual", "mean", "variance", "second_moment"}
    if forecasts.empty or required - set(forecasts):
        raise ResearchError("Forecast table is empty or lacks required columns")
    if forecasts.duplicated(["target_date", "model"]).any():
        raise ResearchError("Duplicate forecast dates per model")
    if "mean" not in set(forecasts["model"]):
        raise ResearchError("Evaluation requires the mean benchmark")
    groups = {
        name: group.sort_values("target_date").set_index("target_date")
        for name, group in forecasts.groupby("model")
    }
    benchmark = groups["mean"]
    scores, losses = [], {}
    for name, group in groups.items():
        if not group.index.equals(benchmark.index) or not np.array_equal(
            group.actual.to_numpy(), benchmark.actual.to_numpy()
        ):
            raise ResearchError("Models must share exactly the same target dates and actuals")
        numeric = group[["actual", "mean", "variance", "second_moment"]].to_numpy()
        if not np.isfinite(numeric).all() or (group.variance <= 0).any():
            raise ResearchError("Nonfinite forecast data or nonpositive variance")
        if not np.allclose(group.second_moment, group.variance + group["mean"] ** 2):
            raise ResearchError("second_moment must equal variance + mean squared")
        error = group.actual.to_numpy() - group["mean"].to_numpy()
        squared = group.actual.to_numpy() ** 2
        moments = group.second_moment.to_numpy()
        losses[name] = {
            "squared_error": error**2,
            "absolute_error": np.abs(error),
            "qlike_second_moment": qlike(squared, moments),
        }
        scores.append(
            {
                "model": name,
                "observations": len(group),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(-error.mean()),
                "qlike_second_moment": float(losses[name]["qlike_second_moment"].mean()),
                "second_moment_rmse": float(np.sqrt(np.mean((squared - moments) ** 2))),
            }
        )
    comparisons = []
    for name in groups:
        if name == "mean":
            continue
        for loss in losses[name]:
            result = diebold_mariano(losses[name][loss], losses["mean"][loss], lags=dm_lags)
            comparisons.append({"model": name, "benchmark": "mean", "loss": loss, **asdict(result)})
    return pd.DataFrame(scores), pd.DataFrame(
        comparisons,
        columns=[
            "model",
            "benchmark",
            "loss",
            *DMResult.__dataclass_fields__,
        ],
    )
