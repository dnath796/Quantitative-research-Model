"""Causal transformations applied before dropping only the leading warmup rows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.config import FeatureSpec
from quant_research.data import validate_monthly
from quant_research.errors import ResearchError


@dataclass(frozen=True)
class ResearchDataset:
    X: pd.DataFrame
    y: pd.Series
    warmup_rows: int


def build_features(
    frame: pd.DataFrame, features: tuple[FeatureSpec, ...], target: str = "wti_chg_pct"
) -> ResearchDataset:
    """Feature row t uses observations no later than t-lag; no full-sample estimates."""
    validate_monthly(frame)
    needed = {target, *(spec.column for spec in features)}
    if needed - set(frame):
        raise ResearchError(f"Missing feature/target columns: {sorted(needed - set(frame))}")
    X = pd.DataFrame(index=frame.index)
    for spec in features:
        series = frame[spec.column]
        if spec.transform == "difference":
            series = series.diff()
        elif spec.transform == "log_return":
            if (series.dropna() <= 0).any():
                raise ResearchError(f"log_return needs positive levels: {spec.column}")
            series = np.log(series).diff()
        X[spec.name] = series.shift(spec.lag)
    y = frame[target].rename("actual")
    valid = X.notna().all(axis=1) & y.notna()
    if not valid.any():
        raise ResearchError("No complete observations after lagging")
    warmup = int(np.argmax(valid.to_numpy()))
    if not valid.iloc[warmup:].all():
        bad = str(valid.iloc[warmup:][~valid.iloc[warmup:]].index[0].date())
        raise ResearchError(f"Missing values inside evaluation sample at {bad}; repair source data")
    X, y = X.iloc[warmup:], y.iloc[warmup:]
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ResearchError("Nonfinite values after transformation")
    return ResearchDataset(X, y, warmup)
