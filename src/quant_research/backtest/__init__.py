"""One-step walk-forward evaluation on a shared, uninterrupted test calendar."""

import logging
from dataclasses import dataclass

import pandas as pd

from quant_research.config import BacktestConfig
from quant_research.data import validate_monthly
from quant_research.errors import ModelFitError, ResearchError
from quant_research.features import ResearchDataset
from quant_research.models import fit_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    forecasts: pd.DataFrame
    diagnostics: list[dict]


def backtest(
    dataset: ResearchDataset, models: tuple[str, ...], config: BacktestConfig
) -> BacktestResult:
    """Refit each model every month, training strictly before the forecast target."""
    X, y = dataset.X, dataset.y
    validate_monthly(X)
    if not X.index.equals(y.index) or len(y) <= config.initial_train:
        raise ResearchError("Aligned data must contain more rows than initial_train")
    if not models or len(set(models)) != len(models):
        raise ResearchError("Specify a nonempty set of distinct models")
    rows, diagnostics = [], []
    for stop in range(config.initial_train, len(y)):
        start = max(0, stop - config.window_size) if config.window == "rolling" else 0
        train_X, train_y = X.iloc[start:stop], y.iloc[start:stop]
        target_date = y.index[stop]
        for name in models:
            try:
                fitted = fit_model(name, train_X, train_y)
                prediction = fitted.forecast(X.iloc[stop])
            except ResearchError as exc:
                raise ModelFitError(f"{name} at {target_date.date()}: {exc}") from exc
            rows.append(
                {
                    "target_date": target_date,
                    "origin_date": y.index[stop - 1],
                    "train_start": train_y.index[0],
                    "train_end": train_y.index[-1],
                    "n_train": len(train_y),
                    "model": name,
                    "actual": float(y.iloc[stop]),
                    "mean": prediction.mean,
                    "variance": prediction.variance,
                    "second_moment": prediction.second_moment,
                }
            )
            diagnostics.append(
                {
                    "target_date": str(target_date.date()),
                    "model": name,
                    **fitted.diagnostics,
                }
            )
        if (stop - config.initial_train) % 25 == 0:
            logger.info(
                "forecast_completed",
                extra={
                    "target_date": str(target_date.date()),
                    "n_train": len(train_y),
                },
            )
    return BacktestResult(pd.DataFrame(rows), diagnostics)
