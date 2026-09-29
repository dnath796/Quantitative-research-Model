"""Training-only mean/OLS fits and one-step ARCH-family forecasts."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from quant_research.errors import ModelFitError, ResearchError

VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class Forecast:
    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.mean, self.variance]).all() or self.variance <= 0:
            raise ModelFitError("Forecast mean/variance must be finite, with positive variance")

    @property
    def second_moment(self) -> float:
        return self.variance + self.mean**2


@dataclass
class FittedModel:
    name: str
    mean: float
    variance: float
    diagnostics: dict[str, Any]
    columns: tuple[str, ...] = ()
    coefficients: np.ndarray | None = field(default=None, repr=False)
    center: np.ndarray | None = field(default=None, repr=False)
    scale: np.ndarray | None = field(default=None, repr=False)

    def forecast(self, features: pd.Series | None = None) -> Forecast:
        """Predict one month ahead using a feature row prepared by build_features."""
        mean = self.mean
        if self.coefficients is not None:
            if features is None or tuple(features.index) != self.columns:
                raise ResearchError("Forecast features must match the fitted column order")
            row = features.to_numpy(dtype=float)
            if not np.isfinite(row).all():
                raise ResearchError("Forecast features must be finite")
            z = (row - self.center) / self.scale
            mean = float(self.coefficients[0] + z @ self.coefficients[1:])
        return Forecast(mean, self.variance)


def _diagnostics(residuals: np.ndarray) -> dict[str, Any]:
    lag1 = None
    if np.std(residuals[:-1]) > 0 and np.std(residuals[1:]) > 0:
        lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    return {"residual_rmse": float(np.sqrt(np.mean(residuals**2))), "residual_acf1": lag1}


def fit_model(name: str, X: pd.DataFrame, y: pd.Series) -> FittedModel:
    """Fit only the supplied training window; reject rank and optimizer failures."""
    if not X.index.equals(y.index):
        raise ResearchError("X and y training indices must match exactly")
    values = y.to_numpy(dtype=float)
    if len(y) < 8 or not np.isfinite(values).all() or not np.isfinite(X.to_numpy()).all():
        raise ResearchError("Training requires >= 8 finite, aligned observations")
    if name == "mean":
        mean = float(values.mean())
        raw_variance = float(values.var(ddof=1))
        return FittedModel(
            name,
            mean,
            max(raw_variance, VARIANCE_FLOOR),
            {
                **_diagnostics(values - mean),
                "variance_floored": raw_variance < VARIANCE_FLOOR,
            },
        )
    if name == "ols":
        if X.shape[1] == 0 or len(y) <= X.shape[1] + 1:
            raise ModelFitError("OLS needs features and positive residual degrees of freedom")
        data = X.to_numpy(dtype=float)
        center, scale = data.mean(axis=0), data.std(axis=0)
        if (scale == 0).any():
            raise ModelFitError("OLS has a constant feature in this training window")
        design = np.column_stack([np.ones(len(y)), (data - center) / scale])
        coef, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
        if rank != design.shape[1]:
            raise ModelFitError("OLS design is rank deficient")
        residuals = values - design @ coef
        raw_variance = float(residuals @ residuals / (len(y) - rank))
        return FittedModel(
            name,
            float(coef[0]),
            max(raw_variance, VARIANCE_FLOOR),
            {
                **_diagnostics(residuals),
                "condition_number": float(singular[0] / singular[-1]),
                "variance_floored": raw_variance < VARIANCE_FLOOR,
                "standardized_coefficients": coef.tolist(),
                "feature_center": center.tolist(),
                "feature_scale": scale.tolist(),
            },
            tuple(X.columns),
            coef,
            center,
            scale,
        )
    if name not in {"garch", "gjr_garch", "egarch"}:
        raise ResearchError(f"Unknown model: {name}")
    if len(y) < 60 or values.std() < 1e-8:
        raise ModelFitError("ARCH-family models require >= 60 observations and nonconstant returns")
    # Decimal returns -> percentage points for numerical conditioning. Explicitly undo scaling.
    model = arch_model(
        values * 100,
        mean="Constant",
        vol="EGARCH" if name == "egarch" else "GARCH",
        p=1,
        o=0 if name == "garch" else 1,
        q=1,
        dist="normal",
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    if result.convergence_flag != 0:
        raise ModelFitError(f"{name} optimizer failed: {result.optimization_result.message}")
    prediction = result.forecast(horizon=1, reindex=False, method="analytic")
    mean = float(prediction.mean.iloc[-1, 0] / 100)
    variance = float(prediction.variance.iloc[-1, 0] / 10000)
    Forecast(mean, variance)
    diagnostics = {
        **_diagnostics(np.asarray(result.resid) / 100),
        "convergence_flag": int(result.convergence_flag),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "parameters_percent_units": {str(k): float(v) for k, v in result.params.items()},
        "variance_floored": False,
    }
    return FittedModel(name, mean, variance, diagnostics)


def forecast(model: FittedModel, features: pd.Series | None = None) -> Forecast:
    """Public functional forecasting API."""
    return model.forecast(features)
