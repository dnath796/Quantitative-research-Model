"""Strict TOML configuration; relative paths are relative to the config file."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from quant_research.errors import ResearchError

MODEL_NAMES = {"mean", "ols", "garch", "gjr_garch", "egarch"}


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    lag: int = 1
    transform: str = "identity"

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ResearchError("Feature column must be a nonempty string")
        if type(self.lag) is not int or self.lag < 1:
            raise ResearchError("Feature lag must be an integer >= 1 to avoid future leakage")
        if self.transform not in {"identity", "difference", "log_return"}:
            raise ResearchError(f"Unknown transform: {self.transform}")

    @property
    def name(self) -> str:
        return f"{self.column}__{self.transform}__lag{self.lag}"


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    path: str | None = None
    sheet: str = "Combined Monthly"
    header_row: int = 2
    date_column: str = "date"
    target: str = "wti_chg_pct"
    start: str | None = None
    end: str | None = None
    seed: int = 42
    periods: int = 180

    def __post_init__(self) -> None:
        if self.source not in {"synthetic", "csv", "excel"}:
            raise ResearchError("Data source must be synthetic, csv, or excel")
        for name in ("sheet", "date_column", "target"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ResearchError(f"{name} must be a nonempty string")
        for name in ("path", "start", "end"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ResearchError(f"{name} must be a nonempty string when provided")
        if self.source != "synthetic" and not self.path:
            raise ResearchError("A data path is required")
        for name, minimum in [("periods", 12), ("seed", 0), ("header_row", 0)]:
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ResearchError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class BacktestConfig:
    window: str = "expanding"
    initial_train: int = 120
    window_size: int = 120
    dm_lags: int = 3
    strategy: bool = False
    cost_bps: float = 5.0

    def __post_init__(self) -> None:
        import math

        if self.window not in {"rolling", "expanding"}:
            raise ResearchError("Window must be rolling or expanding")
        for name in ("initial_train", "window_size", "dm_lags"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "dm_lags" else 8):
                raise ResearchError(f"Invalid {name}")
        if self.window == "rolling" and self.window_size > self.initial_train:
            raise ResearchError("window_size cannot exceed initial_train")
        if type(self.strategy) is not bool:
            raise ResearchError("strategy must be a boolean")
        if type(self.cost_bps) not in (int, float) or not math.isfinite(self.cost_bps):
            raise ResearchError("cost_bps must be finite")
        if self.cost_bps < 0:
            raise ResearchError("cost_bps must be nonnegative")


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: tuple[FeatureSpec, ...] = (
        FeatureSpec("wti_chg_pct"),
        FeatureSpec("dxy_chg_pct"),
        FeatureSpec("sp500_chg_pct"),
        FeatureSpec("vix_close", transform="difference"),
    )
    models: tuple[str, ...] = ("mean", "ols", "garch")
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.models, tuple) or not all(isinstance(m, str) for m in self.models):
            raise ResearchError("models must be a tuple of model names")
        if not isinstance(self.features, tuple) or not all(
            isinstance(f, FeatureSpec) for f in self.features
        ):
            raise ResearchError("features must be a tuple of FeatureSpec objects")
        if not self.models or set(self.models) - MODEL_NAMES:
            raise ResearchError(f"models must contain supported names: {sorted(MODEL_NAMES)}")
        if len(set(self.models)) != len(self.models):
            raise ResearchError("Duplicate models are not allowed")
        if "mean" not in self.models:
            raise ResearchError("Include the mean model as the comparison benchmark")
        if "ols" in self.models and not self.features:
            raise ResearchError("OLS requires at least one feature")
        if len({f.name for f in self.features}) != len(self.features):
            raise ResearchError("Duplicate feature specifications")

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ResearchConfig:
    """Read a TOML specification, rejecting typos rather than ignoring them."""
    path = Path(path).resolve()
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        unknown = set(raw) - {"data", "features", "models", "backtest"}
        if unknown:
            raise ResearchError(f"Unknown config sections: {sorted(unknown)}")
        data = dict(raw.get("data", {}))
        if "models" in raw and not isinstance(raw["models"], list):
            raise ResearchError("models must be an array of names")
        if data.get("path"):
            data["path"] = str((path.parent / data["path"]).resolve())
        defaults = ResearchConfig()
        return ResearchConfig(
            data=DataConfig(**data),
            features=tuple(FeatureSpec(**f) for f in raw["features"])
            if "features" in raw
            else defaults.features,
            models=tuple(raw.get("models", defaults.models)),
            backtest=BacktestConfig(**raw.get("backtest", {})),
        )
    except (TypeError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ResearchError(f"Invalid configuration {path}: {exc}") from exc
