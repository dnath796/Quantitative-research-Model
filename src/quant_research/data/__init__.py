"""Monthly input adapters and reproducible synthetic fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.config import DataConfig
from quant_research.errors import ResearchError


def synthetic_data(periods: int = 180, seed: int = 42) -> pd.DataFrame:
    """Generate decimal returns with lagged drivers and conditional heteroskedasticity."""
    rng = np.random.default_rng(seed)
    dxy = rng.normal(0, 0.02, periods)
    equity = rng.normal(0.005, 0.04, periods)
    vix = 20 + rng.normal(0, 3, periods)
    y = np.zeros(periods)
    variance, innovation = 0.004, 0.0
    for t in range(1, periods):
        variance = 0.0002 + 0.12 * innovation**2 + 0.82 * variance
        innovation = rng.normal() * np.sqrt(variance)
        y[t] = 0.002 + 0.15 * y[t - 1] - 0.5 * dxy[t - 1] + innovation
    return pd.DataFrame(
        {"wti_chg_pct": y, "dxy_chg_pct": dxy, "sp500_chg_pct": equity, "vix_close": vix},
        index=pd.date_range("2000-01-01", periods=periods, freq="MS", name="date"),
    )


def validate_monthly(frame: pd.DataFrame) -> None:
    """Require unique, ordered, contiguous months so positional lags mean calendar lags."""
    if len(frame) == 0 or not isinstance(frame.index, pd.DatetimeIndex):
        raise ResearchError("Data must have a nonempty DatetimeIndex")
    if frame.index.hasnans or frame.index.tz is not None:
        raise ResearchError("Dates must be valid and timezone-naive")
    months = frame.index.to_period("M")
    if months.has_duplicates or not months.is_monotonic_increasing:
        raise ResearchError("Dates must be ordered with exactly one observation per month")
    if len(pd.period_range(months[0], months[-1], freq="M")) != len(months):
        raise ResearchError(
            "Missing calendar months: fill the source explicitly; do not compress time"
        )
    if frame.columns.has_duplicates:
        raise ResearchError("Duplicate data columns")


def load_data(config: DataConfig, columns: list[str] | None = None) -> pd.DataFrame:
    """Read CSV or the existing three-header-row WTI workbook, without filling missing data."""
    if config.source == "synthetic":
        frame = synthetic_data(config.periods, config.seed)
    else:
        path = Path(str(config.path))
        try:
            if config.source == "excel":
                frame = pd.read_excel(path, sheet_name=config.sheet, header=config.header_row)
                # The original workbook's master date column has no third-row field name.
                if (
                    len(frame.columns)
                    and config.date_column not in frame
                    and str(frame.columns[0]).startswith("Unnamed:")
                ):
                    frame = frame.rename(columns={frame.columns[0]: config.date_column})
            else:
                frame = pd.read_csv(path)
        except (ValueError, pd.errors.ParserError) as exc:
            raise ResearchError(f"Cannot read input {path}: {exc}") from exc
        if config.date_column not in frame:
            raise ResearchError(f"Missing date column: {config.date_column}")
        frame = frame.dropna(how="all")  # Trailing empty workbook rows only.
        try:
            dates = pd.to_datetime(frame.pop(config.date_column), errors="raise")
        except (ValueError, TypeError) as exc:
            raise ResearchError(f"Invalid dates: {exc}") from exc
        frame.index = pd.DatetimeIndex(dates, name="date")
    validate_monthly(frame)
    frame.index = frame.index.to_period("M").to_timestamp().rename("date")
    if config.start or config.end:
        try:
            frame = frame.loc[config.start : config.end]
        except (ValueError, KeyError, TypeError) as exc:
            raise ResearchError(f"Invalid sample bounds: {exc}") from exc
    validate_monthly(frame)
    wanted = list(dict.fromkeys([config.target, *(columns or [])]))
    missing = set(wanted) - set(frame.columns)
    if missing:
        raise ResearchError(f"Missing required columns: {sorted(missing)}")
    frame = frame[wanted].copy()
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ResearchError(f"Nonnumeric data: {exc}") from exc
    if np.isinf(frame.to_numpy()).any():
        raise ResearchError("Infinite data values are not allowed")
    return frame
