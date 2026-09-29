"""Composable research pipeline and atomic artifact publication."""

import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from quant_research.backtest import BacktestResult, backtest
from quant_research.config import ResearchConfig
from quant_research.data import load_data
from quant_research.errors import ResearchError
from quant_research.evaluation import evaluate
from quant_research.features import build_features
from quant_research.report import write_report
from quant_research.risk import strategy_returns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    backtest: BacktestResult
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    output: Path


def _git_state() -> dict:
    # Anchor to this module, not an unrelated caller's working directory.
    location = Path(__file__).resolve().parent
    try:
        revision = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=location,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=location,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def run_research(config: ResearchConfig, output: str | Path) -> RunResult:
    """Run a complete experiment; publish a new results directory only after successful writes."""
    started = perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise ResearchError(f"Output already exists; choose a new run directory: {output}")
    frame = load_data(config.data, [spec.column for spec in config.features])
    dataset = build_features(frame, config.features, config.data.target)
    result = backtest(dataset, config.models, config.backtest)
    metrics, comparisons = evaluate(result.forecasts, config.backtest.dm_lags)
    strategy = (
        strategy_returns(result.forecasts, config.backtest.cost_bps)
        if config.backtest.strategy
        else None
    )
    dependencies = {}
    for name in (
        "wti-quant-research",
        "numpy",
        "pandas",
        "scipy",
        "arch",
        "openpyxl",
        "matplotlib",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "uninstalled"
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".research-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        frame.to_csv(staging / "input.csv", index_label="date", float_format="%.17g")
        result.forecasts.to_csv(staging / "forecasts.csv", index=False, float_format="%.17g")
        metrics.to_csv(staging / "metrics.csv", index=False)
        comparisons.to_csv(staging / "comparisons.csv", index=False)
        if strategy is not None:
            strategy.to_csv(staging / "strategy.csv", index=False)
        (staging / "diagnostics.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in result.diagnostics),
            encoding="utf-8",
        )
        (staging / "config.json").write_text(
            json.dumps(config.as_dict(), indent=2), encoding="utf-8"
        )
        write_report(staging, result.forecasts, metrics, comparisons)
        metadata = {
            "created_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
            "git": _git_state(),
            "source_sha256": source_hash.hexdigest(),
            "input_sha256": hashlib.sha256((staging / "input.csv").read_bytes()).hexdigest(),
            "source_file_sha256": hashlib.sha256(Path(config.data.path).read_bytes()).hexdigest()
            if config.data.path
            else None,
            "warmup_rows": dataset.warmup_rows,
            "input_rows": len(frame),
            "forecast_months": result.forecasts.target_date.nunique(),
            "elapsed_seconds": perf_counter() - started,
            "status": "complete",
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if output.exists():
            raise ResearchError(f"Output appeared during run: {output}")
        staging.rename(output)
    logger.info(
        "run_completed",
        extra={"output": str(output), "observations": metadata["forecast_months"]},
    )
    return RunResult(result, metrics, comparisons, output)
