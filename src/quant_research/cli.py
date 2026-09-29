"""Command-line entry point; imports have no data-loading or logging side effects."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from quant_research.config import ResearchConfig, load_config
from quant_research.errors import ResearchError
from quant_research.pipeline import run_research
from quant_research.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a one-month-ahead forecasting experiment")
    parser.add_argument(
        "--config", type=Path, help="TOML config; omit for an offline synthetic demo"
    )
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--window", choices=["rolling", "expanding"], help="Override window mode")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        config = load_config(args.config) if args.config else ResearchConfig()
        if args.window:
            config = replace(config, backtest=replace(config.backtest, window=args.window))
        result = run_research(config, args.output)
    except (ResearchError, OSError) as exc:
        logging.getLogger("quant_research").error("run_failed: %s", exc)
        return 2
    print(result.metrics.to_string(index=False))
    print(f"Report: {result.output / 'report.md'}")
    return 0
