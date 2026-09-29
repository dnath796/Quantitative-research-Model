"""Profile a complete research run with identical semantics to the public API."""

import argparse
import cProfile
import io
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter

from quant_research import load_config, run_research


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    profiler = cProfile.Profile()
    start = perf_counter()
    result = profiler.runcall(run_research, config, args.output)
    elapsed = perf_counter() - start
    profiler.dump_stats(result.output / "profile.pstats")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    (result.output / "profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {
        "elapsed_seconds": elapsed,
        "model_forecasts": len(result.backtest.forecasts),
        "model_forecasts_per_second": len(result.backtest.forecasts) / elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (result.output / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
