import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from quant_research.config import BacktestConfig, DataConfig, ResearchConfig
from quant_research.errors import ResearchError
from quant_research.pipeline import run_research


def test_pipeline_stores_reproducible_complete_artifacts(tmp_path):
    config = ResearchConfig(
        data=DataConfig(periods=90),
        models=("mean", "ols"),
        backtest=BacktestConfig(initial_train=70, window_size=60, strategy=True),
    )
    first = run_research(config, tmp_path / "first")
    second = run_research(config, tmp_path / "second")
    assert_frame_equal(first.backtest.forecasts, second.backtest.forecasts)
    expected = {
        "input.csv",
        "forecasts.csv",
        "metrics.csv",
        "comparisons.csv",
        "metadata.json",
        "config.json",
        "diagnostics.jsonl",
        "strategy.csv",
        "report.md",
        "forecasts.png",
    }
    assert expected == {p.name for p in first.output.iterdir()}
    metadata = json.loads((first.output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["forecast_months"] == 18
    assert (
        metadata["input_sha256"]
        == json.loads((second.output / "metadata.json").read_text())["input_sha256"]
    )
    with pytest.raises(ResearchError, match="already exists"):
        run_research(config, first.output)
    # Re-running the saved input panel reproduces forecasts within CSV round-trip precision.
    replay = replace(config, data=DataConfig(source="csv", path=str(first.output / "input.csv")))
    third = run_research(replay, tmp_path / "replay")
    assert_frame_equal(first.backtest.forecasts, third.backtest.forecasts, atol=1e-12, rtol=1e-9)


def test_failed_run_does_not_publish_output(tmp_path):
    bad = ResearchConfig(data=DataConfig(periods=20))
    with pytest.raises(ResearchError):
        run_research(bad, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_installed_cli_runs_outside_repository(tmp_path):
    config = tmp_path / "small.toml"
    config.write_text(
        'models=["mean", "ols"]\n[data]\nperiods=80\n[backtest]\ninitial_train=60\nwindow_size=60\n'
    )
    result = subprocess.run(
        [sys.executable, "-m", "quant_research", "--config", str(config), "--output", "run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "report.md").is_file()
    assert '"event": "run_completed"' in result.stderr
    failure = subprocess.run(
        [sys.executable, "-m", "quant_research", "--config", "missing.toml", "--output", "bad"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode == 2
    assert "Traceback" not in failure.stderr


def test_import_does_not_read_data_or_write_outputs(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import quant_research"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(
    not Path("master_datasheet.xlsx").exists(), reason="Workbook not distributed in wheel"
)
def test_existing_workbook_adapter_and_feature_coverage():
    from quant_research import build_features, load_config, load_data

    cfg = load_config("configs/wti.toml")
    panel = load_data(cfg.data, [f.column for f in cfg.features])
    dataset = build_features(panel, cfg.features, cfg.data.target)
    assert len(panel) == 420
    assert len(dataset.y) == 418
    assert str(dataset.y.index[-1].date()) == "2025-12-01"
