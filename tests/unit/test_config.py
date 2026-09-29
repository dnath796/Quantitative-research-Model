import pytest

from quant_research.config import (
    BacktestConfig,
    DataConfig,
    FeatureSpec,
    ResearchConfig,
    load_config,
)
from quant_research.errors import ResearchError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": "random"},
        {"initial_train": 0},
        {"dm_lags": -1},
        {"cost_bps": float("nan")},
        {"cost_bps": -1},
        {"cost_bps": True},
        {"strategy": "yes"},
        {"window": "rolling", "initial_train": 60, "window_size": 120},
    ],
)
def test_invalid_backtest_settings(kwargs):
    with pytest.raises(ResearchError):
        BacktestConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "online"},
        {"source": "csv"},
        {"target": 3},
        {"start": 2020},
        {"periods": 1},
        {"header_row": -1},
        {"seed": True},
    ],
)
def test_invalid_data_settings(kwargs):
    with pytest.raises(ResearchError):
        DataConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"models": ()},
        {"models": ("ols",)},
        {"models": ("mean", "mean")},
        {"models": ("mean", "missing")},
        {"features": (), "models": ("mean", "ols")},
        {"features": (FeatureSpec("x"), FeatureSpec("x"))},
    ],
)
def test_invalid_research_specification(kwargs):
    with pytest.raises(ResearchError):
        ResearchConfig(**kwargs)


def test_models_string_is_not_treated_as_character_list(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('models="mean"')
    with pytest.raises(ResearchError, match="array"):
        load_config(path)
