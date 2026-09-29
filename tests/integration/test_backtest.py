from dataclasses import replace

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from quant_research.backtest import backtest
from quant_research.config import BacktestConfig
from quant_research.features import build_features


@pytest.mark.parametrize("window", ["expanding", "rolling"])
def test_training_boundaries_and_mean_forecasts(dataset, window):
    cfg = BacktestConfig(window=window, initial_train=70, window_size=60)
    result = backtest(dataset, ("mean", "ols"), cfg).forecasts
    assert (result.train_end < result.target_date).all()
    assert (result.origin_date == result.train_end).all()
    means = result[result.model == "mean"]
    for row in means.itertuples():
        historical = dataset.y.loc[row.train_start : row.train_end]
        assert row.mean == pytest.approx(historical.mean())
        assert row.n_train == len(historical)
    expected = np.repeat(60, len(means)) if window == "rolling" else np.arange(70, len(dataset.y))
    np.testing.assert_array_equal(means.n_train, expected)


@pytest.mark.parametrize("window", ["expanding", "rolling"])
def test_future_perturbation_cannot_change_forecasts(monthly_data, specs, window):
    cfg = BacktestConfig(window=window, initial_train=70, window_size=60)
    cutoff = monthly_data.index[80]
    data = build_features(monthly_data, specs)
    before = backtest(data, ("mean", "ols", "garch"), cfg).forecasts
    altered = monthly_data.copy()
    altered.loc[cutoff:] += 0.1
    after = backtest(build_features(altered, specs), ("mean", "ols", "garch"), cfg).forecasts
    columns = [c for c in before if c != "actual"]
    assert_frame_equal(
        before.loc[before.target_date <= cutoff, columns],
        after.loc[after.target_date <= cutoff, columns],
    )


def test_rolling_training_discards_old_targets(dataset):
    cfg = BacktestConfig(window="rolling", initial_train=80, window_size=60)
    before = backtest(dataset, ("mean",), cfg).forecasts
    changed_y = dataset.y.copy()
    changed_y.iloc[:20] = 999
    altered = replace(dataset, y=changed_y)
    after = backtest(altered, ("mean",), cfg).forecasts
    assert_frame_equal(before, after)


def test_mean_only_research_needs_no_predictor_columns(monthly_data):
    dataset = build_features(monthly_data, ())
    forecasts = backtest(
        dataset, ("mean",), BacktestConfig(initial_train=80, window_size=60)
    ).forecasts
    assert len(forecasts) == 20
    assert forecasts.iloc[0]["mean"] == pytest.approx(monthly_data.wti_chg_pct.iloc[:80].mean())
