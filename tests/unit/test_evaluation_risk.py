import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quant_research.errors import ResearchError
from quant_research.evaluation import diebold_mariano, evaluate, qlike
from quant_research.risk import strategy_returns


def test_qlike_reference_and_zero_realization():
    actual = np.array([0, 0.04, 0.02])
    moment = np.array([0.01, 0.02, 0.03])
    np.testing.assert_allclose(qlike(actual, moment), np.log(moment) + actual / moment)
    with pytest.raises(ResearchError):
        qlike(actual, np.zeros(3))


def test_dm_against_hand_calculated_hac():
    a, b = np.array([1, 4, 2, 8, 3, 7, 9, 2.0]), np.ones(8)
    d = a - b
    centered = d - d.mean()
    lrv = centered @ centered / 8 + (centered[1:] @ centered[:-1]) / 8
    expected = d.mean() / np.sqrt(lrv / 8)
    result = diebold_mariano(a, b, lags=1)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(2 * norm.sf(expected))
    assert diebold_mariano(b, a, lags=1).statistic == pytest.approx(-expected)


def test_dm_degenerate_cases():
    assert diebold_mariano(np.ones(10), np.ones(10)).p_value == 1
    assert diebold_mariano(np.ones(10), np.zeros(10)).status == "degenerate_loss_variance"
    assert diebold_mariano(np.ones(2), np.ones(2)).status == "insufficient_observations"
    with pytest.raises(ResearchError):
        diebold_mariano(np.array([np.nan]), np.ones(1))


def test_costs_charge_entry_reversal_and_final_exit():
    frame = pd.DataFrame(
        {
            "model": ["ols"] * 3,
            "target_date": pd.date_range("2020-01", periods=3, freq="MS"),
            "mean": [0.01, -0.02, -0.01],
            "actual": [0.02, -0.01, 0.03],
        }
    )
    result = strategy_returns(frame, cost_bps=10)
    np.testing.assert_allclose(result.turnover, [1, 2, 1])
    np.testing.assert_allclose(result.net_return, [0.019, 0.008, -0.031])
    assert result.wealth.iloc[-1] == pytest.approx(1.019 * 1.008 * 0.969)
    assert strategy_returns(frame, 0).wealth.iloc[-1] > result.wealth.iloc[-1]


def test_evaluation_rejects_misaligned_comparisons():
    frame = pd.DataFrame(
        {
            "model": ["mean", "ols"],
            "target_date": ["2020-01", "2020-02"],
            "actual": [0.1, 0.1],
            "mean": [0, 0],
            "variance": [0.01, 0.01],
            "second_moment": [0.01, 0.01],
        }
    )
    with pytest.raises(ResearchError, match="same target dates"):
        evaluate(frame)
