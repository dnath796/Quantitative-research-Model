from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_research.config import DataConfig, FeatureSpec, load_config
from quant_research.data import load_data, validate_monthly
from quant_research.errors import ResearchError
from quant_research.features import build_features


def test_lagged_transform_uses_only_available_observations(monthly_data):
    specs = (FeatureSpec("vix_close", lag=2, transform="difference"),)
    data = build_features(monthly_data, specs)
    date = monthly_data.index[10]
    assert data.X.loc[date].iloc[0] == pytest.approx(
        monthly_data.vix_close.iloc[8] - monthly_data.vix_close.iloc[7]
    )
    assert data.y.loc[date] == monthly_data.wti_chg_pct.iloc[10]
    assert data.warmup_rows == 3


def test_features_are_invariant_to_future_changes(monthly_data, specs):
    before = build_features(monthly_data, specs)
    modified = monthly_data.copy()
    modified.iloc[60:] *= 100
    after = build_features(modified, specs)
    assert_frame_equal(
        before.X.loc[: monthly_data.index[60]], after.X.loc[: monthly_data.index[60]]
    )


@pytest.mark.parametrize("lag", [0, -1, 1.5, True])
def test_forbid_contemporaneous_or_invalid_lags(lag):
    with pytest.raises(ResearchError, match="lag"):
        FeatureSpec("x", lag=lag)


def test_log_returns_and_nonpositive_level(monthly_data):
    data = build_features(monthly_data, (FeatureSpec("vix_close", transform="log_return"),))
    assert data.X.iloc[0, 0] == pytest.approx(
        np.log(monthly_data.vix_close.iloc[1] / monthly_data.vix_close.iloc[0])
    )
    monthly_data.iloc[5, monthly_data.columns.get_loc("vix_close")] = 0
    with pytest.raises(ResearchError, match="positive"):
        build_features(monthly_data, (FeatureSpec("vix_close", transform="log_return"),))


@pytest.mark.parametrize("corruption", ["gap", "duplicate", "reverse"])
def test_calendar_validation(monthly_data, corruption):
    if corruption == "gap":
        monthly_data = monthly_data.drop(monthly_data.index[10])
    elif corruption == "duplicate":
        monthly_data = pd.concat([monthly_data.iloc[:1], monthly_data])
    else:
        monthly_data = monthly_data.iloc[::-1]
    with pytest.raises(ResearchError):
        validate_monthly(monthly_data)


def test_internal_missing_data_is_not_silently_dropped(monthly_data, specs):
    monthly_data.iloc[20, 0] = np.nan
    with pytest.raises(ResearchError, match="Missing values inside"):
        build_features(monthly_data, specs)


def test_csv_round_trip_and_bad_numeric(tmp_path, monthly_data):
    path = tmp_path / "input.csv"
    monthly_data.to_csv(path)
    cfg = DataConfig(source="csv", path=str(path))
    data = load_data(cfg, list(monthly_data.columns))
    assert_frame_equal(data, monthly_data, check_freq=False)
    path.write_text("date,wti_chg_pct\n2020-01-01,nope\n")
    with pytest.raises(ResearchError, match="Nonnumeric"):
        load_data(cfg)


def test_workbook_adapter_preserves_decimal_units(tmp_path):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "Combined Monthly"
    sheet.append(["Master date", "WTI"])
    sheet.append(["Date", "Change %"])
    sheet.append([None, "wti_chg_pct"])
    for date, value in zip(
        pd.date_range("2020-01-01", periods=3, freq="MS"), [0.01, -0.02, 0.03], strict=True
    ):
        sheet.append([date.to_pydatetime(), value])
    path = tmp_path / "sample.xlsx"
    book.save(path)
    data = load_data(DataConfig(source="excel", path=str(path)))
    np.testing.assert_allclose(data.wti_chg_pct, [0.01, -0.02, 0.03])


def test_config_resolves_paths_and_rejects_typo(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[data]\nsource="csv"\npath="input.csv"\n')
    assert load_config(path).data.path == str(tmp_path / "input.csv")
    path.write_text("[backtest]\ninitial_trian=20\n")
    with pytest.raises(ResearchError, match="Invalid configuration"):
        load_config(path)


def test_empty_sample_and_infinite_values(tmp_path, monthly_data):
    path = tmp_path / "input.csv"
    monthly_data.to_csv(path)
    cfg = DataConfig(source="csv", path=str(path))
    with pytest.raises(ResearchError, match="nonempty"):
        load_data(replace(cfg, start="2100"))
    monthly_data.iloc[10, 0] = np.inf
    monthly_data.to_csv(path)
    with pytest.raises(ResearchError, match="Infinite"):
        load_data(cfg)
