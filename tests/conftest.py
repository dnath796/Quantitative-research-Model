import pytest

from quant_research.config import FeatureSpec
from quant_research.data import synthetic_data
from quant_research.features import build_features


@pytest.fixture
def monthly_data():
    return synthetic_data(periods=100, seed=123)


@pytest.fixture
def specs():
    return (FeatureSpec("wti_chg_pct"), FeatureSpec("dxy_chg_pct", lag=2))


@pytest.fixture
def dataset(monthly_data, specs):
    return build_features(monthly_data, specs)
