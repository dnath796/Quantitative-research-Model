import numpy as np
import pytest

from derivatives_engine.options_chain import CHAIN_COLUMNS, filter_options_chain, generate_options_chain, option_moneyness


def test_generated_chain_has_quotes_greeks_and_liquidity():
    chain = generate_options_chain(100, 0.04, 0.22, 0.01, expiration_days=(30, 90), strike_count=9, seed=7)
    assert list(chain.columns) == CHAIN_COLUMNS
    assert set(chain["option_type"]) == {"call", "put"}
    assert set(chain["expiration_days"]) == {30, 90}
    assert (chain["ask"] >= chain["bid"]).all()
    assert (chain["spread"] >= 0).all()
    assert (chain["volume"] >= 0).all()
    assert (chain["open_interest"] >= chain["volume"]).all()
    assert np.isfinite(chain[["delta", "gamma", "theta", "vega", "rho"]]).all().all()


def test_chain_is_reproducible_and_filters_work():
    first = generate_options_chain(100, 0.03, 0.2, expiration_days=(30,), seed=42)
    second = generate_options_chain(100, 0.03, 0.2, expiration_days=(30,), seed=42)
    assert first.equals(second)
    calls = filter_options_chain(first, 30, "call", "OTM")
    assert (calls["option_type"] == "call").all()
    assert (calls["moneyness"] == "OTM").all()
    assert (calls["strike"] > 100).all()


def test_moneyness_labels_are_option_aware():
    assert option_moneyness(100, 90, "call") == "ITM"
    assert option_moneyness(100, 90, "put") == "OTM"
    assert option_moneyness(100, 101, "call") == "ATM"
    with pytest.raises(ValueError):
        filter_options_chain(generate_options_chain(100, 0.03, 0.2), moneyness="invalid")
