"""FX helper and utility tests: forwards, delta-convention conversions and
the historical-vol estimator."""

import math

import numpy as np
import pytest

from dpe import (
    forward_delta_to_spot,
    fx_forward,
    gk_greeks,
    historical_vol,
    spot_delta_to_forward,
)


def test_fx_forward_covered_interest_parity() -> None:
    assert fx_forward(1.10, 0.5, 0.03, 0.02) == pytest.approx(1.10 * math.exp(0.005), abs=1e-12)
    # Negative domestic rate pushes the forward below spot when rf > rd.
    assert fx_forward(0.95, 1.0, -0.005, 0.01) < 0.95


def test_delta_conversion_roundtrip_and_value() -> None:
    s, k, t, sigma, rd, rf = 1.10, 1.10, 0.5, 0.10, 0.03, 0.02
    g = gk_greeks(s, k, t, sigma, rd, rf, "call")
    fwd_delta = spot_delta_to_forward(g.delta, t, rf)
    # Forward delta of a call is exactly N(d1): recover it independently.
    from dpe import bs_d1_d2, norm_cdf

    d1, _ = bs_d1_d2(s, k, t, sigma, rd, rf)
    assert fwd_delta == pytest.approx(norm_cdf(d1), abs=1e-12)
    # Round-trip is exact.
    assert forward_delta_to_spot(fwd_delta, t, rf) == pytest.approx(g.delta, abs=1e-15)


def test_delta_conversion_put_sign_preserved() -> None:
    delta_spot = -0.45
    fwd = spot_delta_to_forward(delta_spot, 1.0, 0.02)
    assert fwd < delta_spot < 0.0  # magnitude grows by e^{rf t}
    assert forward_delta_to_spot(fwd, 1.0, 0.02) == pytest.approx(delta_spot, abs=1e-15)


def test_historical_vol_recovers_generating_sigma() -> None:
    """Estimate on a long synthetic GBM series must recover sigma closely."""
    rng = np.random.default_rng(123)
    sigma, dt = 0.25, 1.0 / 252.0
    log_ret = (0.05 - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(200_000)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    est = historical_vol(prices, 252)
    assert est == pytest.approx(sigma, rel=1e-2)


def test_historical_vol_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        historical_vol([100.0, 101.0])  # too short
    with pytest.raises(ValueError):
        historical_vol([100.0, -5.0, 101.0])  # non-positive price
    with pytest.raises(ValueError):
        historical_vol([100.0, math.nan, 101.0])
    with pytest.raises(ValueError, match="periods_per_year"):
        historical_vol([100.0, 101.0, 99.0, 102.0], periods_per_year=0)
    with pytest.raises(ValueError, match="periods_per_year"):
        historical_vol([100.0, 101.0, 99.0, 102.0], periods_per_year=math.nan)  # silent NaN before
    with pytest.raises(ValueError, match="periods_per_year"):
        historical_vol([100.0, 101.0, 99.0, 102.0], periods_per_year=252.0)


def test_historical_vol_accepts_numpy_integer_periods() -> None:
    prices = [100.0, 101.0, 99.5, 102.0, 101.0]
    assert historical_vol(prices, np.int64(252)) == historical_vol(prices, 252)
    # Sanity anchor derived by hand: sample std of the four log returns x sqrt(252).
    lr = np.diff(np.log(prices))
    assert historical_vol(prices, 252) == pytest.approx(float(np.std(lr, ddof=1)) * math.sqrt(252), abs=1e-15)


def test_fx_helpers_reject_non_finite() -> None:
    with pytest.raises(ValueError, match="rd"):
        fx_forward(1.1, 0.5, math.nan, 0.02)
    with pytest.raises(ValueError, match="t"):
        spot_delta_to_forward(0.5, -0.1, 0.02)
    with pytest.raises(ValueError, match="delta_forward"):
        forward_delta_to_spot(math.inf, 0.5, 0.02)
