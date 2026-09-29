"""Analytic Black-Scholes/Garman-Kohlhagen tests: values, parity, Greeks vs
finite differences, monotonicity, edge cases and input validation."""

import math

import pytest

from dpe import (
    Greeks,
    bs_greeks,
    bs_price,
    gk_greeks,
    gk_price,
)


def test_known_atm_call_value() -> None:
    # Textbook reference value (Hull): s=100, k=100, T=1, sigma=20%, r=5%.
    assert bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, "call") == pytest.approx(10.450584, abs=1e-6)


def test_put_call_parity_grid() -> None:
    """Property check: C - P = s e^{-qt} - k e^{-rt} over a full grid."""
    r, q, sigma = 0.03, 0.015, 0.25
    for s in (50.0, 100.0, 150.0):
        for k in (60.0, 100.0, 140.0):
            for t in (1 / 52, 0.5, 3.0):
                c = bs_price(s, k, t, sigma, r, q, "call")
                p = bs_price(s, k, t, sigma, r, q, "put")
                fwd = s * math.exp(-q * t) - k * math.exp(-r * t)
                assert c - p == pytest.approx(fwd, abs=1e-10), (s, k, t)


def _fd_greeks(s: float, k: float, t: float, sigma: float, r: float, q: float, ot: str) -> Greeks:
    """Central finite differences of the price for every Greek."""
    hs = s * 1e-5
    hv = 1e-5
    ht = 1e-6
    hr = 1e-6

    def price(s_=s, sig_=sigma, t_=t, r_=r) -> float:
        return bs_price(s_, k, t_, sig_, r_, q, ot)

    delta = (price(s_=s + hs) - price(s_=s - hs)) / (2 * hs)
    gamma = (price(s_=s + hs) - 2 * price() + price(s_=s - hs)) / (hs * hs)
    vega = (price(sig_=sigma + hv) - price(sig_=sigma - hv)) / (2 * hv)
    theta = -(price(t_=t + ht) - price(t_=t - ht)) / (2 * ht)  # calendar time
    rho = (price(r_=r + hr) - price(r_=r - hr)) / (2 * hr)
    vanna = (
        price(s_=s + hs, sig_=sigma + hv)
        - price(s_=s + hs, sig_=sigma - hv)
        - price(s_=s - hs, sig_=sigma + hv)
        + price(s_=s - hs, sig_=sigma - hv)
    ) / (4 * hs * hv)
    volga = (price(sig_=sigma + hv) - 2 * price() + price(sig_=sigma - hv)) / (hv * hv)
    return Greeks(price(), delta, gamma, vega, theta, rho, vanna, volga)


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize(
    "s,k,t,sigma,r,q",
    [
        (100.0, 100.0, 1.0, 0.2, 0.05, 0.02),  # equity ATM with dividends
        (100.0, 140.0, 0.5, 0.35, 0.03, 0.0),  # OTM call / ITM put
        (1.10, 1.05, 2.0, 0.10, -0.005, 0.01),  # FX-style, negative domestic rate
        (25.0, 20.0, 1.5, 0.80, 0.04, 0.0),  # high-vol small cap
    ],
)
def test_greeks_match_finite_differences(
    ot: str, s: float, k: float, t: float, sigma: float, r: float, q: float
) -> None:
    g = bs_greeks(s, k, t, sigma, r, q, ot)
    fd = _fd_greeks(s, k, t, sigma, r, q, ot)
    assert g.delta == pytest.approx(fd.delta, rel=1e-4)
    assert g.vega == pytest.approx(fd.vega, rel=1e-4)
    assert g.gamma == pytest.approx(fd.gamma, rel=1e-3)
    assert g.theta == pytest.approx(fd.theta, rel=1e-3, abs=1e-6)
    assert g.rho == pytest.approx(fd.rho, rel=1e-3)
    assert g.vanna == pytest.approx(fd.vanna, rel=1e-3, abs=1e-6)
    assert g.volga == pytest.approx(fd.volga, rel=1e-3, abs=1e-4)


def test_call_price_monotone_in_spot_and_strike() -> None:
    """Property check: calls increase in s and decrease in k on a grid."""
    t, sigma, r, q = 0.75, 0.3, 0.02, 0.01
    spots = [60.0 + 5.0 * i for i in range(17)]
    prices = [bs_price(s, 100.0, t, sigma, r, q, "call") for s in spots]
    assert all(b > a for a, b in zip(prices, prices[1:]))
    strikes = [60.0 + 5.0 * i for i in range(17)]
    kprices = [bs_price(100.0, k, t, sigma, r, q, "call") for k in strikes]
    assert all(b < a for a, b in zip(kprices, kprices[1:]))


def test_t_zero_is_exact_intrinsic() -> None:
    assert bs_price(110, 100, 0.0, 0.2, 0.05, 0.0, "call") == 10.0
    assert bs_price(90, 100, 0.0, 0.2, 0.05, 0.0, "put") == 10.0
    assert bs_price(90, 100, 0.0, 0.2, 0.05, 0.0, "call") == 0.0
    g = bs_greeks(110, 100, 0.0, 0.2, 0.05, 0.0, "call")
    assert g.delta == 1.0 and g.gamma == 0.0 and g.vega == 0.0


def test_sigma_zero_is_discounted_forward_intrinsic() -> None:
    s, k, t, r, q = 100.0, 90.0, 1.0, 0.05, 0.02
    expected = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert bs_price(s, k, t, 0.0, r, q, "call") == pytest.approx(expected, abs=1e-12)
    assert bs_price(s, k, t, 0.0, r, q, "put") == 0.0
    # Continuity: tiny sigma approaches the sigma=0 branch.
    assert bs_price(s, k, t, 1e-9, r, q, "call") == pytest.approx(expected, abs=1e-8)


def test_zero_strike_call_is_discounted_forward() -> None:
    assert bs_price(100, 0.0, 1.0, 0.2, 0.05, 0.02, "call") == pytest.approx(
        100 * math.exp(-0.02), abs=1e-12
    )
    assert bs_price(100, 0.0, 1.0, 0.2, 0.05, 0.02, "put") == 0.0


def test_deep_itm_otm_limits() -> None:
    # Deep ITM call ~ discounted forward minus strike; deep OTM ~ 0.
    itm = bs_price(100, 1.0, 1.0, 0.2, 0.05, 0.0, "call")
    assert itm == pytest.approx(100 - 1.0 * math.exp(-0.05), abs=1e-8)
    otm = bs_price(100, 10_000.0, 1.0, 0.2, 0.05, 0.0, "call")
    assert 0.0 <= otm < 1e-12
    assert bs_greeks(100, 1.0, 1.0, 0.2, 0.05, 0.0, "call").delta == pytest.approx(1.0, abs=1e-9)


def test_negative_rates_supported() -> None:
    p = bs_price(1.0, 1.0, 1.0, 0.1, -0.005, -0.01, "call")
    assert p > 0.0
    c = bs_price(1.0, 1.0, 1.0, 0.1, -0.005, -0.01, "call")
    pp = bs_price(1.0, 1.0, 1.0, 0.1, -0.005, -0.01, "put")
    fwd = math.exp(0.01) - math.exp(0.005)
    assert c - pp == pytest.approx(fwd, abs=1e-12)


def test_gk_equals_bsm_with_q_rf() -> None:
    assert gk_price(1.10, 1.08, 0.5, 0.1, 0.03, 0.02, "put") == pytest.approx(
        bs_price(1.10, 1.08, 0.5, 0.1, 0.03, 0.02, "put"), abs=0.0
    )
    assert gk_greeks(1.10, 1.08, 0.5, 0.1, 0.03, 0.02, "call") == bs_greeks(
        1.10, 1.08, 0.5, 0.1, 0.03, 0.02, "call"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(s=-100.0),
        dict(s=0.0),
        dict(k=-1.0),
        dict(t=-0.1),
        dict(sigma=-0.2),
        dict(r=math.nan),
        dict(q=math.inf),
    ],
)
def test_invalid_inputs_raise(kwargs: dict) -> None:
    base = dict(s=100.0, k=100.0, t=1.0, sigma=0.2, r=0.05, q=0.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        bs_price(base["s"], base["k"], base["t"], base["sigma"], base["r"], base["q"], "call")


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, "straddle")


# ---------------------------------------------------------------------------
# Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
# ---------------------------------------------------------------------------


def test_greeks_at_expiry_tie_break_contract() -> None:
    """API_SPEC §3.2: at the exactly-at-the-money boundary of the degenerate
    region the call takes the ITM branch (delta e^{-qt}, theta q s e^{-qt} -
    r k e^{-rt}, rho k t e^{-rt}) and the put the OTM branch (all zero).
    Checked at t = 0 and at sigma = 0 with the forward exactly at the strike
    (r = q, s = k makes s e^{-qt} - k e^{-rt} exactly zero in floating point)."""
    s, k, r, q = 100.0, 100.0, 0.05, 0.02
    c = bs_greeks(s, k, 0.0, 0.2, r, q, "call")
    assert c == Greeks(0.0, 1.0, 0.0, 0.0, q * s - r * k, 0.0, 0.0, 0.0)
    p = bs_greeks(s, k, 0.0, 0.2, r, q, "put")
    assert p == Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    t, rate = 1.0, 0.03
    df = math.exp(-rate * t)
    c0 = bs_greeks(s, k, t, 0.0, rate, rate, "call")
    assert c0.price == 0.0 and c0.delta == pytest.approx(df, abs=1e-15)
    assert c0.theta == pytest.approx(rate * s * df - rate * k * df, abs=1e-15)  # = 0
    assert c0.rho == pytest.approx(k * t * df, abs=1e-12)
    assert (c0.gamma, c0.vega, c0.vanna, c0.volga) == (0.0, 0.0, 0.0, 0.0)
    p0 = bs_greeks(s, k, t, 0.0, rate, rate, "put")
    assert p0 == Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert math.copysign(1.0, p0.price) > 0  # +0.0, never -0.0


def test_expiry_and_sigma_zero_prices_are_floats() -> None:
    """Integer inputs at expiry must not leak an int (max(s - k, 0.0) returns
    its first argument on ties)."""
    for v in (bs_price(100, 100, 0.0, 0.2, 0.05, 0.0, "call"),
              bs_price(100, 100, 0.0, 0.2, 0.05, 0.0, "put"),
              bs_price(100, 100, 1.0, 0.0, 0.03, 0.03, "put")):
        assert isinstance(v, float) and v == 0.0 and math.copysign(1.0, v) > 0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rates_rejected_everywhere(bad: float) -> None:
    with pytest.raises(ValueError, match="r"):
        bs_price(100.0, 100.0, 1.0, 0.2, bad, 0.0, "call")
    with pytest.raises(ValueError, match="q"):
        bs_greeks(100.0, 100.0, 1.0, 0.2, 0.05, bad, "put")
    # gk_* is *exactly* bs_* with r = rd (API_SPEC §3.3), so the message names r.
    with pytest.raises(ValueError, match="r must be finite"):
        gk_price(1.1, 1.1, 1.0, 0.1, bad, 0.0, "call")
