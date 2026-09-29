"""Implied-vol solver tests: round-trips across the surface, deep ITM/OTM,
degenerate prices and no-arbitrage bound violations."""

import math

import pytest

from dpe import bs_price, implied_vol


def test_roundtrip_grid() -> None:
    """Property check: sigma -> price -> implied sigma over a broad grid."""
    s, r, q = 100.0, 0.03, 0.01
    for ot in ("call", "put"):
        for sigma in (0.05, 0.2, 0.8):
            for k in (70.0, 100.0, 130.0):
                for t in (1 / 52, 1.0, 3.0):
                    price = bs_price(s, k, t, sigma, r, q, ot)
                    lower = (
                        max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
                        if ot == "call"
                        else max(k * math.exp(-r * t) - s * math.exp(-q * t), 0.0)
                    )
                    if price - lower < 1e-12:
                        continue  # numerically pinned to intrinsic: no vol identifiable
                    iv = implied_vol(price, s, k, t, r, q, ot)
                    assert iv == pytest.approx(sigma, abs=1e-7), (ot, sigma, k, t)


def test_deep_itm_and_otm_roundtrip() -> None:
    # Deep ITM: price is nearly all intrinsic, vega is tiny -> bisection path.
    p_itm = bs_price(100.0, 40.0, 0.5, 0.3, 0.05, 0.0, "call")
    assert implied_vol(p_itm, 100.0, 40.0, 0.5, 0.05, 0.0, "call") == pytest.approx(0.3, abs=1e-5)
    # Deep OTM: tiny price.
    p_otm = bs_price(100.0, 250.0, 0.5, 0.3, 0.05, 0.0, "call")
    assert implied_vol(p_otm, 100.0, 250.0, 0.5, 0.05, 0.0, "call") == pytest.approx(0.3, abs=1e-6)


def test_negative_rate_roundtrip() -> None:
    price = bs_price(1.10, 1.10, 1.0, 0.1, -0.005, 0.01, "put")
    assert implied_vol(price, 1.10, 1.10, 1.0, -0.005, 0.01, "put") == pytest.approx(0.1, abs=1e-8)


def test_price_below_intrinsic_raises() -> None:
    # Discounted intrinsic of this call is 100 - 80 e^{-0.05} ~ 23.9; ask for less.
    with pytest.raises(ValueError, match="lower bound"):
        implied_vol(5.0, 100.0, 80.0, 1.0, 0.05, 0.0, "call")


def test_price_above_upper_bound_raises() -> None:
    # A call can never be worth more than the (dividend-discounted) spot.
    with pytest.raises(ValueError, match="upper bound"):
        implied_vol(101.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call")


def test_t_zero_raises() -> None:
    with pytest.raises(ValueError):
        implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, 0.0, "call")


def test_negative_price_raises() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        implied_vol(-1.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call")


# ---------------------------------------------------------------------------
# Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
# ---------------------------------------------------------------------------

ATM_PRICE = 10.45058357  # bs_price(100, 100, 1, 0.2, 0.05, 0, "call") to 8 dp


def test_implied_vol_reports_non_convergence() -> None:
    """A budget that cannot meet tol must raise, never return a half-converged
    root: one Newton step from the bracket midpoint lands at ~0.198, 2 vol
    points off, and the old code returned it silently."""
    with pytest.raises(ValueError, match="did not converge"):
        implied_vol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, "call", tol=1e-10, max_iter=1)
    # The solver is deterministic: the same call with a big enough budget is fine.
    assert implied_vol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, "call", max_iter=20) == pytest.approx(0.2, abs=1e-8)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(max_iter=0), "max_iter"),
        (dict(max_iter=-3), "max_iter"),
        (dict(max_iter=5.0), "max_iter"),
        (dict(tol=0.0), "tol"),
        (dict(tol=-1e-10), "tol"),
        (dict(tol=math.nan), "tol"),
    ],
)
def test_implied_vol_validates_tol_and_max_iter(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        implied_vol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, "call", **kwargs)


def test_implied_vol_high_vol_bracket_expansion() -> None:
    """sigma = 1.7 lies beyond the initial [0, 1] bracket: the doubling
    expansion (1 -> 2) must engage and the round-trip still hold."""
    price = bs_price(100.0, 100.0, 1.0, 1.7, 0.02, 0.0, "call")
    assert implied_vol(price, 100.0, 100.0, 1.0, 0.02, 0.0, "call") == pytest.approx(1.7, abs=1e-6)
    # Between 16 and 20 the bracket is clipped to the 20.0 cap and still solves.
    price = bs_price(100.0, 100.0, 0.05, 17.0, 0.02, 0.0, "call")
    assert implied_vol(price, 100.0, 100.0, 0.05, 0.02, 0.0, "call") == pytest.approx(17.0, abs=1e-6)


def test_implied_vol_tiny_tol_terminates_on_bracket_width() -> None:
    """tol far below the price's floating-point resolution cannot be met in
    price; the bracket-width rule (< 1e-12) must terminate the loop with an
    essentially exact vol rather than raising or looping forever."""
    price = bs_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    iv = implied_vol(price, 100.0, 100.0, 1.0, 0.05, 0.0, "call", tol=1e-300, max_iter=200)
    assert iv == pytest.approx(0.2, abs=1e-10)


def test_implied_vol_is_deterministic() -> None:
    price = bs_price(100.0, 90.0, 0.5, 0.3, 0.02, 0.01, "put")
    a = implied_vol(price, 100.0, 90.0, 0.5, 0.02, 0.01, "put")
    b = implied_vol(price, 100.0, 90.0, 0.5, 0.02, 0.01, "put")
    assert a == b
