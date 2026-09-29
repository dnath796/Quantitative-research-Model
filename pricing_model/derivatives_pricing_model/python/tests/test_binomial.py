"""Binomial-tree tests: convergence to Black-Scholes, American exercise
properties, tree Greeks, Richardson averaging and degenerate limits."""

import math

import pytest

from dpe import (
    binomial_greeks,
    binomial_price,
    bs_greeks,
    bs_price,
)


S, K, T, SIG, R, Q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02


def test_crr_european_converges_to_bs() -> None:
    bs = bs_price(S, K, T, SIG, R, Q, "call")
    tree = binomial_price(S, K, T, SIG, R, Q, "call", "european", steps=2000, method="crr")
    assert tree == pytest.approx(bs, abs=1e-3)


def test_jr_european_converges_to_bs() -> None:
    bs = bs_price(S, K, T, SIG, R, Q, "put")
    tree = binomial_price(S, K, T, SIG, R, Q, "put", "european", steps=2000, method="jr")
    assert tree == pytest.approx(bs, abs=1e-3)


def test_richardson_averaging_beats_plain_tree() -> None:
    """Odd/even averaging should cut the CRR oscillation error substantially."""
    bs = bs_price(S, K, T, SIG, R, Q, "call")
    plain = abs(binomial_price(S, K, T, SIG, R, Q, "call", "european", 200, "crr") - bs)
    rich = abs(
        binomial_price(S, K, T, SIG, R, Q, "call", "european", 200, "crr", richardson=True) - bs
    )
    assert rich < plain


def test_american_geq_european_grid() -> None:
    """Property check: early exercise can only add value."""
    for ot in ("call", "put"):
        for k in (80.0, 100.0, 120.0):
            eur = binomial_price(S, k, T, SIG, R, Q, ot, "european", 200)
            amer = binomial_price(S, k, T, SIG, R, Q, ot, "american", 200)
            assert amer >= eur - 1e-12, (ot, k)


def test_american_put_premium_positive_and_increasing_in_strike() -> None:
    """With r > 0 the American put carries a strictly positive early-exercise
    premium that grows with moneyness (deeper ITM -> more interest earned by
    exercising and investing the strike now)."""
    premiums = []
    for k in (90.0, 100.0, 110.0, 120.0, 130.0):
        eur = binomial_price(S, k, T, SIG, R, 0.0, "put", "european", 500)
        amer = binomial_price(S, k, T, SIG, R, 0.0, "put", "american", 500)
        premiums.append(amer - eur)
    assert all(p > 0.0 for p in premiums)
    assert all(b > a for a, b in zip(premiums, premiums[1:]))


def test_american_call_no_dividends_equals_european() -> None:
    """Merton: never exercise an American call on a non-dividend stock early."""
    eur = binomial_price(S, K, T, SIG, R, 0.0, "call", "european", 500)
    amer = binomial_price(S, K, T, SIG, R, 0.0, "call", "american", 500)
    assert amer == pytest.approx(eur, abs=1e-10)


def test_american_put_vs_high_step_reference() -> None:
    """A 5000-step CRR value is the reference; 500 steps must be close."""
    ref = binomial_price(S, K, T, SIG, R, Q, "put", "american", 5000, "crr")
    got = binomial_price(S, K, T, SIG, R, Q, "put", "american", 500, "crr")
    assert got == pytest.approx(ref, abs=2e-3)


def test_tree_greeks_close_to_analytic() -> None:
    g_tree = binomial_greeks(S, K, T, SIG, R, Q, "call", "european", 500, "crr")
    g_bs = bs_greeks(S, K, T, SIG, R, Q, "call")
    assert g_tree.delta == pytest.approx(g_bs.delta, abs=2e-3)
    assert g_tree.gamma == pytest.approx(g_bs.gamma, rel=2e-2)
    assert g_tree.theta == pytest.approx(g_bs.theta, rel=2e-2)


def test_jr_tree_greeks_close_to_analytic() -> None:
    g_tree = binomial_greeks(S, K, T, SIG, R, Q, "put", "european", 500, "jr")
    g_bs = bs_greeks(S, K, T, SIG, R, Q, "put")
    assert g_tree.delta == pytest.approx(g_bs.delta, abs=3e-3)
    assert g_tree.theta == pytest.approx(g_bs.theta, rel=5e-2)


def test_t_zero_and_sigma_zero_limits() -> None:
    assert binomial_price(110, 100, 0.0, 0.2, R, Q, "call", "european", 100) == 10.0
    # sigma=0 European: discounted forward intrinsic, matches analytic limit.
    assert binomial_price(S, 90.0, T, 0.0, R, Q, "call", "european", 100) == pytest.approx(
        bs_price(S, 90.0, T, 0.0, R, Q, "call"), abs=1e-12
    )
    # sigma=0 American put with r > 0: immediate exercise of ITM is optimal,
    # value equals today's intrinsic.
    assert binomial_price(80.0, 100.0, T, 0.0, 0.05, 0.0, "put", "american", 100) == pytest.approx(
        20.0, abs=1e-12
    )


def test_negative_rate_tree_matches_bs() -> None:
    got = binomial_price(1.10, 1.10, 1.0, 0.1, -0.005, 0.01, "put", "european", 2000)
    assert got == pytest.approx(bs_price(1.10, 1.10, 1.0, 0.1, -0.005, 0.01, "put"), abs=1e-3)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(steps=0), "steps"),
        (dict(method="trinomial"), "method"),
        (dict(style="bermudan"), "style"),
        (dict(sigma=-0.1), "sigma"),
        (dict(t=-1.0), "t"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict, match: str) -> None:
    base = dict(s=S, k=K, t=T, sigma=SIG, r=R, q=Q, option_type="call", style="european", steps=100, method="crr")
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        binomial_price(
            base["s"], base["k"], base["t"], base["sigma"], base["r"], base["q"],
            base["option_type"], base["style"], base["steps"], base["method"],
        )


def test_tree_greeks_require_two_steps_and_positive_vol() -> None:
    with pytest.raises(ValueError, match="steps"):
        binomial_greeks(S, K, T, SIG, R, Q, "call", steps=1)
    with pytest.raises(ValueError, match="sigma"):
        binomial_greeks(S, K, T, 0.0, R, Q, "call", steps=100)


# ---------------------------------------------------------------------------
# Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ot,style,method",
    [("call", "european", "crr"), ("call", "european", "jr"), ("put", "american", "crr")],
)
def test_tree_greeks_two_steps_are_finite(ot: str, style: str, method: str) -> None:
    """steps = 2 is the smallest tree the contract allows for Greeks (level 2 is
    the terminal level); it must return finite values, not crash."""
    g = binomial_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, ot, style, 2, method)
    assert all(math.isfinite(x) for x in (g.price, g.delta, g.gamma, g.theta))
    assert g.gamma > 0.0
    assert (0.0 < g.delta < 1.0) if ot == "call" else (-1.0 < g.delta < 0.0)
    assert g.price == binomial_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, ot, style, 2, method)


def test_crr_tree_put_call_parity_exact() -> None:
    """CRR matches the first moment exactly (p u + (1-p) d = e^{(r-q)dt}), so
    the European tree call minus put equals s e^{-qt} - k e^{-rt} to
    round-off at every step count; JR does not match the first moment, so
    its parity error is only O(dt)."""
    fwd = S * math.exp(-Q * T) - K * math.exp(-R * T)
    for n in (1, 2, 101, 500):
        c = binomial_price(S, K, T, SIG, R, Q, "call", "european", n, "crr")
        p = binomial_price(S, K, T, SIG, R, Q, "put", "european", n, "crr")
        assert (c - p) == pytest.approx(fwd, abs=1e-10), n
    c = binomial_price(S, K, T, SIG, R, Q, "call", "european", 101, "jr")
    p = binomial_price(S, K, T, SIG, R, Q, "put", "european", 101, "jr")
    assert (c - p) == pytest.approx(fwd, abs=1e-3)
    assert abs((c - p) - fwd) > 1e-6  # JR is genuinely not a martingale lattice


def test_crr_one_step_closed_form() -> None:
    """A single CRR step is a hand-computable two-state model:
    e^{-rt} [p (s u - k)^+ + (1 - p) (s d - k)^+] with u = e^{sigma sqrt t},
    d = 1/u, p = (e^{(r-q)t} - d) / (u - d)."""
    u = math.exp(SIG * math.sqrt(T))
    d = 1.0 / u
    p = (math.exp((R - Q) * T) - d) / (u - d)
    expected = math.exp(-R * T) * (p * max(S * u - K, 0.0) + (1.0 - p) * max(S * d - K, 0.0))
    assert expected == pytest.approx(11.073540703840242, abs=1e-12)  # derived, not copied
    got = binomial_price(S, K, T, SIG, R, Q, "call", "european", 1, "crr")
    assert got == pytest.approx(expected, abs=1e-12)


def test_american_call_with_dividend_strictly_exceeds_european() -> None:
    """With q > r the early-exercise premium of a deep-ITM American call is
    material (the dividend leakage outweighs the interest on the strike)."""
    eur = binomial_price(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, "call", "european", 500)
    amer = binomial_price(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, "call", "american", 500)
    assert amer - eur > 1.0
    assert amer >= 20.0  # never below immediate intrinsic


def test_richardson_is_average_of_n_and_n_plus_one() -> None:
    n = 200
    rich = binomial_price(S, K, T, SIG, R, Q, "call", "european", n, "crr", richardson=True)
    a = binomial_price(S, K, T, SIG, R, Q, "call", "european", n, "crr")
    b = binomial_price(S, K, T, SIG, R, Q, "call", "european", n + 1, "crr")
    assert rich == pytest.approx(0.5 * (a + b), abs=1e-14)


def test_tree_overflow_guard() -> None:
    """sigma sqrt(t steps) = 707 > 700 would overflow the terminal spots to inf;
    the contract says invalid-input error, never inf/NaN."""
    with pytest.raises(ValueError, match="overflow"):
        binomial_price(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, "call", "european", 200, "crr")
    with pytest.raises(ValueError, match="overflow"):
        binomial_greeks(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, "call", "european", 200, "jr")
    # Just inside the guard (4.6 + 200 + 400 = 605 < 700) is fine and finite.
    v = binomial_price(100.0, 100.0, 1.0, 20.0, 0.05, 0.0, "call", "european", 400, "crr")
    assert math.isfinite(v) and 0.0 < v < 100.0


def test_integer_types_accepted_for_steps() -> None:
    np = pytest.importorskip("numpy")
    ref = binomial_price(S, K, T, SIG, R, Q, "call", "european", 100, "crr")
    assert binomial_price(S, K, T, SIG, R, Q, "call", "european", np.int64(100), "crr") == ref
    assert binomial_price(S, K, T, SIG, R, Q, "call", "european", np.int32(100), "crr") == ref
    g_ref = binomial_greeks(S, K, T, SIG, R, Q, "call", "european", 50, "crr")
    assert binomial_greeks(S, K, T, SIG, R, Q, "call", "european", np.int16(50), "crr") == g_ref
    for bad in (100.0, True, "100", None):
        with pytest.raises(ValueError, match="steps"):
            binomial_price(S, K, T, SIG, R, Q, "call", "european", bad, "crr")


def test_expiry_price_is_float_zero_not_int_or_negative_zero() -> None:
    v = binomial_price(100, 100, 0.0, 0.2, 0.05, 0.0, "put", "european", 10)
    assert isinstance(v, float) and v == 0.0 and math.copysign(1.0, v) > 0
