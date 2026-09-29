"""Monte Carlo tests: agreement with analytics within standard errors,
variance-reduction effectiveness, exotic-payoff orderings, Greek estimators,
determinism and input validation. Path counts are trimmed to keep the suite
fast; assertions are made in units of the reported standard error."""

import numpy as np
import pytest

from dpe import (
    bs_d1_d2,
    bs_greeks,
    bs_price,
    geometric_asian_price,
    mc_asian_arithmetic,
    mc_barrier_up_out,
    mc_delta_fd_crn,
    mc_delta_pathwise,
    mc_european,
    mc_lookback_floating,
    norm_cdf,
    simulate_gbm_paths,
)

S, K, T, SIG, R, Q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02
N = 20_000  # trimmed path count for test speed


def test_mc_european_within_3se_of_bs() -> None:
    bs = bs_price(S, K, T, SIG, R, Q, "call")
    res = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=42)
    assert abs(res.value - bs) < 3.0 * res.std_error
    assert res.ci_low < res.value < res.ci_high
    assert res.n_paths == N


def test_mc_european_put_within_3se_of_bs() -> None:
    bs = bs_price(S, K, T, SIG, R, Q, "put")
    res = mc_european(S, K, T, SIG, R, Q, "put", n_paths=N, seed=1)
    assert abs(res.value - bs) < 3.0 * res.std_error


def test_antithetic_and_control_variate_reduce_std_error() -> None:
    plain = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=5, antithetic=False)
    anti = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=5, antithetic=True)
    cv = mc_european(
        S, K, T, SIG, R, Q, "call", n_paths=N, seed=5, antithetic=True, control_variate=True
    )
    assert anti.std_error < plain.std_error
    assert cv.std_error < anti.std_error


def test_mc_deterministic_given_seed() -> None:
    a = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=99)
    b = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=99)
    c = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=100)
    assert a == b
    assert a.value != c.value


def test_gbm_paths_shape_and_moments() -> None:
    paths = simulate_gbm_paths(S, T, SIG, R, Q, n_paths=N, n_steps=10, seed=3, antithetic=True)
    assert paths.shape == (N, 11)
    assert np.all(paths[:, 0] == S)
    assert np.all(paths > 0.0)
    # Martingale check: E[S_T e^{-(r-q)T}] = S0; antithetics make this tight.
    disc_mean = float(np.mean(paths[:, -1])) * np.exp(-(R - Q) * T)
    assert disc_mean == pytest.approx(S, rel=5e-3)


def test_asian_leq_vanilla_and_within_ci() -> None:
    """Averaging lowers effective volatility, so Asian <= vanilla (same params)."""
    res = mc_asian_arithmetic(S, K, T, SIG, R, Q, "call", n_paths=N, n_steps=50, seed=11)
    vanilla = bs_price(S, K, T, SIG, R, Q, "call")
    assert res.value < vanilla
    # Geometric mean <= arithmetic mean => geometric Asian is a lower bound.
    geo = geometric_asian_price(S, K, T, SIG, R, Q, "call", 50)
    assert res.value > geo - 3.0 * res.std_error


def test_asian_control_variate_reduces_std_error() -> None:
    no_cv = mc_asian_arithmetic(
        S, K, T, SIG, R, Q, "call", n_paths=N, n_steps=50, seed=11, control_variate=False
    )
    cv = mc_asian_arithmetic(
        S, K, T, SIG, R, Q, "call", n_paths=N, n_steps=50, seed=11, control_variate=True
    )
    assert cv.std_error < 0.2 * no_cv.std_error  # the geometric control is strong


def test_geometric_asian_analytic_vs_mc() -> None:
    """The closed form itself must agree with a direct simulation."""
    n_steps = 12
    paths = simulate_gbm_paths(S, T, SIG, R, Q, n_paths=N, n_steps=n_steps, seed=17, antithetic=True)
    geo = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    payoff = np.exp(-R * T) * np.maximum(geo - K, 0.0)
    half = N // 2
    pair = 0.5 * (payoff[:half] + payoff[half:])
    se = float(np.std(pair, ddof=1) / np.sqrt(half))
    analytic = geometric_asian_price(S, K, T, SIG, R, Q, "call", n_steps)
    assert abs(float(np.mean(pair)) - analytic) < 3.0 * se


def test_barrier_leq_vanilla_and_knocked_out_is_zero() -> None:
    res = mc_barrier_up_out(S, K, T, SIG, R, Q, "call", barrier=130.0, n_paths=N, n_steps=50, seed=2)
    vanilla = bs_price(S, K, T, SIG, R, Q, "call")
    assert 0.0 < res.value < vanilla
    dead = mc_barrier_up_out(S, K, T, SIG, R, Q, "call", barrier=90.0, n_paths=N, n_steps=50, seed=2)
    assert dead.value == 0.0 and dead.std_error == 0.0


def test_barrier_increases_with_barrier_level() -> None:
    """Property check: a higher up-and-out barrier can only add value."""
    values = [
        mc_barrier_up_out(S, K, T, SIG, R, Q, "call", barrier=b, n_paths=N, n_steps=50, seed=8).value
        for b in (110.0, 130.0, 150.0, 200.0)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_lookback_bounds() -> None:
    """Floating-strike lookback call pays S_T - min >= S_T - S0, so it must
    price above the ATM European call on the same parameters."""
    res = mc_lookback_floating(S, T, SIG, R, Q, "call", n_paths=N, n_steps=50, seed=4)
    atm_call = bs_price(S, S, T, SIG, R, Q, "call")
    assert res.value > atm_call
    put = mc_lookback_floating(S, T, SIG, R, Q, "put", n_paths=N, n_steps=50, seed=4)
    assert put.value > 0.0


def test_pathwise_delta_matches_analytic() -> None:
    bs_delta = bs_greeks(S, K, T, SIG, R, Q, "call").delta
    res = mc_delta_pathwise(S, K, T, SIG, R, Q, "call", n_paths=N, seed=21)
    assert abs(res.value - bs_delta) < 3.0 * res.std_error
    put = mc_delta_pathwise(S, K, T, SIG, R, Q, "put", n_paths=N, seed=21)
    bs_put_delta = bs_greeks(S, K, T, SIG, R, Q, "put").delta
    assert abs(put.value - bs_put_delta) < 3.0 * put.std_error


def test_fd_crn_delta_matches_analytic() -> None:
    bs_delta = bs_greeks(S, K, T, SIG, R, Q, "call").delta
    res = mc_delta_fd_crn(S, K, T, SIG, R, Q, "call", n_paths=N, seed=21)
    assert abs(res.value - bs_delta) < 3.0 * res.std_error
    # CRN keeps the FD noise comparable to the pathwise estimator's.
    pw = mc_delta_pathwise(S, K, T, SIG, R, Q, "call", n_paths=N, seed=21)
    assert res.std_error < 2.0 * pw.std_error


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_paths=1),
        dict(n_paths=1001),  # odd with antithetic
        dict(seed=1.5),
        dict(t=0.0),
        dict(sigma=-0.2),
    ],
)
def test_mc_invalid_inputs_raise(kwargs: dict) -> None:
    base = dict(s=S, k=K, t=T, sigma=SIG, r=R, q=Q, n_paths=1000, seed=1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        mc_european(
            base["s"], base["k"], base["t"], base["sigma"], base["r"], base["q"],
            "call", n_paths=base["n_paths"], seed=base["seed"],
        )


def test_barrier_requires_positive_barrier() -> None:
    with pytest.raises(ValueError, match="barrier"):
        mc_barrier_up_out(S, K, T, SIG, R, Q, "call", barrier=-1.0, n_paths=1000, seed=1)


# ---------------------------------------------------------------------------
# Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
# ---------------------------------------------------------------------------


def test_integer_types_accepted() -> None:
    """numpy/pandas integers (what a DataFrame-driven batch passes) must be
    accepted for every integer parameter and give the same numbers as ints."""
    ref = mc_european(S, K, T, SIG, R, Q, "call", n_paths=1000, seed=1)
    got = mc_european(S, K, T, SIG, R, Q, "call", n_paths=np.int64(1000), seed=np.int64(1))
    assert got == ref
    ref_a = mc_asian_arithmetic(S, K, T, SIG, R, Q, "call", n_paths=1000, n_steps=12, seed=1)
    got_a = mc_asian_arithmetic(
        S, K, T, SIG, R, Q, "call", n_paths=np.int32(1000), n_steps=np.int16(12), seed=np.uint8(1)
    )
    assert got_a == ref_a
    assert geometric_asian_price(S, K, T, SIG, R, Q, "call", np.int32(12)) == geometric_asian_price(
        S, K, T, SIG, R, Q, "call", 12
    )
    paths = simulate_gbm_paths(S, T, SIG, R, Q, np.int64(10), np.int64(3), np.int64(0))
    assert paths.shape == (10, 4)
    # Integral-valued floats and bools are still rejected (with the parameter named).
    for bad in (1000.0, True):
        with pytest.raises(ValueError, match="n_paths"):
            mc_european(S, K, T, SIG, R, Q, "call", n_paths=bad, seed=1)
    with pytest.raises(ValueError, match="n_fixings"):
        geometric_asian_price(S, K, T, SIG, R, Q, "call", 12.0)


@pytest.mark.parametrize("seed", [-1, -(2**63), 2**63, 2**64, 1.0, "42", None, True])
def test_seed_domain_is_pinned(seed: object) -> None:
    """Seeds are integers in [0, 2^63 - 1] in every port; anything else is an
    invalid-input error naming `seed` (not a numpy error, not a silent wrap)."""
    with pytest.raises(ValueError, match="seed"):
        mc_european(S, K, T, SIG, R, Q, "call", n_paths=1000, seed=seed)
    with pytest.raises(ValueError, match="seed"):
        simulate_gbm_paths(S, T, SIG, R, Q, 10, 2, seed)


def test_seed_domain_endpoints_accepted() -> None:
    lo = mc_european(S, K, T, SIG, R, Q, "call", n_paths=1000, seed=0)
    hi = mc_european(S, K, T, SIG, R, Q, "call", n_paths=1000, seed=2**63 - 1)
    assert np.isfinite(lo.value) and np.isfinite(hi.value) and lo.value != hi.value


@pytest.mark.parametrize("rel_bump", [1.0, 2.0, 0.0, -1e-4, np.nan, np.inf])
def test_rel_bump_domain(rel_bump: float) -> None:
    """rel_bump >= 1 would price a non-positive spot silently; the contract pins (0, 1)."""
    with pytest.raises(ValueError, match="rel_bump"):
        mc_delta_fd_crn(S, K, T, SIG, R, Q, "call", n_paths=1000, seed=1, rel_bump=rel_bump)


def test_mc_asian_single_fixing_equals_european() -> None:
    """With one fixing the arithmetic Asian IS the European vanilla; the two
    engines consume the same draws in the same order, so the results must be
    bit-identical (catches an S0-counted-as-fixing bug or RNG drift)."""
    asian = mc_asian_arithmetic(
        S, K, T, SIG, R, Q, "call", n_paths=N, n_steps=1, seed=7, antithetic=True, control_variate=False
    )
    euro = mc_european(S, K, T, SIG, R, Q, "call", n_paths=N, seed=7, antithetic=True, control_variate=False)
    assert asian == euro


def test_mc_barrier_single_monitoring_matches_analytic() -> None:
    """One monitoring date (t = T, plus t = 0 where s < B): the up-and-out call
    pays (S_T - k) 1{k < S_T < B}, i.e. C(k) - C(B) - (B - k) e^{-rt} N(d2(B))."""
    barrier = 130.0
    d2_b = bs_d1_d2(S, barrier, T, SIG, R, Q)[1]
    analytic = (
        bs_price(S, K, T, SIG, R, Q, "call")
        - bs_price(S, barrier, T, SIG, R, Q, "call")
        - (barrier - K) * np.exp(-R * T) * norm_cdf(d2_b)
    )
    res = mc_barrier_up_out(S, K, T, SIG, R, Q, "call", barrier, n_paths=200_000, n_steps=1, seed=3)
    assert abs(res.value - analytic) < 3.0 * res.std_error
    assert res.value < bs_price(S, K, T, SIG, R, Q, "call")


def test_mc_lookback_single_step_is_atm_vanilla() -> None:
    """One step: call pays S_T - min(s, S_T) = (S_T - s)^+ and put pays
    max(s, S_T) - S_T = (s - S_T)^+, i.e. the ATM European call / put."""
    call = mc_lookback_floating(S, T, SIG, R, Q, "call", n_paths=200_000, n_steps=1, seed=3)
    assert abs(call.value - bs_price(S, S, T, SIG, R, Q, "call")) < 3.0 * call.std_error
    put = mc_lookback_floating(S, T, SIG, R, Q, "put", n_paths=200_000, n_steps=1, seed=3)
    assert abs(put.value - bs_price(S, S, T, SIG, R, Q, "put")) < 3.0 * put.std_error


def test_mc_sigma_zero_is_deterministic() -> None:
    """sigma = 0: every path is the forward, the payoff is a constant, the
    control variate is degenerate (var = 0 -> beta = 0 branch) and the
    result equals the analytic sigma = 0 limit with (numerically) zero SE."""
    res = mc_european(S, K, T, 0.0, R, Q, "call", n_paths=1000, seed=1, antithetic=True, control_variate=True)
    assert res.value == pytest.approx(bs_price(S, K, T, 0.0, R, Q, "call"), abs=1e-12)
    assert res.std_error <= 1e-12  # pairwise summation of a constant: ~1e-17, not exactly 0
    assert res.ci_high - res.ci_low <= 1e-11
    assert np.isfinite(res.value)
    asian = mc_asian_arithmetic(S, K, T, 0.0, R, Q, "put", n_paths=1000, n_steps=4, seed=1)
    assert asian.value == pytest.approx(geometric_asian_price(S, K, T, 0.0, R, Q, "put", 4), abs=1e-12)
    assert asian.std_error <= 1e-12


def test_mc_results_are_finite_and_ci_ordered() -> None:
    for fn in (
        lambda: mc_european(S, K, T, SIG, R, Q, "put", n_paths=2000, seed=5, control_variate=True),
        lambda: mc_asian_arithmetic(S, K, T, SIG, R, Q, "put", n_paths=2000, n_steps=6, seed=5),
        lambda: mc_barrier_up_out(S, K, T, SIG, R, Q, "put", 140.0, n_paths=2000, n_steps=6, seed=5),
        lambda: mc_lookback_floating(S, T, SIG, R, Q, "put", n_paths=2000, n_steps=6, seed=5),
        lambda: mc_delta_pathwise(S, K, T, SIG, R, Q, "put", n_paths=2000, seed=5),
        lambda: mc_delta_fd_crn(S, K, T, SIG, R, Q, "put", n_paths=2000, seed=5),
    ):
        r = fn()
        assert np.isfinite([r.value, r.std_error, r.ci_low, r.ci_high]).all()
        assert r.ci_low <= r.value <= r.ci_high and r.std_error >= 0.0
