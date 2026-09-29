"""Monte Carlo engine: exact-step GBM simulation, exotics, variance reduction.

Simulation scheme
-----------------
GBM has the exact solution

    S_{t+h} = S_t * exp((r - q - sigma^2/2) h + sigma sqrt(h) Z),  Z ~ N(0,1)

so stepping with this recursion is **exact in distribution** at the grid
dates — there is no Euler discretisation error in the marginals. Path
functionals that look *between* grid dates (barrier crossings, lookback
extrema) still carry monitoring bias; see the individual pricers.

Variance reduction
------------------
* **Antithetic variates**: each normal draw ``Z`` is paired with ``-Z``.
  Because the payoff is monotone in ``Z`` for the payoffs here, the pair
  averages are negatively correlated and the estimator variance drops.
  Statistically each *pair average* is one i.i.d. sample, so standard
  errors are computed over pair averages (n_paths/2 samples), never over
  raw correlated paths.
* **Control variates**: for a payoff ``Y`` and a control ``X`` with known
  mean ``E[X]``, the adjusted estimator ``Y - beta (X - E[X])`` with
  ``beta = Cov(Y, X) / Var(X)`` (estimated in-sample) is unbiased up to the
  O(1/n) bias from estimating beta and has variance reduced by the squared
  correlation. Controls used:
    - European vanilla: discounted terminal spot ``e^{-rt} S_T`` with known
      mean ``s e^{-qt}`` (the forward is a perfect martingale control, and
      its price is the k=0 Black-Scholes value — the "BS analytic" control);
    - arithmetic Asian: the *geometric* Asian payoff, whose discrete-fixing
      price is known in closed form (:func:`geometric_asian_price`) and
      whose correlation with the arithmetic payoff is typically > 0.99.

Every pricer takes an integer ``seed`` in ``[0, 2^63 - 1]`` (the domain
pinned by API_SPEC §5 for every port) and is fully deterministic given it
(numpy PCG64 via ``default_rng``). Results are reported as
:class:`MCResult` with the standard error and a 95% normal CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._validation import (
    CALL,
    validate_market_inputs,
    validate_option_type,
    require_integer,
    require_positive,
    require_seed,
)
from .black_scholes import bs_price, norm_cdf

__all__ = [
    "MCResult",
    "simulate_gbm_paths",
    "geometric_asian_price",
    "mc_european",
    "mc_asian_arithmetic",
    "mc_barrier_up_out",
    "mc_lookback_floating",
    "mc_delta_pathwise",
    "mc_delta_fd_crn",
]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate with its sampling uncertainty.

    Attributes
    ----------
    value : the point estimate (price or Greek).
    std_error : standard error of the estimate (std of the i.i.d. samples /
        sqrt(n_samples); with antithetics, samples are pair averages).
    ci_low, ci_high : 95% normal confidence interval
        ``value -+ 1.96 * std_error``.
    n_paths : number of simulated paths (raw paths, counting both members
        of each antithetic pair).
    """

    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int


def _validate_mc(n_paths: int, seed: int, antithetic: bool, n_steps: int = 1) -> tuple[int, int, int]:
    """Validate the Monte Carlo controls and return them as built-in ints.

    ``n_paths >= 2`` (``>= 4`` and even with antithetics), ``n_steps >= 1``,
    ``seed`` in ``[0, 2^63 - 1]``; every integer parameter accepts any
    :class:`numbers.Integral` (numpy/pandas integers included).
    """
    n_paths = require_integer("n_paths", n_paths, 2)
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even with antithetic=True, got {n_paths!r}")
    if antithetic and n_paths < 4:
        raise ValueError(f"n_paths must be >= 4 with antithetic=True, got {n_paths!r}")
    seed = require_seed(seed)
    n_steps = require_integer("n_steps", n_steps, 1)
    return n_paths, seed, n_steps


def _validate_rel_bump(rel_bump: float) -> None:
    """``rel_bump`` must be finite and in the open interval (0, 1).

    ``rel_bump >= 1`` would price the down-bumped payoff at a non-positive
    spot ``s (1 - rel_bump)`` and return a meaningless delta without any
    error; the contract (API_SPEC §5.8) therefore pins the open unit interval.
    """
    require_positive("rel_bump", rel_bump)
    if rel_bump >= 1.0:
        raise ValueError(f"rel_bump must be in (0, 1), got {rel_bump!r}")


def _normals(n_paths: int, n_steps: int, seed: int, antithetic: bool) -> np.ndarray:
    """Draw the (n_paths, n_steps) normal increments, antithetic-aware.

    With antithetics the second half of the paths is the negation of the
    first half, so path ``i`` and path ``i + n_paths/2`` form a pair.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.vstack([half, -half])
    return rng.standard_normal((n_paths, n_steps))


def _pair_average(samples: np.ndarray, antithetic: bool) -> np.ndarray:
    """Collapse antithetic pairs into i.i.d. pair averages for statistics."""
    if not antithetic:
        return samples
    half = samples.shape[0] // 2
    return 0.5 * (samples[:half] + samples[half:])


def _stats(samples: np.ndarray, n_paths: int) -> MCResult:
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(samples.shape[0]))
    return MCResult(mean, se, mean - _Z95 * se, mean + _Z95 * se, n_paths)


def _control_adjust(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
    """Apply the optimal-beta control-variate adjustment ``y - b (x - E[x])``."""
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0.0:
        return y  # degenerate control (e.g. sigma = 0): no adjustment possible
    beta = float(np.cov(y, x, ddof=1)[0, 1]) / var_x
    return y - beta * (x - x_mean)


def simulate_gbm_paths(
    s: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM paths with the exact log-Euler step.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first column
    is the spot ``s`` and whose column ``i`` holds ``S(i * t / n_steps)``.
    Deterministic given ``seed``. With ``antithetic=True``, ``n_paths`` must
    be even and paths ``[n_paths/2:]`` use the negated increments of paths
    ``[:n_paths/2]``.
    """
    validate_market_inputs(s, 1.0, t, sigma, r, q)
    require_positive("t", t)
    n_paths, seed, n_steps = _validate_mc(n_paths, seed, antithetic, n_steps)

    dt = t / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    z = _normals(n_paths, n_steps, seed, antithetic)
    log_increments = drift + vol * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s
    paths[:, 1:] = s * np.exp(log_paths)
    return paths


def geometric_asian_price(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_fixings: int,
) -> float:
    """Closed-form price of a discretely-monitored geometric-average Asian.

    The geometric mean ``G = (prod_{i=1..n} S(t_i))^{1/n}`` of GBM sampled at
    equally-spaced dates ``t_i = i t / n`` is lognormal:

        E[ln G]   = ln s + (r - q - sigma^2/2) * t (n+1) / (2n)
        Var[ln G] = sigma^2 t (n+1)(2n+1) / (6 n^2)

    (the variance uses ``sum_{i,j} min(i,j) = n(n+1)(2n+1)/6``). Pricing is
    then Black's formula on the lognormal ``G``:

        call = e^{-rt} (E[G] N(d1) - K N(d2)),
        d1 = (ln(E[G]/K) + Var/2) / sqrt(Var),  d2 = d1 - sqrt(Var)

    with ``E[G] = exp(E[ln G] + Var[ln G]/2)``. Used as the analytic control
    variate for the arithmetic Asian and as a deterministic golden value.
    Requires ``t > 0``, ``k > 0`` and ``n_fixings >= 1``; ``sigma = 0``
    returns the discounted deterministic-average intrinsic.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    require_positive("k", k)
    n = require_integer("n_fixings", n_fixings, 1)
    mean_ln = math.log(s) + (r - q - 0.5 * sigma * sigma) * t * (n + 1) / (2.0 * n)
    var_ln = sigma * sigma * t * (n + 1) * (2 * n + 1) / (6.0 * n * n)
    df = math.exp(-r * t)
    eg = math.exp(mean_ln + 0.5 * var_ln)
    if var_ln <= 0.0:
        payoff = max(eg - k, 0.0) if ot == CALL else max(k - eg, 0.0)
        return df * payoff
    sd = math.sqrt(var_ln)
    d1 = (math.log(eg / k) + 0.5 * var_ln) / sd
    d2 = d1 - sd
    if ot == CALL:
        return df * (eg * norm_cdf(d1) - k * norm_cdf(d2))
    return df * (k * norm_cdf(-d2) - eg * norm_cdf(-d1))


def _vanilla_payoff(st: np.ndarray, k: float, ot: str) -> np.ndarray:
    return np.maximum(st - k, 0.0) if ot == CALL else np.maximum(k - st, 0.0)


def mc_european(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
    control_variate: bool = False,
) -> MCResult:
    """European vanilla by Monte Carlo (single exact step to expiry).

    The terminal spot needs only one exact GBM step, so no time grid is
    simulated. Optional variance reduction: antithetic pairs and the
    discounted-terminal-spot control variate (known mean ``s e^{-qt}``, the
    Black-Scholes k=0 analytic value). Requires ``t > 0``.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    n_paths, seed, _ = _validate_mc(n_paths, seed, antithetic)

    z = _normals(n_paths, 1, seed, antithetic)[:, 0]
    st = s * np.exp((r - q - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z)
    df = math.exp(-r * t)
    payoff = df * _vanilla_payoff(st, k, ot)

    y = _pair_average(payoff, antithetic)
    if control_variate:
        x = _pair_average(df * st, antithetic)
        y = _control_adjust(y, x, s * math.exp(-q * t))
    return _stats(y, n_paths)


def mc_asian_arithmetic(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_paths: int = 100_000,
    n_steps: int = 50,
    seed: int = 42,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Arithmetic-average Asian option by Monte Carlo.

    The average is taken over the ``n_steps`` grid dates ``t_i = i t / n``
    for ``i = 1..n`` (the spot at ``t = 0`` is *not* a fixing). With
    ``control_variate=True`` the geometric-average Asian on the same fixing
    dates is used as control, with its exact price from
    :func:`geometric_asian_price`; correlation with the arithmetic payoff
    is typically >99%, cutting the standard error by an order of magnitude.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    require_positive("k", k)
    n_paths, seed, n_steps = _validate_mc(n_paths, seed, antithetic, n_steps)

    paths = simulate_gbm_paths(s, t, sigma, r, q, n_paths, n_steps, seed, antithetic)
    fixings = paths[:, 1:]  # exclude S0
    df = math.exp(-r * t)

    arith = np.mean(fixings, axis=1)
    payoff = df * _vanilla_payoff(arith, k, ot)
    y = _pair_average(payoff, antithetic)

    if control_variate:
        geo = np.exp(np.mean(np.log(fixings), axis=1))
        geo_payoff = df * _vanilla_payoff(geo, k, ot)
        x = _pair_average(geo_payoff, antithetic)
        x_mean = geometric_asian_price(s, k, t, sigma, r, q, ot, n_steps)
        y = _control_adjust(y, x, x_mean)
    return _stats(y, n_paths)


def mc_barrier_up_out(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    barrier: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
    antithetic: bool = True,
) -> MCResult:
    """Up-and-out barrier option, discretely monitored on the step grid.

    The option knocks out (pays 0) if the simulated spot touches or exceeds
    ``barrier`` at **any grid date** (including ``t = 0``: if ``s >=
    barrier`` the option is born dead and the price is exactly 0).

    Discretisation bias
    -------------------
    A continuously-monitored barrier can be breached *between* grid dates,
    which discrete monitoring never sees, so this estimator systematically
    **over-prices** an up-and-out relative to the continuous contract; the
    bias shrinks like O(1/sqrt(n_steps)). The Broadie-Glasserman-Kou
    continuity correction (shift the barrier to
    ``B exp(0.5826 sigma sqrt(dt))`` for an up barrier) removes the leading
    bias term if a continuous contract is intended; this function prices the
    *discrete* contract as specified and applies no correction.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    require_positive("barrier", barrier)
    n_paths, seed, n_steps = _validate_mc(n_paths, seed, antithetic, n_steps)

    if s >= barrier:
        return MCResult(0.0, 0.0, 0.0, 0.0, n_paths)

    paths = simulate_gbm_paths(s, t, sigma, r, q, n_paths, n_steps, seed, antithetic)
    df = math.exp(-r * t)
    alive = np.max(paths, axis=1) < barrier
    payoff = df * _vanilla_payoff(paths[:, -1], k, ot) * alive
    return _stats(_pair_average(payoff, antithetic), n_paths)


def mc_lookback_floating(
    s: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
    antithetic: bool = True,
) -> MCResult:
    """Floating-strike lookback option, discretely monitored on the step grid.

    Payoffs (extrema over the grid dates, ``t = 0`` included):

    * call: ``S_T - min_i S(t_i)`` — buy at the observed low;
    * put:  ``max_i S(t_i) - S_T`` — sell at the observed high.

    Both payoffs are non-negative by construction. Discrete monitoring
    *under*-states the true continuous extremum, so this under-prices the
    continuously-monitored contract with O(1/sqrt(n_steps)) bias — the
    mirror image of the barrier case.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, 1.0, t, sigma, r, q)
    require_positive("t", t)
    n_paths, seed, n_steps = _validate_mc(n_paths, seed, antithetic, n_steps)

    paths = simulate_gbm_paths(s, t, sigma, r, q, n_paths, n_steps, seed, antithetic)
    df = math.exp(-r * t)
    terminal = paths[:, -1]
    if ot == CALL:
        payoff = df * (terminal - np.min(paths, axis=1))
    else:
        payoff = df * (np.max(paths, axis=1) - terminal)
    return _stats(_pair_average(payoff, antithetic), n_paths)


def mc_delta_pathwise(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
) -> MCResult:
    """Pathwise-derivative delta of a European vanilla.

    Differentiating the payoff along each path (valid because the vanilla
    payoff is Lipschitz and ``dS_T/dS_0 = S_T / S_0`` for GBM):

    * call: ``delta = e^{-rt} E[ 1{S_T > K} S_T / S_0 ]``
    * put:  ``delta = -e^{-rt} E[ 1{S_T < K} S_T / S_0 ]``

    This is unbiased and typically far lower-variance than finite
    differences, but it cannot handle discontinuous payoffs (e.g. digital),
    where the likelihood-ratio method would be needed instead.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    n_paths, seed, _ = _validate_mc(n_paths, seed, antithetic)

    z = _normals(n_paths, 1, seed, antithetic)[:, 0]
    st = s * np.exp((r - q - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z)
    df = math.exp(-r * t)
    if ot == CALL:
        samples = df * np.where(st > k, st / s, 0.0)
    else:
        samples = -df * np.where(st < k, st / s, 0.0)
    return _stats(_pair_average(samples, antithetic), n_paths)


def mc_delta_fd_crn(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    n_paths: int = 100_000,
    seed: int = 42,
    rel_bump: float = 1e-4,
    antithetic: bool = True,
) -> MCResult:
    """Central finite-difference delta with common random numbers (CRN).

    The same normal draws (same ``seed``) price the payoff at ``s (1 -+
    rel_bump)`` and the delta is the per-path central difference

        (payoff(s(1+h)) - payoff(s(1-h))) / (2 s h)

    Reusing the randomness makes the difference of the two estimators
    nearly noiseless (their sampling errors cancel path by path); without
    CRN the variance would explode as ``O(1/h^2)``. The standard error
    reported is that of the per-path difference quotient. ``rel_bump`` must
    lie in the open interval (0, 1) so that both bumped spots stay positive.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)
    require_positive("t", t)
    _validate_rel_bump(rel_bump)
    n_paths, seed, _ = _validate_mc(n_paths, seed, antithetic)

    z = _normals(n_paths, 1, seed, antithetic)[:, 0]
    growth = np.exp((r - q - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z)
    df = math.exp(-r * t)
    h = s * rel_bump
    pay_up = df * _vanilla_payoff((s + h) * growth, k, ot)
    pay_dn = df * _vanilla_payoff((s - h) * growth, k, ot)
    samples = (pay_up - pay_dn) / (2.0 * h)
    return _stats(_pair_average(samples, antithetic), n_paths)
