"""Binomial-tree pricing: Cox-Ross-Rubinstein and Jarrow-Rudd lattices.

Both lattices discretise GBM over ``n`` steps of length ``dt = t / n``.

Cox-Ross-Rubinstein (CRR)
    ``u = exp(sigma sqrt(dt))``, ``d = 1/u``; the risk-neutral up
    probability ``p = (exp((r - q) dt) - d) / (u - d)`` matches the first
    moment of GBM; the tree recombines around the spot (the middle node at
    every even step equals ``s``), which is what makes tree-based delta,
    gamma and theta natural.

Jarrow-Rudd (JR, "equal-probability")
    ``u = exp((r - q - sigma^2/2) dt + sigma sqrt(dt))``,
    ``d = exp((r - q - sigma^2/2) dt - sigma sqrt(dt))``, ``p = 1/2``.
    The drift is absorbed into the node placement so both branches are
    equally likely; the lattice drifts with the forward instead of
    recentering on the spot.

Both converge to Black-Scholes at rate O(1/n) for European payoffs, with a
well-known oscillation between adjacent step counts (the strike moves
relative to the node grid). ``richardson=True`` prices at ``n`` and
``n + 1`` steps and averages, which cancels the leading oscillating error
term; this is two-point Richardson/odd-even averaging, and typically buys
one to two extra digits at the same cost order.

Backward induction values American exercise by taking
``max(continuation, intrinsic)`` at every node — the discrete dynamic
program for the optimal stopping problem.

The ``sigma = 0`` and ``t = 0`` limits are handled without building a tree
(the lattice degenerates: ``u = d`` makes ``p`` ill-defined). For
``sigma = 0`` the spot rides the deterministic forward, so a European
option is worth discounted forward intrinsic and an American option is the
best discounted intrinsic over the deterministic path, evaluated on the
step grid (the exact deterministic-limit dynamic program).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._validation import (
    AMERICAN,
    CALL,
    require_integer,
    validate_exercise_style,
    validate_market_inputs,
    validate_option_type,
)
from .black_scholes import bs_price

__all__ = ["BinomialGreeks", "binomial_price", "binomial_greeks", "CRR", "JARROW_RUDD"]

#: Canonical method strings (part of the cross-language contract).
CRR = "crr"
JARROW_RUDD = "jr"

#: Largest admissible magnitude of the log-spot excursion across the lattice.
#: exp() overflows a double at ~709.78, so a terminal node with
#: ``|ln s| + |nu n| + sigma sqrt(t n) > 700`` would be inf (or an inf*0 NaN
#: in ``s u^j d^(n-j)``); the guard turns that into an invalid-input error.
#: Identical in every language port.
LATTICE_LOG_LIMIT = 700.0


def _validate_method(method: str) -> str:
    if not isinstance(method, str) or method.lower() not in (CRR, JARROW_RUDD):
        raise ValueError(f"method must be 'crr' or 'jr', got {method!r}")
    return method.lower()


def _validate_steps(steps: int) -> int:
    """``steps`` must be an integer >= 1 (any :class:`numbers.Integral`)."""
    return require_integer("steps", steps, 1)


def _check_lattice_range(s: float, t: float, sigma: float, r: float, q: float, steps: int) -> None:
    """Reject lattices whose terminal spots would overflow a double.

    The extreme terminal log-spot is ``ln s + nu n +- sigma sqrt(t n)`` with
    ``nu n = (r - q - sigma^2/2) t`` for JR (zero for CRR, whose nodes carry
    no drift); the guard bounds the worst case for either method so the two
    ports never disagree on what is admissible.
    """
    nu_n = abs((r - q - 0.5 * sigma * sigma) * t)
    excursion = abs(math.log(s)) + nu_n + sigma * math.sqrt(t * steps)
    if excursion > LATTICE_LOG_LIMIT:
        raise ValueError(
            f"lattice overflow: |ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) = "
            f"{excursion!r} exceeds {LATTICE_LOG_LIMIT}; reduce steps, sigma or t"
        )


def _payoff(spots: np.ndarray, k: float, ot: str) -> np.ndarray:
    return np.maximum(spots - k, 0.0) if ot == CALL else np.maximum(k - spots, 0.0)


def _degenerate_sigma_zero(
    s: float, k: float, t: float, r: float, q: float, ot: str, style: str, steps: int
) -> float:
    """Exact ``sigma = 0`` limit on the ``steps + 1`` point time grid.

    The spot is deterministic: ``S(t_i) = s exp((r - q) t_i)``. A European
    option pays intrinsic on the terminal forward; an American option is
    exercised at whichever grid date maximises the discounted intrinsic.
    """
    if style == AMERICAN:
        times = np.linspace(0.0, t, steps + 1)
        spots = s * np.exp((r - q) * times)
        payoffs = np.exp(-r * times) * _payoff(spots, k, ot)
        return float(np.max(payoffs))
    return bs_price(s, k, t, 0.0, r, q, ot)  # discounted forward intrinsic


def _tree_params(method: str, sigma: float, r: float, q: float, dt: float) -> tuple[float, float, float]:
    """Return ``(u, d, p)`` for the requested lattice."""
    sq = sigma * math.sqrt(dt)
    if method == CRR:
        u = math.exp(sq)
        d = 1.0 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)
    else:  # Jarrow-Rudd
        nu = (r - q - 0.5 * sigma * sigma) * dt
        u = math.exp(nu + sq)
        d = math.exp(nu - sq)
        p = 0.5
    if not 0.0 < p < 1.0:
        # Happens when the per-step drift outruns the vol spacing (huge
        # (r-q)dt vs sigma sqrt(dt)); the discrete measure would not be a
        # probability, so the lattice is invalid at this step count.
        raise ValueError(
            f"risk-neutral probability p={p!r} outside (0, 1); "
            f"increase steps or use smaller drift/vol ratio"
        )
    return u, d, p


def _roll_back(
    s: float,
    k: float,
    t: float,
    r: float,
    ot: str,
    style: str,
    steps: int,
    u: float,
    d: float,
    p: float,
    keep_levels: int = 0,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Backward induction; optionally keep the option/spot values of the
    first ``keep_levels`` time levels (level 0 = today) for tree Greeks.

    Returns ``(value_at_root, option_levels, spot_levels)``.
    """
    dt = t / steps
    disc = math.exp(-r * dt)

    # Terminal spots: s * u^j * d^(n-j), j = 0..n (ascending).
    j = np.arange(steps + 1)
    spots = s * (u ** j) * (d ** (steps - j))
    values = _payoff(spots, k, ot)

    opt_levels: list[np.ndarray] = []
    spot_levels: list[np.ndarray] = []
    if steps < keep_levels:
        # The terminal level itself is one of the requested levels (only
        # for very small trees, e.g. tree Greeks at steps = 2): store it
        # here, because the backward loop only visits levels < steps.
        opt_levels.append(values.copy())
        spot_levels.append(spots.copy())
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        spots = spots[:-1] / d  # spots at level i: s * u^j * d^(i-j)
        if style == AMERICAN:
            values = np.maximum(values, _payoff(spots, k, ot))
        if i < keep_levels:
            opt_levels.append(values.copy())
            spot_levels.append(spots.copy())
    opt_levels.reverse()  # level 0 first
    spot_levels.reverse()
    return float(values[0]), opt_levels, spot_levels


def binomial_price(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    style: str = "european",
    steps: int = 500,
    method: str = "crr",
    richardson: bool = False,
) -> float:
    """Binomial-tree price of a European or American vanilla option.

    Parameters follow the package-wide conventions (see
    :mod:`dpe.black_scholes`); additionally:

    style : ``"european"`` or ``"american"``.
    steps : number of time steps, >= 1.
    method : ``"crr"`` (Cox-Ross-Rubinstein) or ``"jr"`` (Jarrow-Rudd).
    richardson : if True, average the ``steps`` and ``steps + 1`` prices to
        damp the odd/even oscillation of the convergence to the continuous
        limit (two-point Richardson averaging; see module docstring).

    Degenerate limits: ``t = 0`` returns intrinsic; ``sigma = 0`` returns
    the deterministic-forward limit (for American style, the best discounted
    intrinsic over the step-grid dates). Lattices whose terminal spots would
    overflow a double (``|ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps)
    > 700``) raise ``ValueError`` instead of returning ``inf``/NaN.
    """
    ot = validate_option_type(option_type)
    st = validate_exercise_style(style)
    mth = _validate_method(method)
    steps = _validate_steps(steps)
    validate_market_inputs(s, k, t, sigma, r, q)

    if t <= 0.0:
        return float(max(0.0, s - k)) if ot == CALL else float(max(0.0, k - s))
    if sigma <= 0.0:
        return _degenerate_sigma_zero(s, k, t, r, q, ot, st, steps)
    _check_lattice_range(s, t, sigma, r, q, steps + 1 if richardson else steps)

    def one(n: int) -> float:
        u, d, p = _tree_params(mth, sigma, r, q, t / n)
        v, _, _ = _roll_back(s, k, t, r, ot, st, n, u, d, p)
        return v

    if richardson:
        return 0.5 * (one(steps) + one(steps + 1))
    return one(steps)


@dataclass(frozen=True)
class BinomialGreeks:
    """Tree-based price and Greeks (delta, gamma, theta per year)."""

    price: float
    delta: float
    gamma: float
    theta: float


def binomial_greeks(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    style: str = "european",
    steps: int = 500,
    method: str = "crr",
) -> BinomialGreeks:
    """Delta, gamma and theta read directly off the lattice.

    With node values ``V(i, j)`` and node spots ``S(i, j)`` (level i, j ups):

    * ``delta = (V(1,1) - V(1,0)) / (S(1,1) - S(1,0))`` — the discrete hedge
      ratio one step ahead;
    * ``gamma`` = difference of the two one-sided deltas at level 2 divided
      by the half-spread of the outer level-2 spots;
    * ``theta = (V(2,1) - V(0,0) - dV) / (2 dt)`` where
      ``dV = delta (S(2,1) - s) + gamma (S(2,1) - s)^2 / 2`` removes the
      value change caused by the middle node's spot displacement. For CRR
      the middle level-2 node has spot exactly ``s`` (``ud = 1``) so
      ``dV = 0`` and this is the classic pure calendar-time difference; for
      JR the lattice drifts (``S(2,1) = s e^{2 nu dt}``) and the correction
      strips the delta/gamma contamination out of the time difference.

    Requires ``t > 0`` and ``sigma > 0`` (otherwise the lattice degenerates);
    ``steps`` must be >= 2 so that level 2 exists.
    """
    ot = validate_option_type(option_type)
    st = validate_exercise_style(style)
    mth = _validate_method(method)
    steps = _validate_steps(steps)
    validate_market_inputs(s, k, t, sigma, r, q)
    if steps < 2:
        raise ValueError(f"steps must be >= 2 for tree Greeks, got {steps!r}")
    if t <= 0.0 or sigma <= 0.0:
        raise ValueError("tree Greeks require t > 0 and sigma > 0")
    _check_lattice_range(s, t, sigma, r, q, steps)

    dt = t / steps
    u, d, p = _tree_params(mth, sigma, r, q, dt)
    price, opt, spots = _roll_back(s, k, t, r, ot, st, steps, u, d, p, keep_levels=3)

    v0 = opt[0][0]
    v1, s1 = opt[1], spots[1]
    v2, s2 = opt[2], spots[2]

    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1])
    delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]))
    # Correct for the middle node's spot displacement (zero for CRR, O(dt)
    # for the drifting JR lattice): strip delta/gamma value change so the
    # remaining difference is purely calendar time.
    ds = s2[1] - s
    theta = (v2[1] - v0 - delta * ds - 0.5 * gamma * ds * ds) / (2.0 * dt)

    return BinomialGreeks(price=price, delta=float(delta), gamma=float(gamma), theta=float(theta))
