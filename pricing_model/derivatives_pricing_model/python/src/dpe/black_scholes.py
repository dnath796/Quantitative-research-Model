"""Analytic Black-Scholes-Merton and Garman-Kohlhagen pricing.

Model
-----
Under the risk-neutral measure the underlying follows geometric Brownian
motion with continuous dividend yield ``q``:

    dS/S = (r - q) dt + sigma dW

For an FX rate quoted as domestic-per-foreign, the foreign risk-free rate
plays exactly the role of a continuous dividend yield, so Garman-Kohlhagen
is Black-Scholes-Merton with ``r = rd`` (domestic) and ``q = rf`` (foreign).

With ``F = s * exp((r - q) * t)`` the forward, the European prices are

    call = exp(-r t) * (F N(d1) - K N(d2))
    put  = exp(-r t) * (K N(-d2) - F N(-d1))

    d1 = [ln(s/K) + (r - q + sigma^2 / 2) t] / (sigma sqrt(t))
    d2 = d1 - sigma sqrt(t)

Conventions used throughout the package (and by every language port):

* ``t`` is time to expiry in **years** (ACT/365-style year fraction).
* ``sigma`` is a decimal (0.20 means 20% annualised volatility).
* ``r``, ``q`` are continuously-compounded decimal rates.
* ``theta`` is the derivative w.r.t. *calendar time* per **year**
  (divide by 365 for a per-day theta).
* ``vega``, ``vanna``, ``volga`` are per unit of vol (per 1.00 = 100 vol
  points; divide by 100 for per-vol-point).
* ``rho`` is per unit of rate (divide by 100 for per-percentage-point).

Degenerate limits (``t = 0``, ``sigma = 0``, ``k = 0``) are handled exactly:
the price collapses to (discounted) intrinsic value and the second-order
Greeks vanish; see :func:`bs_price` and :func:`bs_greeks`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._validation import (
    CALL,
    PUT,
    validate_market_inputs,
    validate_option_type,
    require_finite,
    require_integer,
    require_positive,
)

__all__ = [
    "Greeks",
    "norm_cdf",
    "norm_pdf",
    "bs_d1_d2",
    "bs_price",
    "bs_greeks",
    "gk_price",
    "gk_greeks",
    "implied_vol",
]


# ---------------------------------------------------------------------------
# Standard normal helpers (math.erf keeps the core dependency-free and makes
# the values bit-reproducible across language ports that use erf/erfc).
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF ``N(x) = 0.5 * erfc(-x / sqrt(2))``.

    ``erfc`` is used instead of ``0.5 * (1 + erf(...))`` because it keeps full
    relative precision in the deep left tail (deep OTM options), where
    ``1 + erf`` would suffer catastrophic cancellation.
    """
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    """Standard normal density ``phi(x) = exp(-x^2/2) / sqrt(2 pi)``."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def bs_d1_d2(s: float, k: float, t: float, sigma: float, r: float, q: float) -> tuple[float, float]:
    """Return ``(d1, d2)`` for the regular region ``t > 0, sigma > 0, k > 0``.

    Raises ``ValueError`` if called in a degenerate region, because there is
    no finite d1/d2 there — the price/Greek functions handle those limits
    explicitly instead.
    """
    validate_market_inputs(s, k, t, sigma, r, q)
    if t <= 0.0 or sigma <= 0.0 or k <= 0.0:
        raise ValueError("d1/d2 require t > 0, sigma > 0 and k > 0; use bs_price for the limits")
    sq_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sq_t)
    d2 = d1 - sigma * sq_t
    return d1, d2


@dataclass(frozen=True)
class Greeks:
    """Container for the full set of first/second-order Greeks.

    Attributes
    ----------
    price : option present value.
    delta : dV/dS.
    gamma : d2V/dS2 (identical for call and put).
    vega  : dV/dsigma, per unit of vol (identical for call and put).
    theta : dV/dt per **year**, calendar-time convention (negative of the
            derivative w.r.t. time-to-expiry). Usually negative for calls.
    rho   : dV/dr (domestic rate for FX). Dividend/foreign-rate rho is not
            reported; ports must match this exact set.
    vanna : d2V/(dS dsigma) — sensitivity of delta to vol.
    volga : d2V/dsigma2 (a.k.a. vomma) — sensitivity of vega to vol.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float


def _intrinsic(s: float, k: float, option_type: str) -> float:
    """Undiscounted intrinsic value at expiry (always a ``float``).

    ``max(0.0, x)`` (zero first) returns ``0.0`` on ties, so an at-the-money
    tie never leaks an ``int`` (integer inputs) or a ``-0.0`` into the float
    API.
    """
    return float(max(0.0, s - k)) if option_type == CALL else float(max(0.0, k - s))


def bs_price(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
) -> float:
    """European Black-Scholes-Merton price.

    Parameters
    ----------
    s : spot (> 0).
    k : strike (>= 0; ``k = 0`` call is a forward claim worth ``s e^{-qt}``).
    t : time to expiry in years (>= 0).
    sigma : annualised volatility, decimal (>= 0).
    r : continuously-compounded risk-free (domestic) rate; may be negative.
    q : continuous dividend yield (or foreign rate); may be negative.
    option_type : ``"call"`` or ``"put"``.

    Degenerate limits
    -----------------
    * ``t = 0``: value is the intrinsic ``max(+-(s - k), 0)``.
    * ``sigma = 0`` (t > 0): the terminal spot is the deterministic forward,
      so the value is the discounted forward intrinsic
      ``exp(-r t) * max(+-(F - k), 0)`` with ``F = s exp((r - q) t)``.
    * ``k = 0``: call = ``s exp(-q t)`` (a pure forward claim), put = 0.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)

    if t <= 0.0:
        return _intrinsic(s, k, ot)

    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)

    if k <= 0.0:
        return s * df_q if ot == CALL else 0.0

    if sigma <= 0.0:
        # Deterministic world: S_T = F almost surely.
        fwd_intrinsic = s * df_q - k * df_r
        if ot == CALL:
            return float(max(0.0, fwd_intrinsic))
        return float(max(0.0, -fwd_intrinsic))

    d1, d2 = bs_d1_d2(s, k, t, sigma, r, q)
    if ot == CALL:
        return s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2)
    return k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1)


def bs_greeks(
    s: float,
    k: float,
    t: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
) -> Greeks:
    """Full analytic Greeks for a European BSM option.

    Formulas (call; ``phi`` = normal pdf, ``N`` = normal cdf):

    * ``delta = e^{-qt} N(d1)``            (put: ``e^{-qt} (N(d1) - 1)``)
    * ``gamma = e^{-qt} phi(d1) / (s sigma sqrt(t))``
    * ``vega  = s e^{-qt} phi(d1) sqrt(t)``
    * ``theta = -s e^{-qt} phi(d1) sigma / (2 sqrt(t))
                - r k e^{-rt} N(d2) + q s e^{-qt} N(d1)``   (per year)
    * ``rho   = k t e^{-rt} N(d2)``        (put: ``-k t e^{-rt} N(-d2)``)
    * ``vanna = -e^{-qt} phi(d1) d2 / sigma``
    * ``volga = vega * d1 * d2 / sigma``

    Degenerate limits (``t = 0``, ``sigma = 0``, ``k = 0``) return the
    pointwise limits of the formulas: gamma/vega/vanna/volga are 0, delta is
    the discounted forward-moneyness indicator (``e^{-qt}`` if in the money,
    else 0; the exactly-at-the-money boundary maps to the ITM branch), and
    theta/rho are the derivatives of the limiting price. These conventions
    are part of the cross-language contract.
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, sigma, r, q)

    price = bs_price(s, k, t, sigma, r, q, ot)
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)

    if t <= 0.0 or sigma <= 0.0 or k <= 0.0:
        # Limit region: the payoff is (discounted) intrinsic on the forward.
        fwd = s * df_q - k * df_r  # sign of forward moneyness
        if ot == CALL:
            itm = fwd >= 0.0 or k <= 0.0
            delta = df_q if itm else 0.0
            theta = (q * s * df_q - r * k * df_r) if itm else 0.0
            rho = k * t * df_r if itm else 0.0
        else:
            itm = fwd < 0.0 and k > 0.0
            delta = -df_q if itm else 0.0
            theta = (r * k * df_r - q * s * df_q) if itm else 0.0
            rho = -k * t * df_r if itm else 0.0
        return Greeks(price, delta, 0.0, 0.0, theta, rho, 0.0, 0.0)

    d1, d2 = bs_d1_d2(s, k, t, sigma, r, q)
    sq_t = math.sqrt(t)
    pdf_d1 = norm_pdf(d1)

    gamma = df_q * pdf_d1 / (s * sigma * sq_t)
    vega = s * df_q * pdf_d1 * sq_t
    vanna = -df_q * pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    if ot == CALL:
        delta = df_q * norm_cdf(d1)
        theta = (
            -s * df_q * pdf_d1 * sigma / (2.0 * sq_t)
            - r * k * df_r * norm_cdf(d2)
            + q * s * df_q * norm_cdf(d1)
        )
        rho = k * t * df_r * norm_cdf(d2)
    else:
        delta = df_q * (norm_cdf(d1) - 1.0)
        theta = (
            -s * df_q * pdf_d1 * sigma / (2.0 * sq_t)
            + r * k * df_r * norm_cdf(-d2)
            - q * s * df_q * norm_cdf(-d1)
        )
        rho = -k * t * df_r * norm_cdf(-d2)

    return Greeks(price, delta, gamma, vega, theta, rho, vanna, volga)


# ---------------------------------------------------------------------------
# Garman-Kohlhagen: FX wrappers. Spot is domestic-per-foreign; rd discounts
# the payoff (domestic currency), rf is the yield earned by holding the
# foreign currency, mathematically identical to a dividend yield.
# ---------------------------------------------------------------------------


def gk_price(
    s: float,
    k: float,
    t: float,
    sigma: float,
    rd: float,
    rf: float,
    option_type: str,
) -> float:
    """Garman-Kohlhagen FX option price: BSM with ``r = rd``, ``q = rf``.

    ``s`` and ``k`` are FX rates quoted domestic-per-foreign (e.g. USD per
    EUR for EURUSD); the price is in domestic currency per unit of foreign
    notional.
    """
    return bs_price(s, k, t, sigma, rd, rf, option_type)


def gk_greeks(
    s: float,
    k: float,
    t: float,
    sigma: float,
    rd: float,
    rf: float,
    option_type: str,
) -> Greeks:
    """Garman-Kohlhagen Greeks: BSM Greeks with ``r = rd``, ``q = rf``.

    The reported ``delta`` is the **spot delta** (premium-excluded); see
    :mod:`dpe.fx` for spot/forward delta-convention conversions. ``rho`` is
    the sensitivity to the domestic rate ``rd``.
    """
    return bs_greeks(s, k, t, sigma, rd, rf, option_type)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

_IV_MAX_SIGMA = 20.0  # 2000% vol: beyond any market; used as a hard bracket cap.


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    option_type: str,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Implied Black-Scholes volatility via bracketed Newton with bisection fallback.

    Robustness strategy
    -------------------
    1. **No-arbitrage bounds first.** A European call must satisfy
       ``max(s e^{-qt} - k e^{-rt}, 0) < price < s e^{-qt}`` (mirrored for
       puts). A price at or outside these bounds has no finite implied vol,
       so we raise ``ValueError`` with a message naming the violated bound
       rather than letting a root-finder wander.
    2. **Bracketing.** The BSM price is strictly increasing in sigma on the
       open no-arb interval, so a root is bracketed in ``[lo, hi]`` where
       ``hi`` is doubled from 1.0 until the model price exceeds the target
       (clipped at, and tested at, the 2000% cap; beyond it the price is
       numerically indistinguishable from the upper bound and an error is
       raised).
    3. **Safeguarded Newton.** Newton steps use analytic vega; whenever a
       step would leave the bracket or vega is too small (deep ITM/OTM,
       where vega underflows), the step falls back to bisection on the
       maintained bracket. This converges for any admissible input.

    Convergence: stops when the price error is below ``tol`` (absolute) or
    the bracket width is below 1e-12. Returns the implied sigma as a decimal.
    ``tol`` must be > 0 and ``max_iter`` >= 1; if neither stopping rule is
    met within ``max_iter`` iterations the function **raises** — it never
    returns a half-converged root as if it had converged.

    Raises
    ------
    ValueError
        If inputs are invalid (including ``tol <= 0`` or ``max_iter < 1``),
        ``t <= 0`` (no vol is identifiable at expiry), the price violates
        the no-arbitrage bounds, or the iteration budget is exhausted before
        convergence (message contains ``"did not converge"`` and the
        residual).
    """
    ot = validate_option_type(option_type)
    validate_market_inputs(s, k, t, 0.0, r, q)
    require_finite("price", price)
    require_positive("t", t)
    require_positive("k", k)
    require_positive("tol", tol)
    max_iter = require_integer("max_iter", max_iter, 1)

    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    if ot == CALL:
        lower = max(s * df_q - k * df_r, 0.0)
        upper = s * df_q
    else:
        lower = max(k * df_r - s * df_q, 0.0)
        upper = k * df_r

    if price <= lower:
        raise ValueError(
            f"price {price!r} violates the no-arbitrage lower bound {lower!r} "
            f"(discounted intrinsic); no implied vol exists"
        )
    if price >= upper:
        raise ValueError(
            f"price {price!r} violates the no-arbitrage upper bound {upper!r}; "
            f"no implied vol exists"
        )

    def f(sig: float) -> float:
        return bs_price(s, k, t, sig, r, q, ot) - price

    # Bracket the root: price is monotone increasing in sigma. hi doubles
    # 1 -> 2 -> 4 -> 8 -> 16 -> 20 (clipped to the cap) and the cap itself is
    # tested before giving up, so the effective cap is exactly _IV_MAX_SIGMA.
    lo = 0.0  # f(lo) = discounted intrinsic - price < 0 by the bound check
    hi = 1.0
    while f(hi) < 0.0:
        if hi >= _IV_MAX_SIGMA:
            raise ValueError(
                f"implied vol exceeds cap {_IV_MAX_SIGMA}; price {price!r} is "
                f"numerically indistinguishable from the upper bound"
            )
        hi = min(2.0 * hi, _IV_MAX_SIGMA)

    # Safeguarded Newton: max_iter Newton/bisection steps from the bracket
    # midpoint; the residual is checked before every step and once more after
    # the last one, so a returned sigma always satisfies a stopping rule.
    sigma = 0.5 * (lo + hi)  # initial guess: bracket midpoint
    diff = f(sigma)
    for _ in range(max_iter):
        if abs(diff) < tol:
            return sigma
        # Maintain the bracket around the root.
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        vega = bs_greeks(s, k, t, sigma, r, q, ot).vega
        if vega > 1e-12:
            candidate = sigma - diff / vega
            if lo < candidate < hi:
                sigma = candidate
                diff = f(sigma)
                continue
        # Newton unusable (tiny vega or step outside bracket): bisect.
        sigma = 0.5 * (lo + hi)
        if hi - lo < 1e-12:
            return sigma
        diff = f(sigma)
    if abs(diff) < tol:
        return sigma
    # Iteration budget exhausted without meeting either stopping rule: report
    # honestly rather than hand back a half-converged root as if converged.
    raise ValueError(
        f"implied vol did not converge in {max_iter} iterations "
        f"(|model - price| = {abs(diff)!r} > tol {tol!r}, bracket [{lo!r}, {hi!r}]); "
        f"increase max_iter or loosen tol"
    )
