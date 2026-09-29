"""FX-specific helpers: forwards and delta-convention conversions.

FX options are quoted and hedged under several delta conventions. This
module implements the two premium-excluded ("pips") conventions:

* **Spot delta** — the sensitivity of the domestic-currency premium to the
  spot: for a call, ``delta_spot = e^{-rf t} N(d1)``. This is what
  :func:`dpe.black_scholes.gk_greeks` reports.
* **Forward delta** — the hedge in forward contracts rather than spot:
  ``delta_fwd = N(d1)`` for a call. Because one forward contract has spot
  sensitivity ``e^{-rf t}`` (a forward on one unit of foreign currency is
  replicated by holding ``e^{-rf t}`` units of foreign cash), the two
  conventions differ by exactly that factor:

      delta_fwd = e^{+rf t} * delta_spot

  The conversion is payoff-independent — it is pure discounting — so the
  same factor applies to calls and puts and at any strike. Interbank
  markets typically quote forward delta for maturities beyond ~1y and spot
  delta for short dates; converting between them correctly is essential
  when interpolating a vol surface quoted in deltas.

Premium-included deltas (relevant when the premium is paid in foreign
currency) are out of scope for this project.
"""

from __future__ import annotations

import math

from ._validation import require_non_negative, require_positive, require_finite

__all__ = ["fx_forward", "spot_delta_to_forward", "forward_delta_to_spot"]


def fx_forward(s: float, t: float, rd: float, rf: float) -> float:
    """Covered-interest-parity FX forward: ``F = s * exp((rd - rf) t)``.

    ``s`` is spot domestic-per-foreign; ``rd``/``rf`` are continuously
    compounded domestic/foreign rates (either may be negative).
    """
    require_positive("s", s)
    require_non_negative("t", t)
    require_finite("rd", rd)
    require_finite("rf", rf)
    return s * math.exp((rd - rf) * t)


def spot_delta_to_forward(delta_spot: float, t: float, rf: float) -> float:
    """Convert a premium-excluded spot delta to the forward-delta convention.

    ``delta_fwd = exp(rf * t) * delta_spot``. Valid for calls (positive
    delta) and puts (negative delta) alike.
    """
    require_finite("delta_spot", delta_spot)
    require_non_negative("t", t)
    require_finite("rf", rf)
    return math.exp(rf * t) * delta_spot


def forward_delta_to_spot(delta_forward: float, t: float, rf: float) -> float:
    """Convert a premium-excluded forward delta to the spot-delta convention.

    ``delta_spot = exp(-rf * t) * delta_forward`` — the exact inverse of
    :func:`spot_delta_to_forward`.
    """
    require_finite("delta_forward", delta_forward)
    require_non_negative("t", t)
    require_finite("rf", rf)
    return math.exp(-rf * t) * delta_forward
