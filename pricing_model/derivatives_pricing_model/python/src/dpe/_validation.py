"""Shared input validation for the dpe package.

All public pricing functions funnel their inputs through these helpers so that
error behaviour is uniform: bad input always raises ``ValueError`` with a
message naming the offending parameter (the cross-language contract maps this
to ``std::invalid_argument`` in C++, ``Err(DpeError::InvalidInput)`` in Rust
and ``IllegalArgumentException`` in Java; the implied-vol solver's
no-arbitrage and non-convergence failures are also ``ValueError`` here and
``DpeError::NoArbitrage`` / ``DpeError::NoConvergence`` in Rust — see
API_SPEC §8).
"""

from __future__ import annotations

import math
from numbers import Integral

__all__ = [
    "CALL",
    "PUT",
    "EUROPEAN",
    "AMERICAN",
    "SEED_MAX",
    "validate_option_type",
    "validate_exercise_style",
    "validate_market_inputs",
    "require_positive",
    "require_non_negative",
    "require_finite",
    "require_integer",
    "require_seed",
]

#: Largest admissible RNG seed (2^63 - 1): the seed domain is ``[0, SEED_MAX]``
#: in every language port, so a seed means the same thing in Python (numpy
#: PCG64), C++ (``std::int64_t``), Rust (``u64`` checked against this bound)
#: and Java (``long``).
SEED_MAX = 2**63 - 1

#: Canonical option-type strings used across every language port.
CALL = "call"
PUT = "put"

#: Canonical exercise-style strings used across every language port.
EUROPEAN = "european"
AMERICAN = "american"


def require_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` if *value* is NaN or infinite.

    NaN/inf guards sit at the boundary so that no NaN can silently propagate
    through a pricing formula and surface as a nonsense Greek downstream.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def require_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` unless *value* is finite and strictly positive."""
    require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def require_non_negative(name: str, value: float) -> None:
    """Raise ``ValueError`` unless *value* is finite and >= 0."""
    require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def require_integer(name: str, value: object, minimum: int) -> int:
    """Return *value* as a built-in ``int`` or raise ``ValueError``.

    Accepts any :class:`numbers.Integral` (built-in ``int``, ``numpy.int32``,
    ``numpy.int64``, pandas-derived integers, ...) so that batch drivers fed
    from DataFrames work; rejects ``bool`` (a valid ``Integral`` in Python
    but never a sensible step/path count), floats — even integral ones like
    ``500.0`` — strings and ``None``. Values below *minimum* are invalid.
    """
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    iv = int(value)
    if iv < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return iv


def require_seed(seed: object) -> int:
    """Validate an RNG seed: an integer in ``[0, SEED_MAX]`` (``2^63 - 1``).

    The domain is pinned by the cross-language contract (API_SPEC §5) so a
    seed is never silently wrapped (C++), rejected by the RNG library with a
    message that does not name the parameter (numpy), or accepted in one
    port and rejected in another.
    """
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer in [0, {SEED_MAX}], got {seed!r}")
    iv = int(seed)
    if iv < 0 or iv > SEED_MAX:
        raise ValueError(f"seed must be an integer in [0, {SEED_MAX}], got {seed!r}")
    return iv


def validate_option_type(option_type: str) -> str:
    """Normalise and validate an option-type string.

    Accepts ``"call"``/``"put"`` case-insensitively and returns the canonical
    lower-case form. Anything else raises ``ValueError``.
    """
    if not isinstance(option_type, str):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    ot = option_type.lower()
    if ot not in (CALL, PUT):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return ot


def validate_exercise_style(style: str) -> str:
    """Normalise and validate an exercise-style string.

    Accepts ``"european"``/``"american"`` case-insensitively and returns the
    canonical lower-case form. Anything else raises ``ValueError``.
    """
    if not isinstance(style, str):
        raise ValueError(f"style must be 'european' or 'american', got {style!r}")
    st = style.lower()
    if st not in (EUROPEAN, AMERICAN):
        raise ValueError(f"style must be 'european' or 'american', got {style!r}")
    return st


def validate_market_inputs(s: float, k: float, t: float, sigma: float, r: float, q: float) -> None:
    """Validate the common Black-Scholes market inputs.

    Rules (identical in every language port):

    * ``s``      spot, must be > 0 (a zero/negative equity or FX spot is
                 meaningless under geometric Brownian motion);
    * ``k``      strike, must be >= 0 (``k = 0`` is allowed: a zero-strike
                 call is a forward on the asset, a zero-strike put is worthless);
    * ``t``      time to expiry in **years**, must be >= 0 (``t = 0`` means
                 "at expiry" and prices collapse to intrinsic value);
    * ``sigma``  volatility as a **decimal per sqrt-year** (0.20 = 20%),
                 must be >= 0 (``sigma = 0`` gives the deterministic limit);
    * ``r, q``   continuously-compounded rates as decimals; may be negative
                 (negative-rate regimes are in scope) but must be finite.
    """
    require_positive("s", s)
    require_non_negative("k", k)
    require_non_negative("t", t)
    require_non_negative("sigma", sigma)
    require_finite("r", r)
    require_finite("q", q)
