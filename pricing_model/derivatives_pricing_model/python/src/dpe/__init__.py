"""dpe — Derivatives Pricing Engine (reference Python implementation).

Modules
-------
black_scholes : analytic BSM / Garman-Kohlhagen prices, full Greeks
                (delta, gamma, vega, theta, rho, vanna, volga) and a
                robust implied-vol solver.
binomial      : CRR and Jarrow-Rudd lattices, European and American
                exercise, tree Greeks, optional Richardson averaging.
monte_carlo   : exact-step GBM simulation; European, arithmetic Asian,
                up-and-out barrier and floating-strike lookback pricing;
                antithetic and control variates; pathwise and CRN-FD
                deltas; standard errors and 95% CIs.
fx            : FX forward and spot/forward delta-convention conversions.
utils         : historical (realised) volatility estimator.

Shared conventions: time in years, vol and rates as decimals, theta per
year; ``ValueError`` on invalid input. See each module's docstring and
``API_SPEC.md`` at the project root for the full cross-language contract.
"""

from ._validation import AMERICAN, CALL, EUROPEAN, PUT
from .black_scholes import (
    Greeks,
    bs_d1_d2,
    bs_greeks,
    bs_price,
    gk_greeks,
    gk_price,
    implied_vol,
    norm_cdf,
    norm_pdf,
)
from .binomial import CRR, JARROW_RUDD, BinomialGreeks, binomial_greeks, binomial_price
from .fx import forward_delta_to_spot, fx_forward, spot_delta_to_forward
from .monte_carlo import (
    MCResult,
    geometric_asian_price,
    mc_asian_arithmetic,
    mc_barrier_up_out,
    mc_delta_fd_crn,
    mc_delta_pathwise,
    mc_european,
    mc_lookback_floating,
    simulate_gbm_paths,
)
from .utils import historical_vol

__version__ = "1.0.0"

__all__ = [
    "AMERICAN",
    "CALL",
    "CRR",
    "EUROPEAN",
    "JARROW_RUDD",
    "PUT",
    "BinomialGreeks",
    "Greeks",
    "MCResult",
    "binomial_greeks",
    "binomial_price",
    "bs_d1_d2",
    "bs_greeks",
    "bs_price",
    "forward_delta_to_spot",
    "fx_forward",
    "geometric_asian_price",
    "gk_greeks",
    "gk_price",
    "historical_vol",
    "implied_vol",
    "mc_asian_arithmetic",
    "mc_barrier_up_out",
    "mc_delta_fd_crn",
    "mc_delta_pathwise",
    "mc_european",
    "mc_lookback_floating",
    "norm_cdf",
    "norm_pdf",
    "simulate_gbm_paths",
    "spot_delta_to_forward",
    "__version__",
]
