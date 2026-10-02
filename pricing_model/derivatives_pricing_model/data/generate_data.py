"""Regenerate the bundled datasets and cross-language golden values.

Run from anywhere:  python3 data/generate_data.py

Outputs (all deterministic — fixed RNG seeds, analytic or fixed-seed values):

* ``portfolio.csv``        — small mixed equity/FX option book for the demo
                             risk report. For FX rows, ``r`` holds the
                             domestic rate rd and ``q`` the foreign rate rf.
* ``spots_timeseries.csv`` — 2 years (504 business-day steps) of synthetic
                             GBM closes for 3 underliers, used by the demo's
                             historical-vol estimate. Seed fixed at 20260827.
* ``golden/golden.json``   — named reference cases (inputs + expected
                             outputs + absolute tolerance) that every
                             language port must reproduce. Values are
                             produced by the *validated* Python
                             implementation: this script re-checks the
                             textbook ATM value, put-call parity and the
                             implied-vol round-trip before writing anything,
                             and aborts if any check fails.

The golden schema is flat (see API_SPEC.md §9.1): every value inside
``inputs`` and ``expect`` is a scalar number or string. Expected values are
written at full double precision (``json.dump`` round-trips doubles
exactly), so the case ``tol`` is entirely the *port* tolerance — none of it
is consumed by rounding in this generator.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

import dpe  # noqa: E402  (path set up above)

SEED_TIMESERIES = 20260827


# ---------------------------------------------------------------------------
# Validation gate: never write golden values from an unvalidated pricer.
# ---------------------------------------------------------------------------

def _validate_implementation() -> None:
    """Abort unless the pricer reproduces analytic anchors exactly."""
    # 1) Textbook ATM call (Hull): 10.4506 to 4 dp.
    atm = dpe.bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, "call")
    assert abs(atm - 10.450584) < 1e-6, f"ATM anchor failed: {atm}"

    # 2) Put-call parity across a grid.
    for s in (80.0, 100.0, 120.0):
        for k in (90.0, 100.0, 110.0):
            for t in (0.1, 1.0, 3.0):
                c = dpe.bs_price(s, k, t, 0.25, 0.03, 0.015, "call")
                p = dpe.bs_price(s, k, t, 0.25, 0.03, 0.015, "put")
                fwd = s * math.exp(-0.015 * t) - k * math.exp(-0.03 * t)
                assert abs((c - p) - fwd) < 1e-10, f"parity failed at {(s, k, t)}"

    # 3) Implied-vol round-trip.
    iv = dpe.implied_vol(atm, 100, 100, 1.0, 0.05, 0.0, "call")
    assert abs(iv - 0.2) < 1e-8, f"implied-vol round-trip failed: {iv}"

    # 4) CRR converges to BS.
    tree = dpe.binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, "call", "european", 2000, "crr")
    assert abs(tree - atm) < 1e-3, f"CRR convergence failed: {tree} vs {atm}"

    # 5) MC vanilla within 3 SE of BS.
    mc = dpe.mc_european(100, 100, 1.0, 0.2, 0.05, 0.0, "call", n_paths=100_000, seed=42)
    assert abs(mc.value - atm) < 3 * mc.std_error, f"MC check failed: {mc}"


# ---------------------------------------------------------------------------
# CSV datasets
# ---------------------------------------------------------------------------

def write_portfolio() -> None:
    """A small mixed book exercising every real-life scenario in the spec."""
    rows = [
        # id, underlying_type, s, k, t, sigma, r(=rd), q(=rf), type, style, quantity
        ("EQ_DIV_C1", "equity", 105.0, 100.0, 0.75, 0.22, 0.04, 0.02, "call", "european", 100),
        ("EQ_DIV_P1", "equity", 105.0, 110.0, 0.75, 0.22, 0.04, 0.02, "put", "european", -50),
        ("EQ_GROWTH_C1", "equity", 250.0, 260.0, 1.0, 0.35, 0.04, 0.0, "call", "european", 40),
        ("EQ_SMALLCAP_C1", "equity", 18.0, 20.0, 1.0, 0.80, 0.04, 0.0, "call", "european", 500),
        ("EQ_WEEKLY_P1", "equity", 100.0, 98.0, 1.0 / 52.0, 0.25, 0.03, 0.0, "put", "european", 200),
        ("EQ_LEAPS_C1", "equity", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, "call", "european", 75),
        ("EQ_AMER_P1", "equity", 95.0, 100.0, 1.0, 0.30, 0.05, 0.0, "put", "american", 60),
        ("FX_EURUSD_C1", "fx", 1.10, 1.12, 0.5, 0.10, 0.03, 0.02, "call", "european", 1_000_000),
        ("FX_EURUSD_P1", "fx", 1.10, 1.05, 1.0, 0.11, 0.03, 0.02, "put", "european", -500_000),
        ("FX_NEGRATE_P1", "fx", 0.95, 0.95, 1.0, 0.12, -0.005, 0.001, "put", "european", 750_000),
    ]
    with open(DATA_DIR / "portfolio.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")  # LF, as committed
        w.writerow(["id", "underlying_type", "s", "k", "t", "sigma", "r", "q", "type", "style", "quantity"])
        w.writerows(rows)


def write_spots_timeseries() -> None:
    """2y of daily synthetic GBM closes for 3 underliers, fixed seed."""
    rng = np.random.default_rng(SEED_TIMESERIES)
    n_days = 504  # ~2 business years
    dt = 1.0 / 252.0
    underliers = [  # name, s0, mu (real-world drift), sigma
        ("ACME", 105.0, 0.06, 0.22),
        ("GLOBEX", 250.0, 0.10, 0.35),
        ("PIPCO", 1.10, 0.00, 0.10),
    ]
    series = []
    for _, s0, mu, sigma in underliers:
        z = rng.standard_normal(n_days)
        log_ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
        series.append(s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)])))
    dates = np.busday_offset("2024-08-26", np.arange(n_days + 1), roll="forward")
    with open(DATA_DIR / "spots_timeseries.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")  # LF, as committed
        w.writerow(["date"] + [u[0] for u in underliers])
        for i, d in enumerate(dates):
            w.writerow([str(d)] + [f"{s[i]:.6f}" for s in series])


# ---------------------------------------------------------------------------
# Golden cases
# ---------------------------------------------------------------------------

def _case(name: str, inputs: dict, expect: dict, tol: float) -> dict:
    for d in (inputs, expect):
        for key, v in d.items():
            assert isinstance(v, (int, float, str)) and not isinstance(v, bool), (
                f"golden case {name}: {key} is not a flat scalar"
            )
    return {"name": name, "inputs": inputs, "expect": expect, "tol": tol}


def build_golden_cases() -> list[dict]:
    cases: list[dict] = []

    def r6(x: float) -> float:
        """Identity: goldens are stored at full precision.

        (An earlier revision rounded to 10 significant digits, which for
        |value| ~ 50 consumed half of a 1e-8 tolerance before any port was
        even run; see the review finding on golden rounding fragility.)
        """
        assert math.isfinite(x), f"golden value is not finite: {x!r}"
        return float(x)

    # -- Analytic BSM prices ------------------------------------------------
    def bs_case(name: str, s: float, k: float, t: float, sigma: float, r: float, q: float, ot: str, tol: float = 1e-8) -> None:
        price = dpe.bs_price(s, k, t, sigma, r, q, ot)
        cases.append(_case(name, {"s": s, "k": k, "t": t, "sigma": sigma, "r": r, "q": q, "type": ot}, {"price": r6(price)}, tol))

    bs_case("bs_call_atm", 100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    bs_case("bs_put_atm", 100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "put")
    bs_case("bs_call_dividend", 105.0, 100.0, 0.75, 0.22, 0.04, 0.02, "call")
    bs_case("bs_put_deep_itm", 100.0, 150.0, 1.0, 0.2, 0.05, 0.0, "put")
    bs_case("bs_call_deep_otm", 100.0, 150.0, 1.0, 0.2, 0.05, 0.0, "call", tol=1e-10)
    bs_case("bs_call_high_vol", 18.0, 20.0, 1.0, 0.8, 0.04, 0.0, "call")
    bs_case("bs_put_weekly", 100.0, 98.0, 1.0 / 52.0, 0.25, 0.03, 0.0, "put")
    bs_case("bs_call_leaps", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, "call")
    bs_case("bs_call_sigma_zero", 100.0, 90.0, 1.0, 0.0, 0.05, 0.02, "call")
    bs_case("bs_put_expiry", 90.0, 100.0, 0.0, 0.2, 0.05, 0.0, "put")
    bs_case("bs_call_zero_strike", 100.0, 0.0, 1.0, 0.2, 0.05, 0.02, "call")

    # -- Greeks -------------------------------------------------------------
    g = dpe.bs_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    cases.append(_case(
        "bs_greeks_call_atm",
        {"s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2, "r": 0.05, "q": 0.0, "type": "call"},
        {"delta": r6(g.delta), "gamma": r6(g.gamma), "vega": r6(g.vega),
         "theta": r6(g.theta), "rho": r6(g.rho), "vanna": r6(g.vanna), "volga": r6(g.volga)},
        1e-8,
    ))
    gp = dpe.bs_greeks(100.0, 110.0, 0.5, 0.3, 0.03, 0.02, "put")
    cases.append(_case(
        "bs_greeks_put_itm",
        {"s": 100.0, "k": 110.0, "t": 0.5, "sigma": 0.3, "r": 0.03, "q": 0.02, "type": "put"},
        {"delta": r6(gp.delta), "gamma": r6(gp.gamma), "vega": r6(gp.vega),
         "theta": r6(gp.theta), "rho": r6(gp.rho), "vanna": r6(gp.vanna), "volga": r6(gp.volga)},
        1e-8,
    ))

    # -- Garman-Kohlhagen FX ------------------------------------------------
    gk_in = {"s": 1.10, "k": 1.12, "t": 0.5, "sigma": 0.10, "rd": 0.03, "rf": 0.02, "type": "call"}
    gk_g = dpe.gk_greeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, "call")
    cases.append(_case(
        "gk_call_eurusd",
        gk_in,
        {"price": r6(dpe.gk_price(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, "call")),
         "delta_spot": r6(gk_g.delta),
         "delta_forward": r6(dpe.spot_delta_to_forward(gk_g.delta, 0.5, 0.02))},
        1e-8,
    ))
    cases.append(_case(
        "gk_put_negative_rate",
        {"s": 0.95, "k": 0.95, "t": 1.0, "sigma": 0.12, "rd": -0.005, "rf": 0.001, "type": "put"},
        {"price": r6(dpe.gk_price(0.95, 0.95, 1.0, 0.12, -0.005, 0.001, "put"))},
        1e-8,
    ))

    # -- Implied vol round-trips -------------------------------------------
    atm_price = dpe.bs_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    cases.append(_case(
        "iv_roundtrip_call_atm",
        {"price": r6(atm_price), "s": 100.0, "k": 100.0, "t": 1.0, "r": 0.05, "q": 0.0, "type": "call"},
        {"sigma": 0.2},
        1e-6,
    ))
    itm_put_price = dpe.bs_price(100.0, 80.0, 0.5, 0.35, 0.03, 0.01, "put")
    cases.append(_case(
        "iv_roundtrip_put_otm",
        {"price": r6(itm_put_price), "s": 100.0, "k": 80.0, "t": 0.5, "r": 0.03, "q": 0.01, "type": "put"},
        {"sigma": 0.35},
        1e-6,
    ))

    # -- Binomial trees -----------------------------------------------------
    def tree_case(name: str, s: float, k: float, t: float, sigma: float, r: float, q: float,
                  ot: str, style: str, steps: int, method: str, tol: float = 1e-8) -> None:
        price = dpe.binomial_price(s, k, t, sigma, r, q, ot, style, steps, method)
        cases.append(_case(
            name,
            {"s": s, "k": k, "t": t, "sigma": sigma, "r": r, "q": q,
             "type": ot, "style": style, "steps": steps, "method": method},
            {"price": r6(price)},
            tol,
        ))

    tree_case("crr_euro_call_2000", 100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call", "european", 2000, "crr")
    tree_case("crr_amer_put_500", 100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "put", "american", 500, "crr")
    tree_case("crr_amer_put_div_500", 100.0, 100.0, 1.0, 0.2, 0.05, 0.03, "put", "american", 500, "crr")
    tree_case("jr_euro_put_500", 100.0, 100.0, 1.0, 0.2, 0.05, 0.02, "put", "european", 500, "jr")

    # Sanity anchor for the American reference: the 500-step value must sit
    # within 2e-3 of a 5000-step reference (checked here, not stored).
    ref = dpe.binomial_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "put", "american", 5000, "crr")
    got = dpe.binomial_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "put", "american", 500, "crr")
    assert abs(ref - got) < 2e-3, "American put 500-step value drifted from 5000-step reference"

    bg = dpe.binomial_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call", "european", 500, "crr")
    cases.append(_case(
        "crr_greeks_call_500",
        {"s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2, "r": 0.05, "q": 0.0,
         "type": "call", "style": "european", "steps": 500, "method": "crr"},
        {"delta": r6(bg.delta), "gamma": r6(bg.gamma), "theta": r6(bg.theta)},
        1e-8,
    ))

    # -- Monte Carlo --------------------------------------------------------
    # Geometric Asian: closed form, exact across languages.
    cases.append(_case(
        "geo_asian_call_analytic",
        {"s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2, "r": 0.05, "q": 0.0,
         "type": "call", "n_fixings": 12},
        {"price": r6(dpe.geometric_asian_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call", 12))},
        1e-8,
    ))
    # MC estimates: ports use their own RNG, so the expected value is the
    # analytic/high-precision target and the tolerance is a few standard
    # errors at the stated path count (see API_SPEC.md).
    cases.append(_case(
        "mc_euro_call_100k",
        {"s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2, "r": 0.05, "q": 0.0,
         "type": "call", "n_paths": 100000, "seed": 42},
        {"price": r6(dpe.bs_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call"))},
        0.15,
    ))
    asian = dpe.mc_asian_arithmetic(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call",
        n_paths=200_000, n_steps=12, seed=42, antithetic=True, control_variate=True,
    )
    assert asian.std_error < 5e-3, f"Asian CV standard error unexpectedly large: {asian}"
    cases.append(_case(
        "mc_asian_call_cv_12fix",
        {"s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2, "r": 0.05, "q": 0.0,
         "type": "call", "n_paths": 100000, "n_steps": 12, "seed": 42},
        {"price": r6(asian.value)},
        0.05,
    ))

    # Single-monitoring-date exotics have closed forms, which anchors the
    # barrier and lookback path engines (knock-out convention, terminal-date
    # monitoring, extremum bookkeeping) across languages. With one step:
    #   up-and-out call, k < B:  C(k) - C(B) - (B - k) e^{-rt} N(d2(B))
    #   floating lookback call:  S_T - min(s, S_T) = (S_T - s)^+  -> ATM call
    #   floating lookback put:   max(s, S_T) - S_T = (s - S_T)^+  -> ATM put
    # Tolerances are ~4.5 standard errors at 200k paths (SE ~0.0134 for the
    # barrier, ~0.023 / ~0.015 for the lookback call / put) so a correct port
    # with its own RNG stream passes with overwhelming probability.
    s0, k0, t0, sig0, r0, q0, bar = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0, 130.0
    d2_b = dpe.bs_d1_d2(s0, bar, t0, sig0, r0, q0)[1]
    barrier_analytic = (
        dpe.bs_price(s0, k0, t0, sig0, r0, q0, "call")
        - dpe.bs_price(s0, bar, t0, sig0, r0, q0, "call")
        - (bar - k0) * math.exp(-r0 * t0) * dpe.norm_cdf(d2_b)
    )
    mc_bar = dpe.mc_barrier_up_out(s0, k0, t0, sig0, r0, q0, "call", bar,
                                   n_paths=200_000, n_steps=1, seed=3)
    assert abs(mc_bar.value - barrier_analytic) < 3 * mc_bar.std_error, mc_bar
    cases.append(_case(
        "mc_barrier_single_step_call",
        {"s": s0, "k": k0, "t": t0, "sigma": sig0, "r": r0, "q": q0, "type": "call",
         "barrier": bar, "n_paths": 200000, "n_steps": 1, "seed": 3},
        {"price": r6(barrier_analytic)},
        0.06,
    ))
    for ot, tol in (("call", 0.10), ("put", 0.07)):
        analytic = dpe.bs_price(s0, s0, t0, sig0, r0, q0, ot)  # ATM vanilla
        mc_lb = dpe.mc_lookback_floating(s0, t0, sig0, r0, q0, ot,
                                         n_paths=200_000, n_steps=1, seed=3)
        assert abs(mc_lb.value - analytic) < 3 * mc_lb.std_error, mc_lb
        cases.append(_case(
            f"mc_lookback_single_step_{ot}",
            {"s": s0, "t": t0, "sigma": sig0, "r": r0, "q": q0, "type": ot,
             "n_paths": 200000, "n_steps": 1, "seed": 3},
            {"price": r6(analytic)},
            tol,
        ))
    return cases


def write_golden() -> None:
    cases = build_golden_cases()
    out = DATA_DIR / "golden" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"cases": cases}, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out} with {len(cases)} cases")


def main() -> None:
    _validate_implementation()
    write_portfolio()
    write_spots_timeseries()
    write_golden()
    print("wrote portfolio.csv, spots_timeseries.csv")


if __name__ == "__main__":
    main()
