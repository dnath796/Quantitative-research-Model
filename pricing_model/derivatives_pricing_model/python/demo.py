"""End-to-end demo of the dpe pricing engine.

Run:  cd python && PYTHONPATH=src python3 demo.py

Sections
--------
1. Analytic BSM / Garman-Kohlhagen prices and full Greeks.
2. Implied-vol round-trip.
3. Binomial trees: convergence to BS, American early-exercise premium.
4. Monte Carlo: variance-reduction comparison, exotics with 95% CIs,
   pathwise vs CRN finite-difference delta.
5. Historical vol from data/spots_timeseries.csv.
6. Risk report for the small book in data/portfolio.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import dpe

DATA = Path(__file__).resolve().parents[1] / "data"
RULE = "-" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def demo_analytic() -> None:
    section("1. Analytic Black-Scholes-Merton and Garman-Kohlhagen")
    rows = [
        ("Equity ATM call (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, "call"),
        ("Equity ATM put  (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, "put"),
        ("High-vol small cap call", 18.0, 20.0, 1.0, 0.80, 0.04, 0.00, "call"),
        ("Weekly put (T=1/52)", 100.0, 98.0, 1 / 52, 0.25, 0.03, 0.00, "put"),
        ("LEAPS call (T=3y)", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, "call"),
        ("FX EURUSD call (rd,rf)", 1.10, 1.12, 0.5, 0.10, 0.03, 0.02, "call"),
        ("FX neg-rate put (rd<0)", 0.95, 0.95, 1.0, 0.12, -0.005, 0.001, "put"),
    ]
    hdr = f"{'instrument':<24}{'price':>10}{'delta':>9}{'gamma':>9}{'vega':>9}{'theta':>9}{'rho':>9}{'vanna':>9}{'volga':>9}"
    print(hdr)
    for name, s, k, t, sig, r, q, ot in rows:
        g = dpe.bs_greeks(s, k, t, sig, r, q, ot)
        print(
            f"{name:<24}{g.price:>10.4f}{g.delta:>9.4f}{g.gamma:>9.4f}"
            f"{g.vega:>9.4f}{g.theta:>9.4f}{g.rho:>9.4f}{g.vanna:>9.4f}{g.volga:>9.4f}"
        )
    g = dpe.gk_greeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, "call")
    fwd_delta = dpe.spot_delta_to_forward(g.delta, 0.5, 0.02)
    print(
        f"\nFX delta conventions (EURUSD call): spot delta = {g.delta:.4f}, "
        f"forward delta = {fwd_delta:.4f}, forward = {dpe.fx_forward(1.10, 0.5, 0.03, 0.02):.4f}"
    )


def demo_implied_vol() -> None:
    section("2. Implied volatility round-trip")
    for sigma_in, k in ((0.15, 120.0), (0.40, 80.0)):
        price = dpe.bs_price(100.0, k, 0.5, sigma_in, 0.03, 0.01, "call")
        iv = dpe.implied_vol(price, 100.0, k, 0.5, 0.03, 0.01, "call")
        print(f"k={k:>6.1f}  price={price:>9.4f}  sigma_in={sigma_in:.4f}  implied={iv:.10f}")
    try:
        dpe.implied_vol(1.0, 100.0, 80.0, 0.5, 0.03, 0.0, "call")
    except ValueError as exc:
        print(f"below-intrinsic price correctly rejected: {exc}")


def demo_binomial() -> None:
    section("3. Binomial trees (CRR / Jarrow-Rudd)")
    s, k, t, sig, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    bs = dpe.bs_price(s, k, t, sig, r, q, "call")
    print(f"{'steps':>6} {'CRR-BS error':>14} {'CRR+Richardson':>16} {'JR-BS error':>13}")
    for n in (50, 200, 800, 2000):
        crr = dpe.binomial_price(s, k, t, sig, r, q, "call", "european", n, "crr")
        rich = dpe.binomial_price(s, k, t, sig, r, q, "call", "european", n, "crr", richardson=True)
        jr = dpe.binomial_price(s, k, t, sig, r, q, "call", "european", n, "jr")
        print(f"{n:>6} {crr - bs:>14.2e} {rich - bs:>16.2e} {jr - bs:>13.2e}")
    print(f"\nAmerican put early-exercise premium (s=100, T=1, sigma=20%, r=5%):")
    print(f"{'strike':>8}{'european':>12}{'american':>12}{'premium':>10}")
    for kk in (90.0, 100.0, 110.0, 120.0):
        eur = dpe.binomial_price(s, kk, t, sig, r, 0.0, "put", "european", 500)
        amer = dpe.binomial_price(s, kk, t, sig, r, 0.0, "put", "american", 500)
        print(f"{kk:>8.1f}{eur:>12.4f}{amer:>12.4f}{amer - eur:>10.4f}")
    bg = dpe.binomial_greeks(s, k, t, sig, r, q, "call", "european", 500, "crr")
    ag = dpe.bs_greeks(s, k, t, sig, r, q, "call")
    print(
        f"\ntree Greeks (500-step CRR) vs analytic: delta {bg.delta:.4f}/{ag.delta:.4f}, "
        f"gamma {bg.gamma:.4f}/{ag.gamma:.4f}, theta {bg.theta:.4f}/{ag.theta:.4f}"
    )


def demo_monte_carlo() -> None:
    section("4. Monte Carlo (100k paths, seed 42)")
    s, k, t, sig, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    n = 100_000
    bs = dpe.bs_price(s, k, t, sig, r, q, "call")
    plain = dpe.mc_european(s, k, t, sig, r, q, "call", n, 42, antithetic=False)
    anti = dpe.mc_european(s, k, t, sig, r, q, "call", n, 42, antithetic=True)
    cv = dpe.mc_european(s, k, t, sig, r, q, "call", n, 42, antithetic=True, control_variate=True)
    print(f"European call, BS analytic = {bs:.4f}")
    print(f"{'estimator':<28}{'price':>10}{'std err':>10}{'95% CI':>24}")
    for label, res in (("plain", plain), ("antithetic", anti), ("antithetic + control", cv)):
        print(
            f"{label:<28}{res.value:>10.4f}{res.std_error:>10.4f}"
            f"{'[' + format(res.ci_low, '.4f') + ', ' + format(res.ci_high, '.4f') + ']':>24}"
        )
    asian = dpe.mc_asian_arithmetic(s, k, t, sig, r, q, "call", n, 12, 42)
    barrier = dpe.mc_barrier_up_out(s, k, t, sig, r, q, "call", 130.0, n, 100, 42)
    lookback = dpe.mc_lookback_floating(s, t, sig, r, q, "call", n, 100, 42)
    print(f"\n{'exotic':<34}{'price':>10}{'std err':>10}{'95% CI':>24}")
    for label, res in (
        ("arithmetic Asian (12 fix, CV)", asian),
        ("up-and-out barrier B=130", barrier),
        ("floating-strike lookback", lookback),
    ):
        print(
            f"{label:<34}{res.value:>10.4f}{res.std_error:>10.4f}"
            f"{'[' + format(res.ci_low, '.4f') + ', ' + format(res.ci_high, '.4f') + ']':>24}"
        )
    pw = dpe.mc_delta_pathwise(s, k, t, sig, r, q, "call", n, 42)
    fd = dpe.mc_delta_fd_crn(s, k, t, sig, r, q, "call", n, 42)
    print(
        f"\ndelta: analytic {dpe.bs_greeks(s, k, t, sig, r, q, 'call').delta:.4f}, "
        f"pathwise {pw.value:.4f} (se {pw.std_error:.4f}), "
        f"FD+CRN {fd.value:.4f} (se {fd.std_error:.4f})"
    )


def demo_historical_vol() -> pd.DataFrame:
    section("5. Historical volatility from data/spots_timeseries.csv")
    ts = pd.read_csv(DATA / "spots_timeseries.csv", parse_dates=["date"])
    print(f"{'underlier':<10}{'last close':>12}{'realised vol':>14}")
    for col in ts.columns[1:]:
        vol = dpe.historical_vol(ts[col].to_numpy())
        print(f"{col:<10}{ts[col].iloc[-1]:>12.4f}{vol:>14.2%}")
    return ts


def demo_portfolio() -> None:
    section("6. Portfolio risk report (data/portfolio.csv)")
    book = pd.read_csv(DATA / "portfolio.csv")
    print(f"{'id':<16}{'typ':<4}{'style':<5}{'price':>10}{'delta':>9}{'gamma':>9}{'vega':>10}{'theta':>10}{'position value':>16}")
    total = 0.0
    for row in book.itertuples(index=False):
        if row.style == "american":
            price = dpe.binomial_price(
                row.s, row.k, row.t, row.sigma, row.r, row.q, row.type, "american", 500
            )
            g = dpe.binomial_greeks(
                row.s, row.k, row.t, row.sigma, row.r, row.q, row.type, "american", 500
            )
            # The tree reports no vega: print "n/a" rather than a NaN, so the
            # report never contains a NaN (API_SPEC section 8).
            delta, gamma, vega, theta = g.delta, g.gamma, "n/a", g.theta
        else:
            g = dpe.bs_greeks(row.s, row.k, row.t, row.sigma, row.r, row.q, row.type)
            price, delta, gamma, vega, theta = g.price, g.delta, g.gamma, g.vega, g.theta
        value = price * row.quantity
        total += value
        vega_txt = f"{vega:>10.4f}" if isinstance(vega, float) else f"{vega:>10}"
        print(
            f"{row.id:<16}{row.type[0].upper():<4}{row.style[:4]:<5}{price:>10.4f}"
            f"{delta:>9.4f}{gamma:>9.4f}{vega_txt}{theta:>10.4f}{value:>16,.2f}"
        )
    print(f"\n{'total book value':<57}{total:>16,.2f}")
    print("(FX rows: r=rd, q=rf; price in domestic ccy per unit foreign; vega n/a for tree)")


def main() -> None:
    print("dpe — Derivatives Pricing Engine demo (Python reference implementation)")
    demo_analytic()
    demo_implied_vol()
    demo_binomial()
    demo_monte_carlo()
    demo_historical_vol()
    demo_portfolio()
    print(f"\n{RULE}\ndone.\n")


if __name__ == "__main__":
    main()
