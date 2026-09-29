"""Golden-value suite: every case in data/golden/golden.json must reproduce
within its stated absolute tolerance. The same file drives the C++/Rust/Java
test suites, so this test is the cross-language contract for Python.

Dispatch is by case-name prefix, exactly as documented in API_SPEC.md."""

import json
from pathlib import Path

import pytest

import dpe

GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden" / "golden.json"

with open(GOLDEN) as _fh:
    CASES = json.load(_fh)["cases"]


def _compute(case: dict) -> dict:
    """Run the function a golden case exercises and return actual outputs."""
    name = case["name"]
    inp = case["inputs"]

    if name.startswith(("bs_greeks", "gk_")) and "delta" in case["expect"]:
        g = dpe.bs_greeks(inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"])
        return {k: getattr(g, k) for k in case["expect"]}
    if name.startswith("gk_"):
        rd, rf = inp["rd"], inp["rf"]
        out = {"price": dpe.gk_price(inp["s"], inp["k"], inp["t"], inp["sigma"], rd, rf, inp["type"])}
        if "delta_spot" in case["expect"]:
            g = dpe.gk_greeks(inp["s"], inp["k"], inp["t"], inp["sigma"], rd, rf, inp["type"])
            out["delta_spot"] = g.delta
            out["delta_forward"] = dpe.spot_delta_to_forward(g.delta, inp["t"], rf)
        return out
    if name.startswith("bs_"):
        return {"price": dpe.bs_price(inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"])}
    if name.startswith("iv_"):
        return {"sigma": dpe.implied_vol(inp["price"], inp["s"], inp["k"], inp["t"], inp["r"], inp["q"], inp["type"])}
    if name.startswith(("crr_", "jr_")) and "delta" in case["expect"]:
        g = dpe.binomial_greeks(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"],
            inp["type"], inp["style"], inp["steps"], inp["method"],
        )
        return {"delta": g.delta, "gamma": g.gamma, "theta": g.theta}
    if name.startswith(("crr_", "jr_")):
        return {"price": dpe.binomial_price(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"],
            inp["type"], inp["style"], inp["steps"], inp["method"],
        )}
    if name.startswith("geo_asian"):
        return {"price": dpe.geometric_asian_price(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"], inp["n_fixings"]
        )}
    if name.startswith("mc_asian"):
        res = dpe.mc_asian_arithmetic(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"],
            n_paths=inp["n_paths"], n_steps=inp["n_steps"], seed=inp["seed"],
            antithetic=True, control_variate=True,
        )
        return {"price": res.value}
    if name.startswith("mc_euro"):
        res = dpe.mc_european(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"],
            n_paths=inp["n_paths"], seed=inp["seed"], antithetic=True,
        )
        return {"price": res.value}
    if name.startswith("mc_barrier"):
        res = dpe.mc_barrier_up_out(
            inp["s"], inp["k"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"],
            inp["barrier"], n_paths=inp["n_paths"], n_steps=inp["n_steps"], seed=inp["seed"],
            antithetic=True,
        )
        return {"price": res.value}
    if name.startswith("mc_lookback"):
        res = dpe.mc_lookback_floating(
            inp["s"], inp["t"], inp["sigma"], inp["r"], inp["q"], inp["type"],
            n_paths=inp["n_paths"], n_steps=inp["n_steps"], seed=inp["seed"], antithetic=True,
        )
        return {"price": res.value}
    raise AssertionError(f"no dispatch rule for golden case {name!r}")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_case(case: dict) -> None:
    actual = _compute(case)
    assert set(actual) == set(case["expect"]), case["name"]
    for key, expected in case["expect"].items():
        assert actual[key] == pytest.approx(expected, abs=case["tol"]), (case["name"], key)


def test_golden_file_shape() -> None:
    """The schema contract: flat scalars only, positive tolerances, unique names."""
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names))
    assert len(CASES) >= 20
    for c in CASES:
        assert set(c) == {"name", "inputs", "expect", "tol"}
        assert c["tol"] > 0.0
        for section in ("inputs", "expect"):
            for v in c[section].values():
                assert isinstance(v, (int, float, str)) and not isinstance(v, bool)
