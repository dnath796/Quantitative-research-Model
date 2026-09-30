//! Golden-value suite: loads `../data/golden/golden.json` (shared with the
//! Python/C++/Java implementations) and asserts every case within its
//! absolute tolerance.
//!
//! Deterministic cases carry tight tolerances (1e-8 / 1e-10 / 1e-6) and must
//! match the reference up to floating-point noise. The two Monte Carlo cases
//! have statistical tolerances several standard errors wide; the JSON `seed`
//! is advisory (RNG streams are language-specific), and we use it as the
//! seed of our own RNG to keep the run deterministic.

use serde_json::Value;

use dpe::{
    binomial_greeks, binomial_price, bs_greeks, bs_price, geometric_asian_price, gk_greeks,
    gk_price, implied_vol, mc_asian_arithmetic, mc_barrier_up_out, mc_european,
    mc_lookback_floating, spot_delta_to_forward, ExerciseStyle, OptionType, TreeMethod,
};

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing/non-numeric key '{key}' in {v}"))
}

fn int(v: &Value, key: &str) -> usize {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("missing/non-integer key '{key}' in {v}")) as usize
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing/non-string key '{key}' in {v}"))
}

fn check(name: &str, key: &str, got: f64, expect: f64, tol: f64) {
    assert!(
        (got - expect).abs() <= tol,
        "golden case '{name}', key '{key}': got {got}, expect {expect} (tol {tol}, \
         diff {:.3e})",
        (got - expect).abs()
    );
}

#[test]
fn golden_suite() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/golden/golden.json");
    let raw = std::fs::read_to_string(path).expect("golden.json must exist at ../data/golden/");
    let root: Value = serde_json::from_str(&raw).expect("golden.json must parse");
    let cases = root["cases"].as_array().expect("cases array");
    // Not pinned to an exact count: adding a golden case must not break only
    // this suite. Every case is dispatched by name and an unknown name panics
    // (see the final `match` arm), so nothing can be silently skipped.
    assert!(cases.len() >= 20, "expected at least 20 golden cases, got {}", cases.len());
    let mut seen = std::collections::HashSet::new();
    for case in cases {
        assert!(seen.insert(text(case, "name")), "duplicate golden case name");
    }

    for case in cases {
        let name = text(case, "name");
        let inp = &case["inputs"];
        let exp = &case["expect"];
        let tol = num(case, "tol");
        run_case(name, inp, exp, tol);
    }
}

fn run_case(name: &str, inp: &Value, exp: &Value, tol: f64) {
    match name {
        // --- Garman-Kohlhagen -------------------------------------------
        "gk_call_eurusd" => {
            let (s, k, t, sigma) = (num(inp, "s"), num(inp, "k"), num(inp, "t"), num(inp, "sigma"));
            let (rd, rf) = (num(inp, "rd"), num(inp, "rf"));
            let ot = OptionType::parse(text(inp, "type")).unwrap();
            let price = gk_price(s, k, t, sigma, rd, rf, ot).unwrap();
            check(name, "price", price, num(exp, "price"), tol);
            let greeks = gk_greeks(s, k, t, sigma, rd, rf, ot).unwrap();
            check(name, "delta_spot", greeks.delta, num(exp, "delta_spot"), tol);
            let fwd = spot_delta_to_forward(greeks.delta, t, rf).unwrap();
            check(name, "delta_forward", fwd, num(exp, "delta_forward"), tol);
        }
        "gk_put_negative_rate" => {
            let price = gk_price(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "rd"),
                num(inp, "rf"),
                OptionType::parse(text(inp, "type")).unwrap(),
            )
            .unwrap();
            check(name, "price", price, num(exp, "price"), tol);
        }

        // --- Implied vol round-trips ------------------------------------
        "iv_roundtrip_call_atm" | "iv_roundtrip_put_otm" => {
            let sigma = implied_vol(
                num(inp, "price"),
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
            )
            .unwrap();
            check(name, "sigma", sigma, num(exp, "sigma"), tol);
        }

        // --- Binomial trees ---------------------------------------------
        "crr_euro_call_2000" | "crr_amer_put_500" | "crr_amer_put_div_500" | "jr_euro_put_500" => {
            let price = binomial_price(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                ExerciseStyle::parse(text(inp, "style")).unwrap(),
                int(inp, "steps"),
                TreeMethod::parse(text(inp, "method")).unwrap(),
                false,
            )
            .unwrap();
            check(name, "price", price, num(exp, "price"), tol);
        }
        "crr_greeks_call_500" => {
            let g = binomial_greeks(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                ExerciseStyle::parse(text(inp, "style")).unwrap(),
                int(inp, "steps"),
                TreeMethod::parse(text(inp, "method")).unwrap(),
            )
            .unwrap();
            check(name, "delta", g.delta, num(exp, "delta"), tol);
            check(name, "gamma", g.gamma, num(exp, "gamma"), tol);
            check(name, "theta", g.theta, num(exp, "theta"), tol);
        }

        // --- Asians ------------------------------------------------------
        "geo_asian_call_analytic" => {
            let price = geometric_asian_price(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                int(inp, "n_fixings"),
            )
            .unwrap();
            check(name, "price", price, num(exp, "price"), tol);
        }

        // --- Monte Carlo (statistical tolerance; seed advisory) ---------
        "mc_euro_call_100k" => {
            let res = mc_european(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                int(inp, "n_paths"),
                num(inp, "seed") as u64,
                true,
                false,
            )
            .unwrap();
            check(name, "price", res.value, num(exp, "price"), tol);
        }
        "mc_barrier_single_step_call" => {
            let res = mc_barrier_up_out(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                num(inp, "barrier"),
                int(inp, "n_paths"),
                int(inp, "n_steps"),
                num(inp, "seed") as u64,
                true,
            )
            .unwrap();
            check(name, "price", res.value, num(exp, "price"), tol);
        }
        "mc_lookback_single_step_call" | "mc_lookback_single_step_put" => {
            let res = mc_lookback_floating(
                num(inp, "s"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                int(inp, "n_paths"),
                int(inp, "n_steps"),
                num(inp, "seed") as u64,
                true,
            )
            .unwrap();
            check(name, "price", res.value, num(exp, "price"), tol);
        }
        "mc_asian_call_cv_12fix" => {
            let res = mc_asian_arithmetic(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
                int(inp, "n_paths"),
                int(inp, "n_steps"),
                num(inp, "seed") as u64,
                true,
                true,
            )
            .unwrap();
            check(name, "price", res.value, num(exp, "price"), tol);
        }

        // --- Analytic BSM Greeks ----------------------------------------
        n if n.starts_with("bs_greeks") => {
            let g = bs_greeks(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
            )
            .unwrap();
            for (key, got) in [
                ("delta", g.delta),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho", g.rho),
                ("vanna", g.vanna),
                ("volga", g.volga),
            ] {
                check(name, key, got, num(exp, key), tol);
            }
        }

        // --- Analytic BSM prices (all remaining bs_* cases) -------------
        n if n.starts_with("bs_") => {
            let price = bs_price(
                num(inp, "s"),
                num(inp, "k"),
                num(inp, "t"),
                num(inp, "sigma"),
                num(inp, "r"),
                num(inp, "q"),
                OptionType::parse(text(inp, "type")).unwrap(),
            )
            .unwrap();
            check(name, "price", price, num(exp, "price"), tol);
        }

        other => panic!("golden case '{other}' is not wired to any function under test"),
    }
}
