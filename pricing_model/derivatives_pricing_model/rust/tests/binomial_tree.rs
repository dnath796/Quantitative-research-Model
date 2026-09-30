//! Binomial-tree tests: convergence to Black-Scholes, Richardson averaging,
//! American exercise properties, tree Greeks, degenerate limits and invalid
//! inputs.

use dpe::{
    binomial_greeks, binomial_price, bs_greeks, bs_price, ExerciseStyle, OptionType, TreeMethod,
};

const CALL: OptionType = OptionType::Call;
const PUT: OptionType = OptionType::Put;
const EURO: ExerciseStyle = ExerciseStyle::European;
const AMER: ExerciseStyle = ExerciseStyle::American;
const CRR: TreeMethod = TreeMethod::Crr;
const JR: TreeMethod = TreeMethod::Jr;

#[test]
fn crr_and_jr_converge_to_black_scholes() {
    let (s, k, t, sigma, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    let bs = bs_price(s, k, t, sigma, r, q, CALL).unwrap();
    let crr = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 2000, CRR, false).unwrap();
    assert!((crr - bs).abs() < 1e-3, "CRR 2000-step error {}", crr - bs);
    let jr = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 2000, JR, false).unwrap();
    assert!((jr - bs).abs() < 1e-3, "JR 2000-step error {}", jr - bs);
    // Error should shrink with steps (allow the odd/even wobble a factor).
    let coarse = (binomial_price(s, k, t, sigma, r, q, CALL, EURO, 50, CRR, false).unwrap() - bs)
        .abs();
    assert!((crr - bs).abs() < coarse);
}

#[test]
fn richardson_averaging_damps_oscillation() {
    let (s, k, t, sigma, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    let bs = bs_price(s, k, t, sigma, r, q, CALL).unwrap();
    let plain = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 200, CRR, false).unwrap();
    let rich = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 200, CRR, true).unwrap();
    assert!(
        (rich - bs).abs() < (plain - bs).abs(),
        "Richardson {rich} should beat plain {plain} vs BS {bs}"
    );
    // And it must equal the average of the n and n+1 prices by definition.
    let n1 = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 201, CRR, false).unwrap();
    assert!((rich - 0.5 * (plain + n1)).abs() < 1e-14);
}

#[test]
fn american_dominates_european_and_premium_increases_with_strike() {
    let (s, t, sigma, r, q) = (100.0, 1.0, 0.2, 0.05, 0.0);
    let mut prev_premium = 0.0;
    for &k in &[90.0, 100.0, 110.0, 120.0] {
        let eur = binomial_price(s, k, t, sigma, r, q, PUT, EURO, 500, CRR, false).unwrap();
        let amer = binomial_price(s, k, t, sigma, r, q, PUT, AMER, 500, CRR, false).unwrap();
        let premium = amer - eur;
        assert!(premium > 0.0, "early-exercise premium not positive at k={k}");
        assert!(
            premium > prev_premium,
            "premium not increasing with k at k={k}"
        );
        prev_premium = premium;
    }
    // American call on a zero-dividend stock is never exercised early:
    // its tree value must coincide with the European one.
    let eur_c = binomial_price(s, 100.0, t, sigma, r, 0.0, CALL, EURO, 300, CRR, false).unwrap();
    let amer_c = binomial_price(s, 100.0, t, sigma, r, 0.0, CALL, AMER, 300, CRR, false).unwrap();
    assert!((amer_c - eur_c).abs() < 1e-12);
    // And American >= European holds for calls with dividends too.
    let eur_cd = binomial_price(s, 100.0, t, sigma, r, 0.04, CALL, EURO, 300, CRR, false).unwrap();
    let amer_cd = binomial_price(s, 100.0, t, sigma, r, 0.04, CALL, AMER, 300, CRR, false).unwrap();
    assert!(amer_cd >= eur_cd);
}

#[test]
fn tree_greeks_agree_with_analytic() {
    let (s, k, t, sigma, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    let tree = binomial_greeks(s, k, t, sigma, r, q, CALL, EURO, 500, CRR).unwrap();
    let ana = bs_greeks(s, k, t, sigma, r, q, CALL).unwrap();
    assert!((tree.delta - ana.delta).abs() < 5e-4);
    assert!((tree.gamma - ana.gamma).abs() < 5e-4);
    assert!((tree.theta - ana.theta).abs() < 2e-2);
    assert!((tree.price - ana.price).abs() < 5e-3);
    // JR Greeks carry the ds spot-displacement correction; they must agree
    // with the analytic values too.
    let jr = binomial_greeks(s, k, t, sigma, r, q, CALL, EURO, 500, JR).unwrap();
    assert!((jr.delta - ana.delta).abs() < 5e-3);
    assert!((jr.theta - ana.theta).abs() < 5e-2);
    // Smallest legal tree for Greeks (level 2 = terminal) must not fail.
    assert!(binomial_greeks(s, k, t, sigma, r, q, CALL, EURO, 2, CRR).is_ok());
}

#[test]
fn degenerate_limits() {
    let (s, r, q) = (100.0, 0.05, 0.02);
    // t = 0: intrinsic, both styles and methods.
    assert_eq!(
        binomial_price(90.0, 100.0, 0.0, 0.2, r, q, PUT, AMER, 100, CRR, false).unwrap(),
        10.0
    );
    // sigma = 0 European: discounted forward intrinsic = BS sigma-0 limit.
    let bs0 = bs_price(s, 90.0, 1.0, 0.0, r, q, CALL).unwrap();
    let tree0 = binomial_price(s, 90.0, 1.0, 0.0, r, q, CALL, EURO, 100, CRR, false).unwrap();
    assert!((tree0 - bs0).abs() < 1e-14);
    // sigma = 0 American put with r > 0: immediate exercise (t_0) is optimal
    // when deep ITM, so the value is the undiscounted intrinsic today.
    let amer0 = binomial_price(50.0, 100.0, 1.0, 0.0, 0.05, 0.0, PUT, AMER, 100, CRR, false)
        .unwrap();
    assert!((amer0 - 50.0).abs() < 1e-14);
    // ... and always at least the European value.
    let eur0 = binomial_price(50.0, 100.0, 1.0, 0.0, 0.05, 0.0, PUT, EURO, 100, CRR, false)
        .unwrap();
    assert!(amer0 >= eur0);
}

#[test]
fn invalid_inputs_are_rejected() {
    let (s, k, t, sigma, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    // steps < 1.
    assert!(binomial_price(s, k, t, sigma, r, q, CALL, EURO, 0, CRR, false).is_err());
    // Tree Greeks need steps >= 2 and a non-degenerate lattice.
    assert!(binomial_greeks(s, k, t, sigma, r, q, CALL, EURO, 1, CRR).is_err());
    assert!(binomial_greeks(s, k, 0.0, sigma, r, q, CALL, EURO, 500, CRR).is_err());
    assert!(binomial_greeks(s, k, t, 0.0, r, q, CALL, EURO, 500, CRR).is_err());
    // Market-input validation flows through.
    assert!(binomial_price(-1.0, k, t, sigma, r, q, CALL, EURO, 100, CRR, false).is_err());
    assert!(binomial_price(s, k, -1.0, sigma, r, q, CALL, EURO, 100, CRR, false).is_err());
    // Drift dominating vol at one step: p outside (0,1) is an error, not a
    // silent clamp.
    assert!(binomial_price(s, k, 1.0, 0.01, 1.5, 0.0, CALL, EURO, 1, CRR, false).is_err());
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

#[test]
fn tree_greeks_two_steps_are_finite() {
    // steps = 2 is the smallest tree the contract allows for Greeks (level 2
    // is the terminal level); every port must return finite values.
    for (ot, style, method) in [(CALL, EURO, CRR), (CALL, EURO, JR), (PUT, AMER, CRR)] {
        let g = binomial_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, ot, style, 2, method).unwrap();
        assert!(g.price.is_finite() && g.delta.is_finite() && g.gamma.is_finite());
        assert!(g.theta.is_finite());
        assert!(g.gamma > 0.0);
        if ot == CALL {
            assert!(g.delta > 0.0 && g.delta < 1.0);
        } else {
            assert!(g.delta > -1.0 && g.delta < 0.0);
        }
        let price =
            binomial_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, ot, style, 2, method, false).unwrap();
        assert_eq!(g.price, price);
    }
}

#[test]
fn crr_tree_put_call_parity_exact() {
    // CRR matches the first moment exactly, so tree call - put equals
    // s e^{-qt} - k e^{-rt} to round-off at every step count; JR does not
    // (its parity error is O(dt)).
    let (s, k, t, sigma, r, q) = (100.0_f64, 100.0_f64, 1.0_f64, 0.2, 0.05, 0.02);
    let fwd = s * (-q * t).exp() - k * (-r * t).exp();
    for n in [1usize, 2, 101, 500] {
        let c = binomial_price(s, k, t, sigma, r, q, CALL, EURO, n, CRR, false).unwrap();
        let p = binomial_price(s, k, t, sigma, r, q, PUT, EURO, n, CRR, false).unwrap();
        assert!(((c - p) - fwd).abs() < 1e-10, "CRR parity at n={n}: {}", (c - p) - fwd);
    }
    let cj = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 101, JR, false).unwrap();
    let pj = binomial_price(s, k, t, sigma, r, q, PUT, EURO, 101, JR, false).unwrap();
    assert!(((cj - pj) - fwd).abs() < 1e-3);
    assert!(((cj - pj) - fwd).abs() > 1e-6, "JR is not a martingale lattice");
}

#[test]
fn crr_one_step_closed_form() {
    // One CRR step is a hand-computable two-state model.
    let (s, k, t, sigma, r, q) = (100.0_f64, 100.0_f64, 1.0_f64, 0.2_f64, 0.05_f64, 0.02_f64);
    let u = (sigma * t.sqrt()).exp();
    let d = 1.0 / u;
    let p = (((r - q) * t).exp() - d) / (u - d);
    let expected = (-r * t).exp() * (p * (s * u - k).max(0.0) + (1.0 - p) * (s * d - k).max(0.0));
    assert!((expected - 11.073540703840242).abs() < 1e-12); // derived independently
    let got = binomial_price(s, k, t, sigma, r, q, CALL, EURO, 1, CRR, false).unwrap();
    assert!((got - expected).abs() < 1e-12, "got {got}, expected {expected}");
}

#[test]
fn american_call_with_dividend_strictly_exceeds_european() {
    // q > r: dividend leakage on a deep-ITM call makes early exercise valuable.
    let eur = binomial_price(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, CALL, EURO, 500, CRR, false).unwrap();
    let amer = binomial_price(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, CALL, AMER, 500, CRR, false).unwrap();
    assert!(amer - eur > 1.0, "premium {}", amer - eur);
    assert!(amer >= 20.0); // never below immediate intrinsic
}

#[test]
fn tree_overflow_guard() {
    // sigma sqrt(t steps) = 707 > 700 would overflow the terminal spots to
    // inf; the contract says invalid-input error, never inf/NaN.
    let err = binomial_price(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, CALL, EURO, 200, CRR, false)
        .unwrap_err();
    assert!(matches!(&err, dpe::DpeError::InvalidInput(m) if m.contains("overflow")), "{err:?}");
    assert!(binomial_greeks(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, CALL, EURO, 200, JR).is_err());
    // Just inside the guard (4.6 + 200 + 400 = 605 < 700) is finite.
    let v = binomial_price(100.0, 100.0, 1.0, 20.0, 0.05, 0.0, CALL, EURO, 400, CRR, false).unwrap();
    assert!(v.is_finite() && v > 0.0 && v < 100.0);
}

#[test]
fn expiry_price_is_positive_zero_on_tie() {
    let v = binomial_price(100.0, 100.0, 0.0, 0.2, 0.05, 0.0, PUT, EURO, 10, CRR, false).unwrap();
    assert_eq!(v, 0.0);
    assert!(!v.is_sign_negative());
}
