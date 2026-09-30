//! Monte Carlo tests: statistical agreement with analytic anchors, variance
//! reduction, determinism, exotic-payoff ordering and invalid inputs.
//!
//! Path counts are kept modest (10k-50k) so the suite stays fast; the
//! full-size 100k-path runs live in the golden suite.

use dpe::{
    bs_greeks, bs_price, geometric_asian_price, mc_asian_arithmetic, mc_barrier_up_out,
    mc_delta_fd_crn, mc_delta_pathwise, mc_european, mc_lookback_floating, simulate_gbm_paths,
    OptionType,
};

const CALL: OptionType = OptionType::Call;
const PUT: OptionType = OptionType::Put;

const S: f64 = 100.0;
const K: f64 = 100.0;
const T: f64 = 1.0;
const SIG: f64 = 0.2;
const R: f64 = 0.05;
const Q: f64 = 0.0;

#[test]
fn european_within_three_standard_errors_of_bs() {
    let bs = bs_price(S, K, T, SIG, R, Q, CALL).unwrap();
    let res = mc_european(S, K, T, SIG, R, Q, CALL, 50_000, 42, true, false).unwrap();
    assert!(
        (res.value - bs).abs() < 3.0 * res.std_error,
        "MC {} vs BS {bs}, SE {}",
        res.value,
        res.std_error
    );
    // CI is the contractual 1.959963984540054 multiplier.
    assert!((res.ci_high - res.value - 1.959_963_984_540_054 * res.std_error).abs() < 1e-12);
    assert!((res.value - res.ci_low - 1.959_963_984_540_054 * res.std_error).abs() < 1e-12);
    assert_eq!(res.n_paths, 50_000);
    // Put too.
    let bsp = bs_price(S, K, T, SIG, R, Q, PUT).unwrap();
    let resp = mc_european(S, K, T, SIG, R, Q, PUT, 50_000, 42, true, false).unwrap();
    assert!((resp.value - bsp).abs() < 3.0 * resp.std_error);
}

#[test]
fn variance_reduction_reduces_standard_error() {
    let n = 20_000;
    let plain = mc_european(S, K, T, SIG, R, Q, CALL, n, 7, false, false).unwrap();
    let anti = mc_european(S, K, T, SIG, R, Q, CALL, n, 7, true, false).unwrap();
    let cv = mc_european(S, K, T, SIG, R, Q, CALL, n, 7, true, true).unwrap();
    assert!(anti.std_error < plain.std_error);
    assert!(cv.std_error < anti.std_error);
    // Asian: geometric control variate cuts the SE by a large factor.
    let no_cv = mc_asian_arithmetic(S, K, T, SIG, R, Q, CALL, n, 12, 7, true, false).unwrap();
    let with_cv = mc_asian_arithmetic(S, K, T, SIG, R, Q, CALL, n, 12, 7, true, true).unwrap();
    assert!(with_cv.std_error < 0.25 * no_cv.std_error);
}

#[test]
fn deterministic_given_seed() {
    let a = mc_european(S, K, T, SIG, R, Q, CALL, 10_000, 123, true, true).unwrap();
    let b = mc_european(S, K, T, SIG, R, Q, CALL, 10_000, 123, true, true).unwrap();
    assert_eq!(a.value, b.value);
    assert_eq!(a.std_error, b.std_error);
    let c = mc_european(S, K, T, SIG, R, Q, CALL, 10_000, 124, true, true).unwrap();
    assert_ne!(a.value, c.value);
}

#[test]
fn exotic_payoff_ordering() {
    let n = 20_000;
    let vanilla = bs_price(S, K, T, SIG, R, Q, CALL).unwrap();
    // Barrier knocks out on some paths: up-and-out <= vanilla.
    let barrier = mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 130.0, n, 50, 42, true).unwrap();
    assert!(barrier.value < vanilla);
    assert!(barrier.value > 0.0);
    // Averaging dampens vol: Asian <= vanilla (well beyond MC noise).
    let asian = mc_asian_arithmetic(S, K, T, SIG, R, Q, CALL, n, 12, 42, true, true).unwrap();
    assert!(asian.value + 3.0 * asian.std_error < vanilla);
    // Geometric mean <= arithmetic mean, so geo-Asian <= arith-Asian.
    let geo = geometric_asian_price(S, K, T, SIG, R, Q, CALL, 12).unwrap();
    assert!(geo < asian.value + 3.0 * asian.std_error);
    // Lookback pays a non-negative payoff at least the vanilla's (the
    // observed minimum is <= K here only statistically; just require > 0
    // and a sane magnitude).
    let lb = mc_lookback_floating(S, T, SIG, R, Q, CALL, n, 50, 42, true).unwrap();
    assert!(lb.value > 0.0);
    assert!(lb.value > vanilla * 0.5);
}

#[test]
fn geometric_asian_analytic_limits() {
    // n_fixings = 1 makes the geometric Asian a plain European on S_T.
    let geo1 = geometric_asian_price(S, K, T, SIG, R, Q, CALL, 1).unwrap();
    let bs = bs_price(S, K, T, SIG, R, Q, CALL).unwrap();
    assert!((geo1 - bs).abs() < 1e-12);
    // sigma = 0: discounted deterministic-average intrinsic.
    let geo0 = geometric_asian_price(S, K, T, 0.0, R, Q, CALL, 12).unwrap();
    let n = 12.0;
    let eg = (S.ln() + R * T * (n + 1.0) / (2.0 * n)).exp();
    assert!((geo0 - (-R * T).exp() * (eg - K).max(0.0)).abs() < 1e-12);
    // MC arithmetic Asian with CV agrees with the geometric price direction:
    // arithmetic mean >= geometric mean pathwise, so price(arith) >= price(geo).
    let mc = mc_asian_arithmetic(S, K, T, SIG, R, Q, CALL, 20_000, 12, 42, true, true).unwrap();
    let geo = geometric_asian_price(S, K, T, SIG, R, Q, CALL, 12).unwrap();
    assert!(mc.value > geo);
}

#[test]
fn barrier_born_dead_and_bias_direction() {
    // s >= barrier: exactly zero without simulating.
    let dead = mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 90.0, 1000, 50, 42, true).unwrap();
    assert_eq!(
        (dead.value, dead.std_error, dead.ci_low, dead.ci_high),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(dead.n_paths, 1000);
    // Coarser monitoring sees fewer breaches, so the discrete up-and-out
    // price must (statistically) decrease as n_steps grows.
    let coarse = mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 120.0, 20_000, 4, 42, true).unwrap();
    let fine = mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 120.0, 20_000, 100, 42, true).unwrap();
    assert!(coarse.value > fine.value);
}

#[test]
fn pathwise_and_crn_fd_delta_match_analytic() {
    let delta = bs_greeks(S, K, T, SIG, R, Q, CALL).unwrap().delta;
    let pw = mc_delta_pathwise(S, K, T, SIG, R, Q, CALL, 50_000, 42, true).unwrap();
    assert!((pw.value - delta).abs() < 3.0 * pw.std_error.max(1e-3));
    let fd = mc_delta_fd_crn(S, K, T, SIG, R, Q, CALL, 50_000, 42, 1e-4, true).unwrap();
    assert!((fd.value - delta).abs() < 3.0 * fd.std_error.max(1e-3));
    // Put deltas are negative.
    let pwp = mc_delta_pathwise(S, K, T, SIG, R, Q, PUT, 50_000, 42, true).unwrap();
    let deltap = bs_greeks(S, K, T, SIG, R, Q, PUT).unwrap().delta;
    assert!(pwp.value < 0.0);
    assert!((pwp.value - deltap).abs() < 3.0 * pwp.std_error.max(1e-3));
}

#[test]
fn simulated_paths_shape_and_antithetic_mirror() {
    let n_paths = 8;
    let n_steps = 5;
    let paths = simulate_gbm_paths(S, T, SIG, R, Q, n_paths, n_steps, 9, true).unwrap();
    assert_eq!(paths.len(), n_paths);
    for p in &paths {
        assert_eq!(p.len(), n_steps + 1);
        assert_eq!(p[0], S);
        assert!(p.iter().all(|v| v.is_finite() && *v > 0.0));
    }
    // Antithetic rows mirror the log-increments around the drift:
    // ln(S_i+1/S_i) - drift for row j equals -(same) for row j + n/2.
    let dt = T / n_steps as f64;
    let drift = (R - Q - 0.5 * SIG * SIG) * dt;
    for j in 0..n_paths / 2 {
        for i in 0..n_steps {
            let inc_a = (paths[j][i + 1] / paths[j][i]).ln() - drift;
            let inc_b = (paths[j + n_paths / 2][i + 1] / paths[j + n_paths / 2][i]).ln() - drift;
            assert!(
                (inc_a + inc_b).abs() < 1e-12,
                "row {j} step {i}: {inc_a} vs {inc_b}"
            );
        }
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    // n_paths constraints.
    assert!(mc_european(S, K, T, SIG, R, Q, CALL, 1, 42, false, false).is_err());
    assert!(mc_european(S, K, T, SIG, R, Q, CALL, 101, 42, true, false).is_err()); // odd + antithetic
    assert!(mc_european(S, K, T, SIG, R, Q, CALL, 2, 42, true, false).is_err()); // < 4 + antithetic
    // t must be strictly positive for every MC pricer.
    assert!(mc_european(S, K, 0.0, SIG, R, Q, CALL, 100, 42, false, false).is_err());
    assert!(mc_lookback_floating(S, 0.0, SIG, R, Q, CALL, 100, 10, 42, false).is_err());
    // n_steps >= 1.
    assert!(simulate_gbm_paths(S, T, SIG, R, Q, 10, 0, 42, false).is_err());
    // barrier > 0, k > 0 for the Asian, rel_bump > 0.
    assert!(mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 0.0, 100, 10, 42, false).is_err());
    assert!(mc_asian_arithmetic(S, 0.0, T, SIG, R, Q, CALL, 100, 10, 42, false, true).is_err());
    assert!(mc_delta_fd_crn(S, K, T, SIG, R, Q, CALL, 100, 42, 0.0, false).is_err());
    // n_fixings >= 1 and k > 0 for the geometric Asian closed form.
    assert!(geometric_asian_price(S, K, T, SIG, R, Q, CALL, 0).is_err());
    assert!(geometric_asian_price(S, 0.0, T, SIG, R, Q, CALL, 12).is_err());
    // Market validation flows through.
    assert!(mc_european(-1.0, K, T, SIG, R, Q, CALL, 100, 42, false, false).is_err());
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

#[test]
fn seed_domain_is_pinned() {
    // Seeds are integers in [0, 2^63 - 1] in every port. Rust's u64 cannot be
    // negative, so the only out-of-domain values are those above i64::MAX;
    // they are an invalid-input error naming `seed`, not a Rust-only extra.
    for bad in [dpe::SEED_MAX + 1, u64::MAX] {
        let err = mc_european(S, K, T, SIG, R, Q, CALL, 1000, bad, true, false).unwrap_err();
        assert!(matches!(&err, dpe::DpeError::InvalidInput(m) if m.contains("seed")), "{err:?}");
        assert!(simulate_gbm_paths(S, T, SIG, R, Q, 10, 2, bad, false).is_err());
        assert!(mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, 130.0, 100, 4, bad, true).is_err());
    }
    // Both endpoints of the domain are accepted and give different streams.
    let lo = mc_european(S, K, T, SIG, R, Q, CALL, 1000, 0, true, false).unwrap();
    let hi = mc_european(S, K, T, SIG, R, Q, CALL, 1000, dpe::SEED_MAX, true, false).unwrap();
    assert!(lo.value.is_finite() && hi.value.is_finite() && lo.value != hi.value);
}

#[test]
fn rel_bump_domain() {
    // rel_bump >= 1 would price a non-positive spot silently; the contract
    // pins the open interval (0, 1).
    for bad in [1.0, 2.0, 0.0, -1e-4, f64::NAN, f64::INFINITY] {
        let err = mc_delta_fd_crn(S, K, T, SIG, R, Q, CALL, 1000, 42, bad, true).unwrap_err();
        assert!(matches!(&err, dpe::DpeError::InvalidInput(m) if m.contains("rel_bump")), "{err:?}");
    }
}

#[test]
fn asian_single_fixing_equals_european() {
    // One fixing: the arithmetic Asian IS the vanilla, and both engines
    // consume the same draws in the same order -> bit-identical results.
    let asian = mc_asian_arithmetic(S, K, T, SIG, R, Q, CALL, 20_000, 1, 7, true, false).unwrap();
    let euro = mc_european(S, K, T, SIG, R, Q, CALL, 20_000, 7, true, false).unwrap();
    assert_eq!(asian, euro);
}

#[test]
fn barrier_single_monitoring_matches_analytic() {
    // One monitoring date (t = T, plus t = 0 where s < B): the up-and-out
    // call pays (S_T - k) 1{k < S_T < B} = C(k) - C(B) - (B - k) e^{-rt} N(d2(B)).
    let barrier = 130.0_f64;
    let d2_b = ((S / barrier).ln() + (R - Q - 0.5 * SIG * SIG) * T) / (SIG * T.sqrt());
    let analytic = bs_price(S, K, T, SIG, R, Q, CALL).unwrap()
        - bs_price(S, barrier, T, SIG, R, Q, CALL).unwrap()
        - (barrier - K) * (-R * T).exp() * dpe::norm_cdf(d2_b);
    assert!((analytic - 5.310827113020051).abs() < 1e-12); // derived independently
    let res = mc_barrier_up_out(S, K, T, SIG, R, Q, CALL, barrier, 200_000, 1, 3, true).unwrap();
    assert!(
        (res.value - analytic).abs() < 3.0 * res.std_error,
        "barrier {} vs analytic {analytic} (se {})",
        res.value,
        res.std_error
    );
    assert!(res.value < bs_price(S, K, T, SIG, R, Q, CALL).unwrap());
}

#[test]
fn lookback_single_step_is_atm_vanilla() {
    // One step: call pays (S_T - s)^+, put pays (s - S_T)^+ — the ATM vanillas.
    let call = mc_lookback_floating(S, T, SIG, R, Q, CALL, 200_000, 1, 3, true).unwrap();
    let atm_call = bs_price(S, S, T, SIG, R, Q, CALL).unwrap();
    assert!((call.value - atm_call).abs() < 3.0 * call.std_error, "{} vs {atm_call}", call.value);
    let put = mc_lookback_floating(S, T, SIG, R, Q, PUT, 200_000, 1, 3, true).unwrap();
    let atm_put = bs_price(S, S, T, SIG, R, Q, PUT).unwrap();
    assert!((put.value - atm_put).abs() < 3.0 * put.std_error, "{} vs {atm_put}", put.value);
}

#[test]
fn sigma_zero_is_deterministic() {
    // sigma = 0: every path is the forward, the control is degenerate (var_x
    // = 0 -> beta = 0 branch) and the value equals the analytic limit with
    // (numerically) zero standard error.
    let res = mc_european(S, K, T, 0.0, R, Q, CALL, 1000, 1, true, true).unwrap();
    let bs0 = bs_price(S, K, T, 0.0, R, Q, CALL).unwrap();
    assert!((res.value - bs0).abs() < 1e-12, "{} vs {bs0}", res.value);
    assert!(res.std_error <= 1e-12);
    assert!(res.ci_high - res.ci_low <= 1e-11);
    let asian = mc_asian_arithmetic(S, K, T, 0.0, R, Q, PUT, 1000, 4, 1, true, true).unwrap();
    let geo0 = geometric_asian_price(S, K, T, 0.0, R, Q, PUT, 4).unwrap();
    assert!((asian.value - geo0).abs() < 1e-12);
    assert!(asian.std_error <= 1e-12);
}

#[test]
fn results_finite_and_ci_ordered() {
    let results = [
        mc_european(S, K, T, SIG, R, Q, PUT, 2000, 5, true, true).unwrap(),
        mc_asian_arithmetic(S, K, T, SIG, R, Q, PUT, 2000, 6, 5, true, true).unwrap(),
        mc_barrier_up_out(S, K, T, SIG, R, Q, PUT, 140.0, 2000, 6, 5, true).unwrap(),
        mc_lookback_floating(S, T, SIG, R, Q, PUT, 2000, 6, 5, true).unwrap(),
        mc_delta_pathwise(S, K, T, SIG, R, Q, PUT, 2000, 5, true).unwrap(),
        mc_delta_fd_crn(S, K, T, SIG, R, Q, PUT, 2000, 5, 1e-4, true).unwrap(),
    ];
    for r in results {
        assert!(r.value.is_finite() && r.std_error.is_finite());
        assert!(r.ci_low.is_finite() && r.ci_high.is_finite());
        assert!(r.ci_low <= r.value && r.value <= r.ci_high && r.std_error >= 0.0);
    }
}
