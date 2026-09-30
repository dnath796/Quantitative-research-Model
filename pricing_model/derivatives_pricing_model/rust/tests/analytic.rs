//! Analytic module tests: put-call parity grid, Greeks vs central finite
//! differences, monotonicity, degenerate limits, implied vol, FX helpers,
//! historical vol and invalid-input behaviour.

use dpe::{
    bs_greeks, bs_price, forward_delta_to_spot, fx_forward, gk_greeks, gk_price, historical_vol,
    implied_vol, norm_cdf, spot_delta_to_forward, DpeError, OptionType,
};

const CALL: OptionType = OptionType::Call;
const PUT: OptionType = OptionType::Put;

#[test]
fn put_call_parity_grid() {
    // C - P = s e^{-qt} - k e^{-rt} across a spot/strike/expiry grid.
    let (r, q, sigma) = (0.04, 0.015, 0.25);
    for &s in &[60.0, 100.0, 145.0] {
        for &k in &[70.0, 100.0, 130.0] {
            for &t in &[0.1, 0.75, 2.5] {
                let c = bs_price(s, k, t, sigma, r, q, CALL).unwrap();
                let p = bs_price(s, k, t, sigma, r, q, PUT).unwrap();
                let parity = s * (-q * t).exp() - k * (-r * t).exp();
                assert!(
                    (c - p - parity).abs() < 1e-10,
                    "parity violated at s={s} k={k} t={t}: C-P={} vs {parity}",
                    c - p
                );
            }
        }
    }
}

#[test]
fn greeks_match_central_finite_differences() {
    // delta, vega (rel err < 1e-4 per the spec), plus gamma/theta/rho checks.
    let (s, k, t, sigma, r, q) = (105.0, 100.0, 0.8, 0.27, 0.04, 0.015);
    for ot in [CALL, PUT] {
        let g = bs_greeks(s, k, t, sigma, r, q, ot).unwrap();
        let h = 1e-5;

        let price = |s_, sig_, t_, r_| bs_price(s_, k, t_, sig_, r_, q, ot).unwrap();

        let delta_fd =
            (price(s + h, sigma, t, r) - price(s - h, sigma, t, r)) / (2.0 * h);
        assert!(
            ((g.delta - delta_fd) / delta_fd).abs() < 1e-4,
            "{ot}: delta {} vs FD {delta_fd}",
            g.delta
        );

        let vega_fd = (price(s, sigma + h, t, r) - price(s, sigma - h, t, r)) / (2.0 * h);
        assert!(
            ((g.vega - vega_fd) / vega_fd).abs() < 1e-4,
            "{ot}: vega {} vs FD {vega_fd}",
            g.vega
        );

        // Second difference needs a larger bump to stay clear of roundoff.
        let h2 = 1e-3;
        let gamma_fd = (price(s + h2, sigma, t, r) - 2.0 * price(s, sigma, t, r)
            + price(s - h2, sigma, t, r))
            / (h2 * h2);
        assert!(((g.gamma - gamma_fd) / gamma_fd).abs() < 1e-4);

        // theta is -dV/dT (calendar-time convention).
        let theta_fd = -(price(s, sigma, t + h, r) - price(s, sigma, t - h, r)) / (2.0 * h);
        assert!(((g.theta - theta_fd) / theta_fd).abs() < 1e-4);

        let rho_fd = (price(s, sigma, t, r + h) - price(s, sigma, t, r - h)) / (2.0 * h);
        assert!(((g.rho - rho_fd) / rho_fd).abs() < 1e-4);
    }
}

#[test]
fn second_order_greeks_match_finite_differences() {
    let (s, k, t, sigma, r, q) = (100.0, 110.0, 1.2, 0.3, 0.02, 0.01);
    let g = bs_greeks(s, k, t, sigma, r, q, CALL).unwrap();
    let h = 1e-5;
    // vanna = d(delta)/d(sigma)
    let vanna_fd = (bs_greeks(s, k, t, sigma + h, r, q, CALL).unwrap().delta
        - bs_greeks(s, k, t, sigma - h, r, q, CALL).unwrap().delta)
        / (2.0 * h);
    assert!(((g.vanna - vanna_fd) / vanna_fd).abs() < 1e-4);
    // volga = d(vega)/d(sigma)
    let volga_fd = (bs_greeks(s, k, t, sigma + h, r, q, CALL).unwrap().vega
        - bs_greeks(s, k, t, sigma - h, r, q, CALL).unwrap().vega)
        / (2.0 * h);
    assert!(((g.volga - volga_fd) / volga_fd).abs() < 1e-4);
}

#[test]
fn call_price_monotonic_in_spot_and_strike() {
    let (t, sigma, r, q) = (1.0, 0.2, 0.03, 0.01);
    // Increasing in s at fixed k.
    let mut prev = 0.0;
    for i in 0..30 {
        let s = 50.0 + 5.0 * i as f64;
        let c = bs_price(s, 100.0, t, sigma, r, q, CALL).unwrap();
        assert!(c >= prev, "call not increasing in s at s={s}");
        prev = c;
    }
    // Decreasing in k at fixed s.
    let mut prev = f64::INFINITY;
    for i in 0..30 {
        let k = 50.0 + 5.0 * i as f64;
        let c = bs_price(100.0, k, t, sigma, r, q, CALL).unwrap();
        assert!(c <= prev, "call not decreasing in k at k={k}");
        prev = c;
    }
}

#[test]
fn degenerate_limits_exact_intrinsic() {
    // t = 0: undiscounted intrinsic.
    assert_eq!(bs_price(90.0, 100.0, 0.0, 0.2, 0.05, 0.0, PUT).unwrap(), 10.0);
    assert_eq!(bs_price(120.0, 100.0, 0.0, 0.2, 0.05, 0.0, CALL).unwrap(), 20.0);
    assert_eq!(bs_price(90.0, 100.0, 0.0, 0.2, 0.05, 0.0, CALL).unwrap(), 0.0);
    // sigma = 0: discounted forward intrinsic.
    let (s, k, t, r, q) = (100.0_f64, 90.0_f64, 1.0_f64, 0.05_f64, 0.02_f64);
    let want = s * (-q * t).exp() - k * (-r * t).exp();
    assert!((bs_price(s, k, t, 0.0, r, q, CALL).unwrap() - want).abs() < 1e-14);
    assert_eq!(bs_price(s, k, t, 0.0, r, q, PUT).unwrap(), 0.0);
    // k = 0: call is a forward claim, put worthless.
    assert!(
        (bs_price(s, 0.0, t, 0.2, r, q, CALL).unwrap() - s * (-q * t).exp()).abs() < 1e-14
    );
    assert_eq!(bs_price(s, 0.0, t, 0.2, r, q, PUT).unwrap(), 0.0);
    // Degenerate Greeks: second-order all zero, delta = +-e^{-qt} if ITM.
    let g = bs_greeks(s, k, t, 0.0, r, q, CALL).unwrap();
    assert_eq!((g.gamma, g.vega, g.vanna, g.volga), (0.0, 0.0, 0.0, 0.0));
    assert!((g.delta - (-q * t).exp()).abs() < 1e-15);
    let g_otm = bs_greeks(80.0, 100.0, 0.5, 0.0, 0.01, 0.0, CALL).unwrap();
    assert_eq!(g_otm.delta, 0.0);
    assert_eq!(g_otm.theta, 0.0);
}

#[test]
fn invalid_inputs_are_rejected() {
    // Every violation must be an Err naming the parameter, never a panic.
    assert!(bs_price(-1.0, 100.0, 1.0, 0.2, 0.05, 0.0, CALL).is_err()); // s <= 0
    assert!(bs_price(0.0, 100.0, 1.0, 0.2, 0.05, 0.0, CALL).is_err());
    assert!(bs_price(100.0, -5.0, 1.0, 0.2, 0.05, 0.0, CALL).is_err()); // k < 0
    assert!(bs_price(100.0, 100.0, -0.1, 0.2, 0.05, 0.0, CALL).is_err()); // t < 0
    assert!(bs_price(100.0, 100.0, 1.0, -0.2, 0.05, 0.0, CALL).is_err()); // sigma < 0
    assert!(bs_price(100.0, 100.0, 1.0, 0.2, f64::NAN, 0.0, CALL).is_err());
    assert!(bs_price(100.0, 100.0, 1.0, 0.2, 0.05, f64::INFINITY, CALL).is_err());
    assert!(bs_price(f64::NAN, 100.0, 1.0, 0.2, 0.05, 0.0, CALL).is_err());

    // Messages name the offending parameter.
    let err = bs_price(-1.0, 100.0, 1.0, 0.2, 0.05, 0.0, CALL).unwrap_err();
    assert!(matches!(&err, DpeError::InvalidInput(m) if m.contains('s')));
    let err = bs_price(100.0, 100.0, 1.0, -0.2, 0.05, 0.0, CALL).unwrap_err();
    assert!(matches!(&err, DpeError::InvalidInput(m) if m.contains("sigma")));

    // Unknown enum spellings.
    assert!(OptionType::parse("straddle").is_err());
    assert!("CALL".parse::<OptionType>().is_ok()); // case-insensitive
}

#[test]
fn implied_vol_round_trips_across_moneyness() {
    let (s, t, r, q) = (100.0, 0.5, 0.03, 0.01);
    for &sigma in &[0.05, 0.2, 0.5, 0.8] {
        for &k in &[60.0, 100.0, 160.0] {
            for ot in [CALL, PUT] {
                let price = bs_price(s, k, t, sigma, r, q, ot).unwrap();
                // Skip prices numerically at intrinsic (deep ITM, tiny vol):
                // no vol is recoverable there and the solver must refuse.
                let lower = match ot {
                    OptionType::Call => (s * (-q * t).exp() - k * (-r * t).exp()).max(0.0),
                    OptionType::Put => (k * (-r * t).exp() - s * (-q * t).exp()).max(0.0),
                };
                if price - lower < 1e-12 {
                    continue;
                }
                let iv = implied_vol(price, s, k, t, r, q, ot).unwrap();
                assert!(
                    (iv - sigma).abs() < 1e-6,
                    "round-trip failed: sigma={sigma} k={k} {ot}: got {iv}"
                );
            }
        }
    }
}

#[test]
fn implied_vol_rejects_out_of_bounds_prices() {
    let (s, k, t, r, q) = (100.0_f64, 80.0_f64, 0.5_f64, 0.03_f64, 0.0_f64);
    let intrinsic = s * (-q * t).exp() - k * (-r * t).exp();
    // Below (discounted) intrinsic: lower-bound error.
    let err = implied_vol(intrinsic - 1.0, s, k, t, r, q, CALL).unwrap_err();
    assert!(
        matches!(&err, DpeError::NoArbitrage(m) if m.contains("lower bound")),
        "got {err:?}"
    );
    // Above the upper bound s e^{-qt}.
    let err = implied_vol(s + 1.0, s, k, t, r, q, CALL).unwrap_err();
    assert!(
        matches!(&err, DpeError::NoArbitrage(m) if m.contains("upper bound")),
        "got {err:?}"
    );
    // t = 0 / k = 0 are invalid for the solver.
    assert!(implied_vol(5.0, s, k, 0.0, r, q, CALL).is_err());
    assert!(implied_vol(5.0, s, 0.0, t, r, q, CALL).is_err());
    assert!(implied_vol(f64::NAN, s, k, t, r, q, CALL).is_err());
}

#[test]
fn gk_matches_bsm_and_delta_conventions_invert() {
    let (s, k, t, sigma, rd, rf) = (1.1, 1.12, 0.5, 0.1, 0.03, 0.02);
    // GK is exactly BSM with r=rd, q=rf.
    assert_eq!(
        gk_price(s, k, t, sigma, rd, rf, CALL).unwrap(),
        bs_price(s, k, t, sigma, rd, rf, CALL).unwrap()
    );
    let g = gk_greeks(s, k, t, sigma, rd, rf, CALL).unwrap();
    // Forward delta of a call is exactly N(d1).
    let (d1, _) = dpe::bs_d1_d2(s, k, t, sigma, rd, rf).unwrap();
    let fwd_delta = spot_delta_to_forward(g.delta, t, rf).unwrap();
    assert!((fwd_delta - norm_cdf(d1)).abs() < 1e-14);
    // The conversions are exact mutual inverses (calls and puts).
    for delta in [0.44, -0.31] {
        let there = spot_delta_to_forward(delta, t, rf).unwrap();
        let back = forward_delta_to_spot(there, t, rf).unwrap();
        assert!((back - delta).abs() < 1e-15);
    }
    // Covered interest parity forward.
    let f = fx_forward(s, t, rd, rf).unwrap();
    assert!((f - s * ((rd - rf) * t).exp()).abs() < 1e-15);
    assert!(fx_forward(-1.0, t, rd, rf).is_err());
}

#[test]
fn historical_vol_estimator_and_errors() {
    // A constant-log-return series has zero realised vol.
    let flat: Vec<f64> = (0..10).map(|i| 100.0 * 1.01f64.powi(i)).collect();
    assert!(historical_vol(&flat, 252).unwrap() < 1e-12);
    // Alternating returns: hand-computed sample std of log returns.
    let prices = [100.0, 110.0, 99.0, 108.9, 98.01];
    let vol = historical_vol(&prices, 252).unwrap();
    let lr: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = lr.iter().sum::<f64>() / lr.len() as f64;
    let sd = (lr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (lr.len() - 1) as f64).sqrt();
    assert!((vol - sd * 252f64.sqrt()).abs() < 1e-14);
    // Errors: too short, non-positive price, non-finite, bad periods.
    assert!(historical_vol(&[100.0, 101.0], 252).is_err());
    assert!(historical_vol(&[100.0, -1.0, 102.0], 252).is_err());
    assert!(historical_vol(&[100.0, f64::NAN, 102.0], 252).is_err());
    assert!(historical_vol(&prices, 0).is_err());
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

const ATM_PRICE: f64 = 10.45058357; // bs_price(100,100,1,0.2,0.05,0,call) to 8 dp

#[test]
fn implied_vol_reports_non_convergence() {
    use dpe::implied_vol_with;
    // One Newton step from the bracket midpoint lands ~2 vol points off; the
    // solver must return Err (NoConvergence), never that root.
    let err = implied_vol_with(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, CALL, 1e-10, 1).unwrap_err();
    assert!(
        matches!(&err, DpeError::NoConvergence(m) if m.contains("did not converge")),
        "got {err:?}"
    );
    // A sufficient budget converges as usual.
    let iv = implied_vol_with(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, CALL, 1e-10, 20).unwrap();
    assert!((iv - 0.2).abs() < 1e-8);
    // tol / max_iter validation names the parameter.
    for bad_iter in [0usize] {
        let err = implied_vol_with(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, CALL, 1e-10, bad_iter)
            .unwrap_err();
        assert!(matches!(&err, DpeError::InvalidInput(m) if m.contains("max_iter")), "{err:?}");
    }
    for bad_tol in [0.0, -1e-10, f64::NAN] {
        let err = implied_vol_with(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0, CALL, bad_tol, 100)
            .unwrap_err();
        assert!(matches!(&err, DpeError::InvalidInput(m) if m.contains("tol")), "{err:?}");
    }
}

#[test]
fn implied_vol_high_vol_bracket_expansion() {
    // sigma = 1.7 lies beyond the initial [0, 1] bracket: the doubling
    // expansion must engage and the round-trip still hold.
    let price = bs_price(100.0, 100.0, 1.0, 1.7, 0.02, 0.0, CALL).unwrap();
    let iv = implied_vol(price, 100.0, 100.0, 1.0, 0.02, 0.0, CALL).unwrap();
    assert!((iv - 1.7).abs() < 1e-6, "iv {iv}");
    // Between 16 and 20 the bracket is clipped to the 20.0 cap and still solves.
    let p17 = bs_price(100.0, 100.0, 0.05, 17.0, 0.02, 0.0, CALL).unwrap();
    let iv17 = implied_vol(p17, 100.0, 100.0, 0.05, 0.02, 0.0, CALL).unwrap();
    assert!((iv17 - 17.0).abs() < 1e-6, "iv {iv17}");
    // tol below the price's floating-point resolution: the bracket-width rule
    // terminates with an essentially exact vol.
    let p = bs_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, CALL).unwrap();
    let iv = dpe::implied_vol_with(p, 100.0, 100.0, 1.0, 0.05, 0.0, CALL, 1e-300, 200).unwrap();
    assert!((iv - 0.2).abs() < 1e-10);
    // Deterministic.
    let pp = bs_price(100.0, 90.0, 0.5, 0.3, 0.02, 0.01, PUT).unwrap();
    assert_eq!(
        implied_vol(pp, 100.0, 90.0, 0.5, 0.02, 0.01, PUT).unwrap(),
        implied_vol(pp, 100.0, 90.0, 0.5, 0.02, 0.01, PUT).unwrap()
    );
}

#[test]
fn greeks_at_expiry_tie_break_contract() {
    // API_SPEC §3.2: exactly at the money in the degenerate region the call
    // takes the ITM branch and the put the OTM branch.
    let (s, k, r, q) = (100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64);
    let c = bs_greeks(s, k, 0.0, 0.2, r, q, CALL).unwrap();
    assert_eq!(c.price, 0.0);
    assert_eq!(c.delta, 1.0);
    assert!((c.theta - (q * s - r * k)).abs() < 1e-15);
    assert_eq!((c.rho, c.gamma, c.vega, c.vanna, c.volga), (0.0, 0.0, 0.0, 0.0, 0.0));
    let p = bs_greeks(s, k, 0.0, 0.2, r, q, PUT).unwrap();
    assert_eq!((p.price, p.delta, p.theta, p.rho), (0.0, 0.0, 0.0, 0.0));
    // sigma = 0 with the forward exactly at the strike (r = q, s = k makes
    // s e^{-qt} - k e^{-rt} exactly zero in floating point).
    let (t, rate) = (1.0_f64, 0.03_f64);
    let df = (-rate * t).exp();
    let c0 = bs_greeks(s, k, t, 0.0, rate, rate, CALL).unwrap();
    assert_eq!(c0.price, 0.0);
    assert!((c0.delta - df).abs() < 1e-15);
    assert!(c0.theta.abs() < 1e-15); // q s e^{-qt} - r k e^{-rt} = 0 here
    assert!((c0.rho - k * t * df).abs() < 1e-12);
    let p0 = bs_greeks(s, k, t, 0.0, rate, rate, PUT).unwrap();
    assert_eq!((p0.price, p0.delta, p0.theta, p0.rho), (0.0, 0.0, 0.0, 0.0));
    assert!(!p0.price.is_sign_negative()); // +0.0, never -0.0
}

#[test]
fn non_finite_rates_rejected_everywhere() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = bs_price(100.0, 100.0, 1.0, 0.2, bad, 0.0, CALL).unwrap_err();
        assert!(matches!(&err, DpeError::InvalidInput(m) if m.starts_with("r must")), "{err:?}");
        let err = bs_greeks(100.0, 100.0, 1.0, 0.2, 0.05, bad, PUT).unwrap_err();
        assert!(matches!(&err, DpeError::InvalidInput(m) if m.starts_with("q must")), "{err:?}");
        assert!(gk_price(1.1, 1.1, 1.0, 0.1, bad, 0.0, CALL).is_err());
        assert!(fx_forward(1.1, 0.5, bad, 0.02).is_err());
        assert!(spot_delta_to_forward(bad, 0.5, 0.02).is_err());
    }
    // historical_vol: periods_per_year >= 1.
    assert!(historical_vol(&[100.0, 101.0, 99.0, 102.0], 0).is_err());
}
