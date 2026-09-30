//! End-to-end demo of the dpe pricing engine (Rust port).
//!
//! Run: `cd rust && cargo run --bin demo`
//!
//! Sections
//! 1. Analytic BSM / Garman-Kohlhagen prices and full Greeks.
//! 2. Implied-vol round-trip.
//! 3. Binomial trees: convergence to BS, American early-exercise premium.
//! 4. Monte Carlo: variance-reduction comparison, exotics with 95% CIs,
//!    pathwise vs CRN finite-difference delta.
//! 5. Historical vol from data/spots_timeseries.csv.
//! 6. Risk report for the small book in data/portfolio.csv.

use std::path::{Path, PathBuf};

use dpe::{
    binomial_greeks, binomial_price, bs_greeks, bs_price, fx_forward, gk_greeks, historical_vol,
    implied_vol, mc_asian_arithmetic, mc_barrier_up_out, mc_delta_fd_crn, mc_delta_pathwise,
    mc_european, mc_lookback_floating, spot_delta_to_forward, ExerciseStyle, MCResult, OptionType,
    TreeMethod,
};

const RULE: &str =
    "------------------------------------------------------------------------------";

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn section(title: &str) {
    println!("\n{RULE}\n{title}\n{RULE}");
}

fn ci(res: &MCResult) -> String {
    format!("[{:.4}, {:.4}]", res.ci_low, res.ci_high)
}

fn demo_analytic() {
    section("1. Analytic Black-Scholes-Merton and Garman-Kohlhagen");
    let rows: [(&str, f64, f64, f64, f64, f64, f64, OptionType); 7] = [
        ("Equity ATM call (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::Call),
        ("Equity ATM put  (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::Put),
        ("High-vol small cap call", 18.0, 20.0, 1.0, 0.80, 0.04, 0.00, OptionType::Call),
        ("Weekly put (T=1/52)", 100.0, 98.0, 1.0 / 52.0, 0.25, 0.03, 0.00, OptionType::Put),
        ("LEAPS call (T=3y)", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, OptionType::Call),
        ("FX EURUSD call (rd,rf)", 1.10, 1.12, 0.5, 0.10, 0.03, 0.02, OptionType::Call),
        ("FX neg-rate put (rd<0)", 0.95, 0.95, 1.0, 0.12, -0.005, 0.001, OptionType::Put),
    ];
    println!(
        "{:<24}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
        "instrument", "price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"
    );
    for (name, s, k, t, sig, r, q, ot) in rows {
        let g = bs_greeks(s, k, t, sig, r, q, ot).expect("valid demo inputs");
        println!(
            "{name:<24}{:>10.4}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>9.4}",
            g.price, g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga
        );
    }
    let g = gk_greeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, OptionType::Call).unwrap();
    let fwd_delta = spot_delta_to_forward(g.delta, 0.5, 0.02).unwrap();
    let fwd = fx_forward(1.10, 0.5, 0.03, 0.02).unwrap();
    println!(
        "\nFX delta conventions (EURUSD call): spot delta = {:.4}, forward delta = {:.4}, \
         forward = {:.4}",
        g.delta, fwd_delta, fwd
    );
}

fn demo_implied_vol() {
    section("2. Implied volatility round-trip");
    for (sigma_in, k) in [(0.15, 120.0), (0.40, 80.0)] {
        let price = bs_price(100.0, k, 0.5, sigma_in, 0.03, 0.01, OptionType::Call).unwrap();
        let iv = implied_vol(price, 100.0, k, 0.5, 0.03, 0.01, OptionType::Call).unwrap();
        println!("k={k:>6.1}  price={price:>9.4}  sigma_in={sigma_in:.4}  implied={iv:.10}");
    }
    match implied_vol(1.0, 100.0, 80.0, 0.5, 0.03, 0.0, OptionType::Call) {
        Err(err) => println!("below-intrinsic price correctly rejected: {err}"),
        Ok(_) => unreachable!("inadmissible price must be rejected"),
    }
}

fn demo_binomial() {
    section("3. Binomial trees (CRR / Jarrow-Rudd)");
    let (s, k, t, sig, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    let bs = bs_price(s, k, t, sig, r, q, OptionType::Call).unwrap();
    println!("{:>6} {:>14} {:>16} {:>13}", "steps", "CRR-BS error", "CRR+Richardson", "JR-BS error");
    for n in [50usize, 200, 800, 2000] {
        let euro = ExerciseStyle::European;
        let call = OptionType::Call;
        let crr = binomial_price(s, k, t, sig, r, q, call, euro, n, TreeMethod::Crr, false).unwrap();
        let rich = binomial_price(s, k, t, sig, r, q, call, euro, n, TreeMethod::Crr, true).unwrap();
        let jr = binomial_price(s, k, t, sig, r, q, call, euro, n, TreeMethod::Jr, false).unwrap();
        println!("{n:>6} {:>14.2e} {:>16.2e} {:>13.2e}", crr - bs, rich - bs, jr - bs);
    }
    println!("\nAmerican put early-exercise premium (s=100, T=1, sigma=20%, r=5%):");
    println!("{:>8}{:>12}{:>12}{:>10}", "strike", "european", "american", "premium");
    for kk in [90.0, 100.0, 110.0, 120.0] {
        let put = OptionType::Put;
        let eur = binomial_price(s, kk, t, sig, r, 0.0, put, ExerciseStyle::European, 500, TreeMethod::Crr, false).unwrap();
        let amer = binomial_price(s, kk, t, sig, r, 0.0, put, ExerciseStyle::American, 500, TreeMethod::Crr, false).unwrap();
        println!("{kk:>8.1}{eur:>12.4}{amer:>12.4}{:>10.4}", amer - eur);
    }
    let bg = binomial_greeks(s, k, t, sig, r, q, OptionType::Call, ExerciseStyle::European, 500, TreeMethod::Crr).unwrap();
    let ag = bs_greeks(s, k, t, sig, r, q, OptionType::Call).unwrap();
    println!(
        "\ntree Greeks (500-step CRR) vs analytic: delta {:.4}/{:.4}, gamma {:.4}/{:.4}, \
         theta {:.4}/{:.4}",
        bg.delta, ag.delta, bg.gamma, ag.gamma, bg.theta, ag.theta
    );
}

fn demo_monte_carlo() {
    section("4. Monte Carlo (100k paths, seed 42)");
    let (s, k, t, sig, r, q) = (100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
    let n = 100_000;
    let call = OptionType::Call;
    let bs = bs_price(s, k, t, sig, r, q, call).unwrap();
    let plain = mc_european(s, k, t, sig, r, q, call, n, 42, false, false).unwrap();
    let anti = mc_european(s, k, t, sig, r, q, call, n, 42, true, false).unwrap();
    let cv = mc_european(s, k, t, sig, r, q, call, n, 42, true, true).unwrap();
    println!("European call, BS analytic = {bs:.4}");
    println!("{:<28}{:>10}{:>10}{:>24}", "estimator", "price", "std err", "95% CI");
    for (label, res) in [("plain", &plain), ("antithetic", &anti), ("antithetic + control", &cv)] {
        println!("{label:<28}{:>10.4}{:>10.4}{:>24}", res.value, res.std_error, ci(res));
    }
    let asian = mc_asian_arithmetic(s, k, t, sig, r, q, call, n, 12, 42, true, true).unwrap();
    let barrier = mc_barrier_up_out(s, k, t, sig, r, q, call, 130.0, n, 100, 42, true).unwrap();
    let lookback = mc_lookback_floating(s, t, sig, r, q, call, n, 100, 42, true).unwrap();
    println!("\n{:<34}{:>10}{:>10}{:>24}", "exotic", "price", "std err", "95% CI");
    for (label, res) in [
        ("arithmetic Asian (12 fix, CV)", &asian),
        ("up-and-out barrier B=130", &barrier),
        ("floating-strike lookback", &lookback),
    ] {
        println!("{label:<34}{:>10.4}{:>10.4}{:>24}", res.value, res.std_error, ci(res));
    }
    let pw = mc_delta_pathwise(s, k, t, sig, r, q, call, n, 42, true).unwrap();
    let fd = mc_delta_fd_crn(s, k, t, sig, r, q, call, n, 42, 1e-4, true).unwrap();
    let delta = bs_greeks(s, k, t, sig, r, q, call).unwrap().delta;
    println!(
        "\ndelta: analytic {delta:.4}, pathwise {:.4} (se {:.4}), FD+CRN {:.4} (se {:.4})",
        pw.value, pw.std_error, fd.value, fd.std_error
    );
}

/// Minimal CSV reader for the bundled comma-separated files (no quoting).
fn read_csv(path: &Path) -> Vec<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|c| c.trim().to_string()).collect())
        .collect()
}

fn demo_historical_vol() {
    section("5. Historical volatility from data/spots_timeseries.csv");
    let rows = read_csv(&data_dir().join("spots_timeseries.csv"));
    let header = &rows[0];
    println!("{:<10}{:>12}{:>14}", "underlier", "last close", "realised vol");
    for col in 1..header.len() {
        let prices: Vec<f64> = rows[1..]
            .iter()
            .map(|r| r[col].parse::<f64>().expect("numeric close"))
            .collect();
        let vol = historical_vol(&prices, 252).unwrap();
        println!(
            "{:<10}{:>12.4}{:>13.2}%",
            header[col],
            prices[prices.len() - 1],
            vol * 100.0
        );
    }
}

fn demo_portfolio() {
    section("6. Portfolio risk report (data/portfolio.csv)");
    let rows = read_csv(&data_dir().join("portfolio.csv"));
    println!(
        "{:<16}{:<4}{:<5}{:>10}{:>9}{:>9}{:>10}{:>10}{:>16}",
        "id", "typ", "style", "price", "delta", "gamma", "vega", "theta", "position value"
    );
    let mut total = 0.0;
    for row in &rows[1..] {
        let (id, s, k, t, sigma, r, q) = (
            &row[0],
            row[2].parse::<f64>().unwrap(),
            row[3].parse::<f64>().unwrap(),
            row[4].parse::<f64>().unwrap(),
            row[5].parse::<f64>().unwrap(),
            row[6].parse::<f64>().unwrap(),
            row[7].parse::<f64>().unwrap(),
        );
        let ot = OptionType::parse(&row[8]).unwrap();
        let style = ExerciseStyle::parse(&row[9]).unwrap();
        let quantity = row[10].parse::<f64>().unwrap();
        let (price, delta, gamma, vega, theta) = if style == ExerciseStyle::American {
            // American style is priced on the tree (500-step CRR); the tree
            // reports no vega.
            let g = binomial_greeks(s, k, t, sigma, r, q, ot, style, 500, TreeMethod::Crr).unwrap();
            (g.price, g.delta, g.gamma, f64::NAN, g.theta)
        } else {
            let g = bs_greeks(s, k, t, sigma, r, q, ot).unwrap();
            (g.price, g.delta, g.gamma, g.vega, g.theta)
        };
        let value = price * quantity;
        total += value;
        let vega_str = if vega.is_nan() {
            format!("{:>10}", "n/a")
        } else {
            format!("{vega:>10.4}")
        };
        println!(
            "{id:<16}{:<4}{:<5}{price:>10.4}{delta:>9.4}{gamma:>9.4}{vega_str}{theta:>10.4}{value:>16.2}",
            row[8][..1].to_uppercase(),
            &row[9][..4]
        );
    }
    println!("\n{:<57}{total:>16.2}", "total book value");
    println!("(FX rows: r=rd, q=rf; price in domestic ccy per unit foreign; vega n/a for tree)");
}

fn main() {
    println!("dpe — Derivatives Pricing Engine demo (Rust port)");
    demo_analytic();
    demo_implied_vol();
    demo_binomial();
    demo_monte_carlo();
    demo_historical_vol();
    demo_portfolio();
    println!("\n{RULE}\ndone.\n");
}
