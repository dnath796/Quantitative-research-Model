//! Monte Carlo engine: exact-step GBM simulation, exotics, variance reduction.
//!
//! # Simulation scheme
//!
//! GBM has the exact solution
//!
//! ```text
//! S_{t+h} = S_t * exp((r - q - sigma^2/2) h + sigma sqrt(h) Z),  Z ~ N(0,1)
//! ```
//!
//! so stepping with this recursion is **exact in distribution** at the grid
//! dates — there is no Euler discretisation error in the marginals. Path
//! functionals that look *between* grid dates (barrier crossings, lookback
//! extrema) still carry monitoring bias; see the individual pricers.
//!
//! # Variance reduction
//!
//! * **Antithetic variates**: each normal draw `Z` is paired with `-Z`.
//!   Because the payoffs here are monotone in `Z`, the pair averages are
//!   negatively correlated and the estimator variance drops. Statistically
//!   each *pair average* is one i.i.d. sample, so standard errors are
//!   computed over pair averages (`n_paths / 2` samples), never over raw
//!   correlated paths.
//! * **Control variates**: for a payoff `Y` and a control `X` with known
//!   mean `E[X]`, the adjusted estimator `Y − β (X − E[X])` with
//!   `β = Cov(Y, X) / Var(X)` (estimated in-sample; `β = 0` when
//!   `Var(X) = 0`) has variance reduced by the squared correlation.
//!   Controls used: the discounted terminal spot `e^{−rt} S_T` (known mean
//!   `s e^{−qt}`, the Black-Scholes `k = 0` analytic value) for the European
//!   vanilla, and the geometric-Asian payoff (exact mean from
//!   [`geometric_asian_price`]) for the arithmetic Asian.
//!
//! Every pricer takes an integer `seed` in `[0, 2^63 - 1]` (the domain
//! pinned by API_SPEC §5 for every port; larger `u64` values are rejected)
//! and is fully deterministic given it within this crate for the `rand`
//! version locked in `Cargo.lock` (`StdRng`'s stream is not guaranteed
//! stable across `rand` major versions; RNG streams are **not** required to
//! match other language ports either — the golden MC tolerances are several
//! standard errors wide). Results are reported as [`MCResult`] with the standard error and a
//! 95% normal confidence interval.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::DpeError;
use crate::math::norm_cdf;
use crate::types::OptionType;
use crate::validation::{require_positive, validate_market_inputs};

/// Two-sided 95% normal quantile (contractual CI multiplier).
const Z95: f64 = 1.959_963_984_540_054;

/// A Monte Carlo estimate with its sampling uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MCResult {
    /// Point estimate (price or Greek).
    pub value: f64,
    /// Standard error of the estimate: sample std (ddof = 1) of the i.i.d.
    /// samples over sqrt(sample count); with antithetics the samples are
    /// pair averages.
    pub std_error: f64,
    /// `value - 1.959963984540054 * std_error`.
    pub ci_low: f64,
    /// `value + 1.959963984540054 * std_error`.
    pub ci_high: f64,
    /// Raw simulated paths (both antithetic halves).
    pub n_paths: usize,
}

/// Largest admissible RNG seed (2^63 - 1). The seed domain `[0, SEED_MAX]`
/// is pinned by the cross-language contract (API_SPEC §5) so a seed means
/// the same thing in Python, C++, Rust and Java; a `u64` above it is
/// rejected here rather than accepted in Rust alone.
pub const SEED_MAX: u64 = i64::MAX as u64;

fn validate_mc(
    n_paths: usize,
    seed: u64,
    antithetic: bool,
    n_steps: usize,
) -> Result<(), DpeError> {
    if n_paths < 2 {
        return Err(DpeError::InvalidInput(format!(
            "n_paths must be an integer >= 2, got {n_paths}"
        )));
    }
    if antithetic && n_paths % 2 != 0 {
        return Err(DpeError::InvalidInput(format!(
            "n_paths must be even with antithetic=true, got {n_paths}"
        )));
    }
    if antithetic && n_paths < 4 {
        return Err(DpeError::InvalidInput(format!(
            "n_paths must be >= 4 with antithetic=true, got {n_paths}"
        )));
    }
    if seed > SEED_MAX {
        return Err(DpeError::InvalidInput(format!(
            "seed must be an integer in [0, {SEED_MAX}], got {seed}"
        )));
    }
    if n_steps < 1 {
        return Err(DpeError::InvalidInput(format!(
            "n_steps must be an integer >= 1, got {n_steps}"
        )));
    }
    Ok(())
}

/// `rel_bump` must be finite and in the open interval (0, 1): `rel_bump >= 1`
/// would price the down bump at a non-positive spot and return a meaningless
/// delta without any error (API_SPEC §5.8).
fn validate_rel_bump(rel_bump: f64) -> Result<(), DpeError> {
    require_positive("rel_bump", rel_bump)?;
    if rel_bump >= 1.0 {
        return Err(DpeError::InvalidInput(format!(
            "rel_bump must be in (0, 1), got {rel_bump}"
        )));
    }
    Ok(())
}

/// Mean, then [`MCResult`] with ddof = 1 standard error and 95% CI.
fn stats(samples: &[f64], n_paths: usize) -> MCResult {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    let se = (var / n).sqrt();
    MCResult {
        value: mean,
        std_error: se,
        ci_low: mean - Z95 * se,
        ci_high: mean + Z95 * se,
        n_paths,
    }
}

/// In-place optimal-beta control-variate adjustment `y -= beta (x - E[x])`.
///
/// `beta = cov(y, x) / var(x)` with ddof = 1; a degenerate control
/// (`var(x) = 0`, e.g. `sigma = 0`) leaves `y` untouched.
fn control_adjust(y: &mut [f64], x: &[f64], x_mean: f64) {
    let n = y.len() as f64;
    let my = y.iter().sum::<f64>() / n;
    let mx = x.iter().sum::<f64>() / n;
    let mut var_x = 0.0;
    let mut cov = 0.0;
    for (yi, xi) in y.iter().zip(x) {
        let dx = xi - mx;
        var_x += dx * dx;
        cov += (yi - my) * dx;
    }
    var_x /= n - 1.0;
    cov /= n - 1.0;
    if var_x <= 0.0 {
        return;
    }
    let beta = cov / var_x;
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi -= beta * (xi - x_mean);
    }
}

/// Pair-averaged samples from a single-terminal-draw estimator.
///
/// `f(z)` maps one standard normal to one sample; with antithetics the
/// returned samples are the i.i.d. pair averages `(f(z) + f(-z)) / 2`.
fn single_step_samples<F: FnMut(f64) -> f64>(
    n_paths: usize,
    seed: u64,
    antithetic: bool,
    mut f: F,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    if antithetic {
        (0..n_paths / 2)
            .map(|_| {
                let z: f64 = rng.sample(StandardNormal);
                0.5 * (f(z) + f(-z))
            })
            .collect()
    } else {
        (0..n_paths)
            .map(|_| f(rng.sample(StandardNormal)))
            .collect()
    }
}

/// Fill `path` (length `n_steps + 1`, `path[0] = s`) from the normal draws
/// `z` scaled by `sign` (`-1` for the antithetic partner), using the exact
/// GBM step.
fn fill_path(path: &mut [f64], s: f64, drift: f64, vol: f64, z: &[f64], sign: f64) {
    path[0] = s;
    let mut log_acc = 0.0;
    for (i, zi) in z.iter().enumerate() {
        log_acc += drift + vol * sign * zi;
        path[i + 1] = s * log_acc.exp();
    }
}

/// Pair-averaged samples from a whole-path functional.
///
/// The closure sees the full simulated path (grid `t_i = i t / n_steps`,
/// `path[0] = s`) and returns one sample; with antithetics the `+z`/`-z`
/// paths of a pair are averaged into one i.i.d. sample.
#[allow(clippy::too_many_arguments)]
fn path_samples<F: FnMut(&[f64]) -> f64>(
    s: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
    mut functional: F,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dt = t / n_steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let mut z = vec![0.0_f64; n_steps];
    let mut path = vec![0.0_f64; n_steps + 1];
    if antithetic {
        (0..n_paths / 2)
            .map(|_| {
                for zi in z.iter_mut() {
                    *zi = rng.sample(StandardNormal);
                }
                fill_path(&mut path, s, drift, vol, &z, 1.0);
                let a = functional(&path);
                fill_path(&mut path, s, drift, vol, &z, -1.0);
                let b = functional(&path);
                0.5 * (a + b)
            })
            .collect()
    } else {
        (0..n_paths)
            .map(|_| {
                for zi in z.iter_mut() {
                    *zi = rng.sample(StandardNormal);
                }
                fill_path(&mut path, s, drift, vol, &z, 1.0);
                functional(&path)
            })
            .collect()
    }
}

/// Simulate GBM paths with the exact log-Euler step.
///
/// Returns a matrix of shape `(n_paths, n_steps + 1)` whose first column is
/// the spot `s` and whose column `i` holds `S(i · t / n_steps)`.
/// Deterministic given `seed`. With `antithetic = true`, `n_paths` must be
/// even and rows `[n_paths/2 ..)` use the negated increments of rows
/// `[0 .. n_paths/2)`.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`, or a bad
/// Monte Carlo configuration (see crate conventions).
#[allow(clippy::too_many_arguments)]
pub fn simulate_gbm_paths(
    s: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
) -> Result<Vec<Vec<f64>>, DpeError> {
    validate_market_inputs(s, 1.0, t, sigma, r, q)?;
    require_positive("t", t)?;
    validate_mc(n_paths, seed, antithetic, n_steps)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let dt = t / n_steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let mut paths: Vec<Vec<f64>> = Vec::with_capacity(n_paths);
    if antithetic {
        let half = n_paths / 2;
        let mut z_rows: Vec<Vec<f64>> = Vec::with_capacity(half);
        for _ in 0..half {
            let z: Vec<f64> = (0..n_steps).map(|_| rng.sample(StandardNormal)).collect();
            let mut path = vec![0.0_f64; n_steps + 1];
            fill_path(&mut path, s, drift, vol, &z, 1.0);
            paths.push(path);
            z_rows.push(z);
        }
        for z in &z_rows {
            let mut path = vec![0.0_f64; n_steps + 1];
            fill_path(&mut path, s, drift, vol, z, -1.0);
            paths.push(path);
        }
    } else {
        let mut z = vec![0.0_f64; n_steps];
        for _ in 0..n_paths {
            for zi in z.iter_mut() {
                *zi = rng.sample(StandardNormal);
            }
            let mut path = vec![0.0_f64; n_steps + 1];
            fill_path(&mut path, s, drift, vol, &z, 1.0);
            paths.push(path);
        }
    }
    Ok(paths)
}

/// Closed-form price of a discretely-monitored geometric-average Asian.
///
/// The geometric mean of GBM sampled at the equally-spaced fixings
/// `t_i = i t / n`, `i = 1..n` (the spot at 0 is **not** a fixing), is
/// lognormal with
///
/// ```text
/// E[ln G]   = ln s + (r - q - sigma^2/2) t (n+1) / (2n)
/// Var[ln G] = sigma^2 t (n+1)(2n+1) / (6 n^2)
/// ```
///
/// and the price is Black's formula on that lognormal. Used as the analytic
/// control variate for the arithmetic Asian and as a deterministic golden
/// value. If `Var = 0` the discounted deterministic intrinsic on `E[G]` is
/// returned.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid inputs, `t <= 0`, `k <= 0` or
/// `n_fixings < 1`.
#[allow(clippy::too_many_arguments)]
pub fn geometric_asian_price(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_fixings: usize,
) -> Result<f64, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    require_positive("k", k)?;
    if n_fixings < 1 {
        return Err(DpeError::InvalidInput(format!(
            "n_fixings must be an integer >= 1, got {n_fixings}"
        )));
    }

    let n = n_fixings as f64;
    let mean_ln = s.ln() + (r - q - 0.5 * sigma * sigma) * t * (n + 1.0) / (2.0 * n);
    let var_ln = sigma * sigma * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    let df = (-r * t).exp();
    let eg = (mean_ln + 0.5 * var_ln).exp();
    if var_ln <= 0.0 {
        return Ok(df * option_type.payoff(eg, k));
    }
    let sd = var_ln.sqrt();
    let d1 = ((eg / k).ln() + 0.5 * var_ln) / sd;
    let d2 = d1 - sd;
    Ok(match option_type {
        OptionType::Call => df * (eg * norm_cdf(d1) - k * norm_cdf(d2)),
        OptionType::Put => df * (k * norm_cdf(-d2) - eg * norm_cdf(-d1)),
    })
}

/// European vanilla by Monte Carlo (single exact step to expiry).
///
/// Optional variance reduction: antithetic pairs and the
/// discounted-terminal-spot control variate (known mean `s e^{−qt}`, the
/// Black-Scholes `k = 0` analytic value).
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`, or a bad
/// Monte Carlo configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_european(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    seed: u64,
    antithetic: bool,
    control_variate: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    validate_mc(n_paths, seed, antithetic, 1)?;

    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let df = (-r * t).exp();

    if control_variate {
        // Collect payoff and control per draw, pair-averaged in lockstep.
        let mut y: Vec<f64> = Vec::with_capacity(if antithetic { n_paths / 2 } else { n_paths });
        let mut x: Vec<f64> = Vec::with_capacity(y.capacity());
        let mut rng = StdRng::seed_from_u64(seed);
        let sample = |z: f64| -> (f64, f64) {
            let st = s * (drift + vol * z).exp();
            (df * option_type.payoff(st, k), df * st)
        };
        if antithetic {
            for _ in 0..n_paths / 2 {
                let z: f64 = rng.sample(StandardNormal);
                let (ya, xa) = sample(z);
                let (yb, xb) = sample(-z);
                y.push(0.5 * (ya + yb));
                x.push(0.5 * (xa + xb));
            }
        } else {
            for _ in 0..n_paths {
                let (yi, xi) = sample(rng.sample(StandardNormal));
                y.push(yi);
                x.push(xi);
            }
        }
        control_adjust(&mut y, &x, s * (-q * t).exp());
        return Ok(stats(&y, n_paths));
    }

    let samples = single_step_samples(n_paths, seed, antithetic, |z| {
        let st = s * (drift + vol * z).exp();
        df * option_type.payoff(st, k)
    });
    Ok(stats(&samples, n_paths))
}

/// Arithmetic-average Asian option by Monte Carlo.
///
/// The average is taken over the `n_steps` grid dates `t_i = i t / n` for
/// `i = 1..n` (the spot at `t = 0` is **not** a fixing). With
/// `control_variate = true` the geometric-average Asian on the same fixing
/// dates is the control, with its exact price from
/// [`geometric_asian_price`]; the correlation with the arithmetic payoff is
/// typically >99%, cutting the standard error by an order of magnitude.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`, `k <= 0`,
/// or a bad Monte Carlo configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_arithmetic(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
    control_variate: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    require_positive("k", k)?;
    validate_mc(n_paths, seed, antithetic, n_steps)?;

    let df = (-r * t).exp();
    let inv_n = 1.0 / n_steps as f64;

    // Per-path arithmetic payoff and (when needed) geometric-control payoff.
    let payoffs = |path: &[f64]| -> (f64, f64) {
        let fixings = &path[1..]; // exclude S0
        let arith = fixings.iter().sum::<f64>() * inv_n;
        let y = df * option_type.payoff(arith, k);
        if !control_variate {
            return (y, 0.0);
        }
        let geo = (fixings.iter().map(|&v| v.ln()).sum::<f64>() * inv_n).exp();
        (y, df * option_type.payoff(geo, k))
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let dt = t / n_steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let mut z = vec![0.0_f64; n_steps];
    let mut path = vec![0.0_f64; n_steps + 1];
    let n_samples = if antithetic { n_paths / 2 } else { n_paths };
    let mut y: Vec<f64> = Vec::with_capacity(n_samples);
    let mut x: Vec<f64> = Vec::with_capacity(if control_variate { n_samples } else { 0 });
    for _ in 0..n_samples {
        for zi in z.iter_mut() {
            *zi = rng.sample(StandardNormal);
        }
        fill_path(&mut path, s, drift, vol, &z, 1.0);
        let (mut yi, mut xi) = payoffs(&path);
        if antithetic {
            fill_path(&mut path, s, drift, vol, &z, -1.0);
            let (yb, xb) = payoffs(&path);
            yi = 0.5 * (yi + yb);
            xi = 0.5 * (xi + xb);
        }
        y.push(yi);
        if control_variate {
            x.push(xi);
        }
    }

    if control_variate {
        let x_mean = geometric_asian_price(s, k, t, sigma, r, q, option_type, n_steps)?;
        control_adjust(&mut y, &x, x_mean);
    }
    Ok(stats(&y, n_paths))
}

/// Up-and-out barrier option, discretely monitored on the step grid.
///
/// The option knocks out (pays 0) if the simulated spot touches or exceeds
/// `barrier` at **any grid date** including `t_0 = 0`: if `s >= barrier`
/// the option is born dead and `MCResult{0, 0, 0, 0, n_paths}` is returned
/// without simulating.
///
/// # Discretisation bias
///
/// A continuously-monitored barrier can be breached *between* grid dates,
/// which discrete monitoring never sees, so this estimator systematically
/// **over-prices** an up-and-out relative to the continuous contract; the
/// bias shrinks like O(1/sqrt(n_steps)). The Broadie-Glasserman-Kou
/// continuity correction (shift the barrier to `B e^{0.5826 sigma sqrt(dt)}`
/// for an up barrier) removes the leading bias term if a continuous contract
/// is intended; this function prices the **discrete** contract as specified
/// and applies no correction.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`,
/// `barrier <= 0`, or a bad Monte Carlo configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_barrier_up_out(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    barrier: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    require_positive("barrier", barrier)?;
    validate_mc(n_paths, seed, antithetic, n_steps)?;

    if s >= barrier {
        return Ok(MCResult {
            value: 0.0,
            std_error: 0.0,
            ci_low: 0.0,
            ci_high: 0.0,
            n_paths,
        });
    }

    let df = (-r * t).exp();
    let samples = path_samples(
        s,
        t,
        sigma,
        r,
        q,
        n_paths,
        n_steps,
        seed,
        antithetic,
        |path| {
            let alive = path.iter().all(|&v| v < barrier);
            if alive {
                df * option_type.payoff(path[path.len() - 1], k)
            } else {
                0.0
            }
        },
    );
    Ok(stats(&samples, n_paths))
}

/// Floating-strike lookback option, discretely monitored on the step grid.
///
/// Payoffs (extrema over the grid dates, `t = 0` included):
///
/// * call: `S_T − min_i S(t_i)` — buy at the observed low;
/// * put:  `max_i S(t_i) − S_T` — sell at the observed high.
///
/// Both are non-negative by construction. Discrete monitoring *under*-states
/// the true continuous extremum, so this under-prices the
/// continuously-monitored contract with O(1/sqrt(n_steps)) bias — the mirror
/// image of the barrier case. There is no strike parameter.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`, or a bad
/// Monte Carlo configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_lookback_floating(
    s: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, 1.0, t, sigma, r, q)?;
    require_positive("t", t)?;
    validate_mc(n_paths, seed, antithetic, n_steps)?;

    let df = (-r * t).exp();
    let samples = path_samples(
        s,
        t,
        sigma,
        r,
        q,
        n_paths,
        n_steps,
        seed,
        antithetic,
        |path| {
            let terminal = path[path.len() - 1];
            match option_type {
                OptionType::Call => {
                    let low = path.iter().cloned().fold(f64::INFINITY, f64::min);
                    df * (terminal - low)
                }
                OptionType::Put => {
                    let high = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    df * (high - terminal)
                }
            }
        },
    );
    Ok(stats(&samples, n_paths))
}

/// Pathwise-derivative delta of a European vanilla.
///
/// Differentiating the payoff along each path (valid because the vanilla
/// payoff is Lipschitz and `dS_T/dS_0 = S_T / S_0` for GBM):
///
/// * call: `delta = e^{−rt} E[ 1{S_T > K} S_T / S_0 ]`;
/// * put:  `delta = −e^{−rt} E[ 1{S_T < K} S_T / S_0 ]`.
///
/// Unbiased and typically far lower-variance than finite differences, but
/// invalid for discontinuous payoffs (e.g. digitals), where the
/// likelihood-ratio method would be needed instead.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`, or a bad
/// Monte Carlo configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_delta_pathwise(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    seed: u64,
    antithetic: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    validate_mc(n_paths, seed, antithetic, 1)?;

    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let df = (-r * t).exp();
    let samples = single_step_samples(n_paths, seed, antithetic, |z| {
        let st = s * (drift + vol * z).exp();
        match option_type {
            OptionType::Call => {
                if st > k {
                    df * st / s
                } else {
                    0.0
                }
            }
            OptionType::Put => {
                if st < k {
                    -df * st / s
                } else {
                    0.0
                }
            }
        }
    });
    Ok(stats(&samples, n_paths))
}

/// Central finite-difference delta with common random numbers (CRN).
///
/// The same normal draws (same `seed`) price the payoff at `s (1 ± rel_bump)`
/// and the delta is the per-path central difference
/// `(payoff_up − payoff_down) / (2 s rel_bump)`, discounted. Reusing the
/// randomness makes the difference of the two estimators nearly noiseless
/// (their sampling errors cancel path by path); without CRN the variance
/// would explode as O(1/h²). The reported standard error is that of the
/// per-path difference quotient.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `t <= 0`,
/// `rel_bump` outside the open interval (0, 1), or a bad Monte Carlo
/// configuration.
#[allow(clippy::too_many_arguments)]
pub fn mc_delta_fd_crn(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    seed: u64,
    rel_bump: f64,
    antithetic: bool,
) -> Result<MCResult, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    require_positive("t", t)?;
    validate_rel_bump(rel_bump)?;
    validate_mc(n_paths, seed, antithetic, 1)?;

    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let df = (-r * t).exp();
    let h = s * rel_bump;
    let samples = single_step_samples(n_paths, seed, antithetic, |z| {
        let growth = (drift + vol * z).exp();
        let pay_up = df * option_type.payoff((s + h) * growth, k);
        let pay_dn = df * option_type.payoff((s - h) * growth, k);
        (pay_up - pay_dn) / (2.0 * h)
    });
    Ok(stats(&samples, n_paths))
}
