//! Binomial-tree pricing: Cox-Ross-Rubinstein and Jarrow-Rudd lattices.
//!
//! Both lattices discretise GBM over `n` steps of length `dt = t / n`:
//!
//! * **CRR** — `u = e^{sigma sqrt(dt)}`, `d = 1/u`, risk-neutral
//!   `p = (e^{(r-q)dt} - d)/(u - d)`; the tree recombines around the spot,
//!   which is what makes tree-based delta/gamma/theta natural.
//! * **Jarrow-Rudd** — `u = e^{nu + sigma sqrt(dt)}`,
//!   `d = e^{nu - sigma sqrt(dt)}` with `nu = (r - q - sigma^2/2) dt`, and
//!   `p = 1/2`; the drift is absorbed into the node placement so the lattice
//!   drifts with the forward instead of recentering on the spot.
//!
//! Both converge to Black-Scholes at rate O(1/n) for European payoffs, with
//! the well-known oscillation between adjacent step counts (the strike moves
//! relative to the node grid). `richardson = true` prices at `n` and `n + 1`
//! steps and averages, cancelling the leading oscillating error term
//! (two-point odd/even Richardson averaging) — typically one to two extra
//! digits at the same cost order.
//!
//! Backward induction values American exercise by taking
//! `max(continuation, intrinsic)` at **every** node — the discrete dynamic
//! program for the optimal stopping problem.
//!
//! The `sigma = 0` and `t = 0` limits are handled without building a tree
//! (the lattice degenerates: `u = d` makes `p` ill-defined). For `sigma = 0`
//! the spot rides the deterministic forward, so a European option is worth
//! discounted forward intrinsic and an American option is the best
//! discounted intrinsic over the deterministic path evaluated on the step
//! grid (the exact deterministic-limit dynamic program).

use crate::black_scholes::bs_price;
use crate::error::DpeError;
use crate::types::{ExerciseStyle, OptionType, TreeMethod};
use crate::validation::validate_market_inputs;

/// Tree-based price and Greeks (delta, gamma, theta per year).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialGreeks {
    /// Option value at the root node.
    pub price: f64,
    /// Discrete hedge ratio from the two level-1 nodes.
    pub delta: f64,
    /// Difference of the two one-sided level-2 deltas over the half-spread.
    pub gamma: f64,
    /// Calendar-time decay per year, spot-displacement corrected (see
    /// [`binomial_greeks`]).
    pub theta: f64,
}

fn validate_steps(steps: usize) -> Result<(), DpeError> {
    if steps < 1 {
        return Err(DpeError::InvalidInput(format!(
            "steps must be an integer >= 1, got {steps}"
        )));
    }
    Ok(())
}

/// Largest admissible magnitude of the log-spot excursion across the lattice
/// (identical in every port). `exp()` overflows a double at ~709.78, so a
/// terminal node with `|ln s| + |nu n| + sigma sqrt(t n) > 700` would be
/// `inf` (or an `inf * 0` NaN in `s u^j d^(n-j)`); the guard makes that an
/// error.
const LATTICE_LOG_LIMIT: f64 = 700.0;

/// Reject lattices whose terminal spots would overflow a double. The
/// extreme terminal log-spot is `ln s + nu n ± sigma sqrt(t n)` with
/// `nu n = (r - q - sigma^2/2) t` for JR (zero for CRR); bounding the worst
/// case for either method keeps the admissible domain method-independent.
fn check_lattice_range(
    s: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    steps: usize,
) -> Result<(), DpeError> {
    let nu_n = ((r - q - 0.5 * sigma * sigma) * t).abs();
    let excursion = s.ln().abs() + nu_n + sigma * (t * steps as f64).sqrt();
    if excursion > LATTICE_LOG_LIMIT {
        return Err(DpeError::InvalidInput(format!(
            "lattice overflow: |ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) = \
             {excursion} exceeds {LATTICE_LOG_LIMIT}; reduce steps, sigma or t"
        )));
    }
    Ok(())
}

/// Exact `sigma = 0` limit on the `steps + 1` point time grid.
///
/// The spot is deterministic: `S(t_i) = s e^{(r-q) t_i}`. A European option
/// pays intrinsic on the terminal forward; an American option is exercised
/// at whichever grid date maximises the discounted intrinsic.
fn degenerate_sigma_zero(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    style: ExerciseStyle,
    steps: usize,
) -> Result<f64, DpeError> {
    if style == ExerciseStyle::American {
        let mut best = 0.0_f64;
        for i in 0..=steps {
            let ti = t * i as f64 / steps as f64;
            let spot = s * ((r - q) * ti).exp();
            let value = (-r * ti).exp() * option_type.payoff(spot, k);
            if value > best {
                best = value;
            }
        }
        return Ok(best);
    }
    bs_price(s, k, t, 0.0, r, q, option_type) // discounted forward intrinsic
}

/// Return `(u, d, p)` for the requested lattice.
fn tree_params(
    method: TreeMethod,
    sigma: f64,
    r: f64,
    q: f64,
    dt: f64,
) -> Result<(f64, f64, f64), DpeError> {
    let sq = sigma * dt.sqrt();
    let (u, d, p) = match method {
        TreeMethod::Crr => {
            let u = sq.exp();
            let d = 1.0 / u;
            let p = (((r - q) * dt).exp() - d) / (u - d);
            (u, d, p)
        }
        TreeMethod::Jr => {
            let nu = (r - q - 0.5 * sigma * sigma) * dt;
            ((nu + sq).exp(), (nu - sq).exp(), 0.5)
        }
    };
    if !(p > 0.0 && p < 1.0) {
        // Happens when the per-step drift outruns the vol spacing (huge
        // (r-q)dt vs sigma sqrt(dt)); the discrete measure would not be a
        // probability, so the lattice is invalid at this step count.
        return Err(DpeError::InvalidInput(format!(
            "risk-neutral probability p={p} outside (0, 1); \
             increase steps or use smaller drift/vol ratio"
        )));
    }
    Ok((u, d, p))
}

/// Backward induction; optionally keeps the option/spot values of the first
/// `keep_levels` time levels (level 0 = today) for tree Greeks.
///
/// Returns `(value_at_root, option_levels, spot_levels)` where the level
/// vectors hold levels `0..keep_levels` (a level `i` slice has `i + 1`
/// entries, ascending in the number of up-moves).
#[allow(clippy::too_many_arguments)]
fn roll_back(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    option_type: OptionType,
    style: ExerciseStyle,
    steps: usize,
    u: f64,
    d: f64,
    p: f64,
    keep_levels: usize,
) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let dt = t / steps as f64;
    let disc = (-r * dt).exp();

    // Terminal spots: s * u^j * d^(n-j), j = 0..n (ascending).
    let mut spots: Vec<f64> = (0..=steps)
        .map(|j| s * u.powf(j as f64) * d.powf((steps - j) as f64))
        .collect();
    let mut values: Vec<f64> = spots.iter().map(|&sp| option_type.payoff(sp, k)).collect();

    let kept = keep_levels.min(steps + 1);
    let mut opt_levels: Vec<Vec<f64>> = vec![Vec::new(); kept];
    let mut spot_levels: Vec<Vec<f64>> = vec![Vec::new(); kept];
    if steps < keep_levels {
        // The terminal level itself is one of the requested levels
        // (only possible for very small trees, e.g. Greeks at steps = 2).
        opt_levels[steps] = values.clone();
        spot_levels[steps] = spots.clone();
    }

    for i in (0..steps).rev() {
        for j in 0..=i {
            values[j] = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            spots[j] /= d; // spots at level i: s * u^j * d^(i-j)
        }
        values.truncate(i + 1);
        spots.truncate(i + 1);
        if style == ExerciseStyle::American {
            for j in 0..=i {
                values[j] = values[j].max(option_type.payoff(spots[j], k));
            }
        }
        if i < kept {
            opt_levels[i] = values.clone();
            spot_levels[i] = spots.clone();
        }
    }
    (values[0], opt_levels, spot_levels)
}

/// Binomial-tree price of a European or American vanilla option.
///
/// Parameters follow the crate-wide conventions (see
/// [`crate::black_scholes`]); additionally `steps >= 1` and `method` selects
/// the lattice. With `richardson = true` the `steps` and `steps + 1` prices
/// are averaged to damp the odd/even oscillation of the convergence to the
/// continuous limit.
///
/// Degenerate limits: `t = 0` returns intrinsic; `sigma = 0` returns the
/// deterministic-forward limit (for American style, the best discounted
/// intrinsic over the step-grid dates).
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `steps < 1`, a
/// risk-neutral probability outside (0, 1) at this step size, or a lattice
/// whose terminal spots would overflow a double
/// (`|ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) > 700`).
#[allow(clippy::too_many_arguments)]
pub fn binomial_price(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    style: ExerciseStyle,
    steps: usize,
    method: TreeMethod,
    richardson: bool,
) -> Result<f64, DpeError> {
    validate_steps(steps)?;
    validate_market_inputs(s, k, t, sigma, r, q)?;

    if t <= 0.0 {
        return Ok(option_type.payoff(s, k));
    }
    if sigma <= 0.0 {
        return degenerate_sigma_zero(s, k, t, r, q, option_type, style, steps);
    }
    check_lattice_range(s, t, sigma, r, q, if richardson { steps + 1 } else { steps })?;

    let one = |n: usize| -> Result<f64, DpeError> {
        let (u, d, p) = tree_params(method, sigma, r, q, t / n as f64)?;
        let (v, _, _) = roll_back(s, k, t, r, option_type, style, n, u, d, p, 0);
        Ok(v)
    };

    if richardson {
        return Ok(0.5 * (one(steps)? + one(steps + 1)?));
    }
    one(steps)
}

/// Delta, gamma and theta read directly off the lattice.
///
/// With node values `V(i, j)` and node spots `S(i, j)` (level `i`, `j` ups):
///
/// * `delta = (V(1,1) − V(1,0)) / (S(1,1) − S(1,0))` — the discrete hedge
///   ratio one step ahead;
/// * `gamma` = difference of the two one-sided deltas at level 2 divided by
///   the half-spread of the outer level-2 spots;
/// * `theta = (V(2,1) − V(0,0) − dV) / (2 dt)` where
///   `dV = delta·ds + gamma·ds²/2` with `ds = S(2,1) − s` removes the value
///   change caused by the middle node's spot displacement. For CRR the
///   middle level-2 node has spot exactly `s` (`ud = 1`) so `dV = 0` and
///   this is the classic pure calendar-time difference; for JR the lattice
///   drifts and the correction strips the delta/gamma contamination out of
///   the time difference.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs, `steps < 2` (level 2
/// must exist), or `t <= 0` / `sigma <= 0` (the lattice degenerates).
#[allow(clippy::too_many_arguments)]
pub fn binomial_greeks(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    style: ExerciseStyle,
    steps: usize,
    method: TreeMethod,
) -> Result<BinomialGreeks, DpeError> {
    validate_steps(steps)?;
    validate_market_inputs(s, k, t, sigma, r, q)?;
    if steps < 2 {
        return Err(DpeError::InvalidInput(format!(
            "steps must be >= 2 for tree Greeks, got {steps}"
        )));
    }
    if t <= 0.0 || sigma <= 0.0 {
        return Err(DpeError::InvalidInput(
            "tree Greeks require t > 0 and sigma > 0".to_string(),
        ));
    }
    check_lattice_range(s, t, sigma, r, q, steps)?;

    let dt = t / steps as f64;
    let (u, d, p) = tree_params(method, sigma, r, q, dt)?;
    let (price, opt, spots) = roll_back(s, k, t, r, option_type, style, steps, u, d, p, 3);

    let v0 = opt[0][0];
    let (v1, s1) = (&opt[1], &spots[1]);
    let (v2, s2) = (&opt[2], &spots[2]);

    let delta = (v1[1] - v1[0]) / (s1[1] - s1[0]);
    let delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1]);
    let delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0]);
    let gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]));
    // Correct for the middle node's spot displacement (zero for CRR, O(dt)
    // for the drifting JR lattice): strip the delta/gamma value change so
    // the remaining difference is purely calendar time.
    let ds = s2[1] - s;
    let theta = (v2[1] - v0 - delta * ds - 0.5 * gamma * ds * ds) / (2.0 * dt);

    Ok(BinomialGreeks {
        price,
        delta,
        gamma,
        theta,
    })
}
