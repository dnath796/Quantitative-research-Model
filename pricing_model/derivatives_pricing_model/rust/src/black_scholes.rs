//! Analytic Black-Scholes-Merton and Garman-Kohlhagen pricing.
//!
//! Under the risk-neutral measure the underlying follows geometric Brownian
//! motion with continuous dividend yield `q`:
//!
//! ```text
//! dS/S = (r - q) dt + sigma dW
//! ```
//!
//! For an FX rate quoted domestic-per-foreign the foreign risk-free rate
//! plays exactly the role of a continuous dividend yield, so Garman-Kohlhagen
//! is Black-Scholes-Merton with `r = rd` (domestic) and `q = rf` (foreign).
//!
//! With `d1 = [ln(s/k) + (r - q + sigma^2/2) t] / (sigma sqrt(t))` and
//! `d2 = d1 - sigma sqrt(t)`:
//!
//! ```text
//! call = s e^{-qt} N(d1) - k e^{-rt} N(d2)
//! put  = k e^{-rt} N(-d2) - s e^{-qt} N(-d1)
//! ```
//!
//! Conventions used throughout the crate (and by every language port):
//! `t` in **years**; `sigma`, `r`, `q` as decimals; `theta` per **year**
//! (calendar time); `vega`/`vanna`/`volga` per unit of vol; `rho` per unit
//! of rate. Degenerate limits (`t = 0`, `sigma = 0`, `k = 0`) collapse to
//! (discounted) intrinsic value exactly; see [`bs_price`] and [`bs_greeks`].

use crate::error::DpeError;
use crate::math::{norm_cdf, norm_pdf};
use crate::types::OptionType;
use crate::validation::{require_finite, require_positive, validate_market_inputs};

/// Full set of analytic first/second-order Greeks, plus the price.
///
/// Units are part of the cross-language contract:
///
/// * `delta` = dV/dS (dimensionless);
/// * `gamma` = d²V/dS²;
/// * `vega`  = dV/dsigma per unit of vol (per 1.00 = 100 vol points);
/// * `theta` = dV/dt per **year** in calendar time (= −dV/dT);
/// * `rho`   = dV/dr per unit of rate (`r` = domestic rate for FX);
/// * `vanna` = d²V/(dS dsigma) — sensitivity of delta to vol;
/// * `volga` = d²V/dsigma² (vomma) — sensitivity of vega to vol.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// Option present value.
    pub price: f64,
    /// dV/dS.
    pub delta: f64,
    /// d²V/dS² (identical for call and put).
    pub gamma: f64,
    /// dV/dsigma per unit of vol (identical for call and put).
    pub vega: f64,
    /// dV/dt per year, calendar-time convention.
    pub theta: f64,
    /// dV/dr per unit of rate.
    pub rho: f64,
    /// d²V/(dS dsigma).
    pub vanna: f64,
    /// d²V/dsigma².
    pub volga: f64,
}

/// Return `(d1, d2)` for the regular region `t > 0`, `sigma > 0`, `k > 0`.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] if the market inputs are invalid or if called
/// in a degenerate region (there is no finite d1/d2 there — the price/Greek
/// functions handle those limits explicitly instead).
pub fn bs_d1_d2(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
) -> Result<(f64, f64), DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;
    if t <= 0.0 || sigma <= 0.0 || k <= 0.0 {
        return Err(DpeError::InvalidInput(
            "d1/d2 require t > 0, sigma > 0 and k > 0; use bs_price for the limits".to_string(),
        ));
    }
    let sq_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sq_t);
    let d2 = d1 - sigma * sq_t;
    Ok((d1, d2))
}

/// European Black-Scholes-Merton price with continuous dividend yield.
///
/// Degenerate limits (contractual, checked by golden cases):
///
/// * `t = 0` → intrinsic `max(±(s − k), 0)` (no discounting);
/// * `sigma = 0`, `t > 0` → discounted forward intrinsic
///   `max(±(s e^{−qt} − k e^{−rt}), 0)`;
/// * `k = 0` → call = `s e^{−qt}` (pure forward claim), put = 0.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs (see crate
/// conventions: `s > 0`, `k >= 0`, `t >= 0`, `sigma >= 0`, finite rates).
pub fn bs_price(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;

    if t <= 0.0 {
        return Ok(option_type.payoff(s, k));
    }

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    if k <= 0.0 {
        return Ok(match option_type {
            OptionType::Call => s * df_q,
            OptionType::Put => 0.0,
        });
    }

    if sigma <= 0.0 {
        // Deterministic world: S_T equals the forward almost surely.
        let fwd_intrinsic = s * df_q - k * df_r;
        return Ok(match option_type {
            OptionType::Call => fwd_intrinsic.max(0.0),
            OptionType::Put => (-fwd_intrinsic).max(0.0),
        });
    }

    let (d1, d2) = bs_d1_d2(s, k, t, sigma, r, q)?;
    Ok(match option_type {
        OptionType::Call => s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2),
        OptionType::Put => k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1),
    })
}

/// Full analytic Greeks for a European BSM option (see [`Greeks`] for units).
///
/// In the degenerate region (`t = 0`, `sigma = 0` or `k = 0`) the pointwise
/// limits are returned: `gamma = vega = vanna = volga = 0`, `delta` is the
/// discounted forward-moneyness indicator (`±e^{−qt}` if in the money, else
/// 0; the exactly-at-the-money boundary maps to the call-ITM branch), and
/// `theta`/`rho` are the derivatives of the limiting price. These
/// conventions are part of the cross-language contract.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid market inputs.
pub fn bs_greeks(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
) -> Result<Greeks, DpeError> {
    validate_market_inputs(s, k, t, sigma, r, q)?;

    let price = bs_price(s, k, t, sigma, r, q, option_type)?;
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    if t <= 0.0 || sigma <= 0.0 || k <= 0.0 {
        // Limit region: the payoff is (discounted) intrinsic on the forward.
        let fwd = s * df_q - k * df_r; // sign of forward moneyness
        let (delta, theta, rho) = match option_type {
            OptionType::Call => {
                let itm = fwd >= 0.0 || k <= 0.0;
                if itm {
                    (df_q, q * s * df_q - r * k * df_r, k * t * df_r)
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
            OptionType::Put => {
                let itm = fwd < 0.0 && k > 0.0;
                if itm {
                    (-df_q, r * k * df_r - q * s * df_q, -k * t * df_r)
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
        };
        return Ok(Greeks {
            price,
            delta,
            gamma: 0.0,
            vega: 0.0,
            theta,
            rho,
            vanna: 0.0,
            volga: 0.0,
        });
    }

    let (d1, d2) = bs_d1_d2(s, k, t, sigma, r, q)?;
    let sq_t = t.sqrt();
    let pdf_d1 = norm_pdf(d1);

    let gamma = df_q * pdf_d1 / (s * sigma * sq_t);
    let vega = s * df_q * pdf_d1 * sq_t;
    let vanna = -df_q * pdf_d1 * d2 / sigma;
    let volga = vega * d1 * d2 / sigma;

    let (delta, theta, rho) = match option_type {
        OptionType::Call => (
            df_q * norm_cdf(d1),
            -s * df_q * pdf_d1 * sigma / (2.0 * sq_t) - r * k * df_r * norm_cdf(d2)
                + q * s * df_q * norm_cdf(d1),
            k * t * df_r * norm_cdf(d2),
        ),
        OptionType::Put => (
            df_q * (norm_cdf(d1) - 1.0),
            -s * df_q * pdf_d1 * sigma / (2.0 * sq_t) + r * k * df_r * norm_cdf(-d2)
                - q * s * df_q * norm_cdf(-d1),
            -k * t * df_r * norm_cdf(-d2),
        ),
    };

    Ok(Greeks {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
        vanna,
        volga,
    })
}

/// Garman-Kohlhagen FX option price: BSM with `r = rd`, `q = rf`.
///
/// `s` and `k` are FX rates quoted domestic-per-foreign (e.g. USD per EUR
/// for EURUSD); the price is in domestic currency per unit of foreign
/// notional.
///
/// # Errors
///
/// Same as [`bs_price`].
pub fn gk_price(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    rd: f64,
    rf: f64,
    option_type: OptionType,
) -> Result<f64, DpeError> {
    bs_price(s, k, t, sigma, rd, rf, option_type)
}

/// Garman-Kohlhagen Greeks: BSM Greeks with `r = rd`, `q = rf`.
///
/// The reported `delta` is the premium-excluded **spot delta**
/// `e^{−rf t} N(d1)`; see [`crate::fx`] for spot/forward delta-convention
/// conversions. `rho` is the sensitivity to the domestic rate `rd`.
///
/// # Errors
///
/// Same as [`bs_greeks`].
pub fn gk_greeks(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    rd: f64,
    rf: f64,
    option_type: OptionType,
) -> Result<Greeks, DpeError> {
    bs_greeks(s, k, t, sigma, rd, rf, option_type)
}

/// Hard bracket cap for the implied-vol search: 2000% vol, beyond any market.
const IV_MAX_SIGMA: f64 = 20.0;

/// Implied volatility with the contract defaults `tol = 1e-10`,
/// `max_iter = 100`. See [`implied_vol_with`] for the full-control variant.
///
/// # Errors
///
/// See [`implied_vol_with`].
pub fn implied_vol(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, DpeError> {
    implied_vol_with(price, s, k, t, r, q, option_type, 1e-10, 100)
}

/// Implied Black-Scholes volatility via bracketed Newton with bisection
/// fallback.
///
/// Robustness strategy:
///
/// 1. **No-arbitrage bounds first.** A European call must satisfy
///    `max(s e^{−qt} − k e^{−rt}, 0) < price < s e^{−qt}` (mirrored for
///    puts). A price at or outside these bounds has no finite implied vol,
///    so the function returns [`DpeError::NoArbitrage`] naming the violated
///    bound rather than letting a root-finder wander.
/// 2. **Bracketing.** The BSM price is strictly increasing in sigma on the
///    open no-arb interval, so a root is bracketed in `[lo, hi]` where `hi`
///    doubles from 1.0 until the model price exceeds the target (capped at
///    2000% vol as a NaN guard).
/// 3. **Safeguarded Newton.** Newton steps use analytic vega; whenever a
///    step would leave the bracket or vega is too small (deep ITM/OTM, where
///    vega underflows), the step falls back to bisection on the maintained
///    bracket. This converges for any admissible input.
///
/// Converged when the absolute price error is below `tol` or the bracket
/// width is below 1e-12. Round-trips `sigma → price → sigma` to 1e-6
/// absolute for sigma in [0.05, 0.8]. `tol` must be > 0 and `max_iter >= 1`;
/// if neither stopping rule is met within `max_iter` Newton/bisection steps
/// the function returns an error — it never returns a half-converged root
/// as if it had converged.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on invalid inputs (including `tol <= 0`,
/// `max_iter < 1`) or `t <= 0` / `k <= 0` (no vol is identifiable there);
/// [`DpeError::NoArbitrage`] when `price` is at or outside the no-arbitrage
/// bounds or the vol cap is exceeded; [`DpeError::NoConvergence`] when the
/// iteration budget is exhausted (message contains "did not converge" and
/// the residual).
#[allow(clippy::too_many_arguments)]
pub fn implied_vol_with(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    tol: f64,
    max_iter: usize,
) -> Result<f64, DpeError> {
    validate_market_inputs(s, k, t, 0.0, r, q)?;
    require_finite("price", price)?;
    require_positive("t", t)?;
    require_positive("k", k)?;
    require_positive("tol", tol)?;
    if max_iter < 1 {
        return Err(DpeError::InvalidInput(format!(
            "max_iter must be an integer >= 1, got {max_iter}"
        )));
    }

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    let (lower, upper) = match option_type {
        OptionType::Call => ((s * df_q - k * df_r).max(0.0), s * df_q),
        OptionType::Put => ((k * df_r - s * df_q).max(0.0), k * df_r),
    };

    if price <= lower {
        return Err(DpeError::NoArbitrage(format!(
            "price {price} violates the no-arbitrage lower bound {lower} \
             (below intrinsic, discounted); no implied vol exists"
        )));
    }
    if price >= upper {
        return Err(DpeError::NoArbitrage(format!(
            "price {price} violates the no-arbitrage upper bound {upper}; \
             no implied vol exists"
        )));
    }

    let f = |sig: f64| -> Result<f64, DpeError> {
        Ok(bs_price(s, k, t, sig, r, q, option_type)? - price)
    };

    // Bracket the root: price is monotone increasing in sigma, and
    // f(0) = discounted intrinsic - price < 0 by the bound check above.
    // hi doubles 1 -> 2 -> 4 -> 8 -> 16 -> 20 (clipped to the cap) and the
    // cap itself is tested before giving up, so the effective cap is exactly
    // IV_MAX_SIGMA.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while f(hi)? < 0.0 {
        if hi >= IV_MAX_SIGMA {
            return Err(DpeError::NoArbitrage(format!(
                "implied vol exceeds cap {IV_MAX_SIGMA}; price {price} is \
                 numerically indistinguishable from the upper bound"
            )));
        }
        hi = (2.0 * hi).min(IV_MAX_SIGMA);
    }

    // Safeguarded Newton: max_iter Newton/bisection steps from the bracket
    // midpoint; the residual is checked before every step and once more
    // after the last one, so a returned sigma always satisfies a stopping
    // rule.
    let mut sigma = 0.5 * (lo + hi); // initial guess: bracket midpoint
    let mut diff = f(sigma)?;
    for _ in 0..max_iter {
        if diff.abs() < tol {
            return Ok(sigma);
        }
        // Maintain the bracket around the root.
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        let vega = bs_greeks(s, k, t, sigma, r, q, option_type)?.vega;
        if vega > 1e-12 {
            let candidate = sigma - diff / vega;
            if lo < candidate && candidate < hi {
                sigma = candidate;
                diff = f(sigma)?;
                continue;
            }
        }
        // Newton unusable (tiny vega or step outside bracket): bisect.
        sigma = 0.5 * (lo + hi);
        if hi - lo < 1e-12 {
            return Ok(sigma);
        }
        diff = f(sigma)?;
    }
    if diff.abs() < tol {
        return Ok(sigma);
    }
    // Iteration budget exhausted without meeting either stopping rule:
    // report honestly rather than hand back a half-converged root.
    Err(DpeError::NoConvergence(format!(
        "implied vol did not converge in {max_iter} iterations \
         (|model - price| = {} > tol {tol}, bracket [{lo}, {hi}]); \
         increase max_iter or loosen tol",
        diff.abs()
    )))
}
