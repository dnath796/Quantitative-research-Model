//! FX-specific helpers: forwards and delta-convention conversions.
//!
//! FX options are quoted and hedged under several delta conventions. This
//! module implements the two premium-excluded ("pips") conventions:
//!
//! * **Spot delta** — the sensitivity of the domestic-currency premium to
//!   the spot: for a call, `delta_spot = e^{-rf t} N(d1)`. This is what
//!   [`crate::black_scholes::gk_greeks`] reports.
//! * **Forward delta** — the hedge in forward contracts rather than spot:
//!   `delta_fwd = N(d1)` for a call. One forward contract has spot
//!   sensitivity `e^{-rf t}` (a forward on one unit of foreign currency is
//!   replicated by holding `e^{-rf t}` units of foreign cash), so the two
//!   conventions differ by exactly that factor:
//!   `delta_fwd = e^{+rf t} * delta_spot`.
//!
//! The conversion is payoff-independent — pure discounting — so the same
//! factor applies to calls and puts and at any strike, and the two
//! conversions are exact mutual inverses. Interbank markets typically quote
//! forward delta beyond ~1y and spot delta for short dates; converting
//! between them correctly is essential when interpolating a vol surface
//! quoted in deltas. Premium-included deltas are out of scope.

use crate::error::DpeError;
use crate::validation::{require_finite, require_non_negative, require_positive};

/// Covered-interest-parity FX forward: `F = s * e^{(rd - rf) t}`.
///
/// `s` is spot domestic-per-foreign; `rd`/`rf` are continuously compounded
/// domestic/foreign rates (either may be negative).
///
/// # Errors
///
/// [`DpeError::InvalidInput`] unless `s > 0`, `t >= 0` and both rates are
/// finite.
pub fn fx_forward(s: f64, t: f64, rd: f64, rf: f64) -> Result<f64, DpeError> {
    require_positive("s", s)?;
    require_non_negative("t", t)?;
    require_finite("rd", rd)?;
    require_finite("rf", rf)?;
    Ok(s * ((rd - rf) * t).exp())
}

/// Convert a premium-excluded spot delta to the forward-delta convention:
/// `delta_fwd = e^{rf t} * delta_spot`. Sign-preserving (works for put
/// deltas).
///
/// # Errors
///
/// [`DpeError::InvalidInput`] unless `delta_spot` and `rf` are finite and
/// `t >= 0`.
pub fn spot_delta_to_forward(delta_spot: f64, t: f64, rf: f64) -> Result<f64, DpeError> {
    require_finite("delta_spot", delta_spot)?;
    require_non_negative("t", t)?;
    require_finite("rf", rf)?;
    Ok((rf * t).exp() * delta_spot)
}

/// Convert a premium-excluded forward delta to the spot-delta convention:
/// `delta_spot = e^{-rf t} * delta_forward` — the exact inverse of
/// [`spot_delta_to_forward`].
///
/// # Errors
///
/// [`DpeError::InvalidInput`] unless `delta_forward` and `rf` are finite and
/// `t >= 0`.
pub fn forward_delta_to_spot(delta_forward: f64, t: f64, rf: f64) -> Result<f64, DpeError> {
    require_finite("delta_forward", delta_forward)?;
    require_non_negative("t", t)?;
    require_finite("rf", rf)?;
    Ok((-rf * t).exp() * delta_forward)
}
