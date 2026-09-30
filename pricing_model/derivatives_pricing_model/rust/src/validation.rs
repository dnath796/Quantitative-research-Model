//! Shared input validation.
//!
//! All public pricing functions funnel their inputs through these helpers so
//! that error behaviour is uniform: bad input always yields
//! [`DpeError::InvalidInput`] with a message naming the offending parameter.

use crate::error::DpeError;

/// Fail unless `value` is finite (NaN/inf are always invalid).
///
/// The guard sits at the boundary so no NaN can silently propagate through a
/// pricing formula and surface as a nonsense Greek downstream.
pub(crate) fn require_finite(name: &str, value: f64) -> Result<(), DpeError> {
    if !value.is_finite() {
        return Err(DpeError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(())
}

/// Fail unless `value` is finite and strictly positive.
pub(crate) fn require_positive(name: &str, value: f64) -> Result<(), DpeError> {
    require_finite(name, value)?;
    if value <= 0.0 {
        return Err(DpeError::InvalidInput(format!(
            "{name} must be > 0, got {value}"
        )));
    }
    Ok(())
}

/// Fail unless `value` is finite and non-negative.
pub(crate) fn require_non_negative(name: &str, value: f64) -> Result<(), DpeError> {
    require_finite(name, value)?;
    if value < 0.0 {
        return Err(DpeError::InvalidInput(format!(
            "{name} must be >= 0, got {value}"
        )));
    }
    Ok(())
}

/// Validate the common Black-Scholes market inputs.
///
/// Rules (identical in every language port): `s > 0`; `k >= 0` (`k = 0` is a
/// forward claim); `t >= 0` in years (`t = 0` = at expiry); `sigma >= 0` as a
/// decimal per sqrt-year; `r`, `q` finite (may be negative).
pub(crate) fn validate_market_inputs(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    q: f64,
) -> Result<(), DpeError> {
    require_positive("s", s)?;
    require_non_negative("k", k)?;
    require_non_negative("t", t)?;
    require_non_negative("sigma", sigma)?;
    require_finite("r", r)?;
    require_finite("q", q)?;
    Ok(())
}
