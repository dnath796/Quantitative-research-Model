//! Error type shared by every fallible function in the crate.

use std::error::Error;
use std::fmt;

/// Errors returned by the `dpe` pricing engine.
///
/// The cross-language contract maps invalid input to Python `ValueError`,
/// C++ `std::invalid_argument` and Java `IllegalArgumentException`; in Rust
/// bad input is **never** a panic — every public pricing function returns
/// `Result<_, DpeError>` and the message names the offending parameter or
/// violated bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpeError {
    /// A parameter failed validation (non-finite, out of range, unknown
    /// enum spelling, bad Monte Carlo configuration, ...).
    InvalidInput(String),
    /// A target price passed to the implied-vol solver violates the
    /// no-arbitrage bounds, or the solver's hard vol cap was exceeded.
    NoArbitrage(String),
    /// An iterative solver exhausted its iteration budget without meeting a
    /// stopping rule (the message carries the residual and bracket). A
    /// half-converged root is never returned as if it had converged.
    NoConvergence(String),
}

impl fmt::Display for DpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DpeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            DpeError::NoArbitrage(msg) => write!(f, "no-arbitrage violation: {msg}"),
            DpeError::NoConvergence(msg) => write!(f, "no convergence: {msg}"),
        }
    }
}

impl Error for DpeError {}
