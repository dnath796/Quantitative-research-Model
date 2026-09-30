//! Contract enumerations: option type, exercise style, lattice method.
//!
//! The canonical string spellings (`"call"`, `"put"`, `"european"`,
//! `"american"`, `"crr"`, `"jr"`) are part of the cross-language contract;
//! parsing is case-insensitive and anything else is an invalid input.

use std::fmt;
use std::str::FromStr;

use crate::error::DpeError;

/// Vanilla option type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to buy: payoff `max(S - K, 0)`.
    Call,
    /// Right to sell: payoff `max(K - S, 0)`.
    Put,
}

impl OptionType {
    /// Parse `"call"` / `"put"` case-insensitively.
    pub fn parse(text: &str) -> Result<Self, DpeError> {
        match text.to_ascii_lowercase().as_str() {
            "call" => Ok(OptionType::Call),
            "put" => Ok(OptionType::Put),
            _ => Err(DpeError::InvalidInput(format!(
                "option_type must be 'call' or 'put', got '{text}'"
            ))),
        }
    }

    /// Undiscounted vanilla payoff at spot `s`, strike `k`.
    pub(crate) fn payoff(self, s: f64, k: f64) -> f64 {
        match self {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        }
    }
}

impl FromStr for OptionType {
    type Err = DpeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        OptionType::parse(s)
    }
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OptionType::Call => "call",
            OptionType::Put => "put",
        })
    }
}

/// Exercise style of a vanilla option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExerciseStyle {
    /// Exercisable at expiry only.
    European,
    /// Exercisable at every tree date up to and including expiry.
    American,
}

impl ExerciseStyle {
    /// Parse `"european"` / `"american"` case-insensitively.
    pub fn parse(text: &str) -> Result<Self, DpeError> {
        match text.to_ascii_lowercase().as_str() {
            "european" => Ok(ExerciseStyle::European),
            "american" => Ok(ExerciseStyle::American),
            _ => Err(DpeError::InvalidInput(format!(
                "style must be 'european' or 'american', got '{text}'"
            ))),
        }
    }
}

impl FromStr for ExerciseStyle {
    type Err = DpeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ExerciseStyle::parse(s)
    }
}

impl fmt::Display for ExerciseStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ExerciseStyle::European => "european",
            ExerciseStyle::American => "american",
        })
    }
}

/// Binomial-lattice parameterisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeMethod {
    /// Cox-Ross-Rubinstein: `u = e^{sigma sqrt(dt)}`, `d = 1/u`,
    /// risk-neutral `p` from the first moment; recombines around the spot.
    Crr,
    /// Jarrow-Rudd (equal probability): drift absorbed into node placement,
    /// `p = 1/2`; the lattice drifts with the forward.
    Jr,
}

impl TreeMethod {
    /// Parse `"crr"` / `"jr"` case-insensitively.
    pub fn parse(text: &str) -> Result<Self, DpeError> {
        match text.to_ascii_lowercase().as_str() {
            "crr" => Ok(TreeMethod::Crr),
            "jr" => Ok(TreeMethod::Jr),
            _ => Err(DpeError::InvalidInput(format!(
                "method must be 'crr' or 'jr', got '{text}'"
            ))),
        }
    }
}

impl FromStr for TreeMethod {
    type Err = DpeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TreeMethod::parse(s)
    }
}

impl fmt::Display for TreeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TreeMethod::Crr => "crr",
            TreeMethod::Jr => "jr",
        })
    }
}
