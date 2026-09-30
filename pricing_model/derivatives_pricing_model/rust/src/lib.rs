//! # dpe — Derivatives Pricing Engine (Rust port)
//!
//! Rust implementation of the cross-language `dpe` contract (see
//! `API_SPEC.md` at the project root; the Python package under
//! `python/src/dpe/` is the reference implementation).
//!
//! ## Modules
//!
//! * [`black_scholes`] — analytic BSM / Garman-Kohlhagen prices, full Greeks
//!   (delta, gamma, vega, theta, rho, vanna, volga) and a robust implied-vol
//!   solver.
//! * [`binomial`] — CRR and Jarrow-Rudd lattices, European and American
//!   exercise, tree Greeks, optional Richardson averaging.
//! * [`monte_carlo`] — exact-step GBM simulation; European, arithmetic
//!   Asian, up-and-out barrier and floating-strike lookback pricing;
//!   antithetic and control variates; pathwise and CRN-FD deltas; standard
//!   errors and 95% CIs.
//! * [`fx`] — FX forward and spot/forward delta-convention conversions.
//! * [`utils`] — historical (realised) volatility estimator.
//!
//! ## Conventions
//!
//! Time in **years**, vol and rates as decimals, theta per year, vega/rho
//! per unit; see the module docs for the full list. Invalid input is never a
//! panic: every fallible function returns `Result<_, `[`DpeError`]`>` with a
//! message naming the offending parameter.
//!
//! ## Example
//!
//! ```
//! use dpe::{bs_price, bs_greeks, OptionType};
//!
//! let price = bs_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, OptionType::Call).unwrap();
//! assert!((price - 10.4506).abs() < 1e-4);
//! let greeks = bs_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, OptionType::Call).unwrap();
//! assert!(greeks.delta > 0.5 && greeks.delta < 0.7);
//! ```

pub mod binomial;
pub mod black_scholes;
pub mod error;
pub mod fx;
pub mod math;
pub mod monte_carlo;
pub mod types;
pub mod utils;

mod validation;

pub use binomial::{binomial_greeks, binomial_price, BinomialGreeks};
pub use black_scholes::{
    bs_d1_d2, bs_greeks, bs_price, gk_greeks, gk_price, implied_vol, implied_vol_with, Greeks,
};
pub use error::DpeError;
pub use fx::{forward_delta_to_spot, fx_forward, spot_delta_to_forward};
pub use math::{norm_cdf, norm_pdf};
pub use monte_carlo::{
    geometric_asian_price, mc_asian_arithmetic, mc_barrier_up_out, mc_delta_fd_crn,
    mc_delta_pathwise, mc_european, mc_lookback_floating, simulate_gbm_paths, MCResult,
    SEED_MAX,
};
pub use types::{ExerciseStyle, OptionType, TreeMethod};
pub use utils::historical_vol;
