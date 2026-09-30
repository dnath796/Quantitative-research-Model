//! Small numerical utilities shared by the demo and tests.

use crate::error::DpeError;

/// Annualised close-to-close historical volatility from a price series.
///
/// Computes the sample standard deviation (ddof = 1) of the log returns
/// `ln(P_i / P_{i-1})` and scales by `sqrt(periods_per_year)` — the standard
/// realised-vol estimator under the GBM assumption that log returns are
/// i.i.d. normal.
///
/// # Errors
///
/// [`DpeError::InvalidInput`] on fewer than 3 prices (no meaningful
/// ddof = 1 standard deviation from fewer than 2 returns), on any
/// non-positive or non-finite price (log returns would be undefined), or on
/// `periods_per_year < 1`.
pub fn historical_vol(prices: &[f64], periods_per_year: u32) -> Result<f64, DpeError> {
    if prices.len() < 3 {
        return Err(DpeError::InvalidInput(format!(
            "prices must be a series with >= 3 points, got {}",
            prices.len()
        )));
    }
    if prices.iter().any(|&p| !p.is_finite() || p <= 0.0) {
        return Err(DpeError::InvalidInput(
            "prices must all be finite and > 0".to_string(),
        ));
    }
    if periods_per_year < 1 {
        return Err(DpeError::InvalidInput(format!(
            "periods_per_year must be >= 1, got {periods_per_year}"
        )));
    }
    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    Ok(var.sqrt() * f64::from(periods_per_year).sqrt())
}
