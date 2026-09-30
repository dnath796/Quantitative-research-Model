/// \file utils.hpp
/// \brief Small numerical utilities shared by the demo and tests.

#ifndef DPE_UTILS_HPP
#define DPE_UTILS_HPP

#include <vector>

namespace dpe {

/// Annualised close-to-close historical volatility from a price series.
///
/// Sample standard deviation (ddof = 1) of the log returns ln(P_i / P_{i-1})
/// scaled by sqrt(periods_per_year) — the standard realised-vol estimator
/// under the GBM assumption that log returns are i.i.d. normal.
///
/// \throws std::invalid_argument on fewer than 3 prices (no meaningful
/// ddof = 1 standard deviation from fewer than 2 returns), on any
/// non-positive or non-finite price, or if periods_per_year < 1.
double historical_vol(const std::vector<double>& prices, int periods_per_year = 252);

}  // namespace dpe

#endif  // DPE_UTILS_HPP
