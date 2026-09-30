/// \file types.hpp
/// \brief Public value types and enumerations of the dpe pricing engine.
///
/// The canonical string spellings of the enums ("call", "put", "european",
/// "american", "crr", "jr") are part of the cross-language contract; the
/// parse_* helpers accept them case-insensitively and throw
/// std::invalid_argument on anything else.

#ifndef DPE_TYPES_HPP
#define DPE_TYPES_HPP

#include <cstdint>
#include <string>

namespace dpe {

/// Option payoff type.
enum class OptionType { Call, Put };

/// Exercise style.
enum class ExerciseStyle { European, American };

/// Binomial lattice parameterisation.
enum class TreeMethod {
  CRR,  ///< Cox-Ross-Rubinstein: u = exp(sigma sqrt(dt)), d = 1/u.
  JR    ///< Jarrow-Rudd equal-probability: drift absorbed into the nodes, p = 1/2.
};

/// Parse "call"/"put" (case-insensitive) into an OptionType.
/// \throws std::invalid_argument on any other string.
OptionType parse_option_type(const std::string& option_type);

/// Parse "european"/"american" (case-insensitive) into an ExerciseStyle.
/// \throws std::invalid_argument on any other string.
ExerciseStyle parse_exercise_style(const std::string& style);

/// Parse "crr"/"jr" (case-insensitive) into a TreeMethod.
/// \throws std::invalid_argument on any other string.
TreeMethod parse_tree_method(const std::string& method);

/// Full set of analytic BSM/Garman-Kohlhagen Greeks.
///
/// Units (fixed by the cross-language contract):
///  - delta = dV/dS (dimensionless);
///  - gamma = d2V/dS2;
///  - vega  = dV/dsigma per unit of vol (per 1.00 = 100 vol points);
///  - theta = dV/dt in calendar time per year (divide by 365 for per-day);
///  - rho   = dV/dr per unit of rate (domestic rate for FX);
///  - vanna = d2V/(dS dsigma);
///  - volga = d2V/dsigma2 (vomma).
struct Greeks {
  double price;
  double delta;
  double gamma;
  double vega;
  double theta;
  double rho;
  double vanna;
  double volga;
};

/// Tree-based price and Greeks (delta, gamma, theta per year).
struct BinomialGreeks {
  double price;
  double delta;
  double gamma;
  double theta;
};

/// A Monte Carlo estimate with its sampling uncertainty.
///
/// With antithetic variates the i.i.d. samples are the pair averages
/// (f(Z) + f(-Z))/2, so std_error is computed over n_paths/2 samples;
/// n_paths always reports the raw simulated path count.
struct MCResult {
  double value;      ///< point estimate (price or Greek)
  double std_error;  ///< standard error of the estimate
  double ci_low;     ///< value - 1.959963984540054 * std_error
  double ci_high;    ///< value + 1.959963984540054 * std_error
  std::int64_t n_paths;  ///< raw simulated paths (both antithetic halves)
};

}  // namespace dpe

#endif  // DPE_TYPES_HPP
