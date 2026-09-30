/// \file validation.hpp
/// \brief Internal input validation shared by every pricing module.
///
/// All public functions funnel their inputs through these helpers so error
/// behaviour is uniform: bad input always throws std::invalid_argument with
/// a message naming the offending parameter (the cross-language contract).
/// This header is private to the library sources; it is not installed.

#ifndef DPE_SRC_VALIDATION_HPP
#define DPE_SRC_VALIDATION_HPP

#include <cstdint>

namespace dpe {
namespace detail {

/// Throw std::invalid_argument unless value is finite (NaN/inf guard).
void require_finite(const char* name, double value);

/// Throw std::invalid_argument unless value is finite and > 0.
void require_positive(const char* name, double value);

/// Throw std::invalid_argument unless value is finite and >= 0.
void require_non_negative(const char* name, double value);

/// Validate the common Black-Scholes market inputs:
/// s > 0; k >= 0; t >= 0; sigma >= 0; r, q finite.
void validate_market_inputs(double s, double k, double t, double sigma, double r,
                            double q);

/// Largest admissible RNG seed (2^63 - 1). The seed domain [0, kSeedMax] is
/// pinned by the cross-language contract (API_SPEC §5) so a seed means the
/// same thing in Python, C++, Rust and Java and is never silently wrapped.
constexpr std::int64_t kSeedMax = 9223372036854775807LL;

/// Throw std::invalid_argument unless 0 <= seed <= kSeedMax.
void require_seed(std::int64_t seed);

/// Validate Monte Carlo controls: n_paths >= 2 (>= 4 and even when
/// antithetic), n_steps >= 1, seed in [0, kSeedMax].
void validate_mc(std::int64_t n_paths, std::int64_t seed, bool antithetic,
                 int n_steps = 1);

/// Throw std::invalid_argument unless rel_bump is finite and in (0, 1).
void require_rel_bump(double rel_bump);

}  // namespace detail
}  // namespace dpe

#endif  // DPE_SRC_VALIDATION_HPP
