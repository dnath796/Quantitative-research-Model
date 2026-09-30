/// \file monte_carlo.hpp
/// \brief Monte Carlo engine: exact-step GBM, exotics, variance reduction.
///
/// Simulation uses the exact GBM step
///   S_{t+h} = S_t exp((r - q - sigma^2/2) h + sigma sqrt(h) Z), Z ~ N(0,1),
/// which is exact in distribution at the grid dates (no Euler bias in the
/// marginals). Path functionals that look between grid dates (barrier
/// crossings, lookback extrema) still carry monitoring bias; see the
/// individual pricers.
///
/// Variance reduction:
///  - antithetic variates: each Z is paired with -Z; statistically each pair
///    average is one i.i.d. sample, so standard errors are computed over the
///    n_paths/2 pair averages, never over raw correlated paths;
///  - control variates: adjusted samples y - beta (x - E[x]) with
///    beta = sample-cov(y, x) / sample-var(x) estimated from the same run
///    (beta = 0 if var(x) = 0). Controls: discounted terminal spot with
///    known mean s e^{-qt} for the European vanilla; the geometric-Asian
///    payoff with its closed-form price for the arithmetic Asian.
///
/// Every pricer takes an integer seed in [0, 2^63 - 1] (the domain pinned
/// by API_SPEC §5 for every port; negative seeds throw) and is fully
/// deterministic given it: normals come from std::mt19937_64 (whose output
/// the standard fully specifies) through a fixed Box-Muller recipe rather
/// than std::normal_distribution (whose algorithm is implementation-defined
/// and differs between libstdc++ and libc++). RNG streams are
/// language-specific: identical inputs + seed give identical output within
/// C++, not across ports.

#ifndef DPE_MONTE_CARLO_HPP
#define DPE_MONTE_CARLO_HPP

#include <cstdint>
#include <vector>

#include "dpe/types.hpp"

namespace dpe {

/// Simulate GBM paths with the exact log step.
///
/// Returns a matrix of shape [n_paths][n_steps + 1] whose column 0 is the
/// spot s and whose column i holds S(i * t / n_steps). Deterministic given
/// seed. With antithetic = true, n_paths must be even (>= 4) and rows
/// [n_paths/2 ..) use the negated normal increments of rows [0 .. n_paths/2).
/// \throws std::invalid_argument on invalid inputs (requires t > 0).
std::vector<std::vector<double>> simulate_gbm_paths(double s, double t, double sigma,
                                                    double r, double q,
                                                    std::int64_t n_paths, int n_steps,
                                                    std::int64_t seed,
                                                    bool antithetic = false);

/// Closed-form price of a discretely-monitored geometric-average Asian.
///
/// The geometric mean of GBM sampled at t_i = i t / n, i = 1..n (spot at 0
/// is NOT a fixing) is lognormal with
///   m = ln s + (r - q - sigma^2/2) t (n+1) / (2n)
///   v = sigma^2 t (n+1)(2n+1) / (6 n^2)
/// and the price is Black's formula on EG = e^{m + v/2}. If v = 0 the value
/// is e^{-rt} max(+-(EG - k), 0). Used as the control variate for the
/// arithmetic Asian and as a deterministic golden value.
/// Requires t > 0, k > 0, n_fixings >= 1.
double geometric_asian_price(double s, double k, double t, double sigma, double r,
                             double q, OptionType option_type, int n_fixings);

/// European vanilla by Monte Carlo (single exact step to expiry).
/// Optional control variate: discounted terminal spot e^{-rt} S_T with known
/// mean s e^{-qt} (the k = 0 Black-Scholes analytic value). Requires t > 0.
MCResult mc_european(double s, double k, double t, double sigma, double r, double q,
                     OptionType option_type, std::int64_t n_paths = 100000,
                     std::int64_t seed = 42, bool antithetic = true,
                     bool control_variate = false);

/// Arithmetic-average Asian by Monte Carlo, fixings at t_i = i t / n_steps,
/// i = 1..n_steps (spot at 0 is not a fixing). With control_variate = true
/// the geometric Asian on the same paths is the control, with its exact mean
/// from geometric_asian_price; correlation is typically > 99%. Requires k > 0.
MCResult mc_asian_arithmetic(double s, double k, double t, double sigma, double r,
                             double q, OptionType option_type,
                             std::int64_t n_paths = 100000, int n_steps = 50,
                             std::int64_t seed = 42, bool antithetic = true,
                             bool control_variate = true);

/// Up-and-out barrier option, discretely monitored on the step grid.
///
/// Pays the vanilla payoff at t iff max_i S(t_i) < barrier over ALL grid
/// dates including t_0 = 0; if s >= barrier the option is born dead and
/// {0, 0, 0, 0, n_paths} is returned without simulating.
///
/// Discretisation bias: a continuous barrier can be breached between grid
/// dates, so this discrete estimator over-prices the continuously-monitored
/// contract with O(1/sqrt(n_steps)) bias. The Broadie-Glasserman-Kou
/// correction (shift the barrier to B exp(0.5826 sigma sqrt(dt))) would
/// remove the leading term if a continuous contract were intended; this
/// function prices the discrete contract and applies no correction.
MCResult mc_barrier_up_out(double s, double k, double t, double sigma, double r,
                           double q, OptionType option_type, double barrier,
                           std::int64_t n_paths = 100000, int n_steps = 100,
                           std::int64_t seed = 42, bool antithetic = true);

/// Floating-strike lookback, discretely monitored on the grid (t = 0 included):
/// call pays S_T - min_i S(t_i), put pays max_i S(t_i) - S_T. Discrete
/// monitoring under-states the true extremum, so this under-prices the
/// continuous contract with O(1/sqrt(n_steps)) bias. No strike parameter.
MCResult mc_lookback_floating(double s, double t, double sigma, double r, double q,
                              OptionType option_type, std::int64_t n_paths = 100000,
                              int n_steps = 100, std::int64_t seed = 42,
                              bool antithetic = true);

/// Pathwise-derivative delta of a European vanilla:
/// call: e^{-rt} mean(1{S_T > k} S_T / s); put: -e^{-rt} mean(1{S_T < k} S_T / s).
/// Unbiased and low-variance, but requires an (a.e.) differentiable payoff.
MCResult mc_delta_pathwise(double s, double k, double t, double sigma, double r,
                           double q, OptionType option_type,
                           std::int64_t n_paths = 100000, std::int64_t seed = 42,
                           bool antithetic = true);

/// Central finite-difference delta with common random numbers: the SAME
/// normal draws price the payoff at s(1 +- rel_bump); per-path samples are
/// (payoff_up - payoff_down) / (2 s rel_bump), discounted. Reusing the
/// randomness makes the difference nearly noiseless; without CRN the
/// variance would explode as O(1/h^2). Requires 0 < rel_bump < 1 (both
/// bumped spots must stay positive).
MCResult mc_delta_fd_crn(double s, double k, double t, double sigma, double r,
                         double q, OptionType option_type,
                         std::int64_t n_paths = 100000, std::int64_t seed = 42,
                         double rel_bump = 1e-4, bool antithetic = true);

}  // namespace dpe

#endif  // DPE_MONTE_CARLO_HPP
