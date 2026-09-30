#include "dpe/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "dpe/black_scholes.hpp"
#include "validation.hpp"

namespace dpe {
namespace {

constexpr double kZ95 = 1.959963984540054;  // two-sided 95% normal quantile
constexpr double kTwoPi = 6.283185307179586476925286766559005768;

/// Deterministic standard-normal stream.
///
/// std::mt19937_64 is fully specified by the standard (its output for a
/// given seed is identical on every conforming implementation), but
/// std::normal_distribution is NOT — libstdc++ and libc++ use different
/// algorithms and produce different streams from the same engine state. To
/// make "same seed => same numbers" hold across toolchains, the normals are
/// generated here from the raw 64-bit engine output with a fixed recipe:
/// uniforms u = ((x >> 11) + 1) * 2^-53 in (0, 1], then Box-Muller. The only
/// remaining toolchain dependence is the libm rounding of log/cos/sin (at
/// most an ulp). RNG streams are language-local: they are not required to
/// match other ports, only to be reproducible within C++.
class NormalStream {
 public:
  explicit NormalStream(std::int64_t seed)
      : rng_(static_cast<std::uint64_t>(seed)), has_spare_(false), spare_(0.0) {}

  double next() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = uniform_open_closed();
    const double u2 = uniform_open_closed();
    const double radius = std::sqrt(-2.0 * std::log(u1));  // u1 > 0: finite
    const double angle = kTwoPi * u2;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

 private:
  /// Uniform in (0, 1]: the top 53 bits of the engine output plus one, scaled.
  double uniform_open_closed() {
    const std::uint64_t bits = rng_();
    return static_cast<double>((bits >> 11) + 1u) * 0x1.0p-53;
  }

  std::mt19937_64 rng_;
  bool has_spare_;
  double spare_;
};

/// Mean / ddof-1 standard error over the i.i.d. samples -> MCResult.
MCResult make_stats(const std::vector<double>& samples, std::int64_t n_paths) {
  const auto n = samples.size();
  double sum = 0.0;
  for (double v : samples) sum += v;
  const double mean = sum / static_cast<double>(n);
  double ss = 0.0;
  for (double v : samples) {
    const double d = v - mean;
    ss += d * d;
  }
  const double sd = std::sqrt(ss / static_cast<double>(n - 1));
  const double se = sd / std::sqrt(static_cast<double>(n));
  return MCResult{mean, se, mean - kZ95 * se, mean + kZ95 * se, n_paths};
}

/// Optimal-beta control-variate adjustment y - beta (x - E[x]), with beta
/// estimated in-sample (beta = 0 when the control is degenerate).
void control_adjust(std::vector<double>* y, const std::vector<double>& x,
                    double x_mean) {
  const auto n = y->size();
  double my = 0.0, mx = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    my += (*y)[i];
    mx += x[i];
  }
  my /= static_cast<double>(n);
  mx /= static_cast<double>(n);
  double cov = 0.0, var_x = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - mx;
    cov += ((*y)[i] - my) * dx;
    var_x += dx * dx;
  }
  var_x /= static_cast<double>(n - 1);
  if (var_x <= 0.0) return;  // degenerate control (e.g. sigma = 0)
  const double beta = (cov / static_cast<double>(n - 1)) / var_x;
  for (std::size_t i = 0; i < n; ++i) (*y)[i] -= beta * (x[i] - x_mean);
}

double vanilla_payoff(double st, double k, OptionType ot) {
  return ot == OptionType::Call ? std::max(st - k, 0.0) : std::max(k - st, 0.0);
}

/// Shared validation for the single-step (terminal-spot) pricers.
void validate_single_step(double s, double k, double t, double sigma, double r,
                          double q, std::int64_t n_paths, std::int64_t seed,
                          bool antithetic) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::validate_mc(n_paths, seed, antithetic);
}

}  // namespace

std::vector<std::vector<double>> simulate_gbm_paths(double s, double t, double sigma,
                                                    double r, double q,
                                                    std::int64_t n_paths, int n_steps,
                                                    std::int64_t seed,
                                                    bool antithetic) {
  detail::validate_market_inputs(s, 1.0, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::validate_mc(n_paths, seed, antithetic, n_steps);

  const double dt = t / n_steps;
  const double drift = (r - q - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);
  const auto np = static_cast<std::size_t>(n_paths);
  const auto ns = static_cast<std::size_t>(n_steps);
  const std::size_t fresh = antithetic ? np / 2 : np;

  NormalStream normals(seed);
  std::vector<std::vector<double>> paths(np, std::vector<double>(ns + 1, s));
  for (std::size_t i = 0; i < fresh; ++i) {
    double log_s = 0.0;
    double log_s_anti = 0.0;
    for (std::size_t j = 0; j < ns; ++j) {
      const double z = normals.next();
      log_s += drift + vol * z;
      paths[i][j + 1] = s * std::exp(log_s);
      if (antithetic) {
        log_s_anti += drift - vol * z;
        paths[i + fresh][j + 1] = s * std::exp(log_s_anti);
      }
    }
  }
  return paths;
}

double geometric_asian_price(double s, double k, double t, double sigma, double r,
                             double q, OptionType option_type, int n_fixings) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::require_positive("k", k);
  if (n_fixings < 1) {
    throw std::invalid_argument("n_fixings must be an integer >= 1, got " +
                                std::to_string(n_fixings));
  }

  // The geometric mean of GBM at equally spaced fixings is lognormal; the
  // variance uses sum_{i,j} min(i,j) = n(n+1)(2n+1)/6.
  const double n = static_cast<double>(n_fixings);
  const double mean_ln =
      std::log(s) + (r - q - 0.5 * sigma * sigma) * t * (n + 1.0) / (2.0 * n);
  const double var_ln = sigma * sigma * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
  const double df = std::exp(-r * t);
  const double eg = std::exp(mean_ln + 0.5 * var_ln);
  if (var_ln <= 0.0) {
    return df * vanilla_payoff(eg, k, option_type);
  }
  const double sd = std::sqrt(var_ln);
  const double d1 = (std::log(eg / k) + 0.5 * var_ln) / sd;
  const double d2 = d1 - sd;
  if (option_type == OptionType::Call) {
    return df * (eg * norm_cdf(d1) - k * norm_cdf(d2));
  }
  return df * (k * norm_cdf(-d2) - eg * norm_cdf(-d1));
}

MCResult mc_european(double s, double k, double t, double sigma, double r, double q,
                     OptionType option_type, std::int64_t n_paths, std::int64_t seed,
                     bool antithetic, bool control_variate) {
  validate_single_step(s, k, t, sigma, r, q, n_paths, seed, antithetic);

  const double drift = (r - q - 0.5 * sigma * sigma) * t;
  const double vol = sigma * std::sqrt(t);
  const double df = std::exp(-r * t);
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);

  NormalStream normals(seed);
  std::vector<double> y(n_samples);
  std::vector<double> x;
  if (control_variate) x.resize(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double z = normals.next();
    const double st_up = s * std::exp(drift + vol * z);
    double pay = df * vanilla_payoff(st_up, k, option_type);
    double ctrl = df * st_up;
    if (antithetic) {
      const double st_dn = s * std::exp(drift - vol * z);
      pay = 0.5 * (pay + df * vanilla_payoff(st_dn, k, option_type));
      ctrl = 0.5 * (ctrl + df * st_dn);
    }
    y[i] = pay;
    if (control_variate) x[i] = ctrl;
  }
  if (control_variate) {
    // Discounted terminal spot is a martingale with known mean s e^{-qt}.
    control_adjust(&y, x, s * std::exp(-q * t));
  }
  return make_stats(y, n_paths);
}

MCResult mc_asian_arithmetic(double s, double k, double t, double sigma, double r,
                             double q, OptionType option_type, std::int64_t n_paths,
                             int n_steps, std::int64_t seed, bool antithetic,
                             bool control_variate) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::require_positive("k", k);
  detail::validate_mc(n_paths, seed, antithetic, n_steps);

  const double dt = t / n_steps;
  const double drift = (r - q - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);
  const double df = std::exp(-r * t);
  const double inv_n = 1.0 / static_cast<double>(n_steps);
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);
  const auto ns = static_cast<std::size_t>(n_steps);

  NormalStream normals(seed);
  std::vector<double> z(ns);
  std::vector<double> y(n_samples);
  std::vector<double> x;
  if (control_variate) x.resize(n_samples);

  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < ns; ++j) z[j] = normals.next();
    double pay = 0.0, ctrl = 0.0;
    const int reps = antithetic ? 2 : 1;
    for (int rep = 0; rep < reps; ++rep) {
      const double sign = rep == 0 ? 1.0 : -1.0;
      double log_s = 0.0, sum = 0.0, log_sum = 0.0;
      for (std::size_t j = 0; j < ns; ++j) {
        log_s += drift + sign * vol * z[j];
        sum += s * std::exp(log_s);
        log_sum += log_s;  // ln(S(t_j)/s), for the geometric mean
      }
      pay += df * vanilla_payoff(sum * inv_n, k, option_type);
      ctrl += df * vanilla_payoff(s * std::exp(log_sum * inv_n), k, option_type);
    }
    y[i] = pay / reps;
    if (control_variate) x[i] = ctrl / reps;
  }
  if (control_variate) {
    const double x_mean =
        geometric_asian_price(s, k, t, sigma, r, q, option_type, n_steps);
    control_adjust(&y, x, x_mean);
  }
  return make_stats(y, n_paths);
}

MCResult mc_barrier_up_out(double s, double k, double t, double sigma, double r,
                           double q, OptionType option_type, double barrier,
                           std::int64_t n_paths, int n_steps, std::int64_t seed,
                           bool antithetic) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::require_positive("barrier", barrier);
  detail::validate_mc(n_paths, seed, antithetic, n_steps);

  if (s >= barrier) {
    // Born dead: t_0 = 0 is a monitoring date, so the price is exactly 0.
    return MCResult{0.0, 0.0, 0.0, 0.0, n_paths};
  }

  const double dt = t / n_steps;
  const double drift = (r - q - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);
  const double df = std::exp(-r * t);
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);
  const auto ns = static_cast<std::size_t>(n_steps);

  NormalStream normals(seed);
  std::vector<double> z(ns);
  std::vector<double> y(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < ns; ++j) z[j] = normals.next();
    double pay = 0.0;
    const int reps = antithetic ? 2 : 1;
    for (int rep = 0; rep < reps; ++rep) {
      const double sign = rep == 0 ? 1.0 : -1.0;
      double log_s = 0.0;
      bool alive = true;  // s < barrier already checked at t_0
      double terminal = s;
      for (std::size_t j = 0; j < ns; ++j) {
        log_s += drift + sign * vol * z[j];
        terminal = s * std::exp(log_s);
        if (terminal >= barrier) alive = false;
      }
      if (alive) pay += df * vanilla_payoff(terminal, k, option_type);
    }
    y[i] = pay / reps;
  }
  return make_stats(y, n_paths);
}

MCResult mc_lookback_floating(double s, double t, double sigma, double r, double q,
                              OptionType option_type, std::int64_t n_paths,
                              int n_steps, std::int64_t seed, bool antithetic) {
  detail::validate_market_inputs(s, 1.0, t, sigma, r, q);
  detail::require_positive("t", t);
  detail::validate_mc(n_paths, seed, antithetic, n_steps);

  const double dt = t / n_steps;
  const double drift = (r - q - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);
  const double df = std::exp(-r * t);
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);
  const auto ns = static_cast<std::size_t>(n_steps);

  NormalStream normals(seed);
  std::vector<double> z(ns);
  std::vector<double> y(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < ns; ++j) z[j] = normals.next();
    double pay = 0.0;
    const int reps = antithetic ? 2 : 1;
    for (int rep = 0; rep < reps; ++rep) {
      const double sign = rep == 0 ? 1.0 : -1.0;
      double log_s = 0.0;
      double lo = s, hi = s, terminal = s;  // extrema include S(0) = s
      for (std::size_t j = 0; j < ns; ++j) {
        log_s += drift + sign * vol * z[j];
        terminal = s * std::exp(log_s);
        if (terminal < lo) lo = terminal;
        if (terminal > hi) hi = terminal;
      }
      pay += option_type == OptionType::Call ? df * (terminal - lo)
                                             : df * (hi - terminal);
    }
    y[i] = pay / reps;
  }
  return make_stats(y, n_paths);
}

MCResult mc_delta_pathwise(double s, double k, double t, double sigma, double r,
                           double q, OptionType option_type, std::int64_t n_paths,
                           std::int64_t seed, bool antithetic) {
  validate_single_step(s, k, t, sigma, r, q, n_paths, seed, antithetic);

  const double drift = (r - q - 0.5 * sigma * sigma) * t;
  const double vol = sigma * std::sqrt(t);
  const double df = std::exp(-r * t);
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);

  // dS_T/dS_0 = S_T / S_0 for GBM; the vanilla payoff is Lipschitz, so the
  // derivative may be taken inside the expectation.
  const auto sample_of = [&](double st) {
    if (option_type == OptionType::Call) return st > k ? df * st / s : 0.0;
    return st < k ? -df * st / s : 0.0;
  };

  NormalStream normals(seed);
  std::vector<double> y(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double z = normals.next();
    double v = sample_of(s * std::exp(drift + vol * z));
    if (antithetic) {
      v = 0.5 * (v + sample_of(s * std::exp(drift - vol * z)));
    }
    y[i] = v;
  }
  return make_stats(y, n_paths);
}

MCResult mc_delta_fd_crn(double s, double k, double t, double sigma, double r,
                         double q, OptionType option_type, std::int64_t n_paths,
                         std::int64_t seed, double rel_bump, bool antithetic) {
  validate_single_step(s, k, t, sigma, r, q, n_paths, seed, antithetic);
  detail::require_rel_bump(rel_bump);

  const double drift = (r - q - 0.5 * sigma * sigma) * t;
  const double vol = sigma * std::sqrt(t);
  const double df = std::exp(-r * t);
  const double h = s * rel_bump;
  const std::size_t n_samples =
      static_cast<std::size_t>(antithetic ? n_paths / 2 : n_paths);

  // Common random numbers: the same growth factor prices both bumped spots,
  // so the sampling errors cancel path by path in the central difference.
  const auto diff_of = [&](double growth) {
    const double up = df * vanilla_payoff((s + h) * growth, k, option_type);
    const double dn = df * vanilla_payoff((s - h) * growth, k, option_type);
    return (up - dn) / (2.0 * h);
  };

  NormalStream normals(seed);
  std::vector<double> y(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double z = normals.next();
    double v = diff_of(std::exp(drift + vol * z));
    if (antithetic) {
      v = 0.5 * (v + diff_of(std::exp(drift - vol * z)));
    }
    y[i] = v;
  }
  return make_stats(y, n_paths);
}

}  // namespace dpe
