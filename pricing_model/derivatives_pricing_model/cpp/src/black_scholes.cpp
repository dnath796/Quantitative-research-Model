#include "dpe/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "validation.hpp"

namespace dpe {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
const double kSqrt2 = std::sqrt(2.0);
const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * kPi);

/// 2000% vol: beyond any market; used as a hard bracket cap for implied vol.
constexpr double kIvMaxSigma = 20.0;

/// Undiscounted intrinsic value at expiry (zero first in std::max so an
/// at-the-money tie yields +0.0, never -0.0).
double intrinsic(double s, double k, OptionType ot) {
  return ot == OptionType::Call ? std::max(0.0, s - k) : std::max(0.0, k - s);
}

/// (d1, d2) in the regular region t > 0, sigma > 0, k > 0 (callers guarantee).
void d1_d2(double s, double k, double t, double sigma, double r, double q,
           double* d1, double* d2) {
  const double sq_t = std::sqrt(t);
  *d1 = (std::log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sq_t);
  *d2 = *d1 - sigma * sq_t;
}

std::string fmt(double value) {
  std::ostringstream os;
  os.precision(17);
  os << value;
  return os.str();
}

}  // namespace

double norm_cdf(double x) {
  // erfc keeps full relative precision in the deep left tail, where
  // 0.5 * (1 + erf) would cancel catastrophically (deep-OTM prices).
  return 0.5 * std::erfc(-x / kSqrt2);
}

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double bs_price(double s, double k, double t, double sigma, double r, double q,
                OptionType option_type) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);

  if (t <= 0.0) return intrinsic(s, k, option_type);

  const double df_r = std::exp(-r * t);
  const double df_q = std::exp(-q * t);

  if (k <= 0.0) {
    // Zero strike: a call is a pure forward claim, a put is worthless.
    return option_type == OptionType::Call ? s * df_q : 0.0;
  }

  if (sigma <= 0.0) {
    // Deterministic world: S_T equals the forward almost surely.
    const double fwd_intrinsic = s * df_q - k * df_r;
    return option_type == OptionType::Call ? std::max(0.0, fwd_intrinsic)
                                           : std::max(0.0, -fwd_intrinsic);
  }

  double d1 = 0.0, d2 = 0.0;
  d1_d2(s, k, t, sigma, r, q, &d1, &d2);
  if (option_type == OptionType::Call) {
    return s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2);
  }
  return k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1);
}

Greeks bs_greeks(double s, double k, double t, double sigma, double r, double q,
                 OptionType option_type) {
  detail::validate_market_inputs(s, k, t, sigma, r, q);

  const double price = bs_price(s, k, t, sigma, r, q, option_type);
  const double df_r = std::exp(-r * t);
  const double df_q = std::exp(-q * t);

  if (t <= 0.0 || sigma <= 0.0 || k <= 0.0) {
    // Limit region: the payoff is (discounted) intrinsic on the forward;
    // second-order Greeks vanish and delta/theta/rho are the derivatives of
    // the limiting price (the at-the-money boundary maps to the ITM branch).
    const double fwd = s * df_q - k * df_r;  // sign of forward moneyness
    double delta = 0.0, theta = 0.0, rho = 0.0;
    if (option_type == OptionType::Call) {
      const bool itm = fwd >= 0.0 || k <= 0.0;
      if (itm) {
        delta = df_q;
        theta = q * s * df_q - r * k * df_r;
        rho = k * t * df_r;
      }
    } else {
      const bool itm = fwd < 0.0 && k > 0.0;
      if (itm) {
        delta = -df_q;
        theta = r * k * df_r - q * s * df_q;
        rho = -k * t * df_r;
      }
    }
    return Greeks{price, delta, 0.0, 0.0, theta, rho, 0.0, 0.0};
  }

  double d1 = 0.0, d2 = 0.0;
  d1_d2(s, k, t, sigma, r, q, &d1, &d2);
  const double sq_t = std::sqrt(t);
  const double pdf_d1 = norm_pdf(d1);

  const double gamma = df_q * pdf_d1 / (s * sigma * sq_t);
  const double vega = s * df_q * pdf_d1 * sq_t;
  const double vanna = -df_q * pdf_d1 * d2 / sigma;
  const double volga = vega * d1 * d2 / sigma;

  double delta = 0.0, theta = 0.0, rho = 0.0;
  if (option_type == OptionType::Call) {
    delta = df_q * norm_cdf(d1);
    theta = -s * df_q * pdf_d1 * sigma / (2.0 * sq_t) - r * k * df_r * norm_cdf(d2) +
            q * s * df_q * norm_cdf(d1);
    rho = k * t * df_r * norm_cdf(d2);
  } else {
    delta = df_q * (norm_cdf(d1) - 1.0);
    theta = -s * df_q * pdf_d1 * sigma / (2.0 * sq_t) + r * k * df_r * norm_cdf(-d2) -
            q * s * df_q * norm_cdf(-d1);
    rho = -k * t * df_r * norm_cdf(-d2);
  }

  return Greeks{price, delta, gamma, vega, theta, rho, vanna, volga};
}

double gk_price(double s, double k, double t, double sigma, double rd, double rf,
                OptionType option_type) {
  return bs_price(s, k, t, sigma, rd, rf, option_type);
}

Greeks gk_greeks(double s, double k, double t, double sigma, double rd, double rf,
                 OptionType option_type) {
  return bs_greeks(s, k, t, sigma, rd, rf, option_type);
}

double implied_vol(double price, double s, double k, double t, double r, double q,
                   OptionType option_type, double tol, int max_iter) {
  detail::validate_market_inputs(s, k, t, 0.0, r, q);
  detail::require_finite("price", price);
  detail::require_positive("t", t);
  detail::require_positive("k", k);
  detail::require_positive("tol", tol);
  if (max_iter < 1) {
    throw std::invalid_argument("max_iter must be an integer >= 1, got " +
                                std::to_string(max_iter));
  }

  const double df_r = std::exp(-r * t);
  const double df_q = std::exp(-q * t);
  double lower = 0.0, upper = 0.0;
  if (option_type == OptionType::Call) {
    lower = std::max(s * df_q - k * df_r, 0.0);
    upper = s * df_q;
  } else {
    lower = std::max(k * df_r - s * df_q, 0.0);
    upper = k * df_r;
  }

  if (price <= lower) {
    throw std::invalid_argument(
        "price " + fmt(price) + " violates the no-arbitrage lower bound " +
        fmt(lower) + " (below intrinsic); no implied vol exists");
  }
  if (price >= upper) {
    throw std::invalid_argument("price " + fmt(price) +
                                " violates the no-arbitrage upper bound " +
                                fmt(upper) + "; no implied vol exists");
  }

  const auto f = [&](double sig) {
    return bs_price(s, k, t, sig, r, q, option_type) - price;
  };

  // Bracket the root: the price is strictly increasing in sigma on the open
  // no-arb interval, and f(0) = discounted intrinsic - price < 0 by the
  // bound check above. hi doubles 1 -> 2 -> 4 -> 8 -> 16 -> 20 (clipped to
  // the cap) and the cap itself is tested before giving up, so the
  // effective cap is exactly kIvMaxSigma.
  double lo = 0.0;
  double hi = 1.0;
  while (f(hi) < 0.0) {
    if (hi >= kIvMaxSigma) {
      throw std::invalid_argument(
          "implied vol exceeds cap " + fmt(kIvMaxSigma) + "; price " + fmt(price) +
          " is numerically indistinguishable from the upper bound");
    }
    hi = std::min(2.0 * hi, kIvMaxSigma);
  }

  // Safeguarded Newton: max_iter Newton/bisection steps from the bracket
  // midpoint; the residual is checked before every step and once more after
  // the last one, so a returned sigma always satisfies a stopping rule.
  double sigma = 0.5 * (lo + hi);  // initial guess: bracket midpoint
  double diff = f(sigma);
  for (int i = 0; i < max_iter; ++i) {
    if (std::fabs(diff) < tol) return sigma;
    // Maintain the bracket around the root.
    if (diff > 0.0) {
      hi = sigma;
    } else {
      lo = sigma;
    }
    const double vega = bs_greeks(s, k, t, sigma, r, q, option_type).vega;
    if (vega > 1e-12) {
      const double candidate = sigma - diff / vega;
      if (lo < candidate && candidate < hi) {
        sigma = candidate;
        diff = f(sigma);
        continue;
      }
    }
    // Newton unusable (tiny vega or step outside bracket): bisect.
    sigma = 0.5 * (lo + hi);
    if (hi - lo < 1e-12) return sigma;
    diff = f(sigma);
  }
  if (std::fabs(diff) < tol) return sigma;
  // Iteration budget exhausted without meeting either stopping rule: report
  // honestly rather than hand back a half-converged root as if converged.
  throw std::invalid_argument(
      "implied vol did not converge in " + std::to_string(max_iter) +
      " iterations (|model - price| = " + fmt(std::fabs(diff)) + " > tol " + fmt(tol) +
      ", bracket [" + fmt(lo) + ", " + fmt(hi) + "]); increase max_iter or loosen tol");
}

}  // namespace dpe
