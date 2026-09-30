#include "dpe/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dpe/black_scholes.hpp"
#include "validation.hpp"

namespace dpe {
namespace {

struct TreeParams {
  double u;
  double d;
  double p;
};

/// Largest admissible magnitude of the log-spot excursion across the lattice
/// (identical in every port). exp() overflows a double at ~709.78, so a
/// terminal node with |ln s| + |nu n| + sigma sqrt(t n) > 700 would be inf
/// (or an inf*0 NaN in s u^j d^(n-j)); the guard makes that an error.
constexpr double kLatticeLogLimit = 700.0;

/// Reject lattices whose terminal spots would overflow a double. The extreme
/// terminal log-spot is ln s + nu n +- sigma sqrt(t n) with
/// nu n = (r - q - sigma^2/2) t for JR (zero for CRR); bounding the worst
/// case for either method keeps the admissible domain method-independent.
void check_lattice_range(double s, double t, double sigma, double r, double q,
                         int steps) {
  const double nu_n = std::fabs((r - q - 0.5 * sigma * sigma) * t);
  const double excursion =
      std::fabs(std::log(s)) + nu_n + sigma * std::sqrt(t * static_cast<double>(steps));
  if (excursion > kLatticeLogLimit) {
    std::ostringstream os;
    os.precision(17);
    os << "lattice overflow: |ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) = "
       << excursion << " exceeds " << kLatticeLogLimit
       << "; reduce steps, sigma or t";
    throw std::invalid_argument(os.str());
  }
}

double payoff(double spot, double k, OptionType ot) {
  return ot == OptionType::Call ? std::max(spot - k, 0.0) : std::max(k - spot, 0.0);
}

/// (u, d, p) for the requested lattice; throws if p is not a probability.
TreeParams tree_params(TreeMethod method, double sigma, double r, double q,
                       double dt) {
  const double sq = sigma * std::sqrt(dt);
  TreeParams tp{};
  if (method == TreeMethod::CRR) {
    tp.u = std::exp(sq);
    tp.d = 1.0 / tp.u;
    tp.p = (std::exp((r - q) * dt) - tp.d) / (tp.u - tp.d);
  } else {  // Jarrow-Rudd
    const double nu = (r - q - 0.5 * sigma * sigma) * dt;
    tp.u = std::exp(nu + sq);
    tp.d = std::exp(nu - sq);
    tp.p = 0.5;
  }
  if (!(tp.p > 0.0 && tp.p < 1.0)) {
    // Happens when the per-step drift outruns the vol spacing (huge (r-q)dt
    // vs sigma sqrt(dt)); the discrete measure would not be a probability,
    // so the lattice is invalid at this step count — never a silent clamp.
    std::ostringstream os;
    os.precision(17);
    os << "risk-neutral probability p=" << tp.p
       << " outside (0, 1); increase steps or use smaller drift/vol ratio";
    throw std::invalid_argument(os.str());
  }
  return tp;
}

void validate_steps(int steps) {
  if (steps < 1) {
    throw std::invalid_argument("steps must be an integer >= 1, got " +
                                std::to_string(steps));
  }
}

/// Exact sigma = 0 limit on the steps+1 point time grid: the spot rides the
/// deterministic forward, so a European option is worth discounted forward
/// intrinsic and an American option is exercised at whichever grid date
/// maximises the discounted intrinsic (deterministic dynamic program).
double degenerate_sigma_zero(double s, double k, double t, double r, double q,
                             OptionType ot, ExerciseStyle style, int steps) {
  if (style == ExerciseStyle::American) {
    const double dt = t / steps;
    double best = 0.0;
    for (int i = 0; i <= steps; ++i) {
      const double ti = i * dt;
      const double value = std::exp(-r * ti) * payoff(s * std::exp((r - q) * ti), k, ot);
      if (value > best) best = value;
    }
    return best;
  }
  return bs_price(s, k, t, 0.0, r, q, ot);  // discounted forward intrinsic
}

/// Backward induction; optionally keeps option/spot values of the first
/// keep_levels time levels (level 0 = today) for the tree Greeks.
double roll_back(double s, double k, double t, double r, OptionType ot,
                 ExerciseStyle style, int steps, const TreeParams& tp,
                 int keep_levels, std::vector<std::vector<double>>* opt_levels,
                 std::vector<std::vector<double>>* spot_levels) {
  const double dt = t / steps;
  const double disc = std::exp(-r * dt);
  const bool american = style == ExerciseStyle::American;

  // Terminal spots s * u^j * d^(n-j), j = 0..n (ascending), and payoffs.
  std::vector<double> spots(static_cast<std::size_t>(steps) + 1);
  std::vector<double> values(static_cast<std::size_t>(steps) + 1);
  for (int j = 0; j <= steps; ++j) {
    spots[static_cast<std::size_t>(j)] =
        s * std::pow(tp.u, j) * std::pow(tp.d, steps - j);
    values[static_cast<std::size_t>(j)] = payoff(spots[static_cast<std::size_t>(j)], k, ot);
  }

  if (opt_levels != nullptr) opt_levels->assign(static_cast<std::size_t>(keep_levels), {});
  if (spot_levels != nullptr) spot_levels->assign(static_cast<std::size_t>(keep_levels), {});
  if (steps < keep_levels && opt_levels != nullptr && spot_levels != nullptr) {
    // The terminal level itself is one of the requested levels (only for
    // very small trees, e.g. tree Greeks at steps = 2): store it here,
    // because the backward loop only visits levels < steps.
    (*opt_levels)[static_cast<std::size_t>(steps)] = values;
    (*spot_levels)[static_cast<std::size_t>(steps)] = spots;
  }

  for (int i = steps - 1; i >= 0; --i) {
    for (int j = 0; j <= i; ++j) {
      const auto ju = static_cast<std::size_t>(j);
      // Discounted expectation, then the level-i spot via one division by d
      // (spot(i, j) = spot(i+1, j) / d) — mirrors the reference arithmetic.
      values[ju] = disc * (tp.p * values[ju + 1] + (1.0 - tp.p) * values[ju]);
      spots[ju] = spots[ju] / tp.d;
      if (american) values[ju] = std::max(values[ju], payoff(spots[ju], k, ot));
    }
    if (i < keep_levels && opt_levels != nullptr && spot_levels != nullptr) {
      const auto iu = static_cast<std::size_t>(i);
      (*opt_levels)[iu].assign(values.begin(), values.begin() + i + 1);
      (*spot_levels)[iu].assign(spots.begin(), spots.begin() + i + 1);
    }
  }
  return values[0];
}

}  // namespace

double binomial_price(double s, double k, double t, double sigma, double r, double q,
                      OptionType option_type, ExerciseStyle style, int steps,
                      TreeMethod method, bool richardson) {
  validate_steps(steps);
  detail::validate_market_inputs(s, k, t, sigma, r, q);

  if (t <= 0.0) return payoff(s, k, option_type);
  if (sigma <= 0.0) {
    return degenerate_sigma_zero(s, k, t, r, q, option_type, style, steps);
  }
  check_lattice_range(s, t, sigma, r, q, richardson ? steps + 1 : steps);

  const auto one = [&](int n) {
    const TreeParams tp = tree_params(method, sigma, r, q, t / n);
    return roll_back(s, k, t, r, option_type, style, n, tp, 0, nullptr, nullptr);
  };

  if (richardson) return 0.5 * (one(steps) + one(steps + 1));
  return one(steps);
}

BinomialGreeks binomial_greeks(double s, double k, double t, double sigma, double r,
                               double q, OptionType option_type, ExerciseStyle style,
                               int steps, TreeMethod method) {
  validate_steps(steps);
  detail::validate_market_inputs(s, k, t, sigma, r, q);
  if (steps < 2) {
    throw std::invalid_argument("steps must be >= 2 for tree Greeks, got " +
                                std::to_string(steps));
  }
  if (t <= 0.0 || sigma <= 0.0) {
    throw std::invalid_argument("tree Greeks require t > 0 and sigma > 0");
  }
  check_lattice_range(s, t, sigma, r, q, steps);

  const double dt = t / steps;
  const TreeParams tp = tree_params(method, sigma, r, q, dt);
  std::vector<std::vector<double>> opt, spots;
  const double price =
      roll_back(s, k, t, r, option_type, style, steps, tp, 3, &opt, &spots);

  const double v0 = opt[0][0];
  const std::vector<double>& v1 = opt[1];
  const std::vector<double>& s1 = spots[1];
  const std::vector<double>& v2 = opt[2];
  const std::vector<double>& s2 = spots[2];

  const double delta = (v1[1] - v1[0]) / (s1[1] - s1[0]);
  const double delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1]);
  const double delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0]);
  const double gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]));
  // Correct for the middle node's spot displacement (zero for CRR since
  // u d = 1, O(dt) for the drifting JR lattice): strip the delta/gamma value
  // change so the remaining difference is purely calendar time.
  const double ds = s2[1] - s;
  const double theta = (v2[1] - v0 - delta * ds - 0.5 * gamma * ds * ds) / (2.0 * dt);

  return BinomialGreeks{price, delta, gamma, theta};
}

}  // namespace dpe
