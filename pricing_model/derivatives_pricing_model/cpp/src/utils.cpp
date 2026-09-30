#include "dpe/utils.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace dpe {

double historical_vol(const std::vector<double>& prices, int periods_per_year) {
  if (prices.size() < 3) {
    throw std::invalid_argument("prices must be a series with >= 3 points, got " +
                                std::to_string(prices.size()));
  }
  for (double p : prices) {
    if (!std::isfinite(p) || p <= 0.0) {
      throw std::invalid_argument("prices must all be finite and > 0");
    }
  }
  if (periods_per_year < 1) {
    throw std::invalid_argument("periods_per_year must be >= 1, got " +
                                std::to_string(periods_per_year));
  }

  // Sample standard deviation (ddof = 1) of the log returns.
  const std::size_t n = prices.size() - 1;
  std::vector<double> log_returns(n);
  for (std::size_t i = 0; i < n; ++i) {
    log_returns[i] = std::log(prices[i + 1] / prices[i]);
  }
  double mean = 0.0;
  for (double r : log_returns) mean += r;
  mean /= static_cast<double>(n);
  double ss = 0.0;
  for (double r : log_returns) {
    const double d = r - mean;
    ss += d * d;
  }
  const double sd = std::sqrt(ss / static_cast<double>(n - 1));
  return sd * std::sqrt(static_cast<double>(periods_per_year));
}

}  // namespace dpe
