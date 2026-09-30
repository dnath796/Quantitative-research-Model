// FX helper and utility tests: covered-interest-parity forwards, delta-
// convention conversions (exact mutual inverses, sign-preserving), forward
// delta = N(d1), and the historical-vol estimator with its error cases.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "dpe/black_scholes.hpp"
#include "dpe/fx.hpp"
#include "dpe/types.hpp"
#include "dpe/utils.hpp"

namespace {

TEST(FxForward, CoveredInterestParity) {
  EXPECT_NEAR(dpe::fx_forward(1.10, 0.5, 0.03, 0.02), 1.10 * std::exp(0.01 * 0.5),
              1e-15);
  EXPECT_DOUBLE_EQ(dpe::fx_forward(1.10, 0.0, 0.03, 0.02), 1.10);  // t = 0
  // Negative domestic rate: forward below spot when rd < rf.
  EXPECT_LT(dpe::fx_forward(0.95, 1.0, -0.005, 0.001), 0.95);
  EXPECT_THROW(dpe::fx_forward(0.0, 1.0, 0.03, 0.02), std::invalid_argument);
  EXPECT_THROW(dpe::fx_forward(1.1, -1.0, 0.03, 0.02), std::invalid_argument);
}

TEST(FxDelta, SpotForwardConversionsAreMutualInverses) {
  const double t = 0.75, rf = 0.02;
  for (double delta : {0.25, 0.5, -0.25, -0.6}) {  // calls and puts
    const double fwd = dpe::spot_delta_to_forward(delta, t, rf);
    EXPECT_NEAR(dpe::forward_delta_to_spot(fwd, t, rf), delta, 1e-15);
    // Sign-preserving; |forward delta| > |spot delta| when rf > 0.
    EXPECT_EQ(fwd > 0, delta > 0);
    EXPECT_GT(std::fabs(fwd), std::fabs(delta));
  }
}

TEST(FxDelta, ForwardDeltaOfCallIsNd1) {
  const double s = 1.10, k = 1.12, t = 0.5, sigma = 0.10, rd = 0.03, rf = 0.02;
  const dpe::Greeks g = dpe::gk_greeks(s, k, t, sigma, rd, rf, dpe::OptionType::Call);
  const double d1 = (std::log(s / k) + (rd - rf + 0.5 * sigma * sigma) * t) /
                    (sigma * std::sqrt(t));
  // gk delta is the spot delta e^{-rf t} N(d1); converting to the forward
  // convention must give exactly N(d1).
  EXPECT_NEAR(dpe::spot_delta_to_forward(g.delta, t, rf), dpe::norm_cdf(d1), 1e-14);
}

TEST(HistoricalVol, KnownSeries) {
  // Constant growth: all log returns equal, ddof-1 std = 0 (up to the
  // floating-point noise of ln(121/110) vs ln(110/100)).
  EXPECT_NEAR(dpe::historical_vol({100.0, 110.0, 121.0, 133.1}), 0.0, 1e-12);
  // Returns {+1%, -1%}: mean 0, sample std = 0.01 * sqrt(2), annualised.
  const std::vector<double> prices = {100.0, 100.0 * std::exp(0.01),
                                      100.0 * std::exp(0.01) * std::exp(-0.01)};
  EXPECT_NEAR(dpe::historical_vol(prices, 252),
              0.01 * std::sqrt(2.0) * std::sqrt(252.0), 1e-12);
}

TEST(HistoricalVol, InvalidInputsThrow) {
  EXPECT_THROW(dpe::historical_vol({100.0, 101.0}), std::invalid_argument);
  EXPECT_THROW(dpe::historical_vol({100.0, -1.0, 101.0}), std::invalid_argument);
  EXPECT_THROW(dpe::historical_vol({100.0, 0.0, 101.0}), std::invalid_argument);
  EXPECT_THROW(dpe::historical_vol({100.0, 101.0, 102.0}, 0), std::invalid_argument);
}

}  // namespace
