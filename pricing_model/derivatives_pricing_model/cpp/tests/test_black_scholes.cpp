// Analytic module tests: put-call parity grid, Greeks vs central finite
// differences, monotonicity, degenerate limits and input validation.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "dpe/black_scholes.hpp"
#include "dpe/types.hpp"

namespace {

using dpe::OptionType;

constexpr OptionType kCall = OptionType::Call;
constexpr OptionType kPut = OptionType::Put;

TEST(BsPrice, AtmReference) {
  // Textbook anchor (also golden case bs_call_atm).
  EXPECT_NEAR(dpe::bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall), 10.45058357,
              1e-8);
}

TEST(BsPrice, PutCallParityGrid) {
  // Property check: c - p = s e^{-qt} - k e^{-rt} across a grid.
  const double r = 0.04, q = 0.015, sigma = 0.25;
  for (double s : {80.0, 100.0, 120.0}) {
    for (double k : {70.0, 100.0, 130.0}) {
      for (double t : {0.1, 1.0, 3.0}) {
        const double c = dpe::bs_price(s, k, t, sigma, r, q, kCall);
        const double p = dpe::bs_price(s, k, t, sigma, r, q, kPut);
        const double parity = s * std::exp(-q * t) - k * std::exp(-r * t);
        EXPECT_NEAR(c - p, parity, 1e-10) << "s=" << s << " k=" << k << " t=" << t;
      }
    }
  }
}

TEST(BsPrice, MonotoneInSpotAndStrike) {
  const double t = 1.0, sigma = 0.2, r = 0.05, q = 0.01;
  double prev = dpe::bs_price(60, 100, t, sigma, r, q, kCall);
  for (double s = 65; s <= 140; s += 5) {  // call price increasing in s
    const double cur = dpe::bs_price(s, 100, t, sigma, r, q, kCall);
    EXPECT_GT(cur, prev) << "s=" << s;
    prev = cur;
  }
  prev = dpe::bs_price(100, 60, t, sigma, r, q, kCall);
  for (double k = 65; k <= 140; k += 5) {  // call price decreasing in k
    const double cur = dpe::bs_price(100, k, t, sigma, r, q, kCall);
    EXPECT_LT(cur, prev) << "k=" << k;
    prev = cur;
  }
}

TEST(BsPrice, DegenerateLimits) {
  // t = 0: undiscounted intrinsic.
  EXPECT_DOUBLE_EQ(dpe::bs_price(90, 100, 0.0, 0.2, 0.05, 0.0, kPut), 10.0);
  EXPECT_DOUBLE_EQ(dpe::bs_price(90, 100, 0.0, 0.2, 0.05, 0.0, kCall), 0.0);
  // sigma = 0: discounted forward intrinsic.
  const double expected =
      100.0 * std::exp(-0.02) - 90.0 * std::exp(-0.05);
  EXPECT_NEAR(dpe::bs_price(100, 90, 1.0, 0.0, 0.05, 0.02, kCall), expected, 1e-12);
  EXPECT_DOUBLE_EQ(dpe::bs_price(100, 90, 1.0, 0.0, 0.05, 0.02, kPut), 0.0);
  // k = 0: call is a forward claim, put is worthless.
  EXPECT_NEAR(dpe::bs_price(100, 0.0, 1.0, 0.2, 0.05, 0.02, kCall),
              100.0 * std::exp(-0.02), 1e-12);
  EXPECT_DOUBLE_EQ(dpe::bs_price(100, 0.0, 1.0, 0.2, 0.05, 0.02, kPut), 0.0);
}

TEST(BsGreeks, MatchCentralFiniteDifferences) {
  const double s = 100, k = 105, t = 0.75, sigma = 0.25, r = 0.04, q = 0.015;
  for (OptionType ot : {kCall, kPut}) {
    const dpe::Greeks g = dpe::bs_greeks(s, k, t, sigma, r, q, ot);
    const double hs = s * 1e-5, hv = 1e-5, ht = 1e-6, hr = 1e-6;

    const double delta_fd = (dpe::bs_price(s + hs, k, t, sigma, r, q, ot) -
                             dpe::bs_price(s - hs, k, t, sigma, r, q, ot)) /
                            (2 * hs);
    EXPECT_NEAR(g.delta, delta_fd, std::fabs(delta_fd) * 1e-4);

    const double vega_fd = (dpe::bs_price(s, k, t, sigma + hv, r, q, ot) -
                            dpe::bs_price(s, k, t, sigma - hv, r, q, ot)) /
                           (2 * hv);
    EXPECT_NEAR(g.vega, vega_fd, std::fabs(vega_fd) * 1e-4);

    const double gamma_fd = (dpe::bs_price(s + hs, k, t, sigma, r, q, ot) -
                             2 * dpe::bs_price(s, k, t, sigma, r, q, ot) +
                             dpe::bs_price(s - hs, k, t, sigma, r, q, ot)) /
                            (hs * hs);
    EXPECT_NEAR(g.gamma, gamma_fd, std::fabs(gamma_fd) * 1e-3);

    // theta is the calendar-time derivative: -dV/dT.
    const double theta_fd = -(dpe::bs_price(s, k, t + ht, sigma, r, q, ot) -
                              dpe::bs_price(s, k, t - ht, sigma, r, q, ot)) /
                            (2 * ht);
    EXPECT_NEAR(g.theta, theta_fd, std::fabs(theta_fd) * 1e-4);

    const double rho_fd = (dpe::bs_price(s, k, t, sigma, r + hr, q, ot) -
                           dpe::bs_price(s, k, t, sigma, r - hr, q, ot)) /
                          (2 * hr);
    EXPECT_NEAR(g.rho, rho_fd, std::fabs(rho_fd) * 1e-4);

    // vanna = d(delta)/d(sigma); volga = d(vega)/d(sigma).
    const double vanna_fd = (dpe::bs_greeks(s, k, t, sigma + hv, r, q, ot).delta -
                             dpe::bs_greeks(s, k, t, sigma - hv, r, q, ot).delta) /
                            (2 * hv);
    EXPECT_NEAR(g.vanna, vanna_fd, std::fabs(vanna_fd) * 1e-3);
    const double volga_fd = (dpe::bs_greeks(s, k, t, sigma + hv, r, q, ot).vega -
                             dpe::bs_greeks(s, k, t, sigma - hv, r, q, ot).vega) /
                            (2 * hv);
    EXPECT_NEAR(g.volga, volga_fd, std::fabs(volga_fd) * 1e-3);
  }
}

TEST(BsGreeks, GammaAndVegaIdenticalForCallAndPut) {
  const dpe::Greeks c = dpe::bs_greeks(100, 110, 0.5, 0.3, 0.03, 0.02, kCall);
  const dpe::Greeks p = dpe::bs_greeks(100, 110, 0.5, 0.3, 0.03, 0.02, kPut);
  EXPECT_NEAR(c.gamma, p.gamma, 1e-14);
  EXPECT_NEAR(c.vega, p.vega, 1e-12);
  EXPECT_NEAR(c.delta - p.delta, std::exp(-0.02 * 0.5), 1e-14);  // parity
}

TEST(BsGreeks, DegenerateContractValues) {
  // sigma = 0, call ITM on the forward: delta = e^{-qt}, second order = 0.
  const dpe::Greeks g = dpe::bs_greeks(100, 90, 1.0, 0.0, 0.05, 0.02, kCall);
  EXPECT_DOUBLE_EQ(g.delta, std::exp(-0.02));
  EXPECT_DOUBLE_EQ(g.gamma, 0.0);
  EXPECT_DOUBLE_EQ(g.vega, 0.0);
  EXPECT_DOUBLE_EQ(g.vanna, 0.0);
  EXPECT_DOUBLE_EQ(g.volga, 0.0);
  EXPECT_NEAR(g.theta, 0.02 * 100 * std::exp(-0.02) - 0.05 * 90 * std::exp(-0.05),
              1e-12);
  EXPECT_NEAR(g.rho, 90.0 * std::exp(-0.05), 1e-12);
  // OTM side: delta/theta/rho all zero.
  const dpe::Greeks g2 = dpe::bs_greeks(80, 100, 1.0, 0.0, 0.01, 0.0, kCall);
  EXPECT_DOUBLE_EQ(g2.delta, 0.0);
  EXPECT_DOUBLE_EQ(g2.theta, 0.0);
  EXPECT_DOUBLE_EQ(g2.rho, 0.0);
}

TEST(BsGreeks, ExpiryTieBreakContract) {
  // API_SPEC §3.2: exactly at the money in the degenerate region the call
  // takes the ITM branch and the put the OTM branch. t = 0 first.
  const double s = 100, k = 100, r = 0.05, q = 0.02;
  const dpe::Greeks c = dpe::bs_greeks(s, k, 0.0, 0.2, r, q, kCall);
  EXPECT_EQ(c.price, 0.0);
  EXPECT_EQ(c.delta, 1.0);
  EXPECT_DOUBLE_EQ(c.theta, q * s - r * k);
  EXPECT_EQ(c.rho, 0.0);
  EXPECT_EQ(c.gamma, 0.0);
  EXPECT_EQ(c.vega, 0.0);
  const dpe::Greeks p = dpe::bs_greeks(s, k, 0.0, 0.2, r, q, kPut);
  EXPECT_EQ(p.price, 0.0);
  EXPECT_EQ(p.delta, 0.0);
  EXPECT_EQ(p.theta, 0.0);
  EXPECT_EQ(p.rho, 0.0);
  // sigma = 0 with the forward exactly at the strike (r = q, s = k makes
  // s e^{-qt} - k e^{-rt} exactly zero in floating point).
  const double t = 1.0, rate = 0.03, df = std::exp(-rate * t);
  const dpe::Greeks c0 = dpe::bs_greeks(s, k, t, 0.0, rate, rate, kCall);
  EXPECT_EQ(c0.price, 0.0);
  EXPECT_NEAR(c0.delta, df, 1e-15);
  EXPECT_NEAR(c0.theta, 0.0, 1e-15);  // q s e^{-qt} - r k e^{-rt} = 0 here
  EXPECT_NEAR(c0.rho, k * t * df, 1e-12);
  const dpe::Greeks p0 = dpe::bs_greeks(s, k, t, 0.0, rate, rate, kPut);
  EXPECT_EQ(p0.price, 0.0);
  EXPECT_FALSE(std::signbit(p0.price));  // +0.0, never -0.0
  EXPECT_EQ(p0.delta, 0.0);
  EXPECT_EQ(p0.theta, 0.0);
  EXPECT_EQ(p0.rho, 0.0);
}

TEST(GarmanKohlhagen, MatchesBsWithSwappedRates) {
  const double p_gk = dpe::gk_price(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, kCall);
  const double p_bs = dpe::bs_price(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, kCall);
  EXPECT_DOUBLE_EQ(p_gk, p_bs);
  const dpe::Greeks g = dpe::gk_greeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, kCall);
  EXPECT_DOUBLE_EQ(g.price, p_gk);
}

TEST(Validation, InvalidInputsThrow) {
  EXPECT_THROW(dpe::bs_price(-1, 100, 1, 0.2, 0.05, 0, kCall),
               std::invalid_argument);  // s <= 0
  EXPECT_THROW(dpe::bs_price(0, 100, 1, 0.2, 0.05, 0, kCall),
               std::invalid_argument);
  EXPECT_THROW(dpe::bs_price(100, -1, 1, 0.2, 0.05, 0, kCall),
               std::invalid_argument);  // k < 0
  EXPECT_THROW(dpe::bs_price(100, 100, -0.1, 0.2, 0.05, 0, kCall),
               std::invalid_argument);  // t < 0
  EXPECT_THROW(dpe::bs_price(100, 100, 1, -0.2, 0.05, 0, kCall),
               std::invalid_argument);  // sigma < 0
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(dpe::bs_price(100, 100, 1, 0.2, nan, 0, kCall),
               std::invalid_argument);
  EXPECT_THROW(dpe::bs_price(100, 100, 1, 0.2, 0.05, inf, kCall),
               std::invalid_argument);
  EXPECT_THROW(dpe::bs_greeks(nan, 100, 1, 0.2, 0.05, 0, kCall),
               std::invalid_argument);
}

TEST(Validation, ParseEnumsCaseInsensitive) {
  EXPECT_EQ(dpe::parse_option_type("CALL"), OptionType::Call);
  EXPECT_EQ(dpe::parse_option_type("Put"), OptionType::Put);
  EXPECT_EQ(dpe::parse_exercise_style("American"), dpe::ExerciseStyle::American);
  EXPECT_EQ(dpe::parse_tree_method("JR"), dpe::TreeMethod::JR);
  EXPECT_THROW(dpe::parse_option_type("straddle"), std::invalid_argument);
  EXPECT_THROW(dpe::parse_exercise_style("bermudan"), std::invalid_argument);
  EXPECT_THROW(dpe::parse_tree_method("trinomial"), std::invalid_argument);
}

TEST(NormCdf, TailPrecisionAndSymmetry) {
  EXPECT_DOUBLE_EQ(dpe::norm_cdf(0.0), 0.5);
  EXPECT_NEAR(dpe::norm_cdf(1.0) + dpe::norm_cdf(-1.0), 1.0, 1e-15);
  // Deep left tail keeps relative precision (erfc, not 0.5*(1+erf)).
  EXPECT_GT(dpe::norm_cdf(-37.0), 0.0);
  EXPECT_NEAR(dpe::norm_pdf(0.0), 0.3989422804014327, 1e-15);
}

}  // namespace
