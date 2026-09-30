// Binomial-tree tests: convergence to Black-Scholes, Richardson averaging,
// American early-exercise properties, sigma = 0 / t = 0 limits, tree Greeks
// vs analytic, and lattice-validity errors.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dpe/binomial.hpp"
#include "dpe/black_scholes.hpp"
#include "dpe/types.hpp"

namespace {

using dpe::ExerciseStyle;
using dpe::OptionType;
using dpe::TreeMethod;

constexpr OptionType kCall = OptionType::Call;
constexpr OptionType kPut = OptionType::Put;
constexpr ExerciseStyle kEuro = ExerciseStyle::European;
constexpr ExerciseStyle kAmer = ExerciseStyle::American;

TEST(Binomial, CrrConvergesToBlackScholes) {
  const double bs = dpe::bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall);
  const double crr = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall,
                                         kEuro, 2000, TreeMethod::CRR);
  EXPECT_NEAR(crr, bs, 1e-3);
  const double jr = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall,
                                        kEuro, 2000, TreeMethod::JR);
  EXPECT_NEAR(jr, bs, 1e-3);
}

TEST(Binomial, RichardsonDampsOscillation) {
  const double bs = dpe::bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall);
  const double plain = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall,
                                           kEuro, 200, TreeMethod::CRR);
  const double rich = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall,
                                          kEuro, 200, TreeMethod::CRR, true);
  // Averaging n and n+1 cancels the leading oscillating term: an order of
  // magnitude better than plain at the same step count (reference: plain
  // error -1.0e-2, Richardson -6.3e-4 at n=200).
  EXPECT_LT(std::fabs(rich - bs), 0.2 * std::fabs(plain - bs));
  EXPECT_NEAR(rich, bs, 1e-3);
}

TEST(Binomial, AmericanAtLeastEuropean) {
  for (double k : {90.0, 100.0, 110.0, 120.0}) {
    const double euro =
        dpe::binomial_price(100, k, 1.0, 0.2, 0.05, 0.0, kPut, kEuro, 200);
    const double amer =
        dpe::binomial_price(100, k, 1.0, 0.2, 0.05, 0.0, kPut, kAmer, 200);
    EXPECT_GE(amer, euro - 1e-12) << "k=" << k;
  }
}

TEST(Binomial, AmericanPutPremiumPositiveAndIncreasingInStrike) {
  double prev_premium = -1.0;
  for (double k : {90.0, 100.0, 110.0, 120.0}) {
    const double euro =
        dpe::binomial_price(100, k, 1.0, 0.2, 0.05, 0.0, kPut, kEuro, 500);
    const double amer =
        dpe::binomial_price(100, k, 1.0, 0.2, 0.05, 0.0, kPut, kAmer, 500);
    const double premium = amer - euro;
    EXPECT_GT(premium, 0.0) << "k=" << k;
    EXPECT_GT(premium, prev_premium) << "k=" << k;
    prev_premium = premium;
  }
}

TEST(Binomial, AmericanCallNoDividendEqualsEuropean) {
  // Without dividends, early exercise of a call is never optimal.
  const double euro =
      dpe::binomial_price(100, 95, 1.0, 0.25, 0.05, 0.0, kCall, kEuro, 300);
  const double amer =
      dpe::binomial_price(100, 95, 1.0, 0.25, 0.05, 0.0, kCall, kAmer, 300);
  EXPECT_NEAR(amer, euro, 1e-12);
}

TEST(Binomial, DegenerateLimits) {
  // t = 0: intrinsic.
  EXPECT_DOUBLE_EQ(
      dpe::binomial_price(90, 100, 0.0, 0.2, 0.05, 0.0, kPut, kAmer, 100), 10.0);
  // sigma = 0 European: discounted forward intrinsic = BS sigma-0 limit.
  EXPECT_NEAR(dpe::binomial_price(100, 90, 1.0, 0.0, 0.05, 0.02, kCall, kEuro, 100),
              dpe::bs_price(100, 90, 1.0, 0.0, 0.05, 0.02, kCall), 1e-12);
  // sigma = 0 American put with r > 0: k e^{-r t_i} - s is maximised at
  // t_0 = 0, so immediate exercise gives exactly k - s.
  EXPECT_NEAR(dpe::binomial_price(100, 120, 1.0, 0.0, 0.05, 0.0, kPut, kAmer, 100),
              20.0, 1e-12);
}

TEST(Binomial, GreeksCloseToAnalytic) {
  const dpe::BinomialGreeks tg = dpe::binomial_greeks(100, 100, 1.0, 0.2, 0.05, 0.0,
                                                      kCall, kEuro, 500,
                                                      TreeMethod::CRR);
  const dpe::Greeks ag = dpe::bs_greeks(100, 100, 1.0, 0.2, 0.05, 0.0, kCall);
  EXPECT_NEAR(tg.delta, ag.delta, 1e-3);
  EXPECT_NEAR(tg.gamma, ag.gamma, 1e-3);
  EXPECT_NEAR(tg.theta, ag.theta, 2e-2);
  EXPECT_NEAR(tg.price, ag.price, 5e-3);
}

TEST(Binomial, JrGreeksDriftCorrectionWorks) {
  // The JR lattice drifts (S(2,1) != s); the ds correction must still land
  // the Greeks close to analytic.
  const dpe::BinomialGreeks tg = dpe::binomial_greeks(100, 100, 1.0, 0.2, 0.05, 0.02,
                                                      kPut, kEuro, 500,
                                                      TreeMethod::JR);
  const dpe::Greeks ag = dpe::bs_greeks(100, 100, 1.0, 0.2, 0.05, 0.02, kPut);
  EXPECT_NEAR(tg.delta, ag.delta, 2e-3);
  EXPECT_NEAR(tg.gamma, ag.gamma, 1e-3);
  EXPECT_NEAR(tg.theta, ag.theta, 2e-2);
}

TEST(Binomial, InvalidInputsThrow) {
  EXPECT_THROW(
      dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall, kEuro, 0),
      std::invalid_argument);  // steps < 1
  EXPECT_THROW(
      dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall, kEuro, -5),
      std::invalid_argument);
  // Drift dominates vol at this step size: p outside (0,1) must throw,
  // never clamp (CRR, sigma tiny, r huge, one step).
  EXPECT_THROW(
      dpe::binomial_price(100, 100, 1.0, 0.01, 0.5, 0.0, kCall, kEuro, 1),
      std::invalid_argument);
  // Tree Greeks preconditions.
  EXPECT_THROW(dpe::binomial_greeks(100, 100, 1.0, 0.2, 0.05, 0.0, kCall, kEuro, 1),
               std::invalid_argument);  // steps < 2
  EXPECT_THROW(dpe::binomial_greeks(100, 100, 0.0, 0.2, 0.05, 0.0, kCall, kEuro, 10),
               std::invalid_argument);  // t = 0
  EXPECT_THROW(dpe::binomial_greeks(100, 100, 1.0, 0.0, 0.05, 0.0, kCall, kEuro, 10),
               std::invalid_argument);  // sigma = 0
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

TEST(Binomial, TreeGreeksTwoStepsAreFinite) {
  // steps = 2 is the smallest tree the contract allows for Greeks (level 2
  // is the terminal level). Previously an out-of-bounds read (UB).
  struct Case {
    OptionType ot;
    ExerciseStyle style;
    TreeMethod method;
  };
  for (const Case& c : {Case{kCall, kEuro, TreeMethod::CRR}, Case{kCall, kEuro, TreeMethod::JR},
                        Case{kPut, kAmer, TreeMethod::CRR}}) {
    const dpe::BinomialGreeks g =
        dpe::binomial_greeks(100, 100, 1.0, 0.2, 0.05, 0.0, c.ot, c.style, 2, c.method);
    EXPECT_TRUE(std::isfinite(g.price) && std::isfinite(g.delta) && std::isfinite(g.gamma) &&
                std::isfinite(g.theta));
    EXPECT_GT(g.gamma, 0.0);
    if (c.ot == kCall) {
      EXPECT_GT(g.delta, 0.0);
      EXPECT_LT(g.delta, 1.0);
    } else {
      EXPECT_GT(g.delta, -1.0);
      EXPECT_LT(g.delta, 0.0);
    }
    EXPECT_DOUBLE_EQ(g.price, dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.0, c.ot,
                                                  c.style, 2, c.method));
  }
}

TEST(Binomial, CrrPutCallParityExact) {
  // CRR matches the first moment exactly, so tree call - put equals
  // s e^{-qt} - k e^{-rt} to round-off at every step count; JR does not
  // (its parity error is O(dt)).
  const double s = 100, k = 100, t = 1.0, sigma = 0.2, r = 0.05, q = 0.02;
  const double fwd = s * std::exp(-q * t) - k * std::exp(-r * t);
  for (int n : {1, 2, 101, 500}) {
    const double c = dpe::binomial_price(s, k, t, sigma, r, q, kCall, kEuro, n, TreeMethod::CRR);
    const double p = dpe::binomial_price(s, k, t, sigma, r, q, kPut, kEuro, n, TreeMethod::CRR);
    EXPECT_NEAR(c - p, fwd, 1e-10) << "n=" << n;
  }
  const double cj = dpe::binomial_price(s, k, t, sigma, r, q, kCall, kEuro, 101, TreeMethod::JR);
  const double pj = dpe::binomial_price(s, k, t, sigma, r, q, kPut, kEuro, 101, TreeMethod::JR);
  EXPECT_NEAR(cj - pj, fwd, 1e-3);
  EXPECT_GT(std::fabs((cj - pj) - fwd), 1e-6);  // JR is genuinely not a martingale lattice
}

TEST(Binomial, CrrOneStepClosedForm) {
  // One CRR step is a hand-computable two-state model.
  const double s = 100, k = 100, t = 1.0, sigma = 0.2, r = 0.05, q = 0.02;
  const double u = std::exp(sigma * std::sqrt(t));
  const double d = 1.0 / u;
  const double p = (std::exp((r - q) * t) - d) / (u - d);
  const double expected =
      std::exp(-r * t) * (p * std::max(s * u - k, 0.0) + (1.0 - p) * std::max(s * d - k, 0.0));
  EXPECT_NEAR(expected, 11.073540703840242, 1e-12);  // derived independently
  EXPECT_NEAR(dpe::binomial_price(s, k, t, sigma, r, q, kCall, kEuro, 1, TreeMethod::CRR),
              expected, 1e-12);
}

TEST(Binomial, AmericanCallWithDividendStrictlyExceedsEuropean) {
  // q > r: the dividend leakage on a deep-ITM call makes early exercise
  // materially valuable.
  const double eur = dpe::binomial_price(100, 80, 1.0, 0.2, 0.05, 0.08, kCall, kEuro, 500);
  const double amer = dpe::binomial_price(100, 80, 1.0, 0.2, 0.05, 0.08, kCall, kAmer, 500);
  EXPECT_GT(amer - eur, 1.0);
  EXPECT_GE(amer, 20.0);  // never below immediate intrinsic
}

TEST(Binomial, RichardsonIsAverageOfNAndNPlusOne) {
  const double rich = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.02, kCall, kEuro, 200,
                                          TreeMethod::CRR, true);
  const double a = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.02, kCall, kEuro, 200);
  const double b = dpe::binomial_price(100, 100, 1.0, 0.2, 0.05, 0.02, kCall, kEuro, 201);
  EXPECT_NEAR(rich, 0.5 * (a + b), 1e-14);
}

TEST(Binomial, TreeOverflowGuard) {
  // sigma sqrt(t steps) = 707 > 700 would overflow the terminal spots to
  // inf; the contract says invalid-input error, never inf/NaN.
  EXPECT_THROW(dpe::binomial_price(100, 100, 1.0, 50.0, 0.05, 0.0, kCall, kEuro, 200),
               std::invalid_argument);
  EXPECT_THROW(dpe::binomial_greeks(100, 100, 1.0, 50.0, 0.05, 0.0, kCall, kEuro, 200,
                                    TreeMethod::JR),
               std::invalid_argument);
  // Just inside the guard (4.6 + 200 + 400 = 605 < 700) is finite.
  const double v = dpe::binomial_price(100, 100, 1.0, 20.0, 0.05, 0.0, kCall, kEuro, 400);
  EXPECT_TRUE(std::isfinite(v));
  EXPECT_GT(v, 0.0);
  EXPECT_LT(v, 100.0);
}

TEST(Binomial, ExpiryPriceIsPositiveZeroOnTie) {
  const double v = dpe::binomial_price(100, 100, 0.0, 0.2, 0.05, 0.0, kPut, kEuro, 10);
  EXPECT_EQ(v, 0.0);
  EXPECT_FALSE(std::signbit(v));
}

}  // namespace
