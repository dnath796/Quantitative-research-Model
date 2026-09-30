// Monte Carlo tests: statistical agreement with analytic anchors (3-SE
// bands), variance-reduction effectiveness, exotic ordering properties,
// determinism, path-matrix layout and input validation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "dpe/black_scholes.hpp"
#include "dpe/monte_carlo.hpp"
#include "dpe/types.hpp"

namespace {

using dpe::OptionType;

constexpr OptionType kCall = OptionType::Call;
constexpr OptionType kPut = OptionType::Put;

constexpr double kS = 100, kK = 100, kT = 1.0, kSigma = 0.2, kR = 0.05, kQ = 0.0;
constexpr std::int64_t kPaths = 50000;

TEST(McEuropean, WithinThreeStandardErrorsOfBs) {
  const double bs = dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall);
  const dpe::MCResult res =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 42, true);
  EXPECT_NEAR(res.value, bs, 3.0 * res.std_error);
  EXPECT_GT(res.std_error, 0.0);
  EXPECT_EQ(res.n_paths, kPaths);
  // CI construction contract.
  EXPECT_NEAR(res.ci_low, res.value - 1.959963984540054 * res.std_error, 1e-12);
  EXPECT_NEAR(res.ci_high, res.value + 1.959963984540054 * res.std_error, 1e-12);
}

TEST(McEuropean, VarianceReductionOrdering) {
  const dpe::MCResult plain =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 7, false);
  const dpe::MCResult anti =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 7, true);
  const dpe::MCResult cv =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 7, true, true);
  // Vanilla payoff is monotone in Z, so antithetic pairing must help; the
  // forward control is highly correlated with the payoff and helps further.
  EXPECT_LT(anti.std_error, plain.std_error);
  EXPECT_LT(cv.std_error, anti.std_error);
}

TEST(McEuropean, DeterministicGivenSeed) {
  const dpe::MCResult a =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 10000, 123, true);
  const dpe::MCResult b =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 10000, 123, true);
  const dpe::MCResult c =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 10000, 124, true);
  EXPECT_DOUBLE_EQ(a.value, b.value);
  EXPECT_DOUBLE_EQ(a.std_error, b.std_error);
  EXPECT_NE(a.value, c.value);
}

TEST(SimulateGbmPaths, ShapeAndAntitheticLayout) {
  const std::int64_t n_paths = 8;
  const int n_steps = 5;
  const auto paths =
      dpe::simulate_gbm_paths(kS, kT, kSigma, kR, kQ, n_paths, n_steps, 42, true);
  ASSERT_EQ(paths.size(), 8u);
  const double dt = kT / n_steps;
  const double drift = (kR - kQ - 0.5 * kSigma * kSigma) * dt;
  for (const auto& row : paths) {
    ASSERT_EQ(row.size(), 6u);
    EXPECT_DOUBLE_EQ(row[0], kS);
  }
  // Antithetic rows use negated increments: the log-increments of row i and
  // row i + n/2 must sum to 2 * drift at every step.
  for (std::size_t i = 0; i < 4; ++i) {
    for (int j = 1; j <= n_steps; ++j) {
      const auto ju = static_cast<std::size_t>(j);
      const double inc = std::log(paths[i][ju] / paths[i][ju - 1]);
      const double inc_anti = std::log(paths[i + 4][ju] / paths[i + 4][ju - 1]);
      EXPECT_NEAR(inc + inc_anti, 2.0 * drift, 1e-12);
    }
  }
}

TEST(GeometricAsian, SingleFixingIsEuropean) {
  // With one fixing at t the geometric average IS the terminal spot.
  const double geo = dpe::geometric_asian_price(kS, kK, kT, kSigma, kR, kQ, kCall, 1);
  const double bs = dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall);
  EXPECT_NEAR(geo, bs, 1e-12);
}

TEST(GeometricAsian, BelowVanillaAndSigmaZeroLimit) {
  // Averaging reduces effective vol: geometric Asian < vanilla.
  const double geo = dpe::geometric_asian_price(kS, kK, kT, kSigma, kR, kQ, kCall, 12);
  EXPECT_LT(geo, dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall));
  // sigma = 0: discounted deterministic-average intrinsic.
  const double det = dpe::geometric_asian_price(100, 90, 1.0, 0.0, 0.05, 0.0, kCall, 4);
  EXPECT_GT(det, 0.0);
}

TEST(McAsian, ControlVariateAgreesAndShrinksError) {
  const dpe::MCResult cv = dpe::mc_asian_arithmetic(kS, kK, kT, kSigma, kR, kQ, kCall,
                                                    kPaths, 12, 42, true, true);
  const dpe::MCResult raw = dpe::mc_asian_arithmetic(kS, kK, kT, kSigma, kR, kQ, kCall,
                                                     kPaths, 12, 42, true, false);
  // Same estimand: the two estimates must agree within joint error bars.
  EXPECT_NEAR(cv.value, raw.value, 3.0 * (cv.std_error + raw.std_error));
  // The geometric control correlates > 99% with the arithmetic payoff.
  EXPECT_LT(cv.std_error, 0.2 * raw.std_error);
  // Arithmetic Asian <= vanilla European (same params, averaging cuts vol).
  EXPECT_LT(cv.value, dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall));
}

TEST(McBarrier, BelowVanillaAndBornDead) {
  const dpe::MCResult barrier = dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ,
                                                       kCall, 130.0, kPaths, 100, 42);
  // Knock-out only removes payoff: barrier price <= vanilla price.
  EXPECT_LT(barrier.value, dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall));
  EXPECT_GT(barrier.value, 0.0);
  // Spot at/above the barrier: born dead, exact zero without simulating.
  const dpe::MCResult dead = dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ, kCall,
                                                    100.0, kPaths, 100, 42);
  EXPECT_DOUBLE_EQ(dead.value, 0.0);
  EXPECT_DOUBLE_EQ(dead.std_error, 0.0);
  EXPECT_EQ(dead.n_paths, kPaths);
}

TEST(McLookback, PayoffsNonNegativeAndAboveAtmVanilla) {
  const dpe::MCResult call =
      dpe::mc_lookback_floating(kS, kT, kSigma, kR, kQ, kCall, kPaths, 100, 42);
  const dpe::MCResult put =
      dpe::mc_lookback_floating(kS, kT, kSigma, kR, kQ, kPut, kPaths, 100, 42);
  EXPECT_GT(call.value, 0.0);
  EXPECT_GT(put.value, 0.0);
  // S_T - min >= S_T - S_0 pathwise, so the lookback call dominates the
  // ATM-struck vanilla call.
  EXPECT_GT(call.value, dpe::bs_price(kS, kS, kT, kSigma, kR, kQ, kCall));
}

TEST(McDelta, PathwiseAndCrnMatchAnalytic) {
  const double analytic = dpe::bs_greeks(kS, kK, kT, kSigma, kR, kQ, kCall).delta;
  const dpe::MCResult pw =
      dpe::mc_delta_pathwise(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 42);
  EXPECT_NEAR(pw.value, analytic, 3.0 * pw.std_error);
  const dpe::MCResult fd =
      dpe::mc_delta_fd_crn(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 42);
  EXPECT_NEAR(fd.value, analytic, 3.0 * fd.std_error + 1e-4);
  // Put deltas are negative.
  const dpe::MCResult pw_put =
      dpe::mc_delta_pathwise(kS, kK, kT, kSigma, kR, kQ, kPut, kPaths, 42);
  EXPECT_LT(pw_put.value, 0.0);
}

TEST(MonteCarlo, InvalidInputsThrow) {
  EXPECT_THROW(dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 1, 42, false),
               std::invalid_argument);  // n_paths < 2
  EXPECT_THROW(dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 1001, 42, true),
               std::invalid_argument);  // odd with antithetic
  EXPECT_THROW(dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 2, 42, true),
               std::invalid_argument);  // < 4 with antithetic
  EXPECT_THROW(dpe::mc_european(kS, kK, 0.0, kSigma, kR, kQ, kCall, 100, 42),
               std::invalid_argument);  // t = 0
  EXPECT_THROW(dpe::mc_asian_arithmetic(kS, kK, kT, kSigma, kR, kQ, kCall, 100, 0, 42),
               std::invalid_argument);  // n_steps < 1
  EXPECT_THROW(dpe::mc_asian_arithmetic(kS, 0.0, kT, kSigma, kR, kQ, kCall, 100, 10, 42),
               std::invalid_argument);  // k = 0
  EXPECT_THROW(dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ, kCall, 0.0, 100, 10, 42),
               std::invalid_argument);  // barrier <= 0
  EXPECT_THROW(dpe::mc_delta_fd_crn(kS, kK, kT, kSigma, kR, kQ, kCall, 100, 42, 0.0),
               std::invalid_argument);  // rel_bump <= 0
  EXPECT_THROW(dpe::geometric_asian_price(kS, kK, kT, kSigma, kR, kQ, kCall, 0),
               std::invalid_argument);  // n_fixings < 1
  EXPECT_THROW(dpe::simulate_gbm_paths(kS, 0.0, kSigma, kR, kQ, 10, 5, 42),
               std::invalid_argument);  // t = 0
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

TEST(MonteCarlo, NegativeSeedRejected) {
  // Seeds are integers in [0, 2^63 - 1] in every port: a negative seed is an
  // invalid-input error naming `seed`, never a silent unsigned wrap.
  for (std::int64_t bad : {std::int64_t{-1}, std::numeric_limits<std::int64_t>::min()}) {
    try {
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 1000, bad);
      FAIL() << "expected std::invalid_argument for seed=" << bad;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("seed"), std::string::npos);
    }
    EXPECT_THROW(dpe::simulate_gbm_paths(kS, kT, kSigma, kR, kQ, 10, 2, bad),
                 std::invalid_argument);
    EXPECT_THROW(dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ, kCall, 130.0, 100, 4, bad),
                 std::invalid_argument);
  }
  // Both endpoints of the domain are accepted and give different streams.
  const dpe::MCResult lo = dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 1000, 0);
  const dpe::MCResult hi = dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, 1000,
                                            std::numeric_limits<std::int64_t>::max());
  EXPECT_TRUE(std::isfinite(lo.value) && std::isfinite(hi.value));
  EXPECT_NE(lo.value, hi.value);
}

TEST(MonteCarlo, RelBumpDomain) {
  for (double bad : {1.0, 2.0, 0.0, -1e-4, std::nan(""), HUGE_VAL}) {
    try {
      dpe::mc_delta_fd_crn(kS, kK, kT, kSigma, kR, kQ, kCall, 1000, 42, bad);
      FAIL() << "expected std::invalid_argument for rel_bump=" << bad;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("rel_bump"), std::string::npos);
    }
  }
}

TEST(MonteCarlo, AsianSingleFixingEqualsEuropean) {
  // One fixing: the arithmetic Asian IS the vanilla, and both engines consume
  // the same draws in the same order, so the results must be bit-identical.
  const dpe::MCResult asian = dpe::mc_asian_arithmetic(kS, kK, kT, kSigma, kR, kQ, kCall,
                                                       kPaths, 1, 7, true, false);
  const dpe::MCResult euro =
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kCall, kPaths, 7, true, false);
  EXPECT_EQ(asian.value, euro.value);
  EXPECT_EQ(asian.std_error, euro.std_error);
  EXPECT_EQ(asian.n_paths, euro.n_paths);
}

TEST(MonteCarlo, BarrierSingleMonitoringMatchesAnalytic) {
  // One monitoring date (t = T, plus t = 0 where s < B): the up-and-out call
  // pays (S_T - k) 1{k < S_T < B} = C(k) - C(B) - (B - k) e^{-rt} N(d2(B)).
  const double barrier = 130.0;
  const double sq_t = std::sqrt(kT);
  const double d2_b = (std::log(kS / barrier) + (kR - kQ - 0.5 * kSigma * kSigma) * kT) /
                      (kSigma * sq_t);
  const double analytic = dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall) -
                          dpe::bs_price(kS, barrier, kT, kSigma, kR, kQ, kCall) -
                          (barrier - kK) * std::exp(-kR * kT) * dpe::norm_cdf(d2_b);
  EXPECT_NEAR(analytic, 5.310827113020051, 1e-12);  // derived independently
  const dpe::MCResult res = dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ, kCall, barrier,
                                                   200000, 1, 3);
  EXPECT_NEAR(res.value, analytic, 3.0 * res.std_error);
  EXPECT_LT(res.value, dpe::bs_price(kS, kK, kT, kSigma, kR, kQ, kCall));
}

TEST(MonteCarlo, LookbackSingleStepIsAtmVanilla) {
  // One step: call pays (S_T - s)^+, put pays (s - S_T)^+ — the ATM vanillas.
  const dpe::MCResult call =
      dpe::mc_lookback_floating(kS, kT, kSigma, kR, kQ, kCall, 200000, 1, 3);
  EXPECT_NEAR(call.value, dpe::bs_price(kS, kS, kT, kSigma, kR, kQ, kCall), 3.0 * call.std_error);
  const dpe::MCResult put = dpe::mc_lookback_floating(kS, kT, kSigma, kR, kQ, kPut, 200000, 1, 3);
  EXPECT_NEAR(put.value, dpe::bs_price(kS, kS, kT, kSigma, kR, kQ, kPut), 3.0 * put.std_error);
}

TEST(MonteCarlo, SigmaZeroIsDeterministic) {
  // sigma = 0: every path is the forward, the control is degenerate (var_x
  // = 0 -> beta = 0 branch) and the value equals the analytic sigma = 0 limit
  // with (numerically) zero standard error.
  const dpe::MCResult res =
      dpe::mc_european(kS, kK, kT, 0.0, kR, kQ, kCall, 1000, 1, true, true);
  EXPECT_NEAR(res.value, dpe::bs_price(kS, kK, kT, 0.0, kR, kQ, kCall), 1e-12);
  EXPECT_LE(res.std_error, 1e-12);
  EXPECT_LE(res.ci_high - res.ci_low, 1e-11);
  EXPECT_TRUE(std::isfinite(res.value));
  const dpe::MCResult asian =
      dpe::mc_asian_arithmetic(kS, kK, kT, 0.0, kR, kQ, kPut, 1000, 4, 1, true, true);
  EXPECT_NEAR(asian.value, dpe::geometric_asian_price(kS, kK, kT, 0.0, kR, kQ, kPut, 4), 1e-12);
  EXPECT_LE(asian.std_error, 1e-12);
}

TEST(MonteCarlo, ResultsFiniteAndCiOrdered) {
  const dpe::MCResult results[] = {
      dpe::mc_european(kS, kK, kT, kSigma, kR, kQ, kPut, 2000, 5, true, true),
      dpe::mc_asian_arithmetic(kS, kK, kT, kSigma, kR, kQ, kPut, 2000, 6, 5),
      dpe::mc_barrier_up_out(kS, kK, kT, kSigma, kR, kQ, kPut, 140.0, 2000, 6, 5),
      dpe::mc_lookback_floating(kS, kT, kSigma, kR, kQ, kPut, 2000, 6, 5),
      dpe::mc_delta_pathwise(kS, kK, kT, kSigma, kR, kQ, kPut, 2000, 5),
      dpe::mc_delta_fd_crn(kS, kK, kT, kSigma, kR, kQ, kPut, 2000, 5),
  };
  for (const dpe::MCResult& r : results) {
    EXPECT_TRUE(std::isfinite(r.value) && std::isfinite(r.std_error) &&
                std::isfinite(r.ci_low) && std::isfinite(r.ci_high));
    EXPECT_LE(r.ci_low, r.value);
    EXPECT_LE(r.value, r.ci_high);
    EXPECT_GE(r.std_error, 0.0);
  }
}

}  // namespace
