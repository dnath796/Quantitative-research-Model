// Implied-vol solver tests: round-trip accuracy across a sigma/strike grid
// (including deep ITM/OTM where Newton must not diverge), and clear errors
// on prices outside the no-arbitrage bounds.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "dpe/black_scholes.hpp"
#include "dpe/types.hpp"

namespace {

using dpe::OptionType;

constexpr OptionType kCall = OptionType::Call;
constexpr OptionType kPut = OptionType::Put;

TEST(ImpliedVol, RoundTripGrid) {
  // sigma -> price -> sigma to 1e-6 absolute for sigma in [0.05, 0.8],
  // strikes from deep ITM to deep OTM (tiny-vega corners take the
  // bisection fallback).
  const double s = 100, t = 0.5, r = 0.03, q = 0.01;
  for (double sigma : {0.05, 0.1, 0.2, 0.4, 0.8}) {
    for (double k : {60.0, 100.0, 150.0}) {
      for (OptionType ot : {kCall, kPut}) {
        const double price = dpe::bs_price(s, k, t, sigma, r, q, ot);
        // Skip corners where the price sits on the no-arb boundary to
        // machine precision (e.g. sigma=5% put struck at 60: worthless).
        const double lower = ot == kCall
                                 ? std::max(s * std::exp(-q * t) - k * std::exp(-r * t), 0.0)
                                 : std::max(k * std::exp(-r * t) - s * std::exp(-q * t), 0.0);
        if (price - lower < 1e-12) continue;
        const double iv = dpe::implied_vol(price, s, k, t, r, q, ot);
        // The solver converges in PRICE to 1e-10; the induced sigma error is
        // ~tol/vega, so in tiny-vega corners (deep OTM, low vol) assert the
        // price round-trip instead of an unattainable sigma accuracy.
        const double vega = dpe::bs_greeks(s, k, t, sigma, r, q, ot).vega;
        if (vega > 1e-4) {
          EXPECT_NEAR(iv, sigma, 1e-6) << "sigma=" << sigma << " k=" << k;
        } else {
          EXPECT_NEAR(dpe::bs_price(s, k, t, iv, r, q, ot), price, 1e-9)
              << "sigma=" << sigma << " k=" << k;
        }
      }
    }
  }
}

TEST(ImpliedVol, HighVolBeyondInitialBracket) {
  // sigma > 1 forces the doubling bracket expansion.
  const double price = dpe::bs_price(100, 100, 1.0, 1.7, 0.02, 0.0, kCall);
  EXPECT_NEAR(dpe::implied_vol(price, 100, 100, 1.0, 0.02, 0.0, kCall), 1.7, 1e-6);
}

TEST(ImpliedVol, BelowIntrinsicMentionsLowerBound) {
  // Deep ITM call: price below discounted intrinsic has no implied vol.
  try {
    dpe::implied_vol(1.0, 100, 80, 0.5, 0.03, 0.0, kCall);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("lower bound"), std::string::npos);
  }
}

TEST(ImpliedVol, AboveUpperBoundMentionsUpperBound) {
  try {
    dpe::implied_vol(101.0, 100, 100, 1.0, 0.05, 0.0, kCall);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("upper bound"), std::string::npos);
  }
}

TEST(ImpliedVol, InvalidInputsThrow) {
  EXPECT_THROW(dpe::implied_vol(5.0, 100, 100, 0.0, 0.05, 0.0, kCall),
               std::invalid_argument);  // t = 0
  EXPECT_THROW(dpe::implied_vol(5.0, 100, 0.0, 1.0, 0.05, 0.0, kCall),
               std::invalid_argument);  // k = 0
  EXPECT_THROW(dpe::implied_vol(5.0, -100, 100, 1.0, 0.05, 0.0, kCall),
               std::invalid_argument);  // s <= 0
}

// ---------------------------------------------------------------------------
// Contract-edge tests added after review (see docs/ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

constexpr double kAtmPrice = 10.45058357;  // bs_price(100,100,1,0.2,0.05,0,call) to 8 dp

TEST(ImpliedVol, ReportsNonConvergence) {
  // One Newton step from the bracket midpoint lands ~2 vol points off; the
  // solver must throw (message names the failure), never return that root.
  try {
    dpe::implied_vol(kAtmPrice, 100, 100, 1.0, 0.05, 0.0, kCall, 1e-10, 1);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("did not converge"), std::string::npos);
  }
  // A sufficient budget converges as usual.
  EXPECT_NEAR(dpe::implied_vol(kAtmPrice, 100, 100, 1.0, 0.05, 0.0, kCall, 1e-10, 20), 0.2,
              1e-8);
}

TEST(ImpliedVol, ValidatesTolAndMaxIter) {
  for (int bad_iter : {0, -3}) {
    try {
      dpe::implied_vol(kAtmPrice, 100, 100, 1.0, 0.05, 0.0, kCall, 1e-10, bad_iter);
      FAIL() << "expected std::invalid_argument for max_iter=" << bad_iter;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("max_iter"), std::string::npos);
    }
  }
  for (double bad_tol : {0.0, -1e-10, std::nan("")}) {
    try {
      dpe::implied_vol(kAtmPrice, 100, 100, 1.0, 0.05, 0.0, kCall, bad_tol, 100);
      FAIL() << "expected std::invalid_argument for tol=" << bad_tol;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("tol"), std::string::npos);
    }
  }
}

TEST(ImpliedVol, CapRegionAndTinyTolTerminate) {
  // Between 16 and 20 the bracket is clipped to the 20.0 cap and still solves.
  const double p17 = dpe::bs_price(100, 100, 0.05, 17.0, 0.02, 0.0, kCall);
  EXPECT_NEAR(dpe::implied_vol(p17, 100, 100, 0.05, 0.02, 0.0, kCall), 17.0, 1e-6);
  // tol below the price's floating-point resolution: the bracket-width rule
  // (< 1e-12) terminates with an essentially exact vol.
  const double p = dpe::bs_price(100, 100, 1.0, 0.2, 0.05, 0.0, kCall);
  EXPECT_NEAR(dpe::implied_vol(p, 100, 100, 1.0, 0.05, 0.0, kCall, 1e-300, 200), 0.2, 1e-10);
  // Deterministic.
  const double pp = dpe::bs_price(100, 90, 0.5, 0.3, 0.02, 0.01, kPut);
  EXPECT_EQ(dpe::implied_vol(pp, 100, 90, 0.5, 0.02, 0.01, kPut),
            dpe::implied_vol(pp, 100, 90, 0.5, 0.02, 0.01, kPut));
}

}  // namespace
