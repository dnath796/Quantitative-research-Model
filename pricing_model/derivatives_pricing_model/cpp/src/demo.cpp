/// \file demo.cpp
/// \brief End-to-end demo of the dpe pricing engine (C++ port).
///
/// Mirrors python/demo.py:
///  1. Analytic BSM / Garman-Kohlhagen prices and full Greeks.
///  2. Implied-vol round-trip.
///  3. Binomial trees: convergence to BS, American early-exercise premium.
///  4. Monte Carlo: variance-reduction comparison, exotics with 95% CIs,
///     pathwise vs CRN finite-difference delta.
///  5. Historical vol from data/spots_timeseries.csv.
///  6. Risk report for the small book in data/portfolio.csv.

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dpe/dpe.hpp"

#ifndef DPE_DATA_DIR
#define DPE_DATA_DIR "../data"
#endif

namespace {

using dpe::OptionType;

const char* kRule =
    "------------------------------------------------------------------------------";

void section(const char* title) { std::printf("\n%s\n%s\n%s\n", kRule, title, kRule); }

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    rows.push_back(std::move(fields));
  }
  return rows;
}

void demo_analytic() {
  section("1. Analytic Black-Scholes-Merton and Garman-Kohlhagen");
  struct Row {
    const char* name;
    double s, k, t, sigma, r, q;
    OptionType ot;
  };
  const Row rows[] = {
      {"Equity ATM call (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::Call},
      {"Equity ATM put  (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::Put},
      {"High-vol small cap call", 18.0, 20.0, 1.0, 0.80, 0.04, 0.00, OptionType::Call},
      {"Weekly put (T=1/52)", 100.0, 98.0, 1.0 / 52, 0.25, 0.03, 0.00, OptionType::Put},
      {"LEAPS call (T=3y)", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, OptionType::Call},
      {"FX EURUSD call (rd,rf)", 1.10, 1.12, 0.5, 0.10, 0.03, 0.02, OptionType::Call},
      {"FX neg-rate put (rd<0)", 0.95, 0.95, 1.0, 0.12, -0.005, 0.001, OptionType::Put},
  };
  std::printf("%-24s%10s%9s%9s%9s%9s%9s%9s%9s\n", "instrument", "price", "delta",
              "gamma", "vega", "theta", "rho", "vanna", "volga");
  for (const Row& row : rows) {
    const dpe::Greeks g =
        dpe::bs_greeks(row.s, row.k, row.t, row.sigma, row.r, row.q, row.ot);
    std::printf("%-24s%10.4f%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f\n", row.name, g.price,
                g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga);
  }
  const dpe::Greeks g = dpe::gk_greeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02,
                                       OptionType::Call);
  const double fwd_delta = dpe::spot_delta_to_forward(g.delta, 0.5, 0.02);
  std::printf(
      "\nFX delta conventions (EURUSD call): spot delta = %.4f, forward delta = "
      "%.4f, forward = %.4f\n",
      g.delta, fwd_delta, dpe::fx_forward(1.10, 0.5, 0.03, 0.02));
}

void demo_implied_vol() {
  section("2. Implied volatility round-trip");
  const double pairs[][2] = {{0.15, 120.0}, {0.40, 80.0}};
  for (const auto& p : pairs) {
    const double sigma_in = p[0], k = p[1];
    const double price =
        dpe::bs_price(100.0, k, 0.5, sigma_in, 0.03, 0.01, OptionType::Call);
    const double iv = dpe::implied_vol(price, 100.0, k, 0.5, 0.03, 0.01,
                                       OptionType::Call);
    std::printf("k=%6.1f  price=%9.4f  sigma_in=%.4f  implied=%.10f\n", k, price,
                sigma_in, iv);
  }
  try {
    dpe::implied_vol(1.0, 100.0, 80.0, 0.5, 0.03, 0.0, OptionType::Call);
  } catch (const std::invalid_argument& exc) {
    std::printf("below-intrinsic price correctly rejected: %s\n", exc.what());
  }
}

void demo_binomial() {
  section("3. Binomial trees (CRR / Jarrow-Rudd)");
  const double s = 100, k = 100, t = 1.0, sig = 0.2, r = 0.05, q = 0.0;
  const double bs = dpe::bs_price(s, k, t, sig, r, q, OptionType::Call);
  std::printf("%6s %14s %16s %13s\n", "steps", "CRR-BS error", "CRR+Richardson",
              "JR-BS error");
  for (int n : {50, 200, 800, 2000}) {
    const double crr = dpe::binomial_price(s, k, t, sig, r, q, OptionType::Call,
                                           dpe::ExerciseStyle::European, n,
                                           dpe::TreeMethod::CRR);
    const double rich = dpe::binomial_price(s, k, t, sig, r, q, OptionType::Call,
                                            dpe::ExerciseStyle::European, n,
                                            dpe::TreeMethod::CRR, true);
    const double jr = dpe::binomial_price(s, k, t, sig, r, q, OptionType::Call,
                                          dpe::ExerciseStyle::European, n,
                                          dpe::TreeMethod::JR);
    std::printf("%6d %14.2e %16.2e %13.2e\n", n, crr - bs, rich - bs, jr - bs);
  }
  std::printf("\nAmerican put early-exercise premium (s=100, T=1, sigma=20%%, r=5%%):\n");
  std::printf("%8s%12s%12s%10s\n", "strike", "european", "american", "premium");
  for (double kk : {90.0, 100.0, 110.0, 120.0}) {
    const double eur = dpe::binomial_price(s, kk, t, sig, r, 0.0, OptionType::Put,
                                           dpe::ExerciseStyle::European, 500);
    const double amer = dpe::binomial_price(s, kk, t, sig, r, 0.0, OptionType::Put,
                                            dpe::ExerciseStyle::American, 500);
    std::printf("%8.1f%12.4f%12.4f%10.4f\n", kk, eur, amer, amer - eur);
  }
  const dpe::BinomialGreeks bg =
      dpe::binomial_greeks(s, k, t, sig, r, q, OptionType::Call,
                           dpe::ExerciseStyle::European, 500, dpe::TreeMethod::CRR);
  const dpe::Greeks ag = dpe::bs_greeks(s, k, t, sig, r, q, OptionType::Call);
  std::printf(
      "\ntree Greeks (500-step CRR) vs analytic: delta %.4f/%.4f, gamma %.4f/%.4f, "
      "theta %.4f/%.4f\n",
      bg.delta, ag.delta, bg.gamma, ag.gamma, bg.theta, ag.theta);
}

void print_mc_row(const char* label, int width, const dpe::MCResult& res) {
  char ci[32];
  std::snprintf(ci, sizeof(ci), "[%.4f, %.4f]", res.ci_low, res.ci_high);
  std::printf("%-*s%10.4f%10.4f%24s\n", width, label, res.value, res.std_error, ci);
}

void demo_monte_carlo() {
  section("4. Monte Carlo (100k paths, seed 42)");
  const double s = 100, k = 100, t = 1.0, sig = 0.2, r = 0.05, q = 0.0;
  const std::int64_t n = 100000;
  const double bs = dpe::bs_price(s, k, t, sig, r, q, OptionType::Call);
  const dpe::MCResult plain =
      dpe::mc_european(s, k, t, sig, r, q, OptionType::Call, n, 42, false);
  const dpe::MCResult anti =
      dpe::mc_european(s, k, t, sig, r, q, OptionType::Call, n, 42, true);
  const dpe::MCResult cv =
      dpe::mc_european(s, k, t, sig, r, q, OptionType::Call, n, 42, true, true);
  std::printf("European call, BS analytic = %.4f\n", bs);
  std::printf("%-28s%10s%10s%24s\n", "estimator", "price", "std err", "95% CI");
  print_mc_row("plain", 28, plain);
  print_mc_row("antithetic", 28, anti);
  print_mc_row("antithetic + control", 28, cv);

  const dpe::MCResult asian =
      dpe::mc_asian_arithmetic(s, k, t, sig, r, q, OptionType::Call, n, 12, 42);
  const dpe::MCResult barrier = dpe::mc_barrier_up_out(s, k, t, sig, r, q,
                                                       OptionType::Call, 130.0, n,
                                                       100, 42);
  const dpe::MCResult lookback =
      dpe::mc_lookback_floating(s, t, sig, r, q, OptionType::Call, n, 100, 42);
  std::printf("\n%-34s%10s%10s%24s\n", "exotic", "price", "std err", "95% CI");
  print_mc_row("arithmetic Asian (12 fix, CV)", 34, asian);
  print_mc_row("up-and-out barrier B=130", 34, barrier);
  print_mc_row("floating-strike lookback", 34, lookback);

  const dpe::MCResult pw =
      dpe::mc_delta_pathwise(s, k, t, sig, r, q, OptionType::Call, n, 42);
  const dpe::MCResult fd =
      dpe::mc_delta_fd_crn(s, k, t, sig, r, q, OptionType::Call, n, 42);
  std::printf(
      "\ndelta: analytic %.4f, pathwise %.4f (se %.4f), FD+CRN %.4f (se %.4f)\n",
      dpe::bs_greeks(s, k, t, sig, r, q, OptionType::Call).delta, pw.value,
      pw.std_error, fd.value, fd.std_error);
}

void demo_historical_vol() {
  section("5. Historical volatility from data/spots_timeseries.csv");
  const auto rows = read_csv(std::string(DPE_DATA_DIR) + "/spots_timeseries.csv");
  const auto& header = rows.front();
  std::printf("%-10s%12s%14s\n", "underlier", "last close", "realised vol");
  for (std::size_t col = 1; col < header.size(); ++col) {
    std::vector<double> prices;
    prices.reserve(rows.size() - 1);
    for (std::size_t i = 1; i < rows.size(); ++i) prices.push_back(std::stod(rows[i][col]));
    const double vol = dpe::historical_vol(prices);
    std::printf("%-10s%12.4f%13.2f%%\n", header[col].c_str(), prices.back(),
                100.0 * vol);
  }
}

void demo_portfolio() {
  section("6. Portfolio risk report (data/portfolio.csv)");
  const auto rows = read_csv(std::string(DPE_DATA_DIR) + "/portfolio.csv");
  std::printf("%-16s%-4s%-5s%10s%9s%9s%10s%10s%16s\n", "id", "typ", "style", "price",
              "delta", "gamma", "vega", "theta", "position value");
  double total = 0.0;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& f = rows[i];
    // Columns: id, underlying_type, s, k, t, sigma, r, q, type, style, quantity.
    const std::string& id = f[0];
    const double s = std::stod(f[2]), k = std::stod(f[3]), t = std::stod(f[4]);
    const double sigma = std::stod(f[5]), r = std::stod(f[6]), q = std::stod(f[7]);
    const OptionType ot = dpe::parse_option_type(f[8]);
    const dpe::ExerciseStyle style = dpe::parse_exercise_style(f[9]);
    const double quantity = std::stod(f[10]);

    double price = 0, delta = 0, gamma = 0, theta = 0;
    char vega_str[16];
    if (style == dpe::ExerciseStyle::American) {
      // American exercise has no closed form: price and Greeks off the tree.
      price = dpe::binomial_price(s, k, t, sigma, r, q, ot,
                                  dpe::ExerciseStyle::American, 500);
      const dpe::BinomialGreeks g = dpe::binomial_greeks(
          s, k, t, sigma, r, q, ot, dpe::ExerciseStyle::American, 500);
      delta = g.delta;
      gamma = g.gamma;
      theta = g.theta;
      std::snprintf(vega_str, sizeof(vega_str), "%10s", "n/a");
    } else {
      const dpe::Greeks g = dpe::bs_greeks(s, k, t, sigma, r, q, ot);
      price = g.price;
      delta = g.delta;
      gamma = g.gamma;
      theta = g.theta;
      std::snprintf(vega_str, sizeof(vega_str), "%10.4f", g.vega);
    }
    const double value = price * quantity;
    total += value;
    std::printf("%-16s%-4c%-5.4s%10.4f%9.4f%9.4f%s%10.4f%16.2f\n", id.c_str(),
                static_cast<char>(std::toupper(static_cast<unsigned char>(f[8][0]))),
                f[9].c_str(), price, delta, gamma, vega_str, theta, value);
  }
  std::printf("\n%-57s%16.2f\n", "total book value", total);
  std::printf(
      "(FX rows: r=rd, q=rf; price in domestic ccy per unit foreign; vega n/a for "
      "tree)\n");
}

}  // namespace

int main() {
  std::printf("dpe — Derivatives Pricing Engine demo (C++ port)\n");
  demo_analytic();
  demo_implied_vol();
  demo_binomial();
  demo_monte_carlo();
  demo_historical_vol();
  demo_portfolio();
  std::printf("\n%s\ndone.\n\n", kRule);
  return 0;
}
