// Golden-value suite: every case in data/golden/golden.json must reproduce
// within its stated absolute tolerance. The same file drives the Python,
// Rust and Java suites, so this test is the cross-language contract for C++.
// Dispatch is by case-name prefix, exactly as documented in API_SPEC.md §9.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "dpe/dpe.hpp"
#include "mini_json.hpp"

#ifndef DPE_GOLDEN_JSON
#define DPE_GOLDEN_JSON "../data/golden/golden.json"
#endif

namespace {

using mini_json::Value;

const Value& golden() {
  static const Value doc = [] {
    std::ifstream in(DPE_GOLDEN_JSON);
    if (!in) throw std::runtime_error("cannot open golden.json at " DPE_GOLDEN_JSON);
    std::ostringstream buf;
    buf << in.rdbuf();
    return mini_json::parse(buf.str());
  }();
  return doc;
}

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

double in_num(const Value& c, const std::string& key) {
  return c.at("inputs").at(key).as_number();
}

dpe::OptionType in_type(const Value& c) {
  return dpe::parse_option_type(c.at("inputs").at("type").as_string());
}

/// Run the function a golden case exercises and return the actual outputs.
std::map<std::string, double> compute(const Value& c) {
  const std::string name = c.at("name").as_string();
  const Value& inp = c.at("inputs");
  const Value& expect = c.at("expect");

  if (starts_with(name, "bs_greeks")) {
    const dpe::Greeks g =
        dpe::bs_greeks(in_num(c, "s"), in_num(c, "k"), in_num(c, "t"),
                       in_num(c, "sigma"), in_num(c, "r"), in_num(c, "q"), in_type(c));
    return {{"delta", g.delta}, {"gamma", g.gamma},   {"vega", g.vega},
            {"theta", g.theta}, {"rho", g.rho},       {"vanna", g.vanna},
            {"volga", g.volga}, {"price", g.price}};
  }
  if (starts_with(name, "gk_")) {
    const double rd = in_num(c, "rd");
    const double rf = in_num(c, "rf");
    std::map<std::string, double> out;
    out["price"] = dpe::gk_price(in_num(c, "s"), in_num(c, "k"), in_num(c, "t"),
                                 in_num(c, "sigma"), rd, rf, in_type(c));
    if (expect.contains("delta_spot")) {
      const dpe::Greeks g = dpe::gk_greeks(in_num(c, "s"), in_num(c, "k"),
                                           in_num(c, "t"), in_num(c, "sigma"), rd, rf,
                                           in_type(c));
      out["delta_spot"] = g.delta;
      out["delta_forward"] = dpe::spot_delta_to_forward(g.delta, in_num(c, "t"), rf);
    }
    return out;
  }
  if (starts_with(name, "bs_")) {
    return {{"price", dpe::bs_price(in_num(c, "s"), in_num(c, "k"), in_num(c, "t"),
                                    in_num(c, "sigma"), in_num(c, "r"),
                                    in_num(c, "q"), in_type(c))}};
  }
  if (starts_with(name, "iv_")) {
    return {{"sigma", dpe::implied_vol(in_num(c, "price"), in_num(c, "s"),
                                       in_num(c, "k"), in_num(c, "t"),
                                       in_num(c, "r"), in_num(c, "q"), in_type(c))}};
  }
  if (starts_with(name, "crr_") || starts_with(name, "jr_")) {
    const auto style = dpe::parse_exercise_style(inp.at("style").as_string());
    const auto method = dpe::parse_tree_method(inp.at("method").as_string());
    const int steps = static_cast<int>(in_num(c, "steps"));
    if (expect.contains("delta")) {
      const dpe::BinomialGreeks g = dpe::binomial_greeks(
          in_num(c, "s"), in_num(c, "k"), in_num(c, "t"), in_num(c, "sigma"),
          in_num(c, "r"), in_num(c, "q"), in_type(c), style, steps, method);
      return {{"delta", g.delta}, {"gamma", g.gamma}, {"theta", g.theta}};
    }
    return {{"price", dpe::binomial_price(in_num(c, "s"), in_num(c, "k"),
                                          in_num(c, "t"), in_num(c, "sigma"),
                                          in_num(c, "r"), in_num(c, "q"), in_type(c),
                                          style, steps, method)}};
  }
  if (starts_with(name, "geo_asian")) {
    return {{"price", dpe::geometric_asian_price(
                          in_num(c, "s"), in_num(c, "k"), in_num(c, "t"),
                          in_num(c, "sigma"), in_num(c, "r"), in_num(c, "q"),
                          in_type(c), static_cast<int>(in_num(c, "n_fixings")))}};
  }
  if (starts_with(name, "mc_asian")) {
    // Per API_SPEC §9.3 the seed is advisory (RNG streams are language-
    // specific); the run stays deterministic with our own mt19937_64 stream.
    const dpe::MCResult res = dpe::mc_asian_arithmetic(
        in_num(c, "s"), in_num(c, "k"), in_num(c, "t"), in_num(c, "sigma"),
        in_num(c, "r"), in_num(c, "q"), in_type(c),
        static_cast<std::int64_t>(in_num(c, "n_paths")),
        static_cast<int>(in_num(c, "n_steps")),
        static_cast<std::int64_t>(in_num(c, "seed")), true, true);
    return {{"price", res.value}};
  }
  if (starts_with(name, "mc_euro")) {
    const dpe::MCResult res = dpe::mc_european(
        in_num(c, "s"), in_num(c, "k"), in_num(c, "t"), in_num(c, "sigma"),
        in_num(c, "r"), in_num(c, "q"), in_type(c),
        static_cast<std::int64_t>(in_num(c, "n_paths")),
        static_cast<std::int64_t>(in_num(c, "seed")), true, false);
    return {{"price", res.value}};
  }
  if (starts_with(name, "mc_barrier")) {
    const dpe::MCResult res = dpe::mc_barrier_up_out(
        in_num(c, "s"), in_num(c, "k"), in_num(c, "t"), in_num(c, "sigma"),
        in_num(c, "r"), in_num(c, "q"), in_type(c), in_num(c, "barrier"),
        static_cast<std::int64_t>(in_num(c, "n_paths")),
        static_cast<int>(in_num(c, "n_steps")),
        static_cast<std::int64_t>(in_num(c, "seed")), true);
    return {{"price", res.value}};
  }
  if (starts_with(name, "mc_lookback")) {
    const dpe::MCResult res = dpe::mc_lookback_floating(
        in_num(c, "s"), in_num(c, "t"), in_num(c, "sigma"), in_num(c, "r"),
        in_num(c, "q"), in_type(c), static_cast<std::int64_t>(in_num(c, "n_paths")),
        static_cast<int>(in_num(c, "n_steps")),
        static_cast<std::int64_t>(in_num(c, "seed")), true);
    return {{"price", res.value}};
  }
  ADD_FAILURE() << "no dispatch rule for golden case " << name;
  return {};
}

}  // namespace

TEST(Golden, FileShape) {
  const auto& cases = golden().at("cases").as_array();
  ASSERT_GE(cases.size(), 20u);
  std::set<std::string> names;
  for (const auto& c : cases) {
    names.insert(c.at("name").as_string());
    EXPECT_GT(c.at("tol").as_number(), 0.0);
  }
  EXPECT_EQ(names.size(), cases.size()) << "duplicate golden case names";
}

TEST(Golden, AllCases) {
  const auto& cases = golden().at("cases").as_array();
  for (const auto& c : cases) {
    const std::string name = c.at("name").as_string();
    const double tol = c.at("tol").as_number();
    const std::map<std::string, double> actual = compute(c);
    for (const auto& [key, expected] : c.at("expect").as_object()) {
      ASSERT_TRUE(actual.count(key) != 0) << name << ": missing key " << key;
      EXPECT_NEAR(actual.at(key), expected.as_number(), tol)
          << "case " << name << ", key " << key;
    }
  }
}
