#include "validation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "dpe/types.hpp"

namespace dpe {
namespace {

/// Lower-case a copy of the input (ASCII; canonical spellings are ASCII).
std::string to_lower(const std::string& s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

std::string fmt(double value) {
  std::ostringstream os;
  os.precision(17);
  os << value;
  return os.str();
}

}  // namespace

OptionType parse_option_type(const std::string& option_type) {
  const std::string ot = to_lower(option_type);
  if (ot == "call") return OptionType::Call;
  if (ot == "put") return OptionType::Put;
  fail("option_type must be 'call' or 'put', got '" + option_type + "'");
}

ExerciseStyle parse_exercise_style(const std::string& style) {
  const std::string st = to_lower(style);
  if (st == "european") return ExerciseStyle::European;
  if (st == "american") return ExerciseStyle::American;
  fail("style must be 'european' or 'american', got '" + style + "'");
}

TreeMethod parse_tree_method(const std::string& method) {
  const std::string m = to_lower(method);
  if (m == "crr") return TreeMethod::CRR;
  if (m == "jr") return TreeMethod::JR;
  fail("method must be 'crr' or 'jr', got '" + method + "'");
}

namespace detail {

void require_finite(const char* name, double value) {
  if (!std::isfinite(value)) {
    fail(std::string(name) + " must be finite, got " + fmt(value));
  }
}

void require_positive(const char* name, double value) {
  require_finite(name, value);
  if (value <= 0.0) {
    fail(std::string(name) + " must be > 0, got " + fmt(value));
  }
}

void require_non_negative(const char* name, double value) {
  require_finite(name, value);
  if (value < 0.0) {
    fail(std::string(name) + " must be >= 0, got " + fmt(value));
  }
}

void validate_market_inputs(double s, double k, double t, double sigma, double r,
                            double q) {
  require_positive("s", s);
  require_non_negative("k", k);
  require_non_negative("t", t);
  require_non_negative("sigma", sigma);
  require_finite("r", r);
  require_finite("q", q);
}

void require_seed(std::int64_t seed) {
  if (seed < 0) {
    fail("seed must be an integer in [0, " + std::to_string(kSeedMax) + "], got " +
         std::to_string(seed));
  }
}

void require_rel_bump(double rel_bump) {
  require_positive("rel_bump", rel_bump);
  if (rel_bump >= 1.0) {
    // rel_bump >= 1 would price the down bump at a non-positive spot and
    // return a meaningless delta without any error (API_SPEC §5.8).
    fail("rel_bump must be in (0, 1), got " + fmt(rel_bump));
  }
}

void validate_mc(std::int64_t n_paths, std::int64_t seed, bool antithetic, int n_steps) {
  if (n_paths < 2) {
    fail("n_paths must be an integer >= 2, got " + std::to_string(n_paths));
  }
  if (antithetic && n_paths % 2 != 0) {
    fail("n_paths must be even with antithetic=true, got " + std::to_string(n_paths));
  }
  if (antithetic && n_paths < 4) {
    fail("n_paths must be >= 4 with antithetic=true, got " + std::to_string(n_paths));
  }
  require_seed(seed);
  if (n_steps < 1) {
    fail("n_steps must be an integer >= 1, got " + std::to_string(n_steps));
  }
}

}  // namespace detail
}  // namespace dpe
