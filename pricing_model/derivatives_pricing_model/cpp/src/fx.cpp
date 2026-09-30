#include "dpe/fx.hpp"

#include <cmath>

#include "validation.hpp"

namespace dpe {

double fx_forward(double s, double t, double rd, double rf) {
  detail::require_positive("s", s);
  detail::require_non_negative("t", t);
  detail::require_finite("rd", rd);
  detail::require_finite("rf", rf);
  return s * std::exp((rd - rf) * t);
}

double spot_delta_to_forward(double delta_spot, double t, double rf) {
  detail::require_finite("delta_spot", delta_spot);
  detail::require_non_negative("t", t);
  detail::require_finite("rf", rf);
  return std::exp(rf * t) * delta_spot;
}

double forward_delta_to_spot(double delta_forward, double t, double rf) {
  detail::require_finite("delta_forward", delta_forward);
  detail::require_non_negative("t", t);
  detail::require_finite("rf", rf);
  return std::exp(-rf * t) * delta_forward;
}

}  // namespace dpe
