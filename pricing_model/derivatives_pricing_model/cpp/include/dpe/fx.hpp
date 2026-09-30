/// \file fx.hpp
/// \brief FX-specific helpers: forwards and delta-convention conversions.
///
/// The two premium-excluded ("pips") conventions implemented here:
///  - spot delta: sensitivity of the domestic premium to the spot
///    (e^{-rf t} N(d1) for a call) — what gk_greeks reports;
///  - forward delta: the hedge in forward contracts (N(d1) for a call).
///
/// One forward on a unit of foreign currency has spot sensitivity e^{-rf t}
/// (it is replicated by e^{-rf t} units of foreign cash), so the conversion
/// is the pure discount factor e^{+-rf t}: payoff-independent, sign-
/// preserving (works for put deltas) and exactly self-inverse. Premium-
/// included deltas are out of scope.

#ifndef DPE_FX_HPP
#define DPE_FX_HPP

namespace dpe {

/// Covered-interest-parity FX forward: F = s * exp((rd - rf) t).
/// s is spot domestic-per-foreign; rd/rf may be negative.
/// \throws std::invalid_argument if s <= 0, t < 0, or any input non-finite.
double fx_forward(double s, double t, double rd, double rf);

/// Convert a premium-excluded spot delta to the forward-delta convention:
/// delta_fwd = exp(rf * t) * delta_spot.
/// \throws std::invalid_argument if t < 0 or any input non-finite.
double spot_delta_to_forward(double delta_spot, double t, double rf);

/// Convert a premium-excluded forward delta to the spot-delta convention:
/// delta_spot = exp(-rf * t) * delta_forward (exact inverse of the above).
/// \throws std::invalid_argument if t < 0 or any input non-finite.
double forward_delta_to_spot(double delta_forward, double t, double rf);

}  // namespace dpe

#endif  // DPE_FX_HPP
