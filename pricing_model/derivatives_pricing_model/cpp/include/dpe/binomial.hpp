/// \file binomial.hpp
/// \brief Binomial-tree pricing: Cox-Ross-Rubinstein and Jarrow-Rudd lattices.
///
/// Both lattices discretise GBM over n steps of dt = t / n:
///  - CRR: u = exp(sigma sqrt(dt)), d = 1/u,
///    p = (exp((r-q) dt) - d) / (u - d); the tree recombines around the spot.
///  - JR ("equal-probability"): with nu = (r - q - sigma^2/2) dt,
///    u = exp(nu + sigma sqrt(dt)), d = exp(nu - sigma sqrt(dt)), p = 1/2;
///    the lattice drifts with the forward.
///
/// Both converge to Black-Scholes at O(1/n) with the well-known odd/even
/// oscillation; richardson = true prices at n and n+1 steps and averages,
/// cancelling the leading oscillating error term (two-point Richardson /
/// odd-even averaging) for one to two extra digits at the same cost order.

#ifndef DPE_BINOMIAL_HPP
#define DPE_BINOMIAL_HPP

#include "dpe/types.hpp"

namespace dpe {

/// Binomial-tree price of a European or American vanilla option.
///
/// \param steps number of time steps, >= 1
/// \param method TreeMethod::CRR or TreeMethod::JR
/// \param richardson average the steps and steps+1 prices to damp the
///        odd/even oscillation of the convergence to the continuous limit
///
/// American style applies max(continuation, intrinsic) at every node.
/// If the risk-neutral probability p falls outside (0, 1) (per-step drift
/// outruns the vol spacing) the lattice is invalid at this step count and
/// std::invalid_argument is thrown — never a silent clamp.
///
/// Degenerate limits: t = 0 -> intrinsic; sigma = 0 -> European: discounted
/// forward intrinsic; American: the maximum over grid dates t_i = i t/steps
/// of e^{-r t_i} payoff(s e^{(r-q) t_i}) (deterministic-path dynamic program).
///
/// Lattices whose terminal spots would overflow a double
/// (|ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) > 700) throw
/// instead of returning inf/NaN.
/// \throws std::invalid_argument on invalid inputs.
double binomial_price(double s, double k, double t, double sigma, double r, double q,
                      OptionType option_type,
                      ExerciseStyle style = ExerciseStyle::European,
                      int steps = 500, TreeMethod method = TreeMethod::CRR,
                      bool richardson = false);

/// Delta, gamma and theta read directly off the lattice.
///
/// With node values V(i,j) and node spots S(i,j) (level i, j up-moves),
/// retained during a single backward induction:
///
///   delta = (V(1,1) - V(1,0)) / (S(1,1) - S(1,0))
///   gamma = [one-sided deltas at level 2 differenced] / ((S(2,2)-S(2,0))/2)
///   theta = (V(2,1) - V(0,0) - delta*ds - gamma*ds^2/2) / (2 dt),
///           ds = S(2,1) - s
///
/// The ds correction strips the spot-displacement contribution out of the JR
/// calendar difference; it is exactly zero for CRR (u d = 1).
///
/// Requires steps >= 2, t > 0 and sigma > 0 (else std::invalid_argument).
BinomialGreeks binomial_greeks(double s, double k, double t, double sigma, double r,
                               double q, OptionType option_type,
                               ExerciseStyle style = ExerciseStyle::European,
                               int steps = 500, TreeMethod method = TreeMethod::CRR);

}  // namespace dpe

#endif  // DPE_BINOMIAL_HPP
