/// \file black_scholes.hpp
/// \brief Analytic Black-Scholes-Merton and Garman-Kohlhagen pricing.
///
/// Conventions (identical to the Python reference and every other port):
///  - t is time to expiry in years; sigma is a decimal (0.20 = 20%);
///  - r and q are continuously-compounded decimal rates (may be negative);
///  - for FX (Garman-Kohlhagen) r = rd (domestic) and q = rf (foreign),
///    with the spot quoted domestic-per-foreign.
///
/// Degenerate limits (t = 0, sigma = 0, k = 0) are exact: the price
/// collapses to (discounted) intrinsic value and second-order Greeks vanish.

#ifndef DPE_BLACK_SCHOLES_HPP
#define DPE_BLACK_SCHOLES_HPP

#include "dpe/types.hpp"

namespace dpe {

/// Standard normal CDF, N(x) = 0.5 * erfc(-x / sqrt(2)).
///
/// erfc is used instead of 0.5*(1 + erf(...)) because it keeps full relative
/// precision in the deep left tail (deep-OTM options), where 1 + erf would
/// suffer catastrophic cancellation (golden case bs_call_deep_otm, tol 1e-10).
double norm_cdf(double x);

/// Standard normal density phi(x) = exp(-x^2/2) / sqrt(2 pi).
double norm_pdf(double x);

/// European Black-Scholes-Merton price with continuous dividend yield.
///
/// \param s spot (> 0)
/// \param k strike (>= 0; k = 0 call is a forward claim worth s e^{-qt})
/// \param t time to expiry in years (>= 0)
/// \param sigma annualised volatility, decimal (>= 0)
/// \param r continuously-compounded risk-free (domestic) rate
/// \param q continuous dividend yield (or foreign rate)
/// \param option_type call or put
///
/// Degenerate limits: t = 0 -> intrinsic max(+-(s-k), 0);
/// sigma = 0 (t > 0) -> discounted forward intrinsic
/// max(+-(s e^{-qt} - k e^{-rt}), 0); k = 0 -> call = s e^{-qt}, put = 0.
/// \throws std::invalid_argument on invalid inputs.
double bs_price(double s, double k, double t, double sigma, double r, double q,
                OptionType option_type);

/// Full analytic Greeks for a European BSM option (units in dpe::Greeks).
///
/// In the degenerate region (t = 0 or sigma = 0 or k = 0):
/// gamma = vega = vanna = volga = 0; with forward moneyness
/// f = s e^{-qt} - k e^{-rt}, an in-the-money call (f >= 0 or k = 0) has
/// delta = e^{-qt}, theta = q s e^{-qt} - r k e^{-rt}, rho = k t e^{-rt}
/// (mirrored for puts, ITM when f < 0 and k > 0); all three are 0 otherwise.
/// \throws std::invalid_argument on invalid inputs.
Greeks bs_greeks(double s, double k, double t, double sigma, double r, double q,
                 OptionType option_type);

/// Garman-Kohlhagen FX option price: exactly bs_price with r = rd, q = rf.
/// s and k are FX rates quoted domestic-per-foreign; the price is in
/// domestic currency per unit of foreign notional.
double gk_price(double s, double k, double t, double sigma, double rd, double rf,
                OptionType option_type);

/// Garman-Kohlhagen Greeks: bs_greeks with r = rd, q = rf.
/// The reported delta is the premium-excluded spot delta e^{-rf t} N(d1);
/// rho is the sensitivity to the domestic rate rd. See fx.hpp for
/// spot/forward delta-convention conversions.
Greeks gk_greeks(double s, double k, double t, double sigma, double rd, double rf,
                 OptionType option_type);

/// Implied Black-Scholes volatility via bracketed Newton with bisection fallback.
///
/// Strategy: (1) reject prices at or outside the no-arbitrage bounds
/// [max(+-(s e^{-qt} - k e^{-rt}), 0), s e^{-qt} (call) / k e^{-rt} (put)]
/// with an error naming the violated bound; (2) bracket the root in [0, hi]
/// with hi doubling from 1.0 until the model price exceeds the target
/// (hard cap 20.0 = 2000% vol); (3) safeguarded Newton with analytic vega,
/// falling back to bisection whenever a step would leave the bracket or vega
/// underflows (deep ITM/OTM). Converged when |model - price| < tol or the
/// bracket width drops below 1e-12. tol must be > 0 and max_iter >= 1; if
/// neither stopping rule is met within max_iter Newton/bisection steps the
/// function throws (message contains "did not converge" and the residual) —
/// it never returns a half-converged root as if it had converged.
///
/// Additionally requires t > 0 and k > 0.
/// \throws std::invalid_argument on invalid inputs (including tol <= 0,
///         max_iter < 1), inadmissible prices, or non-convergence.
double implied_vol(double price, double s, double k, double t, double r, double q,
                   OptionType option_type, double tol = 1e-10, int max_iter = 100);

}  // namespace dpe

#endif  // DPE_BLACK_SCHOLES_HPP
