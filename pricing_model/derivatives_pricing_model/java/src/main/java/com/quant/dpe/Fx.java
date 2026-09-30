package com.quant.dpe;

/**
 * FX-specific helpers: forwards and delta-convention conversions.
 *
 * <p>FX options are quoted and hedged under several delta conventions. This
 * class implements the two premium-excluded ("pips") conventions:</p>
 * <ul>
 *   <li><b>Spot delta</b> — sensitivity of the domestic-currency premium to
 *       the spot: for a call, {@code delta_spot = e^{-rf t} N(d1)}. This is
 *       what {@link BlackScholes#gkGreeks} reports.</li>
 *   <li><b>Forward delta</b> — the hedge in forward contracts rather than
 *       spot: {@code delta_fwd = N(d1)} for a call. Because one forward
 *       contract has spot sensitivity {@code e^{-rf t}} (a forward on one
 *       unit of foreign currency is replicated by holding {@code e^{-rf t}}
 *       units of foreign cash), the two conventions differ by exactly that
 *       factor: {@code delta_fwd = e^{+rf t} * delta_spot}. The conversion is
 *       payoff-independent — pure discounting — so the same factor applies to
 *       calls and puts and at any strike.</li>
 * </ul>
 *
 * <p>Premium-included deltas (premium paid in foreign currency) are out of
 * scope for this project.</p>
 */
public final class Fx {

    private Fx() {
    }

    /**
     * Covered-interest-parity FX forward: {@code F = s * exp((rd - rf) t)}.
     *
     * @param s  FX spot, domestic per foreign (&gt; 0)
     * @param t  time to delivery in years (&gt;= 0)
     * @param rd domestic continuously compounded rate (may be negative)
     * @param rf foreign continuously compounded rate (may be negative)
     * @return the forward FX rate
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double fxForward(double s, double t, double rd, double rf) {
        Validation.requirePositive("s", s);
        Validation.requireNonNegative("t", t);
        Validation.requireFinite("rd", rd);
        Validation.requireFinite("rf", rf);
        return s * Math.exp((rd - rf) * t);
    }

    /**
     * Converts a premium-excluded spot delta to the forward-delta convention:
     * {@code delta_fwd = exp(rf * t) * delta_spot}.
     *
     * <p>Sign-preserving — valid for calls (positive delta) and puts
     * (negative delta) alike; exact inverse of {@link #forwardDeltaToSpot}.</p>
     *
     * @param deltaSpot premium-excluded spot delta
     * @param t         time to expiry in years (&gt;= 0)
     * @param rf        foreign continuously compounded rate
     * @return the equivalent forward delta
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double spotDeltaToForward(double deltaSpot, double t, double rf) {
        Validation.requireFinite("delta_spot", deltaSpot);
        Validation.requireNonNegative("t", t);
        Validation.requireFinite("rf", rf);
        return Math.exp(rf * t) * deltaSpot;
    }

    /**
     * Converts a premium-excluded forward delta to the spot-delta convention:
     * {@code delta_spot = exp(-rf * t) * delta_forward}.
     *
     * @param deltaForward premium-excluded forward delta
     * @param t            time to expiry in years (&gt;= 0)
     * @param rf           foreign continuously compounded rate
     * @return the equivalent spot delta
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double forwardDeltaToSpot(double deltaForward, double t, double rf) {
        Validation.requireFinite("delta_forward", deltaForward);
        Validation.requireNonNegative("t", t);
        Validation.requireFinite("rf", rf);
        return Math.exp(-rf * t) * deltaForward;
    }
}
