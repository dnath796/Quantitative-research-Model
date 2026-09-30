package com.quant.dpe;

/**
 * Analytic Black-Scholes-Merton and Garman-Kohlhagen pricing.
 *
 * <p>Under the risk-neutral measure the underlying follows geometric Brownian
 * motion with continuous dividend yield {@code q}: {@code dS/S = (r - q) dt +
 * sigma dW}. For an FX rate quoted domestic-per-foreign the foreign risk-free
 * rate plays exactly the role of a dividend yield, so Garman-Kohlhagen is BSM
 * with {@code r = rd}, {@code q = rf}.</p>
 *
 * <p>Package-wide conventions (shared with every language port): {@code t} in
 * years; {@code sigma} decimal per sqrt-year; {@code r}, {@code q}
 * continuously compounded decimals (may be negative); theta per calendar
 * year; vega/vanna/volga per unit of vol; rho per unit of rate.</p>
 *
 * <p>Degenerate limits ({@code t = 0}, {@code sigma = 0}, {@code k = 0}) are
 * handled exactly: the price collapses to (discounted) intrinsic value and the
 * second-order Greeks vanish; see {@link #bsPrice} and {@link #bsGreeks}.</p>
 */
public final class BlackScholes {

    private BlackScholes() {
    }

    private static final double SQRT2 = Math.sqrt(2.0);
    private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);
    /** 1 / sqrt(pi), used by the asymptotic erfc branch. */
    private static final double INV_SQRT_PI = 5.6418958354775628695e-1;

    /** 2000% vol: beyond any market; hard cap for the implied-vol bracket. */
    private static final double IV_MAX_SIGMA = 20.0;

    // -----------------------------------------------------------------------
    // Standard normal helpers. Java has no built-in erf/erfc, so we bundle
    // W. J. Cody's rational Chebyshev approximation (SPECFUN "calerf",
    // TOMS/netlib), accurate to < 1e-16 relative error over the whole real
    // line — the same accuracy class as the C library erfc used by the
    // Python reference, which keeps the deep-OTM golden case within 1e-10.
    // -----------------------------------------------------------------------

    // Coefficients for |x| <= 0.46875: erf(x) = x * P(x^2) / Q(x^2).
    private static final double[] ERF_A = {
            3.16112374387056560e00, 1.13864154151050156e02,
            3.77485237685302021e02, 3.20937758913846947e03,
            1.85777706184603153e-1};
    private static final double[] ERF_B = {
            2.36012909523441209e01, 2.44024637934444173e02,
            1.28261652607737228e03, 2.84423683343917062e03};
    // Coefficients for 0.46875 < x <= 4: erfc(x) = exp(-x^2) * P(x) / Q(x).
    private static final double[] ERF_C = {
            5.64188496988670089e-1, 8.88314979438837594e00,
            6.61191906371416295e01, 2.98635138197400131e02,
            8.81952221241769090e02, 1.71204761263407058e03,
            2.05107837782607147e03, 1.23033935479799725e03,
            2.15311535474403846e-8};
    private static final double[] ERF_D = {
            1.57449261107098347e01, 1.17693950891312499e02,
            5.37181101862009858e02, 1.62138957456669019e03,
            3.29079923573345963e03, 4.36261909014324716e03,
            3.43936767414372164e03, 1.23033935480374942e03};
    // Coefficients for x > 4: erfc(x) ~ exp(-x^2)/x * (1/sqrt(pi) + P(1/x^2)/Q(1/x^2)/x^2).
    private static final double[] ERF_P = {
            3.05326634961232344e-1, 3.60344899949804439e-1,
            1.25781726111229246e-1, 1.60837851487422766e-2,
            6.58749161529837803e-4, 1.63153871373020978e-2};
    private static final double[] ERF_Q = {
            2.56852019228982242e00, 1.87295284992346047e00,
            5.27905102951428412e-1, 6.05183413124413191e-2,
            2.33520497626869185e-3};

    /**
     * Complementary error function {@code erfc(x) = 2/sqrt(pi) *
     * integral_x^inf exp(-u^2) du}, full double precision.
     *
     * @param x argument (any finite double)
     * @return erfc(x) in [0, 2]
     */
    public static double erfc(double x) {
        double ax = Math.abs(x);
        if (ax <= 0.46875) {
            // erfc = 1 - erf via the central rational approximation.
            double z = ax * ax;
            double num = ERF_A[4] * z;
            double den = z;
            for (int i = 0; i < 3; i++) {
                num = (num + ERF_A[i]) * z;
                den = (den + ERF_B[i]) * z;
            }
            double erf = x * (num + ERF_A[3]) / (den + ERF_B[3]);
            return 1.0 - erf;
        }
        double result;
        if (ax <= 4.0) {
            double num = ERF_C[8] * ax;
            double den = ax;
            for (int i = 0; i < 7; i++) {
                num = (num + ERF_C[i]) * ax;
                den = (den + ERF_D[i]) * ax;
            }
            result = (num + ERF_C[7]) / (den + ERF_D[7]);
        } else if (ax >= 26.543) {
            // exp(-x^2) underflows: erfc(+x) = 0 to full double precision.
            result = 0.0;
        } else {
            double z = 1.0 / (ax * ax);
            double num = ERF_P[5] * z;
            double den = z;
            for (int i = 0; i < 4; i++) {
                num = (num + ERF_P[i]) * z;
                den = (den + ERF_Q[i]) * z;
            }
            result = z * (num + ERF_P[4]) / (den + ERF_Q[4]);
            result = (INV_SQRT_PI - result) / ax;
        }
        // Scale by exp(-x^2), splitting x^2 to preserve precision in the tail:
        // x^2 = hi^2 + (x - hi)(x + hi) with hi = x rounded to 1/16 keeps the
        // exponent argument error tiny even for large x.
        double hi = Math.floor(ax * 16.0) / 16.0;
        double del = (ax - hi) * (ax + hi);
        result = Math.exp(-hi * hi) * Math.exp(-del) * result;
        return (x < 0.0) ? 2.0 - result : result;
    }

    /**
     * Standard normal CDF {@code N(x) = 0.5 * erfc(-x / sqrt(2))}.
     *
     * <p>Computed via {@code erfc} rather than {@code 0.5 * (1 + erf(...))}
     * because erfc keeps full relative precision in the deep left tail (deep
     * OTM options), where {@code 1 + erf} suffers catastrophic cancellation.</p>
     *
     * @param x quantile
     * @return P(Z &lt;= x) for Z ~ N(0, 1)
     */
    public static double normCdf(double x) {
        return 0.5 * erfc(-x / SQRT2);
    }

    /**
     * Standard normal density {@code phi(x) = exp(-x^2 / 2) / sqrt(2 pi)}.
     *
     * @param x argument
     * @return the density at {@code x}
     */
    public static double normPdf(double x) {
        return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
    }

    /**
     * Returns {@code {d1, d2}} for the regular region {@code t > 0, sigma > 0,
     * k > 0}.
     *
     * @throws IllegalArgumentException if inputs are invalid or the point is
     *         degenerate ({@code t = 0}, {@code sigma = 0} or {@code k = 0}) —
     *         there is no finite d1/d2 there; the price/Greek functions handle
     *         those limits explicitly instead
     */
    public static double[] bsD1D2(double s, double k, double t, double sigma,
                                  double r, double q) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        if (t <= 0.0 || sigma <= 0.0 || k <= 0.0) {
            throw new IllegalArgumentException(
                    "d1/d2 require t > 0, sigma > 0 and k > 0; use bsPrice for the limits");
        }
        double sqT = Math.sqrt(t);
        double d1 = (Math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqT);
        double d2 = d1 - sigma * sqT;
        return new double[]{d1, d2};
    }

    private static double intrinsic(double s, double k, OptionType type) {
        return type == OptionType.CALL ? Math.max(s - k, 0.0) : Math.max(k - s, 0.0);
    }

    /**
     * European Black-Scholes-Merton price with continuous dividend yield.
     *
     * <p>Degenerate limits (part of the cross-language contract):</p>
     * <ul>
     *   <li>{@code t = 0}: intrinsic {@code max(+-(s - k), 0)} (no discounting);</li>
     *   <li>{@code sigma = 0}, {@code t > 0}: the terminal spot is the
     *       deterministic forward, so the value is the discounted forward
     *       intrinsic {@code max(+-(s e^{-qt} - k e^{-rt}), 0)};</li>
     *   <li>{@code k = 0}: call = {@code s e^{-qt}} (a pure forward claim), put = 0.</li>
     * </ul>
     *
     * @param s     spot (&gt; 0)
     * @param k     strike (&gt;= 0)
     * @param t     time to expiry in years (&gt;= 0)
     * @param sigma annualised volatility, decimal (&gt;= 0)
     * @param r     continuously compounded risk-free (domestic) rate; may be negative
     * @param q     continuous dividend yield (or foreign rate); may be negative
     * @param type  call or put
     * @return the option present value
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double bsPrice(double s, double k, double t, double sigma,
                                 double r, double q, OptionType type) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);

        if (t <= 0.0) {
            return intrinsic(s, k, type);
        }
        double dfR = Math.exp(-r * t);
        double dfQ = Math.exp(-q * t);

        if (k <= 0.0) {
            return type == OptionType.CALL ? s * dfQ : 0.0;
        }
        if (sigma <= 0.0) {
            // Deterministic world: S_T = F almost surely.
            double fwdIntrinsic = s * dfQ - k * dfR;
            return type == OptionType.CALL
                    ? Math.max(fwdIntrinsic, 0.0)
                    : Math.max(-fwdIntrinsic, 0.0);
        }
        double[] d = bsD1D2(s, k, t, sigma, r, q);
        if (type == OptionType.CALL) {
            return s * dfQ * normCdf(d[0]) - k * dfR * normCdf(d[1]);
        }
        return k * dfR * normCdf(-d[1]) - s * dfQ * normCdf(-d[0]);
    }

    /**
     * Full analytic Greeks for a European BSM option.
     *
     * <p>Regular-region formulas (call; {@code phi} = normal pdf):
     * {@code delta = e^{-qt} N(d1)}; {@code gamma = e^{-qt} phi(d1) / (s sigma
     * sqrt(t))}; {@code vega = s e^{-qt} phi(d1) sqrt(t)}; theta per calendar
     * year; {@code rho = k t e^{-rt} N(d2)}; {@code vanna = -e^{-qt} phi(d1)
     * d2 / sigma}; {@code volga = vega d1 d2 / sigma}.</p>
     *
     * <p>Degenerate limits ({@code t = 0}, {@code sigma = 0}, {@code k = 0})
     * return the pointwise limits: gamma/vega/vanna/volga are 0; delta is the
     * discounted forward-moneyness indicator ({@code e^{-qt}} if in the money,
     * else 0; the exactly-at-the-money boundary maps to the ITM branch for
     * calls); theta/rho are the derivatives of the limiting price. These
     * conventions are part of the cross-language contract.</p>
     *
     * @param s     spot (&gt; 0)
     * @param k     strike (&gt;= 0)
     * @param t     time to expiry in years (&gt;= 0)
     * @param sigma annualised volatility, decimal (&gt;= 0)
     * @param r     continuously compounded risk-free rate
     * @param q     continuous dividend yield (or foreign rate)
     * @param type  call or put
     * @return the price and full Greeks
     * @throws IllegalArgumentException on invalid inputs
     */
    public static Greeks bsGreeks(double s, double k, double t, double sigma,
                                  double r, double q, OptionType type) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);

        double price = bsPrice(s, k, t, sigma, r, q, type);
        double dfR = Math.exp(-r * t);
        double dfQ = Math.exp(-q * t);

        if (t <= 0.0 || sigma <= 0.0 || k <= 0.0) {
            // Limit region: the payoff is (discounted) intrinsic on the forward.
            double fwd = s * dfQ - k * dfR; // sign of forward moneyness
            double delta;
            double theta;
            double rho;
            if (type == OptionType.CALL) {
                boolean itm = fwd >= 0.0 || k <= 0.0;
                delta = itm ? dfQ : 0.0;
                theta = itm ? q * s * dfQ - r * k * dfR : 0.0;
                rho = itm ? k * t * dfR : 0.0;
            } else {
                boolean itm = fwd < 0.0 && k > 0.0;
                delta = itm ? -dfQ : 0.0;
                theta = itm ? r * k * dfR - q * s * dfQ : 0.0;
                rho = itm ? -k * t * dfR : 0.0;
            }
            return new Greeks(price, delta, 0.0, 0.0, theta, rho, 0.0, 0.0);
        }

        double[] d = bsD1D2(s, k, t, sigma, r, q);
        double d1 = d[0];
        double d2 = d[1];
        double sqT = Math.sqrt(t);
        double pdfD1 = normPdf(d1);

        double gamma = dfQ * pdfD1 / (s * sigma * sqT);
        double vega = s * dfQ * pdfD1 * sqT;
        double vanna = -dfQ * pdfD1 * d2 / sigma;
        double volga = vega * d1 * d2 / sigma;

        double delta;
        double theta;
        double rho;
        if (type == OptionType.CALL) {
            delta = dfQ * normCdf(d1);
            theta = -s * dfQ * pdfD1 * sigma / (2.0 * sqT)
                    - r * k * dfR * normCdf(d2)
                    + q * s * dfQ * normCdf(d1);
            rho = k * t * dfR * normCdf(d2);
        } else {
            delta = dfQ * (normCdf(d1) - 1.0);
            theta = -s * dfQ * pdfD1 * sigma / (2.0 * sqT)
                    + r * k * dfR * normCdf(-d2)
                    - q * s * dfQ * normCdf(-d1);
            rho = -k * t * dfR * normCdf(-d2);
        }
        return new Greeks(price, delta, gamma, vega, theta, rho, vanna, volga);
    }

    /**
     * Garman-Kohlhagen FX option price: exactly BSM with {@code r = rd},
     * {@code q = rf}.
     *
     * <p>{@code s} and {@code k} are FX rates quoted domestic-per-foreign
     * (e.g. USD per EUR for EURUSD); the price is in domestic currency per
     * unit of foreign notional.</p>
     *
     * @param s     FX spot, domestic per foreign (&gt; 0)
     * @param k     strike, same quotation (&gt;= 0)
     * @param t     time to expiry in years (&gt;= 0)
     * @param sigma annualised volatility, decimal (&gt;= 0)
     * @param rd    domestic continuously compounded rate
     * @param rf    foreign continuously compounded rate
     * @param type  call or put
     * @return the premium in domestic currency per unit foreign
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double gkPrice(double s, double k, double t, double sigma,
                                 double rd, double rf, OptionType type) {
        return bsPrice(s, k, t, sigma, rd, rf, type);
    }

    /**
     * Garman-Kohlhagen Greeks: BSM Greeks with {@code r = rd}, {@code q = rf}.
     *
     * <p>The reported delta is the premium-excluded <b>spot delta</b>
     * {@code e^{-rf t} N(d1)}; see {@link Fx} for spot/forward
     * delta-convention conversions. {@code rho} is the sensitivity to the
     * domestic rate {@code rd}.</p>
     *
     * @param s     FX spot, domestic per foreign (&gt; 0)
     * @param k     strike (&gt;= 0)
     * @param t     time to expiry in years (&gt;= 0)
     * @param sigma annualised volatility (&gt;= 0)
     * @param rd    domestic rate
     * @param rf    foreign rate
     * @param type  call or put
     * @return the premium and full Greeks (spot-delta convention)
     * @throws IllegalArgumentException on invalid inputs
     */
    public static Greeks gkGreeks(double s, double k, double t, double sigma,
                                  double rd, double rf, OptionType type) {
        return bsGreeks(s, k, t, sigma, rd, rf, type);
    }

    /**
     * Implied Black-Scholes volatility with default tolerance {@code 1e-10}
     * and at most 100 iterations.
     *
     * @see #impliedVol(double, double, double, double, double, double, OptionType, double, int)
     */
    public static double impliedVol(double price, double s, double k, double t,
                                    double r, double q, OptionType type) {
        return impliedVol(price, s, k, t, r, q, type, 1e-10, 100);
    }

    /**
     * Implied Black-Scholes volatility via bracketed Newton with bisection
     * fallback.
     *
     * <p>Robustness strategy (identical to the Python reference):</p>
     * <ol>
     *   <li><b>No-arbitrage bounds first.</b> A European call must satisfy
     *       {@code max(s e^{-qt} - k e^{-rt}, 0) < price < s e^{-qt}}
     *       (mirrored for puts). A price at or outside these bounds has no
     *       finite implied vol, so the method throws with a message naming the
     *       violated bound rather than letting a root-finder wander.</li>
     *   <li><b>Bracketing.</b> The BSM price is strictly increasing in sigma
     *       on the open no-arb interval, so a root is bracketed in
     *       {@code [0, hi]} where {@code hi} doubles from 1.0 until the model
     *       price exceeds the target (hard cap 2000% vol).</li>
     *   <li><b>Safeguarded Newton.</b> Newton steps use analytic vega;
     *       whenever a step would leave the bracket or vega is too small
     *       (deep ITM/OTM, where vega underflows), the step falls back to
     *       bisection on the maintained bracket.</li>
     * </ol>
     *
     * <p>Converges when the price error is below {@code tol} (absolute) or the
     * bracket width is below 1e-12; round-trips {@code sigma -> price ->
     * sigma} to 1e-6 absolute for sigma in [0.05, 0.8]. {@code tol} must be
     * &gt; 0 and {@code maxIter} &gt;= 1; if neither stopping rule is met
     * within {@code maxIter} Newton/bisection steps the method throws (message
     * contains "did not converge" and the residual) — it never returns a
     * half-converged root as if it had converged.</p>
     *
     * @param price   observed option price
     * @param s       spot (&gt; 0)
     * @param k       strike (&gt; 0)
     * @param t       time to expiry in years (&gt; 0)
     * @param r       risk-free (domestic) rate
     * @param q       dividend yield (foreign rate)
     * @param type    call or put
     * @param tol     absolute price tolerance for convergence
     * @param maxIter maximum Newton/bisection iterations
     * @return the implied volatility as a decimal
     * @throws IllegalArgumentException if inputs are invalid (including
     *         {@code tol <= 0} or {@code maxIter < 1}), {@code t <= 0} (no vol
     *         is identifiable at expiry), the price violates the no-arbitrage
     *         bounds, or the iteration budget is exhausted before convergence
     */
    public static double impliedVol(double price, double s, double k, double t,
                                    double r, double q, OptionType type,
                                    double tol, int maxIter) {
        Validation.validateMarketInputs(s, k, t, 0.0, r, q);
        Validation.requireFinite("price", price);
        Validation.requirePositive("t", t);
        Validation.requirePositive("k", k);
        Validation.requirePositive("tol", tol);
        if (maxIter < 1) {
            throw new IllegalArgumentException("max_iter must be an integer >= 1, got " + maxIter);
        }

        double dfR = Math.exp(-r * t);
        double dfQ = Math.exp(-q * t);
        double lower;
        double upper;
        if (type == OptionType.CALL) {
            lower = Math.max(s * dfQ - k * dfR, 0.0);
            upper = s * dfQ;
        } else {
            lower = Math.max(k * dfR - s * dfQ, 0.0);
            upper = k * dfR;
        }
        if (price <= lower) {
            throw new IllegalArgumentException(
                    "price " + price + " violates the no-arbitrage lower bound " + lower
                            + " (discounted intrinsic); no implied vol exists");
        }
        if (price >= upper) {
            throw new IllegalArgumentException(
                    "price " + price + " violates the no-arbitrage upper bound " + upper
                            + "; no implied vol exists");
        }

        // Bracket the root: price is monotone increasing in sigma. hi doubles
        // 1 -> 2 -> 4 -> 8 -> 16 -> 20 (clipped to the cap) and the cap itself
        // is tested before giving up, so the effective cap is exactly
        // IV_MAX_SIGMA.
        double lo = 0.0; // f(lo) = discounted intrinsic - price < 0 by the bound check
        double hi = 1.0;
        while (bsPrice(s, k, t, hi, r, q, type) - price < 0.0) {
            if (hi >= IV_MAX_SIGMA) {
                throw new IllegalArgumentException(
                        "implied vol exceeds cap " + IV_MAX_SIGMA + "; price " + price
                                + " is numerically indistinguishable from the upper bound");
            }
            hi = Math.min(2.0 * hi, IV_MAX_SIGMA);
        }

        // Safeguarded Newton: maxIter Newton/bisection steps from the bracket
        // midpoint; the residual is checked before every step and once more
        // after the last one, so a returned sigma always satisfies a stopping
        // rule.
        double sigma = 0.5 * (lo + hi); // initial guess: bracket midpoint
        double diff = bsPrice(s, k, t, sigma, r, q, type) - price;
        for (int i = 0; i < maxIter; i++) {
            if (Math.abs(diff) < tol) {
                return sigma;
            }
            // Maintain the bracket around the root.
            if (diff > 0.0) {
                hi = sigma;
            } else {
                lo = sigma;
            }
            double vega = bsGreeks(s, k, t, sigma, r, q, type).vega();
            if (vega > 1e-12) {
                double candidate = sigma - diff / vega;
                if (candidate > lo && candidate < hi) {
                    sigma = candidate;
                    diff = bsPrice(s, k, t, sigma, r, q, type) - price;
                    continue;
                }
            }
            // Newton unusable (tiny vega or step outside bracket): bisect.
            sigma = 0.5 * (lo + hi);
            if (hi - lo < 1e-12) {
                return sigma;
            }
            diff = bsPrice(s, k, t, sigma, r, q, type) - price;
        }
        if (Math.abs(diff) < tol) {
            return sigma;
        }
        // Iteration budget exhausted without meeting either stopping rule:
        // report honestly rather than hand back a half-converged root.
        throw new IllegalArgumentException(
                "implied vol did not converge in " + maxIter + " iterations (|model - price| = "
                        + Math.abs(diff) + " > tol " + tol + ", bracket [" + lo + ", " + hi
                        + "]); increase max_iter or loosen tol");
    }
}
