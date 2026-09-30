package com.quant.dpe;

/**
 * Binomial-tree pricing: Cox-Ross-Rubinstein and Jarrow-Rudd lattices.
 *
 * <p>Both lattices discretise GBM over {@code n} steps of length
 * {@code dt = t / n}:</p>
 * <ul>
 *   <li><b>CRR</b>: {@code u = exp(sigma sqrt(dt))}, {@code d = 1/u}; the
 *       risk-neutral up probability {@code p = (exp((r - q) dt) - d)/(u - d)}
 *       matches the first moment of GBM; the tree recombines around the spot,
 *       which is what makes tree-based delta, gamma and theta natural.</li>
 *   <li><b>JR</b> (equal-probability): with {@code nu = (r - q - sigma^2/2) dt},
 *       {@code u = exp(nu + sigma sqrt(dt))}, {@code d = exp(nu - sigma sqrt(dt))},
 *       {@code p = 1/2}; the drift is absorbed into the node placement, so the
 *       lattice drifts with the forward instead of recentering on the spot.</li>
 * </ul>
 *
 * <p>Both converge to Black-Scholes at rate O(1/n) for European payoffs, with
 * a well-known oscillation between adjacent step counts (the strike moves
 * relative to the node grid). Richardson averaging prices at {@code n} and
 * {@code n + 1} steps and averages, cancelling the leading oscillating error
 * term — two-point odd/even Richardson averaging, typically worth one to two
 * extra digits at the same cost order.</p>
 *
 * <p>Backward induction values American exercise by taking
 * {@code max(continuation, intrinsic)} at every node — the discrete dynamic
 * program for the optimal stopping problem.</p>
 *
 * <p>The {@code sigma = 0} and {@code t = 0} limits are handled without
 * building a tree (the lattice degenerates: {@code u = d} makes {@code p}
 * ill-defined). For {@code sigma = 0} the spot rides the deterministic
 * forward, so a European option is worth discounted forward intrinsic and an
 * American option is the best discounted intrinsic over the deterministic
 * path, evaluated on the step grid.</p>
 */
public final class Binomial {

    private Binomial() {
    }

    /**
     * Largest admissible magnitude of the log-spot excursion across the
     * lattice (identical in every port). {@code exp()} overflows a double at
     * ~709.78, so a terminal node with
     * {@code |ln s| + |nu n| + sigma sqrt(t n) > 700} would be infinite (or an
     * {@code inf * 0} NaN in {@code s u^j d^(n-j)}); the guard makes that an
     * error.
     */
    static final double LATTICE_LOG_LIMIT = 700.0;

    private static void validateSteps(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be an integer >= 1, got " + steps);
        }
    }

    /**
     * Rejects lattices whose terminal spots would overflow a double. The
     * extreme terminal log-spot is {@code ln s + nu n +- sigma sqrt(t n)} with
     * {@code nu n = (r - q - sigma^2/2) t} for JR (zero for CRR); bounding the
     * worst case for either method keeps the admissible domain
     * method-independent.
     */
    private static void checkLatticeRange(double s, double t, double sigma, double r,
                                          double q, int steps) {
        double nuN = Math.abs((r - q - 0.5 * sigma * sigma) * t);
        double excursion = Math.abs(Math.log(s)) + nuN + sigma * Math.sqrt(t * steps);
        if (excursion > LATTICE_LOG_LIMIT) {
            throw new IllegalArgumentException(
                    "lattice overflow: |ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) = "
                            + excursion + " exceeds " + LATTICE_LOG_LIMIT
                            + "; reduce steps, sigma or t");
        }
    }

    private static double payoff(double spot, double k, OptionType type) {
        return type == OptionType.CALL ? Math.max(spot - k, 0.0) : Math.max(k - spot, 0.0);
    }

    /**
     * Exact {@code sigma = 0} limit on the {@code steps + 1} point time grid.
     *
     * <p>The spot is deterministic: {@code S(t_i) = s exp((r - q) t_i)}. A
     * European option pays intrinsic on the terminal forward; an American
     * option is exercised at whichever grid date maximises the discounted
     * intrinsic.</p>
     */
    private static double degenerateSigmaZero(double s, double k, double t, double r,
                                              double q, OptionType type,
                                              ExerciseStyle style, int steps) {
        if (style == ExerciseStyle.AMERICAN) {
            double best = 0.0;
            for (int i = 0; i <= steps; i++) {
                double ti = t * i / steps;
                double spot = s * Math.exp((r - q) * ti);
                double v = Math.exp(-r * ti) * payoff(spot, k, type);
                if (v > best) {
                    best = v;
                }
            }
            return best;
        }
        return BlackScholes.bsPrice(s, k, t, 0.0, r, q, type); // discounted forward intrinsic
    }

    /** Returns {@code {u, d, p}} for the requested lattice at step size {@code dt}. */
    private static double[] treeParams(BinomialMethod method, double sigma, double r,
                                       double q, double dt) {
        double sq = sigma * Math.sqrt(dt);
        double u;
        double d;
        double p;
        if (method == BinomialMethod.CRR) {
            u = Math.exp(sq);
            d = 1.0 / u;
            p = (Math.exp((r - q) * dt) - d) / (u - d);
        } else { // Jarrow-Rudd
            double nu = (r - q - 0.5 * sigma * sigma) * dt;
            u = Math.exp(nu + sq);
            d = Math.exp(nu - sq);
            p = 0.5;
        }
        if (!(p > 0.0 && p < 1.0)) {
            // Happens when the per-step drift outruns the vol spacing (huge
            // (r-q)dt vs sigma sqrt(dt)); the discrete measure would not be a
            // probability, so the lattice is invalid at this step count.
            throw new IllegalArgumentException(
                    "risk-neutral probability p=" + p + " outside (0, 1); "
                            + "increase steps or use smaller drift/vol ratio");
        }
        return new double[]{u, d, p};
    }

    /**
     * Backward induction; optionally keeps the option/spot values of the first
     * {@code keepLevels} time levels (level 0 = today) for tree Greeks.
     *
     * @return {@code [rootValue, optLevels, spotLevels]} packed as arrays:
     *         {@code optLevels[i]}/{@code spotLevels[i]} hold level {@code i}
     *         (length {@code i + 1}), or {@code null} when not kept
     */
    private static double rollBack(double s, double k, double t, double r,
                                   OptionType type, ExerciseStyle style, int steps,
                                   double u, double d, double p,
                                   double[][] optLevels, double[][] spotLevels) {
        double dt = t / steps;
        double disc = Math.exp(-r * dt);
        int keepLevels = optLevels == null ? 0 : optLevels.length;

        // Terminal spots: s * u^j * d^(n-j), j = 0..n (ascending).
        double[] spots = new double[steps + 1];
        double[] values = new double[steps + 1];
        for (int j = 0; j <= steps; j++) {
            spots[j] = s * Math.pow(u, j) * Math.pow(d, steps - j);
            values[j] = payoff(spots[j], k, type);
        }
        if (steps < keepLevels) {
            // The terminal level itself is one of the requested levels (only
            // for very small trees, e.g. tree Greeks at steps = 2): store it
            // here, because the backward loop only visits levels < steps.
            optLevels[steps] = values.clone();
            spotLevels[steps] = spots.clone();
        }

        for (int i = steps - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                values[j] = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
                spots[j] = spots[j] / d; // spots at level i: s * u^j * d^(i-j)
                if (style == ExerciseStyle.AMERICAN) {
                    values[j] = Math.max(values[j], payoff(spots[j], k, type));
                }
            }
            if (i < keepLevels) {
                optLevels[i] = java.util.Arrays.copyOfRange(values, 0, i + 1);
                spotLevels[i] = java.util.Arrays.copyOfRange(spots, 0, i + 1);
            }
        }
        return values[0];
    }

    /**
     * Binomial-tree price without Richardson averaging.
     *
     * @see #binomialPrice(double, double, double, double, double, double,
     *      OptionType, ExerciseStyle, int, BinomialMethod, boolean)
     */
    public static double binomialPrice(double s, double k, double t, double sigma,
                                       double r, double q, OptionType type,
                                       ExerciseStyle style, int steps,
                                       BinomialMethod method) {
        return binomialPrice(s, k, t, sigma, r, q, type, style, steps, method, false);
    }

    /**
     * Binomial-tree price of a European or American vanilla option.
     *
     * @param s          spot (&gt; 0)
     * @param k          strike (&gt;= 0)
     * @param t          time to expiry in years (&gt;= 0)
     * @param sigma      annualised volatility, decimal (&gt;= 0)
     * @param r          continuously compounded risk-free rate
     * @param q          continuous dividend yield (foreign rate for FX)
     * @param type       call or put
     * @param style      european or american
     * @param steps      number of time steps, &gt;= 1
     * @param method     CRR or JR lattice
     * @param richardson if true, average the {@code steps} and {@code steps + 1}
     *                   prices to damp the odd/even oscillation of the
     *                   convergence to the continuous limit (two-point
     *                   Richardson averaging)
     * @return the option value at the root node
     * @throws IllegalArgumentException on invalid inputs, when the
     *         risk-neutral probability falls outside (0, 1) at this step size,
     *         or when the lattice's terminal spots would overflow a double
     *         ({@code |ln s| + |(r - q - sigma^2/2) t| + sigma sqrt(t steps) > 700})
     */
    public static double binomialPrice(double s, double k, double t, double sigma,
                                       double r, double q, OptionType type,
                                       ExerciseStyle style, int steps,
                                       BinomialMethod method, boolean richardson) {
        validateSteps(steps);
        Validation.validateMarketInputs(s, k, t, sigma, r, q);

        if (t <= 0.0) {
            return payoff(s, k, type);
        }
        if (sigma <= 0.0) {
            return degenerateSigmaZero(s, k, t, r, q, type, style, steps);
        }
        checkLatticeRange(s, t, sigma, r, q, richardson ? steps + 1 : steps);
        if (richardson) {
            return 0.5 * (priceOnce(s, k, t, sigma, r, q, type, style, steps, method)
                    + priceOnce(s, k, t, sigma, r, q, type, style, steps + 1, method));
        }
        return priceOnce(s, k, t, sigma, r, q, type, style, steps, method);
    }

    private static double priceOnce(double s, double k, double t, double sigma,
                                    double r, double q, OptionType type,
                                    ExerciseStyle style, int n, BinomialMethod method) {
        double[] udp = treeParams(method, sigma, r, q, t / n);
        return rollBack(s, k, t, r, type, style, n, udp[0], udp[1], udp[2], null, null);
    }

    /**
     * Delta, gamma and theta read directly off the lattice.
     *
     * <p>With node values {@code V(i, j)} and node spots {@code S(i, j)}
     * (level i, j up-moves):</p>
     * <ul>
     *   <li>{@code delta = (V(1,1) - V(1,0)) / (S(1,1) - S(1,0))} — the
     *       discrete hedge ratio one step ahead;</li>
     *   <li>{@code gamma} = difference of the two one-sided deltas at level 2
     *       divided by the half-spread of the outer level-2 spots;</li>
     *   <li>{@code theta = (V(2,1) - V(0,0) - dV) / (2 dt)} where
     *       {@code dV = delta (S(2,1) - s) + gamma (S(2,1) - s)^2 / 2} removes
     *       the value change caused by the middle node's spot displacement.
     *       For CRR the middle level-2 node has spot exactly {@code s}
     *       ({@code ud = 1}) so {@code dV = 0} and this is the classic pure
     *       calendar-time difference; for JR the lattice drifts
     *       ({@code S(2,1) = s e^{2 nu dt}}) and the correction strips the
     *       delta/gamma contamination out of the time difference.</li>
     * </ul>
     *
     * @param s      spot (&gt; 0)
     * @param k      strike (&gt;= 0)
     * @param t      time to expiry in years, must be &gt; 0
     * @param sigma  annualised volatility, must be &gt; 0
     * @param r      risk-free rate
     * @param q      dividend yield
     * @param type   call or put
     * @param style  european or american
     * @param steps  number of steps, &gt;= 2 (so that level 2 exists)
     * @param method CRR or JR lattice
     * @return price plus tree delta, gamma and theta (per year)
     * @throws IllegalArgumentException on invalid inputs, {@code steps < 2},
     *         {@code t <= 0} or {@code sigma <= 0} (the lattice degenerates)
     */
    public static BinomialGreeks binomialGreeks(double s, double k, double t, double sigma,
                                                double r, double q, OptionType type,
                                                ExerciseStyle style, int steps,
                                                BinomialMethod method) {
        validateSteps(steps);
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        if (steps < 2) {
            throw new IllegalArgumentException("steps must be >= 2 for tree Greeks, got " + steps);
        }
        if (t <= 0.0 || sigma <= 0.0) {
            throw new IllegalArgumentException("tree Greeks require t > 0 and sigma > 0");
        }
        checkLatticeRange(s, t, sigma, r, q, steps);

        double dt = t / steps;
        double[] udp = treeParams(method, sigma, r, q, dt);
        double[][] opt = new double[3][];
        double[][] spot = new double[3][];
        double price = rollBack(s, k, t, r, type, style, steps,
                udp[0], udp[1], udp[2], opt, spot);

        double v0 = opt[0][0];
        double[] v1 = opt[1];
        double[] s1 = spot[1];
        double[] v2 = opt[2];
        double[] s2 = spot[2];

        double delta = (v1[1] - v1[0]) / (s1[1] - s1[0]);
        double deltaUp = (v2[2] - v2[1]) / (s2[2] - s2[1]);
        double deltaDn = (v2[1] - v2[0]) / (s2[1] - s2[0]);
        double gamma = (deltaUp - deltaDn) / (0.5 * (s2[2] - s2[0]));
        // Correct for the middle node's spot displacement (zero for CRR, O(dt)
        // for the drifting JR lattice): strip delta/gamma value change so the
        // remaining difference is purely calendar time.
        double ds = s2[1] - s;
        double theta = (v2[1] - v0 - delta * ds - 0.5 * gamma * ds * ds) / (2.0 * dt);

        return new BinomialGreeks(price, delta, gamma, theta);
    }
}
