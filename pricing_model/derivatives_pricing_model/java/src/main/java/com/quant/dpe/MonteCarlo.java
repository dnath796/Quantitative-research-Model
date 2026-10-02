package com.quant.dpe;

import java.util.SplittableRandom;

/**
 * Monte Carlo engine: exact-step GBM simulation, exotics, variance reduction.
 *
 * <p><b>Simulation scheme.</b> GBM has the exact solution
 * {@code S_{t+h} = S_t exp((r - q - sigma^2/2) h + sigma sqrt(h) Z)}, so
 * stepping with this recursion is exact in distribution at the grid dates —
 * there is no Euler discretisation error in the marginals. Path functionals
 * that look <i>between</i> grid dates (barrier crossings, lookback extrema)
 * still carry monitoring bias; see the individual pricers.</p>
 *
 * <p><b>Variance reduction.</b></p>
 * <ul>
 *   <li><i>Antithetic variates</i>: each normal draw {@code Z} is paired with
 *       {@code -Z}. Because the payoffs here are monotone in {@code Z}, the
 *       pair averages are negatively correlated and the estimator variance
 *       drops. Statistically each pair average is one i.i.d. sample, so
 *       standard errors are computed over pair averages (nPaths/2 samples),
 *       never over raw correlated paths.</li>
 *   <li><i>Control variates</i>: for a payoff {@code Y} and control {@code X}
 *       with known mean, the adjusted estimator {@code Y - beta (X - E[X])}
 *       with {@code beta = Cov(Y, X) / Var(X)} (estimated in-sample) has
 *       variance reduced by the squared correlation. Controls: European
 *       vanilla uses the discounted terminal spot {@code e^{-rt} S_T} (known
 *       mean {@code s e^{-qt}}, the k = 0 Black-Scholes value); the
 *       arithmetic Asian uses the geometric-Asian payoff, whose
 *       discrete-fixing price is known in closed form
 *       ({@link #geometricAsianPrice}) and whose correlation with the
 *       arithmetic payoff is typically above 99%.</li>
 * </ul>
 *
 * <p>Every pricer takes a {@code seed} in {@code [0, 2^63 - 1]} (the domain
 * pinned by API_SPEC section 5 for every port; negative seeds throw) and is
 * fully deterministic given it (Java {@link SplittableRandom} Gaussians, whose
 * algorithm is specified since JDK 17). Per the cross-language contract, RNG
 * streams are language-specific: identical inputs and seed give identical
 * results within Java, not across ports — the MC golden tolerances are several
 * standard errors wide to absorb that.</p>
 *
 * <p>Memory: this port materialises the {@code nPaths x nSteps} normal matrix
 * and the {@code nPaths x (nSteps + 1)} path matrix (about 16 bytes per
 * path-step, e.g. ~160 MB for 100 000 paths x 100 steps); the C++ and Rust
 * ports stream paths in O(nSteps) memory instead.</p>
 */
public final class MonteCarlo {

    private MonteCarlo() {
    }

    /** Two-sided 95% normal quantile (fixed by the cross-language contract). */
    static final double Z95 = 1.959963984540054;

    private static void validateMc(int nPaths, long seed, boolean antithetic, int nSteps) {
        if (nPaths < 2) {
            throw new IllegalArgumentException("n_paths must be an integer >= 2, got " + nPaths);
        }
        if (antithetic && nPaths % 2 != 0) {
            throw new IllegalArgumentException(
                    "n_paths must be even with antithetic=true, got " + nPaths);
        }
        if (antithetic && nPaths < 4) {
            throw new IllegalArgumentException(
                    "n_paths must be >= 4 with antithetic=true, got " + nPaths);
        }
        Validation.requireSeed(seed);
        if (nSteps < 1) {
            throw new IllegalArgumentException("n_steps must be an integer >= 1, got " + nSteps);
        }
    }

    /**
     * Draws the {@code nPaths x nSteps} normal increments, antithetic-aware.
     *
     * <p>With antithetics the second half of the paths is the negation of the
     * first half, so path {@code i} and path {@code i + nPaths/2} form a pair.</p>
     */
    private static double[][] normals(int nPaths, int nSteps, long seed, boolean antithetic) {
        SplittableRandom rng = new SplittableRandom(seed);
        double[][] z = new double[nPaths][nSteps];
        int nDraw = antithetic ? nPaths / 2 : nPaths;
        for (int i = 0; i < nDraw; i++) {
            for (int j = 0; j < nSteps; j++) {
                z[i][j] = rng.nextGaussian();
            }
        }
        if (antithetic) {
            int half = nPaths / 2;
            for (int i = 0; i < half; i++) {
                for (int j = 0; j < nSteps; j++) {
                    z[half + i][j] = -z[i][j];
                }
            }
        }
        return z;
    }

    /** Collapses antithetic pairs into i.i.d. pair averages for statistics. */
    private static double[] pairAverage(double[] samples, boolean antithetic) {
        if (!antithetic) {
            return samples;
        }
        int half = samples.length / 2;
        double[] out = new double[half];
        for (int i = 0; i < half; i++) {
            out[i] = 0.5 * (samples[i] + samples[half + i]);
        }
        return out;
    }

    private static MCResult stats(double[] samples, int nPaths) {
        int n = samples.length;
        double mean = 0.0;
        for (double x : samples) {
            mean += x;
        }
        mean /= n;
        double ss = 0.0;
        for (double x : samples) {
            double d = x - mean;
            ss += d * d;
        }
        double se = Math.sqrt(ss / (n - 1) / n); // sample std (ddof=1) / sqrt(n)
        return new MCResult(mean, se, mean - Z95 * se, mean + Z95 * se, nPaths);
    }

    /** Applies the optimal-beta control-variate adjustment {@code y - b (x - E[x])}. */
    private static double[] controlAdjust(double[] y, double[] x, double xMean) {
        int n = y.length;
        double my = 0.0;
        double mx = 0.0;
        for (int i = 0; i < n; i++) {
            my += y[i];
            mx += x[i];
        }
        my /= n;
        mx /= n;
        double varX = 0.0;
        double cov = 0.0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx;
            varX += dx * dx;
            cov += (y[i] - my) * dx;
        }
        varX /= (n - 1);
        cov /= (n - 1);
        if (varX <= 0.0) {
            return y; // degenerate control (e.g. sigma = 0): no adjustment possible
        }
        double beta = cov / varX;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = y[i] - beta * (x[i] - xMean);
        }
        return out;
    }

    private static double vanillaPayoff(double st, double k, OptionType type) {
        return type == OptionType.CALL ? Math.max(st - k, 0.0) : Math.max(k - st, 0.0);
    }

    /**
     * Simulates GBM paths with the exact log-Euler step.
     *
     * @param s          spot (&gt; 0)
     * @param t          horizon in years (&gt; 0)
     * @param sigma      annualised volatility (&gt;= 0)
     * @param r          risk-free rate
     * @param q          dividend yield
     * @param nPaths     number of paths, &gt;= 2 (&gt;= 4 and even if antithetic)
     * @param nSteps     number of time steps, &gt;= 1
     * @param seed       RNG seed in {@code [0, 2^63 - 1]} (deterministic given the seed)
     * @param antithetic if true, paths {@code [nPaths/2:]} use the negated
     *                   increments of paths {@code [:nPaths/2]}
     * @return matrix of shape {@code [nPaths][nSteps + 1]}: column 0 is the
     *         spot {@code s}, column i holds {@code S(i * t / nSteps)}
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double[][] simulateGbmPaths(double s, double t, double sigma,
                                              double r, double q, int nPaths, int nSteps,
                                              long seed, boolean antithetic) {
        Validation.validateMarketInputs(s, 1.0, t, sigma, r, q);
        Validation.requirePositive("t", t);
        validateMc(nPaths, seed, antithetic, nSteps);

        double dt = t / nSteps;
        double drift = (r - q - 0.5 * sigma * sigma) * dt;
        double vol = sigma * Math.sqrt(dt);
        double[][] z = normals(nPaths, nSteps, seed, antithetic);
        double[][] paths = new double[nPaths][nSteps + 1];
        for (int i = 0; i < nPaths; i++) {
            paths[i][0] = s;
            double logPath = 0.0;
            for (int j = 0; j < nSteps; j++) {
                logPath += drift + vol * z[i][j];
                paths[i][j + 1] = s * Math.exp(logPath);
            }
        }
        return paths;
    }

    /**
     * Closed-form price of a discretely monitored geometric-average Asian.
     *
     * <p>The geometric mean {@code G = (prod_{i=1..n} S(t_i))^{1/n}} of GBM
     * sampled at equally spaced dates {@code t_i = i t / n} (the spot at 0 is
     * <i>not</i> a fixing) is lognormal with
     * {@code E[ln G] = ln s + (r - q - sigma^2/2) t (n+1)/(2n)} and
     * {@code Var[ln G] = sigma^2 t (n+1)(2n+1)/(6 n^2)} (via
     * {@code sum min(i,j) = n(n+1)(2n+1)/6}); pricing is then Black's formula
     * on the lognormal {@code G}. Used as the analytic control variate for the
     * arithmetic Asian and as a deterministic golden value.</p>
     *
     * @param s        spot (&gt; 0)
     * @param k        strike (&gt; 0)
     * @param t        time to expiry in years (&gt; 0)
     * @param sigma    annualised volatility (&gt;= 0); {@code sigma = 0}
     *                 returns the discounted deterministic-average intrinsic
     * @param r        risk-free rate
     * @param q        dividend yield
     * @param type     call or put
     * @param nFixings number of averaging dates, &gt;= 1
     * @return the closed-form price
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double geometricAsianPrice(double s, double k, double t, double sigma,
                                             double r, double q, OptionType type,
                                             int nFixings) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        Validation.requirePositive("k", k);
        if (nFixings < 1) {
            throw new IllegalArgumentException(
                    "n_fixings must be an integer >= 1, got " + nFixings);
        }
        double n = nFixings;
        double meanLn = Math.log(s) + (r - q - 0.5 * sigma * sigma) * t * (n + 1.0) / (2.0 * n);
        double varLn = sigma * sigma * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
        double df = Math.exp(-r * t);
        double eg = Math.exp(meanLn + 0.5 * varLn);
        if (varLn <= 0.0) {
            double payoff = type == OptionType.CALL
                    ? Math.max(eg - k, 0.0) : Math.max(k - eg, 0.0);
            return df * payoff;
        }
        double sd = Math.sqrt(varLn);
        double d1 = (Math.log(eg / k) + 0.5 * varLn) / sd;
        double d2 = d1 - sd;
        if (type == OptionType.CALL) {
            return df * (eg * BlackScholes.normCdf(d1) - k * BlackScholes.normCdf(d2));
        }
        return df * (k * BlackScholes.normCdf(-d2) - eg * BlackScholes.normCdf(-d1));
    }

    /**
     * European vanilla by Monte Carlo (single exact step to expiry).
     *
     * <p>The terminal spot needs only one exact GBM step, so no time grid is
     * simulated. Optional variance reduction: antithetic pairs and the
     * discounted-terminal-spot control variate (known mean {@code s e^{-qt}},
     * the Black-Scholes k = 0 analytic value).</p>
     *
     * @param s              spot (&gt; 0)
     * @param k              strike (&gt;= 0)
     * @param t              time to expiry in years (&gt; 0)
     * @param sigma          annualised volatility (&gt;= 0)
     * @param r              risk-free rate
     * @param q              dividend yield
     * @param type           call or put
     * @param nPaths         raw path count
     * @param seed           RNG seed
     * @param antithetic     pair each draw with its negation
     * @param controlVariate use the discounted terminal spot as control
     * @return the estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcEuropean(double s, double k, double t, double sigma,
                                      double r, double q, OptionType type,
                                      int nPaths, long seed, boolean antithetic,
                                      boolean controlVariate) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        validateMc(nPaths, seed, antithetic, 1);

        double[][] z = normals(nPaths, 1, seed, antithetic);
        double df = Math.exp(-r * t);
        double drift = (r - q - 0.5 * sigma * sigma) * t;
        double vol = sigma * Math.sqrt(t);
        double[] payoff = new double[nPaths];
        double[] discSt = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double st = s * Math.exp(drift + vol * z[i][0]);
            payoff[i] = df * vanillaPayoff(st, k, type);
            discSt[i] = df * st;
        }
        double[] y = pairAverage(payoff, antithetic);
        if (controlVariate) {
            double[] x = pairAverage(discSt, antithetic);
            y = controlAdjust(y, x, s * Math.exp(-q * t));
        }
        return stats(y, nPaths);
    }

    /**
     * Arithmetic-average Asian option by Monte Carlo.
     *
     * <p>The average is taken over the {@code nSteps} grid dates
     * {@code t_i = i t / n}, {@code i = 1..n} (the spot at {@code t = 0} is
     * <i>not</i> a fixing). With {@code controlVariate = true} the
     * geometric-average Asian on the same fixing dates is the control, with
     * its exact price from {@link #geometricAsianPrice}; correlation with the
     * arithmetic payoff is typically above 99%, cutting the standard error by
     * an order of magnitude.</p>
     *
     * @param s              spot (&gt; 0)
     * @param k              strike (&gt; 0)
     * @param t              time to expiry in years (&gt; 0)
     * @param sigma          annualised volatility (&gt;= 0)
     * @param r              risk-free rate
     * @param q              dividend yield
     * @param type           call or put
     * @param nPaths         raw path count
     * @param nSteps         number of fixings, &gt;= 1
     * @param seed           RNG seed
     * @param antithetic     pair each draw with its negation
     * @param controlVariate use the geometric-Asian control variate
     * @return the estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcAsianArithmetic(double s, double k, double t, double sigma,
                                             double r, double q, OptionType type,
                                             int nPaths, int nSteps, long seed,
                                             boolean antithetic, boolean controlVariate) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        Validation.requirePositive("k", k);
        validateMc(nPaths, seed, antithetic, nSteps);

        double[][] paths = simulateGbmPaths(s, t, sigma, r, q, nPaths, nSteps, seed, antithetic);
        double df = Math.exp(-r * t);
        double[] payoff = new double[nPaths];
        double[] geoPayoff = controlVariate ? new double[nPaths] : null;
        for (int i = 0; i < nPaths; i++) {
            double sum = 0.0;
            double logSum = 0.0;
            for (int j = 1; j <= nSteps; j++) {
                sum += paths[i][j];
                if (controlVariate) {
                    logSum += Math.log(paths[i][j]);
                }
            }
            payoff[i] = df * vanillaPayoff(sum / nSteps, k, type);
            if (controlVariate) {
                geoPayoff[i] = df * vanillaPayoff(Math.exp(logSum / nSteps), k, type);
            }
        }
        double[] y = pairAverage(payoff, antithetic);
        if (controlVariate) {
            double[] x = pairAverage(geoPayoff, antithetic);
            double xMean = geometricAsianPrice(s, k, t, sigma, r, q, type, nSteps);
            y = controlAdjust(y, x, xMean);
        }
        return stats(y, nPaths);
    }

    /**
     * Up-and-out barrier option, discretely monitored on the step grid.
     *
     * <p>The option knocks out (pays 0) if the simulated spot touches or
     * exceeds {@code barrier} at <b>any grid date</b> (including
     * {@code t = 0}: if {@code s >= barrier} the option is born dead and the
     * price is exactly 0, returned without simulating).</p>
     *
     * <p><b>Discretisation bias.</b> A continuously monitored barrier can be
     * breached <i>between</i> grid dates, which discrete monitoring never
     * sees, so this estimator systematically <b>over-prices</b> an up-and-out
     * relative to the continuous contract; the bias shrinks like
     * O(1/sqrt(nSteps)). The Broadie-Glasserman-Kou continuity correction
     * (shift the barrier to {@code B exp(0.5826 sigma sqrt(dt))} for an up
     * barrier) removes the leading bias term if a continuous contract is
     * intended; this method prices the <i>discrete</i> contract as specified
     * and applies no correction.</p>
     *
     * @param s          spot (&gt; 0)
     * @param k          strike (&gt;= 0)
     * @param t          time to expiry in years (&gt; 0)
     * @param sigma      annualised volatility (&gt;= 0)
     * @param r          risk-free rate
     * @param q          dividend yield
     * @param type       call or put
     * @param barrier    knock-out level (&gt; 0)
     * @param nPaths     raw path count
     * @param nSteps     monitoring dates, &gt;= 1
     * @param seed       RNG seed
     * @param antithetic pair each draw with its negation
     * @return the estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcBarrierUpOut(double s, double k, double t, double sigma,
                                          double r, double q, OptionType type,
                                          double barrier, int nPaths, int nSteps,
                                          long seed, boolean antithetic) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        Validation.requirePositive("barrier", barrier);
        validateMc(nPaths, seed, antithetic, nSteps);

        if (s >= barrier) {
            return new MCResult(0.0, 0.0, 0.0, 0.0, nPaths);
        }
        double[][] paths = simulateGbmPaths(s, t, sigma, r, q, nPaths, nSteps, seed, antithetic);
        double df = Math.exp(-r * t);
        double[] payoff = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double max = 0.0;
            for (int j = 0; j <= nSteps; j++) {
                if (paths[i][j] > max) {
                    max = paths[i][j];
                }
            }
            boolean alive = max < barrier;
            payoff[i] = alive ? df * vanillaPayoff(paths[i][nSteps], k, type) : 0.0;
        }
        return stats(pairAverage(payoff, antithetic), nPaths);
    }

    /**
     * Floating-strike lookback option, discretely monitored on the step grid.
     *
     * <p>Payoffs (extrema over the grid dates, {@code t = 0} included):
     * call pays {@code S_T - min_i S(t_i)} (buy at the observed low); put pays
     * {@code max_i S(t_i) - S_T} (sell at the observed high). Both are
     * non-negative by construction. Discrete monitoring <i>under</i>-states
     * the true continuous extremum, so this under-prices the continuously
     * monitored contract with O(1/sqrt(nSteps)) bias — the mirror image of
     * the barrier case. There is no strike parameter.</p>
     *
     * @param s          spot (&gt; 0)
     * @param t          time to expiry in years (&gt; 0)
     * @param sigma      annualised volatility (&gt;= 0)
     * @param r          risk-free rate
     * @param q          dividend yield
     * @param type       call or put
     * @param nPaths     raw path count
     * @param nSteps     monitoring dates, &gt;= 1
     * @param seed       RNG seed
     * @param antithetic pair each draw with its negation
     * @return the estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcLookbackFloating(double s, double t, double sigma,
                                              double r, double q, OptionType type,
                                              int nPaths, int nSteps, long seed,
                                              boolean antithetic) {
        Validation.validateMarketInputs(s, 1.0, t, sigma, r, q);
        Validation.requirePositive("t", t);
        validateMc(nPaths, seed, antithetic, nSteps);

        double[][] paths = simulateGbmPaths(s, t, sigma, r, q, nPaths, nSteps, seed, antithetic);
        double df = Math.exp(-r * t);
        double[] payoff = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j <= nSteps; j++) {
                double v = paths[i][j];
                if (v < min) {
                    min = v;
                }
                if (v > max) {
                    max = v;
                }
            }
            double terminal = paths[i][nSteps];
            payoff[i] = type == OptionType.CALL
                    ? df * (terminal - min)
                    : df * (max - terminal);
        }
        return stats(pairAverage(payoff, antithetic), nPaths);
    }

    /**
     * Pathwise-derivative delta of a European vanilla.
     *
     * <p>Differentiating the payoff along each path (valid because the
     * vanilla payoff is Lipschitz and {@code dS_T/dS_0 = S_T / S_0} for GBM):
     * call delta {@code = e^{-rt} E[1{S_T > K} S_T / S_0]}; put delta
     * {@code = -e^{-rt} E[1{S_T < K} S_T / S_0]}. Unbiased and typically far
     * lower-variance than finite differences, but it cannot handle
     * discontinuous payoffs (e.g. digitals), where the likelihood-ratio
     * method would be needed instead.</p>
     *
     * @param s          spot (&gt; 0)
     * @param k          strike (&gt;= 0)
     * @param t          time to expiry in years (&gt; 0)
     * @param sigma      annualised volatility (&gt;= 0)
     * @param r          risk-free rate
     * @param q          dividend yield
     * @param type       call or put
     * @param nPaths     raw path count
     * @param seed       RNG seed
     * @param antithetic pair each draw with its negation
     * @return the delta estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcDeltaPathwise(double s, double k, double t, double sigma,
                                           double r, double q, OptionType type,
                                           int nPaths, long seed, boolean antithetic) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        validateMc(nPaths, seed, antithetic, 1);

        double[][] z = normals(nPaths, 1, seed, antithetic);
        double df = Math.exp(-r * t);
        double drift = (r - q - 0.5 * sigma * sigma) * t;
        double vol = sigma * Math.sqrt(t);
        double[] samples = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double st = s * Math.exp(drift + vol * z[i][0]);
            if (type == OptionType.CALL) {
                samples[i] = st > k ? df * st / s : 0.0;
            } else {
                samples[i] = st < k ? -df * st / s : 0.0;
            }
        }
        return stats(pairAverage(samples, antithetic), nPaths);
    }

    /**
     * Central finite-difference delta with common random numbers (CRN).
     *
     * <p>The same normal draws (same seed) price the payoff at
     * {@code s (1 -+ relBump)} and the delta is the per-path central
     * difference {@code (payoff_up - payoff_down) / (2 s relBump)},
     * discounted. Reusing the randomness makes the difference of the two
     * estimators nearly noiseless (their sampling errors cancel path by
     * path); without CRN the variance would explode as O(1/h^2). The standard
     * error reported is that of the per-path difference quotient.</p>
     *
     * @param s          spot (&gt; 0)
     * @param k          strike (&gt;= 0)
     * @param t          time to expiry in years (&gt; 0)
     * @param sigma      annualised volatility (&gt;= 0)
     * @param r          risk-free rate
     * @param q          dividend yield
     * @param type       call or put
     * @param nPaths     raw path count
     * @param seed       RNG seed
     * @param relBump    relative spot bump in the open interval (0, 1) (e.g. 1e-4)
     * @param antithetic pair each draw with its negation
     * @return the delta estimate with standard error and 95% CI
     * @throws IllegalArgumentException on invalid inputs
     */
    public static MCResult mcDeltaFdCrn(double s, double k, double t, double sigma,
                                        double r, double q, OptionType type,
                                        int nPaths, long seed, double relBump,
                                        boolean antithetic) {
        Validation.validateMarketInputs(s, k, t, sigma, r, q);
        Validation.requirePositive("t", t);
        Validation.requireRelBump(relBump);
        validateMc(nPaths, seed, antithetic, 1);

        double[][] z = normals(nPaths, 1, seed, antithetic);
        double df = Math.exp(-r * t);
        double drift = (r - q - 0.5 * sigma * sigma) * t;
        double vol = sigma * Math.sqrt(t);
        double h = s * relBump;
        double[] samples = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double growth = Math.exp(drift + vol * z[i][0]);
            double payUp = df * vanillaPayoff((s + h) * growth, k, type);
            double payDn = df * vanillaPayoff((s - h) * growth, k, type);
            samples[i] = (payUp - payDn) / (2.0 * h);
        }
        return stats(pairAverage(samples, antithetic), nPaths);
    }
}
