package com.quant.dpe;

/**
 * A Monte Carlo estimate together with its sampling uncertainty.
 *
 * <p>With antithetic variates the i.i.d. statistical samples are the pair
 * averages {@code (f(Z) + f(-Z)) / 2}, so {@code stdError} is the sample
 * standard deviation (ddof = 1) of the pair averages divided by the square
 * root of the number of pairs — never of the raw correlated paths.
 * {@code nPaths} always reports the raw path count (both antithetic halves).</p>
 *
 * @param value    point estimate (price or Greek)
 * @param stdError standard error of the estimate
 * @param ciLow    lower 95% bound: {@code value - 1.959963984540054 * stdError}
 * @param ciHigh   upper 95% bound: {@code value + 1.959963984540054 * stdError}
 * @param nPaths   raw simulated paths (counting both members of each antithetic pair)
 */
public record MCResult(double value, double stdError, double ciLow, double ciHigh, int nPaths) {
}
