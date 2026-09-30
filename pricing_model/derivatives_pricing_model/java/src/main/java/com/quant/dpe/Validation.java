package com.quant.dpe;

/**
 * Shared input validation for the dpe package.
 *
 * <p>All public pricing entry points funnel their inputs through these helpers
 * so that error behaviour is uniform: bad input always throws
 * {@link IllegalArgumentException} with a message naming the offending
 * parameter (the Java mapping of the cross-language error contract).</p>
 */
final class Validation {

    private Validation() {
    }

    /**
     * Throws unless {@code value} is finite (rejects NaN and infinities).
     *
     * <p>NaN/inf guards sit at the boundary so that no NaN can silently
     * propagate through a pricing formula and surface as a nonsense Greek
     * downstream.</p>
     */
    static void requireFinite(String name, double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite, got " + value);
        }
    }

    /** Throws unless {@code value} is finite and strictly positive. */
    static void requirePositive(String name, double value) {
        requireFinite(name, value);
        if (value <= 0.0) {
            throw new IllegalArgumentException(name + " must be > 0, got " + value);
        }
    }

    /** Throws unless {@code value} is finite and non-negative. */
    static void requireNonNegative(String name, double value) {
        requireFinite(name, value);
        if (value < 0.0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + value);
        }
    }

    /**
     * Largest admissible RNG seed (2^63 - 1, i.e. {@link Long#MAX_VALUE}). The
     * seed domain {@code [0, SEED_MAX]} is pinned by the cross-language
     * contract (API_SPEC section 5) so a seed means the same thing in Python,
     * C++, Rust and Java and is never silently wrapped.
     */
    static final long SEED_MAX = Long.MAX_VALUE;

    /** Throws unless {@code 0 <= seed <= SEED_MAX}. */
    static void requireSeed(long seed) {
        if (seed < 0L) {
            throw new IllegalArgumentException(
                    "seed must be an integer in [0, " + SEED_MAX + "], got " + seed);
        }
    }

    /**
     * Throws unless {@code relBump} is finite and in the open interval (0, 1):
     * {@code relBump >= 1} would price the down bump at a non-positive spot
     * and return a meaningless delta without any error (API_SPEC section 5.8).
     */
    static void requireRelBump(double relBump) {
        requirePositive("rel_bump", relBump);
        if (relBump >= 1.0) {
            throw new IllegalArgumentException("rel_bump must be in (0, 1), got " + relBump);
        }
    }

    /**
     * Validates the common Black-Scholes market inputs.
     *
     * <p>Rules (identical in every language port): {@code s > 0};
     * {@code k >= 0} ({@code k = 0} is a forward claim); {@code t >= 0} in
     * years; {@code sigma >= 0} as a decimal; {@code r}, {@code q} finite
     * (negative rates are in scope).</p>
     */
    static void validateMarketInputs(double s, double k, double t, double sigma,
                                     double r, double q) {
        requirePositive("s", s);
        requireNonNegative("k", k);
        requireNonNegative("t", t);
        requireNonNegative("sigma", sigma);
        requireFinite("r", r);
        requireFinite("q", q);
    }
}
