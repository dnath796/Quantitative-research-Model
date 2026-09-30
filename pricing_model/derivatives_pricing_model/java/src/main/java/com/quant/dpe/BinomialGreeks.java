package com.quant.dpe;

/**
 * Tree-based price and Greeks read directly off the binomial lattice.
 *
 * @param price option value at the root node
 * @param delta discrete hedge ratio from the two level-1 nodes
 * @param gamma difference of one-sided level-2 deltas over the half-spread
 * @param theta calendar-time decay per <b>year</b>, displacement-corrected
 *              (see {@link Binomial#binomialGreeks})
 */
public record BinomialGreeks(double price, double delta, double gamma, double theta) {
}
