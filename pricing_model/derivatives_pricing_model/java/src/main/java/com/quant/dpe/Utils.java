package com.quant.dpe;

/**
 * Small numerical utilities shared by the demo and tests.
 */
public final class Utils {

    private Utils() {
    }

    /**
     * Annualised close-to-close historical volatility with 252 trading
     * periods per year.
     *
     * @see #historicalVol(double[], int)
     */
    public static double historicalVol(double[] prices) {
        return historicalVol(prices, 252);
    }

    /**
     * Annualised close-to-close historical volatility from a price series.
     *
     * <p>Computes the sample standard deviation (ddof = 1) of the log returns
     * {@code ln(P_i / P_{i-1})} and scales by {@code sqrt(periodsPerYear)} —
     * the standard realised-vol estimator under the GBM assumption that log
     * returns are i.i.d. normal.</p>
     *
     * @param prices         price series, at least 3 points, all finite and &gt; 0
     * @param periodsPerYear observation periods per year (e.g. 252 for daily), &gt;= 1
     * @return annualised realised volatility as a decimal
     * @throws IllegalArgumentException on fewer than 3 prices (no meaningful
     *         ddof = 1 standard deviation from fewer than 2 returns), any
     *         non-positive or non-finite price, or {@code periodsPerYear < 1}
     */
    public static double historicalVol(double[] prices, int periodsPerYear) {
        if (prices == null || prices.length < 3) {
            int n = prices == null ? 0 : prices.length;
            throw new IllegalArgumentException(
                    "prices must be a 1-D series with >= 3 points, got " + n);
        }
        for (double p : prices) {
            if (!Double.isFinite(p) || p <= 0.0) {
                throw new IllegalArgumentException("prices must all be finite and > 0");
            }
        }
        if (periodsPerYear < 1) {
            throw new IllegalArgumentException(
                    "periods_per_year must be >= 1, got " + periodsPerYear);
        }
        int n = prices.length - 1;
        double[] logReturns = new double[n];
        double mean = 0.0;
        for (int i = 0; i < n; i++) {
            logReturns[i] = Math.log(prices[i + 1] / prices[i]);
            mean += logReturns[i];
        }
        mean /= n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) {
            double d = logReturns[i] - mean;
            ss += d * d;
        }
        double std = Math.sqrt(ss / (n - 1));
        return std * Math.sqrt(periodsPerYear);
    }
}
