package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Analytic module tests: put-call parity grid, Greeks vs central finite
 * differences, monotonicity, degenerate limits and input validation.
 */
public class BlackScholesTest {

    private static final double R = 0.05;
    private static final double Q = 0.02;

    @Test
    public void putCallParityGrid() {
        // C - P = s e^{-qt} - k e^{-rt} must hold across the whole grid.
        double[] spots = {60.0, 90.0, 100.0, 140.0};
        double[] strikes = {70.0, 100.0, 130.0};
        double[] times = {0.05, 0.5, 2.0};
        for (double s : spots) {
            for (double k : strikes) {
                for (double t : times) {
                    double call = BlackScholes.bsPrice(s, k, t, 0.25, R, Q, OptionType.CALL);
                    double put = BlackScholes.bsPrice(s, k, t, 0.25, R, Q, OptionType.PUT);
                    double parity = s * Math.exp(-Q * t) - k * Math.exp(-R * t);
                    assertEquals("parity s=" + s + " k=" + k + " t=" + t,
                            parity, call - put, 1e-10);
                }
            }
        }
    }

    @Test
    public void greeksMatchFiniteDifferences() {
        double s = 105.0;
        double k = 100.0;
        double t = 0.75;
        double sigma = 0.22;
        for (OptionType type : OptionType.values()) {
            Greeks g = BlackScholes.bsGreeks(s, k, t, sigma, R, Q, type);

            double hs = s * 1e-5;
            double deltaFd = (BlackScholes.bsPrice(s + hs, k, t, sigma, R, Q, type)
                    - BlackScholes.bsPrice(s - hs, k, t, sigma, R, Q, type)) / (2.0 * hs);
            assertEquals("delta rel err " + type, 0.0,
                    Math.abs(g.delta() - deltaFd) / Math.abs(deltaFd), 1e-4);

            double hv = 1e-5;
            double vegaFd = (BlackScholes.bsPrice(s, k, t, sigma + hv, R, Q, type)
                    - BlackScholes.bsPrice(s, k, t, sigma - hv, R, Q, type)) / (2.0 * hv);
            assertEquals("vega rel err " + type, 0.0,
                    Math.abs(g.vega() - vegaFd) / Math.abs(vegaFd), 1e-4);

            double gammaFd = (BlackScholes.bsPrice(s + hs, k, t, sigma, R, Q, type)
                    - 2.0 * BlackScholes.bsPrice(s, k, t, sigma, R, Q, type)
                    + BlackScholes.bsPrice(s - hs, k, t, sigma, R, Q, type)) / (hs * hs);
            assertEquals("gamma " + type, gammaFd, g.gamma(), 1e-6);

            // theta is -dV/dT (calendar-time convention).
            double ht = 1e-6;
            double thetaFd = -(BlackScholes.bsPrice(s, k, t + ht, sigma, R, Q, type)
                    - BlackScholes.bsPrice(s, k, t - ht, sigma, R, Q, type)) / (2.0 * ht);
            assertEquals("theta " + type, thetaFd, g.theta(), 1e-5);

            double hr = 1e-6;
            double rhoFd = (BlackScholes.bsPrice(s, k, t, sigma, R + hr, Q, type)
                    - BlackScholes.bsPrice(s, k, t, sigma, R - hr, Q, type)) / (2.0 * hr);
            assertEquals("rho " + type, rhoFd, g.rho(), 1e-5);

            double vannaFd = (BlackScholes.bsGreeks(s, k, t, sigma + hv, R, Q, type).delta()
                    - BlackScholes.bsGreeks(s, k, t, sigma - hv, R, Q, type).delta())
                    / (2.0 * hv);
            assertEquals("vanna " + type, vannaFd, g.vanna(), 1e-5);

            double volgaFd = (BlackScholes.bsGreeks(s, k, t, sigma + hv, R, Q, type).vega()
                    - BlackScholes.bsGreeks(s, k, t, sigma - hv, R, Q, type).vega())
                    / (2.0 * hv);
            assertEquals("volga " + type, volgaFd, g.volga(), 1e-4);
        }
    }

    @Test
    public void callMonotonicIncreasingInSpotDecreasingInStrike() {
        double prev = -1.0;
        for (double s = 60.0; s <= 140.0; s += 5.0) {
            double c = BlackScholes.bsPrice(s, 100.0, 1.0, 0.2, R, Q, OptionType.CALL);
            assertTrue("call increasing in s at s=" + s, c > prev);
            prev = c;
        }
        prev = Double.POSITIVE_INFINITY;
        for (double k = 60.0; k <= 140.0; k += 5.0) {
            double c = BlackScholes.bsPrice(100.0, k, 1.0, 0.2, R, Q, OptionType.CALL);
            assertTrue("call decreasing in k at k=" + k, c < prev);
            prev = c;
        }
    }

    @Test
    public void degenerateLimitsAreExactIntrinsic() {
        // t = 0: undiscounted intrinsic.
        assertEquals(10.0, BlackScholes.bsPrice(110.0, 100.0, 0.0, 0.2, R, Q,
                OptionType.CALL), 0.0);
        assertEquals(15.0, BlackScholes.bsPrice(85.0, 100.0, 0.0, 0.2, R, Q,
                OptionType.PUT), 0.0);
        // sigma = 0: discounted forward intrinsic.
        double t = 2.0;
        double expected = Math.max(100.0 * Math.exp(-Q * t) - 90.0 * Math.exp(-R * t), 0.0);
        assertEquals(expected, BlackScholes.bsPrice(100.0, 90.0, t, 0.0, R, Q,
                OptionType.CALL), 1e-14);
        assertEquals(0.0, BlackScholes.bsPrice(100.0, 90.0, t, 0.0, R, Q,
                OptionType.PUT), 1e-14);
        // k = 0: call is a forward claim, put is worthless.
        assertEquals(100.0 * Math.exp(-Q * t),
                BlackScholes.bsPrice(100.0, 0.0, t, 0.2, R, Q, OptionType.CALL), 1e-14);
        assertEquals(0.0, BlackScholes.bsPrice(100.0, 0.0, t, 0.2, R, Q, OptionType.PUT), 0.0);
        // Degenerate Greeks: second-order Greeks vanish, delta is the
        // discounted moneyness indicator.
        Greeks g = BlackScholes.bsGreeks(100.0, 90.0, t, 0.0, R, Q, OptionType.CALL);
        assertEquals(Math.exp(-Q * t), g.delta(), 1e-14);
        assertEquals(0.0, g.gamma(), 0.0);
        assertEquals(0.0, g.vega(), 0.0);
        assertEquals(0.0, g.vanna(), 0.0);
        assertEquals(0.0, g.volga(), 0.0);
    }

    @Test
    public void gkIsBsWithForeignRateAsYield() {
        double gk = BlackScholes.gkPrice(1.25, 1.20, 0.5, 0.11, 0.04, 0.01, OptionType.PUT);
        double bs = BlackScholes.bsPrice(1.25, 1.20, 0.5, 0.11, 0.04, 0.01, OptionType.PUT);
        assertEquals(bs, gk, 0.0);
        Greeks g = BlackScholes.gkGreeks(1.25, 1.20, 0.5, 0.11, 0.04, 0.01, OptionType.PUT);
        assertEquals(BlackScholes.bsGreeks(1.25, 1.20, 0.5, 0.11, 0.04, 0.01,
                OptionType.PUT).delta(), g.delta(), 0.0);
    }

    @Test
    public void invalidInputsThrow() {
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(-1.0, 100.0, 1.0, 0.2, R, Q, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(0.0, 100.0, 1.0, 0.2, R, Q, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100.0, -5.0, 1.0, 0.2, R, Q, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100.0, 100.0, -0.1, 0.2, R, Q, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100.0, 100.0, 1.0, -0.2, R, Q, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100.0, 100.0, 1.0, 0.2, Double.NaN, Q,
                        OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(100.0, 100.0, 1.0, 0.2, R,
                        Double.POSITIVE_INFINITY, OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.bsPrice(Double.NaN, 100.0, 1.0, 0.2, R, Q, OptionType.PUT));
        assertThrows(IllegalArgumentException.class, () -> OptionType.parse("banana"));
        assertThrows(IllegalArgumentException.class, () -> OptionType.parse(null));
        assertThrows(IllegalArgumentException.class, () -> ExerciseStyle.parse("bermudan"));
        assertThrows(IllegalArgumentException.class, () -> BinomialMethod.parse("trinomial"));
    }

    @Test
    public void optionTypeParsingIsCaseInsensitive() {
        assertEquals(OptionType.CALL, OptionType.parse("Call"));
        assertEquals(OptionType.PUT, OptionType.parse("PUT"));
        assertEquals(ExerciseStyle.AMERICAN, ExerciseStyle.parse("American"));
        assertEquals(BinomialMethod.JR, BinomialMethod.parse("JR"));
    }

    @Test
    public void deepTailPricesStayPositiveAndTiny() {
        // Very deep OTM: erfc-based CDF must keep tail precision (no negative
        // or zero price from cancellation).
        double p = BlackScholes.bsPrice(100.0, 300.0, 0.25, 0.15, R, 0.0, OptionType.CALL);
        assertTrue("deep OTM call must be > 0", p > 0.0);
        assertTrue("deep OTM call must be tiny", p < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Contract-edge tests added after review (see docs/ARCHITECTURE.md section 6).
    // -----------------------------------------------------------------------

    @Test
    public void greeksAtExpiryTieBreakContract() {
        // API_SPEC section 3.2: exactly at the money in the degenerate region
        // the call takes the ITM branch and the put the OTM branch.
        double s = 100.0;
        double k = 100.0;
        double r = 0.05;
        double q = 0.02;
        Greeks c = BlackScholes.bsGreeks(s, k, 0.0, 0.2, r, q, OptionType.CALL);
        assertEquals(0.0, c.price(), 0.0);
        assertEquals(1.0, c.delta(), 0.0);
        assertEquals(q * s - r * k, c.theta(), 1e-15);
        assertEquals(0.0, c.rho(), 0.0);
        assertEquals(0.0, c.gamma(), 0.0);
        assertEquals(0.0, c.vega(), 0.0);
        Greeks p = BlackScholes.bsGreeks(s, k, 0.0, 0.2, r, q, OptionType.PUT);
        assertEquals(new Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), p);
        // sigma = 0 with the forward exactly at the strike (r = q, s = k makes
        // s e^{-qt} - k e^{-rt} exactly zero in floating point).
        double t = 1.0;
        double rate = 0.03;
        double df = Math.exp(-rate * t);
        Greeks c0 = BlackScholes.bsGreeks(s, k, t, 0.0, rate, rate, OptionType.CALL);
        assertEquals(0.0, c0.price(), 0.0);
        assertEquals(df, c0.delta(), 1e-15);
        assertEquals(0.0, c0.theta(), 1e-15); // q s e^{-qt} - r k e^{-rt} = 0 here
        assertEquals(k * t * df, c0.rho(), 1e-12);
        Greeks p0 = BlackScholes.bsGreeks(s, k, t, 0.0, rate, rate, OptionType.PUT);
        assertEquals(new Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), p0);
        assertEquals(1.0, Math.copySign(1.0, p0.price()), 0.0); // +0.0, never -0.0
    }

    @Test
    public void nonFiniteRatesRejectedEverywhere() {
        for (double bad : new double[]{Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY}) {
            IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                    () -> BlackScholes.bsPrice(100.0, 100.0, 1.0, 0.2, bad, 0.0, OptionType.CALL));
            assertTrue(exc.getMessage(), exc.getMessage().startsWith("r must"));
            exc = assertThrows(IllegalArgumentException.class,
                    () -> BlackScholes.bsGreeks(100.0, 100.0, 1.0, 0.2, 0.05, bad, OptionType.PUT));
            assertTrue(exc.getMessage(), exc.getMessage().startsWith("q must"));
            assertThrows(IllegalArgumentException.class,
                    () -> BlackScholes.gkPrice(1.1, 1.1, 1.0, 0.1, bad, 0.0, OptionType.CALL));
        }
    }
}
