package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Implied-vol solver tests: round-trips (including deep ITM/OTM where Newton
 * must fall back to bisection) and no-arbitrage bound rejection.
 */
public class ImpliedVolTest {

    @Test
    public void roundTripAcrossVolAndMoneyness() {
        // Property check mirroring the Python reference grid: sigma -> price
        // -> implied sigma, skipping prices numerically pinned to intrinsic
        // (no vol is identifiable there for ANY solver).
        double s = 100.0;
        double r = 0.03;
        double q = 0.01;
        double[] sigmas = {0.05, 0.2, 0.8};
        double[] strikes = {70.0, 100.0, 130.0};
        double[] times = {1.0 / 52.0, 1.0, 3.0};
        for (OptionType type : OptionType.values()) {
            for (double sigma : sigmas) {
                for (double k : strikes) {
                    for (double t : times) {
                        double price = BlackScholes.bsPrice(s, k, t, sigma, r, q, type);
                        double lower = type == OptionType.CALL
                                ? Math.max(s * Math.exp(-q * t) - k * Math.exp(-r * t), 0.0)
                                : Math.max(k * Math.exp(-r * t) - s * Math.exp(-q * t), 0.0);
                        if (price - lower < 1e-12) {
                            continue; // numerically at intrinsic: no invertible vol
                        }
                        double iv = BlackScholes.impliedVol(price, s, k, t, r, q, type);
                        assertEquals("roundtrip sigma=" + sigma + " k=" + k + " t=" + t
                                + " " + type, sigma, iv, 1e-6);
                    }
                }
            }
        }
    }

    @Test
    public void deepItmAndOtmRoundTrip() {
        // Deep ITM: tiny vega, Newton must not diverge (bisection fallback).
        double pItm = BlackScholes.bsPrice(100.0, 40.0, 0.5, 0.3, 0.05, 0.0, OptionType.CALL);
        assertEquals(0.3, BlackScholes.impliedVol(pItm, 100.0, 40.0, 0.5, 0.05, 0.0,
                OptionType.CALL), 1e-5);
        // Deep OTM.
        double pOtm = BlackScholes.bsPrice(100.0, 250.0, 0.5, 0.3, 0.05, 0.0, OptionType.CALL);
        assertEquals(0.3, BlackScholes.impliedVol(pOtm, 100.0, 250.0, 0.5, 0.05, 0.0,
                OptionType.CALL), 1e-6);
    }

    @Test
    public void priceBelowIntrinsicIsRejectedWithLowerBoundMessage() {
        // Deep ITM call: discounted intrinsic is ~ 21.5; price 1.0 is below it.
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(1.0, 100.0, 80.0, 0.5, 0.03, 0.0,
                        OptionType.CALL));
        assertTrue("message should name the lower bound: " + exc.getMessage(),
                exc.getMessage().contains("lower bound"));
    }

    @Test
    public void priceAboveUpperBoundIsRejected() {
        // A call can never be worth more than s e^{-qt}.
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(120.0, 100.0, 100.0, 1.0, 0.05, 0.0,
                        OptionType.CALL));
        assertTrue("message should name the upper bound: " + exc.getMessage(),
                exc.getMessage().contains("upper bound"));
    }

    @Test
    public void zeroPriceAndExpiryAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(0.0, 100.0, 100.0, 1.0, 0.05, 0.0,
                        OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(5.0, 100.0, 100.0, 0.0, 0.05, 0.0,
                        OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(5.0, 100.0, 0.0, 1.0, 0.05, 0.0,
                        OptionType.CALL));
        assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(Double.NaN, 100.0, 100.0, 1.0, 0.05, 0.0,
                        OptionType.CALL));
    }

    // -----------------------------------------------------------------------
    // Contract-edge tests added after review (see docs/ARCHITECTURE.md section 6).
    // -----------------------------------------------------------------------

    private static final double ATM_PRICE = 10.45058357; // bs_price(100,100,1,0.2,0.05,0,call)

    @Test
    public void impliedVolReportsNonConvergence() {
        // One Newton step from the bracket midpoint lands ~2 vol points off;
        // the solver must throw (message names the failure), never return it.
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> BlackScholes.impliedVol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0,
                        OptionType.CALL, 1e-10, 1));
        assertTrue(exc.getMessage(), exc.getMessage().contains("did not converge"));
        // A sufficient budget converges as usual.
        assertEquals(0.2, BlackScholes.impliedVol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0,
                OptionType.CALL, 1e-10, 20), 1e-8);
    }

    @Test
    public void impliedVolValidatesTolAndMaxIter() {
        for (int badIter : new int[]{0, -3}) {
            IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                    () -> BlackScholes.impliedVol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0,
                            OptionType.CALL, 1e-10, badIter));
            assertTrue(exc.getMessage(), exc.getMessage().contains("max_iter"));
        }
        for (double badTol : new double[]{0.0, -1e-10, Double.NaN}) {
            IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                    () -> BlackScholes.impliedVol(ATM_PRICE, 100.0, 100.0, 1.0, 0.05, 0.0,
                            OptionType.CALL, badTol, 100));
            assertTrue(exc.getMessage(), exc.getMessage().contains("tol"));
        }
    }

    @Test
    public void impliedVolHighVolBracketExpansion() {
        // sigma = 1.7 lies beyond the initial [0, 1] bracket: the doubling
        // expansion must engage and the round-trip still hold.
        double price = BlackScholes.bsPrice(100.0, 100.0, 1.0, 1.7, 0.02, 0.0, OptionType.CALL);
        assertEquals(1.7, BlackScholes.impliedVol(price, 100.0, 100.0, 1.0, 0.02, 0.0,
                OptionType.CALL), 1e-6);
        // Between 16 and 20 the bracket is clipped to the 20.0 cap and still solves.
        double p17 = BlackScholes.bsPrice(100.0, 100.0, 0.05, 17.0, 0.02, 0.0, OptionType.CALL);
        assertEquals(17.0, BlackScholes.impliedVol(p17, 100.0, 100.0, 0.05, 0.02, 0.0,
                OptionType.CALL), 1e-6);
    }

    @Test
    public void impliedVolTinyTolTerminatesOnBracketWidthAndIsDeterministic() {
        double p = BlackScholes.bsPrice(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, OptionType.CALL);
        assertEquals(0.2, BlackScholes.impliedVol(p, 100.0, 100.0, 1.0, 0.05, 0.0,
                OptionType.CALL, 1e-300, 200), 1e-10);
        double pp = BlackScholes.bsPrice(100.0, 90.0, 0.5, 0.3, 0.02, 0.01, OptionType.PUT);
        assertEquals(BlackScholes.impliedVol(pp, 100.0, 90.0, 0.5, 0.02, 0.01, OptionType.PUT),
                BlackScholes.impliedVol(pp, 100.0, 90.0, 0.5, 0.02, 0.01, OptionType.PUT), 0.0);
    }
}
