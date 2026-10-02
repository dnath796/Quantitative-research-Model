package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * FX helper and utility tests: forward parity, delta-convention conversions
 * as exact mutual inverses, and the historical-vol estimator.
 */
public class FxUtilsTest {

    @Test
    public void fxForwardCoveredInterestParity() {
        assertEquals(1.10 * Math.exp((0.03 - 0.02) * 0.5),
                Fx.fxForward(1.10, 0.5, 0.03, 0.02), 1e-15);
        // Negative-rate regime works.
        assertEquals(0.95 * Math.exp((-0.005 - 0.001) * 1.0),
                Fx.fxForward(0.95, 1.0, -0.005, 0.001), 1e-15);
        // t = 0: forward equals spot.
        assertEquals(1.25, Fx.fxForward(1.25, 0.0, 0.03, 0.02), 0.0);
    }

    @Test
    public void deltaConversionsAreExactMutualInverses() {
        double t = 0.75;
        double rf = 0.025;
        for (double delta : new double[]{0.25, 0.5, 0.9, -0.25, -0.65}) {
            double fwd = Fx.spotDeltaToForward(delta, t, rf);
            assertEquals("round-trip spot->fwd->spot", delta,
                    Fx.forwardDeltaToSpot(fwd, t, rf), 1e-15);
            // Sign-preserving (works for put deltas).
            assertTrue("sign preserved", Math.signum(fwd) == Math.signum(delta));
        }
        // Forward delta of a call is exactly N(d1): check against gkGreeks.
        double s = 1.10;
        double k = 1.12;
        double sigma = 0.10;
        double rd = 0.03;
        double rfx = 0.02;
        Greeks g = BlackScholes.gkGreeks(s, k, t, sigma, rd, rfx, OptionType.CALL);
        double d1 = BlackScholes.bsD1D2(s, k, t, sigma, rd, rfx)[0];
        assertEquals(BlackScholes.normCdf(d1),
                Fx.spotDeltaToForward(g.delta(), t, rfx), 1e-14);
    }

    @Test
    public void fxInvalidInputsThrow() {
        assertThrows(IllegalArgumentException.class, () -> Fx.fxForward(0.0, 1.0, 0.03, 0.02));
        assertThrows(IllegalArgumentException.class, () -> Fx.fxForward(1.1, -1.0, 0.03, 0.02));
        assertThrows(IllegalArgumentException.class,
                () -> Fx.spotDeltaToForward(Double.NaN, 1.0, 0.02));
        assertThrows(IllegalArgumentException.class,
                () -> Fx.forwardDeltaToSpot(0.5, 1.0, Double.POSITIVE_INFINITY));
    }

    @Test
    public void historicalVolRecoversConstantVolOfSyntheticSeries() {
        // A deterministic alternating log-return series has a known sample std.
        int n = 253;
        double[] prices = new double[n];
        prices[0] = 100.0;
        double ret = 0.01;
        for (int i = 1; i < n; i++) {
            prices[i] = prices[i - 1] * Math.exp(i % 2 == 0 ? -ret : ret);
        }
        // Alternating +-1% log returns: sample mean ~ 0, sample std ~ 1%.
        double vol = Utils.historicalVol(prices, 252);
        assertEquals(0.01 * Math.sqrt(252.0), vol, 1e-3);
    }

    @Test
    public void historicalVolValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> Utils.historicalVol(new double[]{100.0, 101.0}));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.historicalVol(new double[]{100.0, -1.0, 102.0}));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.historicalVol(new double[]{100.0, Double.NaN, 102.0}));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.historicalVol(new double[]{100.0, 101.0, 102.0}, 0));
        assertThrows(IllegalArgumentException.class, () -> Utils.historicalVol(null));
    }
}
