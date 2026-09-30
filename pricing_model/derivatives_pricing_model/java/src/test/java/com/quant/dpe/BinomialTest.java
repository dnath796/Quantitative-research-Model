package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Binomial module tests: convergence to Black-Scholes, Richardson averaging,
 * American exercise properties, tree Greeks, degenerate limits and validation.
 */
public class BinomialTest {

    private static final double S = 100.0;
    private static final double T = 1.0;
    private static final double SIGMA = 0.2;
    private static final double R = 0.05;

    @Test
    public void crrConvergesToBlackScholesWithin1e3At2000Steps() {
        double bs = BlackScholes.bsPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL);
        double crr = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 2000, BinomialMethod.CRR);
        assertEquals(bs, crr, 1e-3);
        double jr = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 2000, BinomialMethod.JR);
        assertEquals(bs, jr, 1e-3);
    }

    @Test
    public void richardsonAveragingImprovesConvergence() {
        double bs = BlackScholes.bsPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL);
        double plain = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 200, BinomialMethod.CRR);
        double rich = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 200, BinomialMethod.CRR, true);
        assertTrue("Richardson should beat plain at 200 steps: |"
                        + (rich - bs) + "| vs |" + (plain - bs) + "|",
                Math.abs(rich - bs) < Math.abs(plain - bs));
    }

    @Test
    public void americanAtLeastEuropeanAndPremiumIncreasesWithStrike() {
        double prevPremium = -1.0;
        for (double k : new double[]{90.0, 100.0, 110.0, 120.0}) {
            double eur = Binomial.binomialPrice(S, k, T, SIGMA, R, 0.0, OptionType.PUT,
                    ExerciseStyle.EUROPEAN, 400, BinomialMethod.CRR);
            double amer = Binomial.binomialPrice(S, k, T, SIGMA, R, 0.0, OptionType.PUT,
                    ExerciseStyle.AMERICAN, 400, BinomialMethod.CRR);
            double premium = amer - eur;
            assertTrue("american >= european at k=" + k, premium >= 0.0);
            assertTrue("early-exercise premium positive at k=" + k, premium > 1e-6);
            assertTrue("premium increasing in k at k=" + k, premium > prevPremium);
            prevPremium = premium;
        }
        // American call on a zero-dividend stock is never exercised early.
        double eurCall = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 400, BinomialMethod.CRR);
        double amerCall = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                ExerciseStyle.AMERICAN, 400, BinomialMethod.CRR);
        assertEquals(eurCall, amerCall, 1e-10);
    }

    @Test
    public void treeGreeksMatchAnalyticForEuropean() {
        BinomialGreeks tg = Binomial.binomialGreeks(S, 100.0, T, SIGMA, R, 0.0,
                OptionType.CALL, ExerciseStyle.EUROPEAN, 500, BinomialMethod.CRR);
        Greeks ag = BlackScholes.bsGreeks(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL);
        assertEquals("tree delta", ag.delta(), tg.delta(), 5e-4);
        assertEquals("tree gamma", ag.gamma(), tg.gamma(), 5e-4);
        assertEquals("tree theta", ag.theta(), tg.theta(), 2e-2);
        // JR lattice drifts; the ds correction must keep theta consistent too.
        BinomialGreeks jg = Binomial.binomialGreeks(S, 100.0, T, SIGMA, R, 0.0,
                OptionType.CALL, ExerciseStyle.EUROPEAN, 500, BinomialMethod.JR);
        assertEquals("JR tree delta", ag.delta(), jg.delta(), 5e-3);
        assertEquals("JR tree theta", ag.theta(), jg.theta(), 5e-2);
    }

    @Test
    public void degenerateLimits() {
        // t = 0: intrinsic.
        assertEquals(10.0, Binomial.binomialPrice(110.0, 100.0, 0.0, SIGMA, R, 0.0,
                OptionType.CALL, ExerciseStyle.AMERICAN, 100, BinomialMethod.CRR), 0.0);
        // sigma = 0 European: discounted forward intrinsic = BS limit.
        double bs0 = BlackScholes.bsPrice(S, 90.0, T, 0.0, R, 0.02, OptionType.CALL);
        assertEquals(bs0, Binomial.binomialPrice(S, 90.0, T, 0.0, R, 0.02, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 100, BinomialMethod.CRR), 1e-14);
        // sigma = 0 American put with r > 0: exercising the ITM put at t = 0
        // beats waiting (interest erodes the discounted strike), so the value
        // is the immediate intrinsic.
        double amer0 = Binomial.binomialPrice(80.0, 100.0, T, 0.0, R, 0.0, OptionType.PUT,
                ExerciseStyle.AMERICAN, 100, BinomialMethod.CRR);
        assertEquals(20.0, amer0, 1e-12);
        // ... and it must dominate the European sigma = 0 value.
        double eur0 = Binomial.binomialPrice(80.0, 100.0, T, 0.0, R, 0.0, OptionType.PUT,
                ExerciseStyle.EUROPEAN, 100, BinomialMethod.CRR);
        assertTrue(amer0 >= eur0);
    }

    @Test
    public void invalidInputsThrow() {
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 0, BinomialMethod.CRR));
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialPrice(-5.0, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 100, BinomialMethod.CRR));
        // Drift dominating vol at one step: p outside (0, 1) must throw, not clamp.
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialPrice(S, 100.0, 10.0, 0.01, 0.5, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 1, BinomialMethod.CRR));
        // Tree Greeks preconditions.
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialGreeks(S, 100.0, T, SIGMA, R, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 1, BinomialMethod.CRR));
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialGreeks(S, 100.0, 0.0, SIGMA, R, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 100, BinomialMethod.CRR));
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialGreeks(S, 100.0, T, 0.0, R, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 100, BinomialMethod.CRR));
    }

    // -----------------------------------------------------------------------
    // Contract-edge tests added after review (see docs/ARCHITECTURE.md section 6).
    // -----------------------------------------------------------------------

    @Test
    public void treeGreeksTwoStepsAreFinite() {
        // steps = 2 is the smallest tree the contract allows for Greeks (level
        // 2 is the terminal level). Previously a NullPointerException.
        Object[][] cases = {
            {OptionType.CALL, ExerciseStyle.EUROPEAN, BinomialMethod.CRR},
            {OptionType.CALL, ExerciseStyle.EUROPEAN, BinomialMethod.JR},
            {OptionType.PUT, ExerciseStyle.AMERICAN, BinomialMethod.CRR},
        };
        for (Object[] c : cases) {
            OptionType ot = (OptionType) c[0];
            ExerciseStyle style = (ExerciseStyle) c[1];
            BinomialMethod method = (BinomialMethod) c[2];
            BinomialGreeks g = Binomial.binomialGreeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0,
                    ot, style, 2, method);
            assertTrue(Double.isFinite(g.price()) && Double.isFinite(g.delta())
                    && Double.isFinite(g.gamma()) && Double.isFinite(g.theta()));
            assertTrue("gamma > 0", g.gamma() > 0.0);
            if (ot == OptionType.CALL) {
                assertTrue(g.delta() > 0.0 && g.delta() < 1.0);
            } else {
                assertTrue(g.delta() > -1.0 && g.delta() < 0.0);
            }
            assertEquals(Binomial.binomialPrice(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, ot, style,
                    2, method), g.price(), 0.0);
        }
    }

    @Test
    public void crrTreePutCallParityExact() {
        // CRR matches the first moment exactly, so tree call - put equals
        // s e^{-qt} - k e^{-rt} to round-off at every step count; JR does not
        // (its parity error is O(dt)).
        double q = 0.02;
        double fwd = S * Math.exp(-q * T) - 100.0 * Math.exp(-R * T);
        for (int n : new int[]{1, 2, 101, 500}) {
            double c = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, q, OptionType.CALL,
                    ExerciseStyle.EUROPEAN, n, BinomialMethod.CRR);
            double p = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, q, OptionType.PUT,
                    ExerciseStyle.EUROPEAN, n, BinomialMethod.CRR);
            assertEquals("CRR parity at n=" + n, fwd, c - p, 1e-10);
        }
        double cj = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, q, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 101, BinomialMethod.JR);
        double pj = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, q, OptionType.PUT,
                ExerciseStyle.EUROPEAN, 101, BinomialMethod.JR);
        assertEquals(fwd, cj - pj, 1e-3);
        assertTrue("JR is not a martingale lattice", Math.abs((cj - pj) - fwd) > 1e-6);
    }

    @Test
    public void crrOneStepClosedForm() {
        // One CRR step is a hand-computable two-state model.
        double q = 0.02;
        double u = Math.exp(SIGMA * Math.sqrt(T));
        double d = 1.0 / u;
        double p = (Math.exp((R - q) * T) - d) / (u - d);
        double expected = Math.exp(-R * T)
                * (p * Math.max(S * u - 100.0, 0.0) + (1.0 - p) * Math.max(S * d - 100.0, 0.0));
        assertEquals(11.073540703840242, expected, 1e-12); // derived independently
        assertEquals(expected, Binomial.binomialPrice(S, 100.0, T, SIGMA, R, q, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 1, BinomialMethod.CRR), 1e-12);
    }

    @Test
    public void americanCallWithDividendStrictlyExceedsEuropean() {
        // q > r: dividend leakage on a deep-ITM call makes early exercise valuable.
        double eur = Binomial.binomialPrice(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 500, BinomialMethod.CRR);
        double amer = Binomial.binomialPrice(100.0, 80.0, 1.0, 0.2, 0.05, 0.08, OptionType.CALL,
                ExerciseStyle.AMERICAN, 500, BinomialMethod.CRR);
        assertTrue("premium " + (amer - eur), amer - eur > 1.0);
        assertTrue(amer >= 20.0); // never below immediate intrinsic
    }

    @Test
    public void richardsonIsAverageOfNAndNPlusOne() {
        double rich = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.02, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 200, BinomialMethod.CRR, true);
        double a = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.02, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 200, BinomialMethod.CRR);
        double b = Binomial.binomialPrice(S, 100.0, T, SIGMA, R, 0.02, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 201, BinomialMethod.CRR);
        assertEquals(0.5 * (a + b), rich, 1e-14);
    }

    @Test
    public void treeOverflowGuard() {
        // sigma sqrt(t steps) = 707 > 700 would overflow the terminal spots to
        // inf; the contract says invalid-input error, never inf/NaN.
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialPrice(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 200, BinomialMethod.CRR));
        assertTrue(exc.getMessage(), exc.getMessage().contains("overflow"));
        assertThrows(IllegalArgumentException.class,
                () -> Binomial.binomialGreeks(100.0, 100.0, 1.0, 50.0, 0.05, 0.0, OptionType.CALL,
                        ExerciseStyle.EUROPEAN, 200, BinomialMethod.JR));
        // Just inside the guard (4.6 + 200 + 400 = 605 < 700) is finite.
        double v = Binomial.binomialPrice(100.0, 100.0, 1.0, 20.0, 0.05, 0.0, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 400, BinomialMethod.CRR);
        assertTrue(Double.isFinite(v) && v > 0.0 && v < 100.0);
    }

    @Test
    public void expiryPriceIsPositiveZeroOnTie() {
        double v = Binomial.binomialPrice(100.0, 100.0, 0.0, 0.2, 0.05, 0.0, OptionType.PUT,
                ExerciseStyle.EUROPEAN, 10, BinomialMethod.CRR);
        assertEquals(0.0, v, 0.0);
        assertEquals(1.0, Math.copySign(1.0, v), 0.0);
    }
}
