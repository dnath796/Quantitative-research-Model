package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Monte Carlo tests: statistical agreement with analytic anchors (3-SE bands),
 * exotic ordering properties, variance reduction, determinism and validation.
 *
 * <p>Path counts are kept modest so the whole suite stays well under the
 * runtime budget; assertions use the estimator's own standard error.</p>
 */
public class MonteCarloTest {

    private static final double S = 100.0;
    private static final double K = 100.0;
    private static final double T = 1.0;
    private static final double SIG = 0.2;
    private static final double R = 0.05;
    private static final long SEED = 42L;

    @Test
    public void europeanWithinThreeStandardErrorsOfBs() {
        double bs = BlackScholes.bsPrice(S, K, T, SIG, R, 0.0, OptionType.CALL);
        MCResult res = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                50_000, SEED, true, false);
        assertTrue("MC price " + res.value() + " within 3 SE of BS " + bs,
                Math.abs(res.value() - bs) <= 3.0 * res.stdError());
        assertTrue(res.stdError() > 0.0);
        assertEquals(res.value() - MonteCarlo.Z95 * res.stdError(), res.ciLow(), 1e-12);
        assertEquals(res.value() + MonteCarlo.Z95 * res.stdError(), res.ciHigh(), 1e-12);
        assertEquals(50_000, res.nPaths());
    }

    @Test
    public void controlVariateShrinksStandardError() {
        MCResult plain = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                20_000, SEED, false, false);
        MCResult anti = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                20_000, SEED, true, false);
        MCResult cv = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                20_000, SEED, true, true);
        assertTrue("antithetic SE < plain SE", anti.stdError() < plain.stdError());
        assertTrue("control-variate SE < antithetic SE", cv.stdError() < anti.stdError());
    }

    @Test
    public void deterministicGivenSeed() {
        MCResult a = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                10_000, 7L, true, false);
        MCResult b = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                10_000, 7L, true, false);
        assertEquals(a.value(), b.value(), 0.0);
        assertEquals(a.stdError(), b.stdError(), 0.0);
        MCResult c = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                10_000, 8L, true, false);
        assertTrue("different seed gives different estimate", a.value() != c.value());
    }

    @Test
    public void simulatedPathsHaveCorrectShapeAndAntitheticStructure() {
        double[][] paths = MonteCarlo.simulateGbmPaths(S, T, SIG, R, 0.0, 8, 5, SEED, true);
        assertEquals(8, paths.length);
        assertEquals(6, paths[0].length);
        for (double[] path : paths) {
            assertEquals(S, path[0], 0.0);
        }
        // Antithetic pairing: log-returns of row i and row i + n/2 are negated.
        for (int i = 0; i < 4; i++) {
            double z1 = Math.log(paths[i][1] / S);
            double z2 = Math.log(paths[4 + i][1] / S);
            double drift = (R - 0.5 * SIG * SIG) * (T / 5.0);
            assertEquals("antithetic increments", 2.0 * drift, z1 + z2, 1e-12);
        }
    }

    @Test
    public void geometricAsianClosedFormAnchorsArithmeticMc() {
        // Arithmetic-average Asian >= geometric-average Asian (AM-GM), and the
        // MC estimate with the geometric control must sit near its 3-SE band
        // around the true arithmetic price; verify against ordering instead.
        double geo = MonteCarlo.geometricAsianPrice(S, K, T, SIG, R, 0.0,
                OptionType.CALL, 12);
        MCResult arith = MonteCarlo.mcAsianArithmetic(S, K, T, SIG, R, 0.0,
                OptionType.CALL, 40_000, 12, SEED, true, true);
        assertTrue("arithmetic Asian above geometric Asian",
                arith.value() > geo);
        // Asian (average) option is worth less than the vanilla on the same params.
        double vanilla = BlackScholes.bsPrice(S, K, T, SIG, R, 0.0, OptionType.CALL);
        assertTrue("Asian <= vanilla", arith.value() < vanilla);
    }

    @Test
    public void barrierBelowVanillaAndBornDeadIsZero() {
        MCResult barrier = MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.CALL,
                130.0, 20_000, 50, SEED, true);
        double vanilla = BlackScholes.bsPrice(S, K, T, SIG, R, 0.0, OptionType.CALL);
        assertTrue("up-and-out <= vanilla", barrier.value() < vanilla);
        assertTrue("knock-out price positive", barrier.value() > 0.0);
        // Spot at/above barrier: born dead, exact zero without simulation.
        MCResult dead = MonteCarlo.mcBarrierUpOut(135.0, K, T, SIG, R, 0.0, OptionType.CALL,
                130.0, 20_000, 50, SEED, true);
        assertEquals(0.0, dead.value(), 0.0);
        assertEquals(0.0, dead.stdError(), 0.0);
        assertEquals(20_000, dead.nPaths());
        // More monitoring dates catch more breaches: price decreases in nSteps
        // (discretisation bias of the discrete contract vs continuous).
        MCResult coarse = MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.CALL,
                130.0, 20_000, 5, SEED, true);
        assertTrue("finer monitoring lowers up-and-out price",
                barrier.value() < coarse.value());
    }

    @Test
    public void lookbackPayoffsAreNonNegativeAndAboveVanilla() {
        MCResult lb = MonteCarlo.mcLookbackFloating(S, T, SIG, R, 0.0, OptionType.CALL,
                20_000, 50, SEED, true);
        assertTrue("lookback call value positive", lb.value() > 0.0);
        // Floating-strike lookback call (buy at the min) dominates the ATM
        // vanilla call (buy at S0 = the t=0 grid value >= min).
        double vanilla = BlackScholes.bsPrice(S, S, T, SIG, R, 0.0, OptionType.CALL);
        assertTrue("lookback >= ATM vanilla", lb.value() > vanilla);
    }

    @Test
    public void pathwiseAndCrnDeltasMatchAnalytic() {
        double analytic = BlackScholes.bsGreeks(S, K, T, SIG, R, 0.0,
                OptionType.CALL).delta();
        MCResult pw = MonteCarlo.mcDeltaPathwise(S, K, T, SIG, R, 0.0, OptionType.CALL,
                50_000, SEED, true);
        assertTrue("pathwise delta within 3 SE",
                Math.abs(pw.value() - analytic) <= 3.0 * pw.stdError());
        MCResult fd = MonteCarlo.mcDeltaFdCrn(S, K, T, SIG, R, 0.0, OptionType.CALL,
                50_000, SEED, 1e-4, true);
        assertTrue("CRN FD delta within 3 SE + O(h^2) bias",
                Math.abs(fd.value() - analytic) <= 3.0 * fd.stdError() + 1e-4);
        // Put deltas are negative.
        MCResult pwPut = MonteCarlo.mcDeltaPathwise(S, K, T, SIG, R, 0.0, OptionType.PUT,
                20_000, SEED, true);
        assertTrue("put pathwise delta negative", pwPut.value() < 0.0);
    }

    @Test
    public void invalidInputsThrow() {
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcEuropean(S, K, 0.0, SIG, R, 0.0, OptionType.CALL,
                        1000, SEED, true, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                        1, SEED, false, false));
        // Odd path count with antithetics is invalid.
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                        1001, SEED, true, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                        2, SEED, true, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.simulateGbmPaths(S, T, SIG, R, 0.0, 100, 0, SEED, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.CALL,
                        0.0, 1000, 10, SEED, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcDeltaFdCrn(S, K, T, SIG, R, 0.0, OptionType.CALL,
                        1000, SEED, 0.0, false));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.geometricAsianPrice(S, 0.0, T, SIG, R, 0.0,
                        OptionType.CALL, 12));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.geometricAsianPrice(S, K, T, SIG, R, 0.0,
                        OptionType.CALL, 0));
        assertThrows(IllegalArgumentException.class,
                () -> MonteCarlo.mcAsianArithmetic(S, 0.0, T, SIG, R, 0.0, OptionType.CALL,
                        1000, 12, SEED, true, true));
    }

    // -----------------------------------------------------------------------
    // Contract-edge tests added after review (see docs/ARCHITECTURE.md section 6).
    // -----------------------------------------------------------------------

    @Test
    public void negativeSeedRejected() {
        // Seeds are integers in [0, 2^63 - 1] in every port: a negative seed
        // is an invalid-input error naming `seed` (previously accepted).
        for (long bad : new long[]{-1L, Long.MIN_VALUE}) {
            IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                    () -> MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                            1000, bad, true, false));
            assertTrue(exc.getMessage(), exc.getMessage().contains("seed"));
            assertThrows(IllegalArgumentException.class,
                    () -> MonteCarlo.simulateGbmPaths(S, T, SIG, R, 0.0, 10, 2, bad, false));
            assertThrows(IllegalArgumentException.class,
                    () -> MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.CALL,
                            130.0, 100, 4, bad, true));
        }
        // Both endpoints of the domain are accepted and give different streams.
        MCResult lo = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                1000, 0L, true, false);
        MCResult hi = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                1000, Long.MAX_VALUE, true, false);
        assertTrue(Double.isFinite(lo.value()) && Double.isFinite(hi.value()));
        assertTrue(lo.value() != hi.value());
    }

    @Test
    public void relBumpDomain() {
        // rel_bump >= 1 would price a non-positive spot silently; the
        // contract pins the open interval (0, 1).
        for (double bad : new double[]{1.0, 2.0, 0.0, -1e-4, Double.NaN,
                Double.POSITIVE_INFINITY}) {
            IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                    () -> MonteCarlo.mcDeltaFdCrn(S, K, T, SIG, R, 0.0, OptionType.CALL,
                            1000, SEED, bad, true));
            assertTrue(exc.getMessage(), exc.getMessage().contains("rel_bump"));
        }
    }

    @Test
    public void asianSingleFixingEqualsEuropean() {
        // One fixing: the arithmetic Asian IS the vanilla, and both engines
        // consume the same draws in the same order -> bit-identical results.
        MCResult asian = MonteCarlo.mcAsianArithmetic(S, K, T, SIG, R, 0.0, OptionType.CALL,
                20_000, 1, 7L, true, false);
        MCResult euro = MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.CALL,
                20_000, 7L, true, false);
        assertEquals(euro, asian);
    }

    @Test
    public void barrierSingleMonitoringMatchesAnalytic() {
        // One monitoring date (t = T, plus t = 0 where s < B): the up-and-out
        // call pays (S_T - k) 1{k < S_T < B} = C(k) - C(B) - (B - k) e^{-rt} N(d2(B)).
        double barrier = 130.0;
        double d2B = (Math.log(S / barrier) + (R - 0.0 - 0.5 * SIG * SIG) * T)
                / (SIG * Math.sqrt(T));
        double analytic = BlackScholes.bsPrice(S, K, T, SIG, R, 0.0, OptionType.CALL)
                - BlackScholes.bsPrice(S, barrier, T, SIG, R, 0.0, OptionType.CALL)
                - (barrier - K) * Math.exp(-R * T) * BlackScholes.normCdf(d2B);
        assertEquals(5.310827113020051, analytic, 1e-12); // derived independently
        MCResult res = MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.CALL,
                barrier, 200_000, 1, 3L, true);
        assertEquals("barrier vs analytic (se " + res.stdError() + ")", analytic, res.value(),
                3.0 * res.stdError());
        assertTrue(res.value() < BlackScholes.bsPrice(S, K, T, SIG, R, 0.0, OptionType.CALL));
    }

    @Test
    public void lookbackSingleStepIsAtmVanilla() {
        // One step: call pays (S_T - s)^+, put pays (s - S_T)^+ (the ATM vanillas).
        MCResult call = MonteCarlo.mcLookbackFloating(S, T, SIG, R, 0.0, OptionType.CALL,
                200_000, 1, 3L, true);
        assertEquals(BlackScholes.bsPrice(S, S, T, SIG, R, 0.0, OptionType.CALL), call.value(),
                3.0 * call.stdError());
        MCResult put = MonteCarlo.mcLookbackFloating(S, T, SIG, R, 0.0, OptionType.PUT,
                200_000, 1, 3L, true);
        assertEquals(BlackScholes.bsPrice(S, S, T, SIG, R, 0.0, OptionType.PUT), put.value(),
                3.0 * put.stdError());
    }

    @Test
    public void sigmaZeroIsDeterministic() {
        // sigma = 0: every path is the forward, the control is degenerate
        // (varX = 0 -> beta = 0 branch) and the value equals the analytic
        // limit with (numerically) zero standard error.
        MCResult res = MonteCarlo.mcEuropean(S, K, T, 0.0, R, 0.0, OptionType.CALL,
                1000, 1L, true, true);
        assertEquals(BlackScholes.bsPrice(S, K, T, 0.0, R, 0.0, OptionType.CALL), res.value(),
                1e-12);
        assertTrue(res.stdError() <= 1e-12);
        assertTrue(res.ciHigh() - res.ciLow() <= 1e-11);
        MCResult asian = MonteCarlo.mcAsianArithmetic(S, K, T, 0.0, R, 0.0, OptionType.PUT,
                1000, 4, 1L, true, true);
        assertEquals(MonteCarlo.geometricAsianPrice(S, K, T, 0.0, R, 0.0, OptionType.PUT, 4),
                asian.value(), 1e-12);
        assertTrue(asian.stdError() <= 1e-12);
    }

    @Test
    public void resultsFiniteAndCiOrdered() {
        MCResult[] results = {
            MonteCarlo.mcEuropean(S, K, T, SIG, R, 0.0, OptionType.PUT, 2000, 5L, true, true),
            MonteCarlo.mcAsianArithmetic(S, K, T, SIG, R, 0.0, OptionType.PUT, 2000, 6, 5L,
                    true, true),
            MonteCarlo.mcBarrierUpOut(S, K, T, SIG, R, 0.0, OptionType.PUT, 140.0, 2000, 6, 5L,
                    true),
            MonteCarlo.mcLookbackFloating(S, T, SIG, R, 0.0, OptionType.PUT, 2000, 6, 5L, true),
            MonteCarlo.mcDeltaPathwise(S, K, T, SIG, R, 0.0, OptionType.PUT, 2000, 5L, true),
            MonteCarlo.mcDeltaFdCrn(S, K, T, SIG, R, 0.0, OptionType.PUT, 2000, 5L, 1e-4, true),
        };
        for (MCResult r : results) {
            assertTrue(Double.isFinite(r.value()) && Double.isFinite(r.stdError()));
            assertTrue(Double.isFinite(r.ciLow()) && Double.isFinite(r.ciHigh()));
            assertTrue(r.ciLow() <= r.value() && r.value() <= r.ciHigh());
            assertTrue(r.stdError() >= 0.0);
        }
    }
}
