package com.quant.dpe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Cross-language golden-value suite: loads {@code ../data/golden/golden.json}
 * (relative to the {@code java/} directory) and asserts every case within its
 * stated absolute tolerance.
 *
 * <p>Per the API spec (section 9.3) the two Monte Carlo cases have advisory
 * seeds — the RNG stream is Java-specific — and several-standard-error-wide
 * tolerances; every other case is deterministic and must match to 1e-6..1e-10.</p>
 */
public class GoldenTest {

    private static Map<String, Object> load() throws IOException {
        Path path = Path.of("../data/golden/golden.json");
        if (!Files.exists(path)) {
            // Fallback when the JVM is started from the repository root.
            path = Path.of("data/golden/golden.json");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> doc = (Map<String, Object>) MiniJson.parse(Files.readString(path));
        return doc;
    }

    private static double in(Map<String, Object> inputs, String key) {
        Object v = inputs.get(key);
        if (v == null) {
            fail("missing input '" + key + "'");
        }
        return (Double) v;
    }

    @Test
    public void allGoldenCasesMatch() throws IOException {
        Map<String, Object> doc = load();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> cases = (List<Map<String, Object>>) doc.get("cases");
        assertTrue("golden file must contain cases", cases.size() >= 20);
        for (Map<String, Object> c : cases) {
            runCase(c);
        }
    }

    private void runCase(Map<String, Object> c) {
        String name = (String) c.get("name");
        @SuppressWarnings("unchecked")
        Map<String, Object> inputs = (Map<String, Object>) c.get("inputs");
        @SuppressWarnings("unchecked")
        Map<String, Object> expect = (Map<String, Object>) c.get("expect");
        double tol = (Double) c.get("tol");

        switch (name) {
            case "bs_call_atm":
            case "bs_put_atm":
            case "bs_call_dividend":
            case "bs_put_deep_itm":
            case "bs_call_deep_otm":
            case "bs_call_high_vol":
            case "bs_put_weekly":
            case "bs_call_leaps":
            case "bs_call_sigma_zero":
            case "bs_put_expiry":
            case "bs_call_zero_strike": {
                double price = BlackScholes.bsPrice(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")));
                assertEquals(name + ".price", (Double) expect.get("price"), price, tol);
                break;
            }
            case "bs_greeks_call_atm":
            case "bs_greeks_put_itm": {
                Greeks g = BlackScholes.bsGreeks(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")));
                assertEquals(name + ".delta", (Double) expect.get("delta"), g.delta(), tol);
                assertEquals(name + ".gamma", (Double) expect.get("gamma"), g.gamma(), tol);
                assertEquals(name + ".vega", (Double) expect.get("vega"), g.vega(), tol);
                assertEquals(name + ".theta", (Double) expect.get("theta"), g.theta(), tol);
                assertEquals(name + ".rho", (Double) expect.get("rho"), g.rho(), tol);
                assertEquals(name + ".vanna", (Double) expect.get("vanna"), g.vanna(), tol);
                assertEquals(name + ".volga", (Double) expect.get("volga"), g.volga(), tol);
                break;
            }
            case "gk_call_eurusd": {
                double t = in(inputs, "t");
                double rf = in(inputs, "rf");
                OptionType type = OptionType.parse((String) inputs.get("type"));
                double price = BlackScholes.gkPrice(in(inputs, "s"), in(inputs, "k"), t,
                        in(inputs, "sigma"), in(inputs, "rd"), rf, type);
                Greeks g = BlackScholes.gkGreeks(in(inputs, "s"), in(inputs, "k"), t,
                        in(inputs, "sigma"), in(inputs, "rd"), rf, type);
                double deltaFwd = Fx.spotDeltaToForward(g.delta(), t, rf);
                assertEquals(name + ".price", (Double) expect.get("price"), price, tol);
                assertEquals(name + ".delta_spot", (Double) expect.get("delta_spot"),
                        g.delta(), tol);
                assertEquals(name + ".delta_forward", (Double) expect.get("delta_forward"),
                        deltaFwd, tol);
                break;
            }
            case "gk_put_negative_rate": {
                double price = BlackScholes.gkPrice(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "rd"), in(inputs, "rf"),
                        OptionType.parse((String) inputs.get("type")));
                assertEquals(name + ".price", (Double) expect.get("price"), price, tol);
                break;
            }
            case "iv_roundtrip_call_atm":
            case "iv_roundtrip_put_otm": {
                double sigma = BlackScholes.impliedVol(in(inputs, "price"), in(inputs, "s"),
                        in(inputs, "k"), in(inputs, "t"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")));
                assertEquals(name + ".sigma", (Double) expect.get("sigma"), sigma, tol);
                break;
            }
            case "crr_euro_call_2000":
            case "crr_amer_put_500":
            case "crr_amer_put_div_500":
            case "jr_euro_put_500": {
                double price = Binomial.binomialPrice(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        ExerciseStyle.parse((String) inputs.get("style")),
                        (int) (double) (Double) inputs.get("steps"),
                        BinomialMethod.parse((String) inputs.get("method")));
                assertEquals(name + ".price", (Double) expect.get("price"), price, tol);
                break;
            }
            case "crr_greeks_call_500": {
                BinomialGreeks g = Binomial.binomialGreeks(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        ExerciseStyle.parse((String) inputs.get("style")),
                        (int) (double) (Double) inputs.get("steps"),
                        BinomialMethod.parse((String) inputs.get("method")));
                assertEquals(name + ".delta", (Double) expect.get("delta"), g.delta(), tol);
                assertEquals(name + ".gamma", (Double) expect.get("gamma"), g.gamma(), tol);
                assertEquals(name + ".theta", (Double) expect.get("theta"), g.theta(), tol);
                break;
            }
            case "geo_asian_call_analytic": {
                double price = MonteCarlo.geometricAsianPrice(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        (int) (double) (Double) inputs.get("n_fixings"));
                assertEquals(name + ".price", (Double) expect.get("price"), price, tol);
                break;
            }
            case "mc_euro_call_100k": {
                // Seed is advisory (RNG streams are language-specific); the
                // tolerance is several standard errors wide by construction.
                MCResult res = MonteCarlo.mcEuropean(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        (int) (double) (Double) inputs.get("n_paths"), 42L, true, false);
                assertEquals(name + ".price", (Double) expect.get("price"), res.value(), tol);
                break;
            }
            case "mc_asian_call_cv_12fix": {
                MCResult res = MonteCarlo.mcAsianArithmetic(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        (int) (double) (Double) inputs.get("n_paths"),
                        (int) (double) (Double) inputs.get("n_steps"), 42L, true, true);
                assertEquals(name + ".price", (Double) expect.get("price"), res.value(), tol);
                break;
            }
            case "mc_barrier_single_step_call": {
                MCResult res = MonteCarlo.mcBarrierUpOut(in(inputs, "s"), in(inputs, "k"),
                        in(inputs, "t"), in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")), in(inputs, "barrier"),
                        (int) (double) (Double) inputs.get("n_paths"),
                        (int) (double) (Double) inputs.get("n_steps"),
                        (long) (double) (Double) inputs.get("seed"), true);
                assertEquals(name + ".price", (Double) expect.get("price"), res.value(), tol);
                break;
            }
            case "mc_lookback_single_step_call":
            case "mc_lookback_single_step_put": {
                MCResult res = MonteCarlo.mcLookbackFloating(in(inputs, "s"), in(inputs, "t"),
                        in(inputs, "sigma"), in(inputs, "r"), in(inputs, "q"),
                        OptionType.parse((String) inputs.get("type")),
                        (int) (double) (Double) inputs.get("n_paths"),
                        (int) (double) (Double) inputs.get("n_steps"),
                        (long) (double) (Double) inputs.get("seed"), true);
                assertEquals(name + ".price", (Double) expect.get("price"), res.value(), tol);
                break;
            }
            default:
                fail("unhandled golden case: " + name);
        }
    }
}
