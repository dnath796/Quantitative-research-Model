package com.quant.dpe;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * End-to-end demo of the dpe pricing engine (Java port).
 *
 * <p>Run via {@code ./demo.sh} from the {@code java/} directory. Sections
 * mirror the Python reference demo:</p>
 * <ol>
 *   <li>Analytic BSM / Garman-Kohlhagen prices and full Greeks;</li>
 *   <li>Implied-vol round-trip;</li>
 *   <li>Binomial trees: convergence to BS, American early-exercise premium;</li>
 *   <li>Monte Carlo: variance-reduction comparison, exotics with 95% CIs,
 *       pathwise vs CRN finite-difference delta;</li>
 *   <li>Historical vol from {@code data/spots_timeseries.csv};</li>
 *   <li>Risk report for the small book in {@code data/portfolio.csv}.</li>
 * </ol>
 */
public final class Demo {

    private Demo() {
    }

    private static final String RULE = "-".repeat(78);

    private static void section(String title) {
        System.out.printf("%n%s%n%s%n%s%n", RULE, title, RULE);
    }

    /**
     * Runs the demo. Optional first argument: path to the {@code data}
     * directory (defaults to {@code ../data} relative to the working
     * directory, i.e. running from {@code java/}).
     *
     * @param args optional data-directory override
     * @throws IOException if a bundled CSV cannot be read
     */
    public static void main(String[] args) throws IOException {
        Path data = Path.of(args.length > 0 ? args[0] : "../data");
        System.out.println("dpe - Derivatives Pricing Engine demo (Java port)");
        demoAnalytic();
        demoImpliedVol();
        demoBinomial();
        demoMonteCarlo();
        demoHistoricalVol(data);
        demoPortfolio(data);
        System.out.printf("%n%s%ndone.%n%n", RULE);
    }

    private static void demoAnalytic() {
        section("1. Analytic Black-Scholes-Merton and Garman-Kohlhagen");
        Object[][] rows = {
                {"Equity ATM call (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType.CALL},
                {"Equity ATM put  (q=2%)", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType.PUT},
                {"High-vol small cap call", 18.0, 20.0, 1.0, 0.80, 0.04, 0.00, OptionType.CALL},
                {"Weekly put (T=1/52)", 100.0, 98.0, 1.0 / 52.0, 0.25, 0.03, 0.00, OptionType.PUT},
                {"LEAPS call (T=3y)", 100.0, 120.0, 3.0, 0.28, 0.04, 0.02, OptionType.CALL},
                {"FX EURUSD call (rd,rf)", 1.10, 1.12, 0.5, 0.10, 0.03, 0.02, OptionType.CALL},
                {"FX neg-rate put (rd<0)", 0.95, 0.95, 1.0, 0.12, -0.005, 0.001, OptionType.PUT},
        };
        System.out.printf("%-24s%10s%9s%9s%9s%9s%9s%9s%9s%n",
                "instrument", "price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga");
        for (Object[] row : rows) {
            Greeks g = BlackScholes.bsGreeks((double) row[1], (double) row[2], (double) row[3],
                    (double) row[4], (double) row[5], (double) row[6], (OptionType) row[7]);
            System.out.printf("%-24s%10.4f%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f%n",
                    row[0], g.price(), g.delta(), g.gamma(), g.vega(), g.theta(), g.rho(),
                    g.vanna(), g.volga());
        }
        Greeks g = BlackScholes.gkGreeks(1.10, 1.12, 0.5, 0.10, 0.03, 0.02, OptionType.CALL);
        double fwdDelta = Fx.spotDeltaToForward(g.delta(), 0.5, 0.02);
        System.out.printf("%nFX delta conventions (EURUSD call): spot delta = %.4f, "
                        + "forward delta = %.4f, forward = %.4f%n",
                g.delta(), fwdDelta, Fx.fxForward(1.10, 0.5, 0.03, 0.02));
    }

    private static void demoImpliedVol() {
        section("2. Implied volatility round-trip");
        double[][] cases = {{0.15, 120.0}, {0.40, 80.0}};
        for (double[] c : cases) {
            double sigmaIn = c[0];
            double k = c[1];
            double price = BlackScholes.bsPrice(100.0, k, 0.5, sigmaIn, 0.03, 0.01, OptionType.CALL);
            double iv = BlackScholes.impliedVol(price, 100.0, k, 0.5, 0.03, 0.01, OptionType.CALL);
            System.out.printf("k=%6.1f  price=%9.4f  sigma_in=%.4f  implied=%.10f%n",
                    k, price, sigmaIn, iv);
        }
        try {
            BlackScholes.impliedVol(1.0, 100.0, 80.0, 0.5, 0.03, 0.0, OptionType.CALL);
        } catch (IllegalArgumentException exc) {
            System.out.println("below-intrinsic price correctly rejected: " + exc.getMessage());
        }
    }

    private static void demoBinomial() {
        section("3. Binomial trees (CRR / Jarrow-Rudd)");
        double s = 100.0;
        double k = 100.0;
        double t = 1.0;
        double sig = 0.2;
        double r = 0.05;
        double q = 0.0;
        double bs = BlackScholes.bsPrice(s, k, t, sig, r, q, OptionType.CALL);
        System.out.printf("%6s %14s %16s %13s%n", "steps", "CRR-BS error", "CRR+Richardson", "JR-BS error");
        for (int n : new int[]{50, 200, 800, 2000}) {
            double crr = Binomial.binomialPrice(s, k, t, sig, r, q, OptionType.CALL,
                    ExerciseStyle.EUROPEAN, n, BinomialMethod.CRR);
            double rich = Binomial.binomialPrice(s, k, t, sig, r, q, OptionType.CALL,
                    ExerciseStyle.EUROPEAN, n, BinomialMethod.CRR, true);
            double jr = Binomial.binomialPrice(s, k, t, sig, r, q, OptionType.CALL,
                    ExerciseStyle.EUROPEAN, n, BinomialMethod.JR);
            System.out.printf("%6d %14.2e %16.2e %13.2e%n", n, crr - bs, rich - bs, jr - bs);
        }
        System.out.printf("%nAmerican put early-exercise premium (s=100, T=1, sigma=20%%, r=5%%):%n");
        System.out.printf("%8s%12s%12s%10s%n", "strike", "european", "american", "premium");
        for (double kk : new double[]{90.0, 100.0, 110.0, 120.0}) {
            double eur = Binomial.binomialPrice(s, kk, t, sig, r, 0.0, OptionType.PUT,
                    ExerciseStyle.EUROPEAN, 500, BinomialMethod.CRR);
            double amer = Binomial.binomialPrice(s, kk, t, sig, r, 0.0, OptionType.PUT,
                    ExerciseStyle.AMERICAN, 500, BinomialMethod.CRR);
            System.out.printf("%8.1f%12.4f%12.4f%10.4f%n", kk, eur, amer, amer - eur);
        }
        BinomialGreeks bg = Binomial.binomialGreeks(s, k, t, sig, r, q, OptionType.CALL,
                ExerciseStyle.EUROPEAN, 500, BinomialMethod.CRR);
        Greeks ag = BlackScholes.bsGreeks(s, k, t, sig, r, q, OptionType.CALL);
        System.out.printf("%ntree Greeks (500-step CRR) vs analytic: delta %.4f/%.4f, "
                        + "gamma %.4f/%.4f, theta %.4f/%.4f%n",
                bg.delta(), ag.delta(), bg.gamma(), ag.gamma(), bg.theta(), ag.theta());
    }

    private static void demoMonteCarlo() {
        section("4. Monte Carlo (100k paths, seed 42)");
        double s = 100.0;
        double k = 100.0;
        double t = 1.0;
        double sig = 0.2;
        double r = 0.05;
        double q = 0.0;
        int n = 100_000;
        double bs = BlackScholes.bsPrice(s, k, t, sig, r, q, OptionType.CALL);
        MCResult plain = MonteCarlo.mcEuropean(s, k, t, sig, r, q, OptionType.CALL, n, 42, false, false);
        MCResult anti = MonteCarlo.mcEuropean(s, k, t, sig, r, q, OptionType.CALL, n, 42, true, false);
        MCResult cv = MonteCarlo.mcEuropean(s, k, t, sig, r, q, OptionType.CALL, n, 42, true, true);
        System.out.printf("European call, BS analytic = %.4f%n", bs);
        System.out.printf("%-28s%10s%10s%24s%n", "estimator", "price", "std err", "95% CI");
        printMcRow("plain", plain);
        printMcRow("antithetic", anti);
        printMcRow("antithetic + control", cv);

        MCResult asian = MonteCarlo.mcAsianArithmetic(s, k, t, sig, r, q, OptionType.CALL,
                n, 12, 42, true, true);
        MCResult barrier = MonteCarlo.mcBarrierUpOut(s, k, t, sig, r, q, OptionType.CALL,
                130.0, n, 100, 42, true);
        MCResult lookback = MonteCarlo.mcLookbackFloating(s, t, sig, r, q, OptionType.CALL,
                n, 100, 42, true);
        System.out.printf("%n%-34s%10s%10s%24s%n", "exotic", "price", "std err", "95% CI");
        printMcRowWide("arithmetic Asian (12 fix, CV)", asian);
        printMcRowWide("up-and-out barrier B=130", barrier);
        printMcRowWide("floating-strike lookback", lookback);

        MCResult pw = MonteCarlo.mcDeltaPathwise(s, k, t, sig, r, q, OptionType.CALL, n, 42, true);
        MCResult fd = MonteCarlo.mcDeltaFdCrn(s, k, t, sig, r, q, OptionType.CALL, n, 42, 1e-4, true);
        System.out.printf("%ndelta: analytic %.4f, pathwise %.4f (se %.4f), FD+CRN %.4f (se %.4f)%n",
                BlackScholes.bsGreeks(s, k, t, sig, r, q, OptionType.CALL).delta(),
                pw.value(), pw.stdError(), fd.value(), fd.stdError());
    }

    private static void printMcRow(String label, MCResult res) {
        System.out.printf("%-28s%10.4f%10.4f%24s%n", label, res.value(), res.stdError(),
                String.format("[%.4f, %.4f]", res.ciLow(), res.ciHigh()));
    }

    private static void printMcRowWide(String label, MCResult res) {
        System.out.printf("%-34s%10.4f%10.4f%24s%n", label, res.value(), res.stdError(),
                String.format("[%.4f, %.4f]", res.ciLow(), res.ciHigh()));
    }

    private static void demoHistoricalVol(Path data) throws IOException {
        section("5. Historical volatility from data/spots_timeseries.csv");
        List<String> lines = Files.readAllLines(data.resolve("spots_timeseries.csv"));
        String[] header = lines.get(0).split(",");
        int nCols = header.length;
        int nRows = lines.size() - 1;
        double[][] cols = new double[nCols - 1][nRows];
        for (int i = 0; i < nRows; i++) {
            String[] parts = lines.get(i + 1).split(",");
            for (int c = 1; c < nCols; c++) {
                cols[c - 1][i] = Double.parseDouble(parts[c]);
            }
        }
        System.out.printf("%-10s%12s%14s%n", "underlier", "last close", "realised vol");
        for (int c = 1; c < nCols; c++) {
            double[] series = cols[c - 1];
            double vol = Utils.historicalVol(series);
            System.out.printf("%-10s%12.4f%13.2f%%%n", header[c], series[nRows - 1], vol * 100.0);
        }
    }

    private static void demoPortfolio(Path data) throws IOException {
        section("6. Portfolio risk report (data/portfolio.csv)");
        List<String> lines = Files.readAllLines(data.resolve("portfolio.csv"));
        System.out.printf("%-16s%-4s%-5s%10s%9s%9s%10s%10s%16s%n",
                "id", "typ", "style", "price", "delta", "gamma", "vega", "theta", "position value");
        double total = 0.0;
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] p = line.split(",");
            String id = p[0];
            double s = Double.parseDouble(p[2]);
            double k = Double.parseDouble(p[3]);
            double t = Double.parseDouble(p[4]);
            double sigma = Double.parseDouble(p[5]);
            double r = Double.parseDouble(p[6]);
            double q = Double.parseDouble(p[7]);
            OptionType type = OptionType.parse(p[8]);
            ExerciseStyle style = ExerciseStyle.parse(p[9]);
            double quantity = Double.parseDouble(p[10]);

            double price;
            double delta;
            double gamma;
            String vega; // the tree reports no vega: print "n/a", never a NaN
            double theta;
            if (style == ExerciseStyle.AMERICAN) {
                price = Binomial.binomialPrice(s, k, t, sigma, r, q, type,
                        ExerciseStyle.AMERICAN, 500, BinomialMethod.CRR);
                BinomialGreeks g = Binomial.binomialGreeks(s, k, t, sigma, r, q, type,
                        ExerciseStyle.AMERICAN, 500, BinomialMethod.CRR);
                delta = g.delta();
                gamma = g.gamma();
                vega = String.format("%10s", "n/a");
                theta = g.theta();
            } else {
                Greeks g = BlackScholes.bsGreeks(s, k, t, sigma, r, q, type);
                price = g.price();
                delta = g.delta();
                gamma = g.gamma();
                vega = String.format("%10.4f", g.vega());
                theta = g.theta();
            }
            double value = price * quantity;
            total += value;
            System.out.printf("%-16s%-4s%-5s%10.4f%9.4f%9.4f%s%10.4f%,16.2f%n",
                    id, p[8].substring(0, 1).toUpperCase(java.util.Locale.ROOT),
                    p[9].substring(0, Math.min(4, p[9].length())),
                    price, delta, gamma, vega, theta, value);
        }
        System.out.printf("%n%-57s%,16.2f%n", "total book value", total);
        System.out.println("(FX rows: r=rd, q=rf; price in domestic ccy per unit foreign; "
                + "vega n/a for tree)");
    }
}
