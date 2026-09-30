package com.quant.dpe;

/**
 * Full set of analytic Black-Scholes-Merton / Garman-Kohlhagen Greeks.
 *
 * <p>Units (fixed by the cross-language contract; never rescaled here):</p>
 * <ul>
 *   <li>{@code delta} = dV/dS (dimensionless);</li>
 *   <li>{@code gamma} = d2V/dS2;</li>
 *   <li>{@code vega}  = dV/dsigma per unit of vol (per 1.00 = 100 vol points);</li>
 *   <li>{@code theta} = dV/dt in calendar time, per <b>year</b> (divide by 365
 *       for per-day theta);</li>
 *   <li>{@code rho}   = dV/dr per unit of rate (domestic rate for FX);</li>
 *   <li>{@code vanna} = d2V/(dS dsigma) — sensitivity of delta to vol;</li>
 *   <li>{@code volga} = d2V/dsigma2 (vomma) — sensitivity of vega to vol.</li>
 * </ul>
 *
 * @param price option present value
 * @param delta dV/dS
 * @param gamma d2V/dS2 (identical for call and put)
 * @param vega  dV/dsigma, per unit of vol (identical for call and put)
 * @param theta dV/dt per year, calendar-time convention
 * @param rho   dV/dr (domestic rate for FX)
 * @param vanna d2V/(dS dsigma)
 * @param volga d2V/dsigma2
 */
public record Greeks(
        double price,
        double delta,
        double gamma,
        double vega,
        double theta,
        double rho,
        double vanna,
        double volga) {
}
