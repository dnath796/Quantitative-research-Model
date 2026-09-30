//! Standard-normal helpers built on a self-contained `erfc`.
//!
//! Rust's standard library exposes no error function, and the crate keeps
//! its dependency list minimal, so `erfc` is implemented here with
//! W. J. Cody's rational Chebyshev approximation (SPECFUN/CALERF, Cody 1969
//! / ACM TOMS Algorithm 715). glibc's `erfc` uses a different rational
//! approximation (Sun's fdlibm `s_erf.c`); both are accurate to about an ulp
//! over the whole double range, which is why the Rust and Python/C++ values
//! agree to the 1e-10 golden tolerance. The normal CDF is then `N(x) = 0.5 * erfc(-x / sqrt(2))` —
//! using `erfc` rather than `0.5 * (1 + erf)` preserves full relative
//! precision in the deep left tail (deep OTM options), which the golden case
//! `bs_call_deep_otm` checks at 1e-10 absolute tolerance.

use std::f64::consts::PI;

/// Cody erf numerator coefficients, |x| <= 0.46875.
const A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_56e2,
    3.774_852_376_853_020_2e2,
    3.209_377_589_138_469_47e3,
    1.857_777_061_846_031_53e-1,
];
/// Cody erf denominator coefficients, |x| <= 0.46875.
const B: [f64; 4] = [
    2.360_129_095_234_412_09e1,
    2.440_246_379_344_441_73e2,
    1.282_616_526_077_372_28e3,
    2.844_236_833_439_170_62e3,
];
/// Cody erfc numerator coefficients, 0.46875 < |x| <= 4.
const C: [f64; 9] = [
    5.641_884_969_886_700_9e-1,
    8.883_149_794_388_375_94e0,
    6.611_919_063_714_162_95e1,
    2.986_351_381_974_001_31e2,
    8.819_522_212_417_690_9e2,
    1.712_047_612_634_070_58e3,
    2.051_078_377_826_071_47e3,
    1.230_339_354_797_997_25e3,
    2.153_115_354_744_038_46e-8,
];
/// Cody erfc denominator coefficients, 0.46875 < |x| <= 4.
const D: [f64; 8] = [
    1.574_492_611_070_983_47e1,
    1.176_939_508_913_124_99e2,
    5.371_811_018_620_098_58e2,
    1.621_389_574_566_690_19e3,
    3.290_799_235_733_459_63e3,
    4.362_619_090_143_247_16e3,
    3.439_367_674_143_721_64e3,
    1.230_339_354_803_749_42e3,
];
/// Cody erfc numerator coefficients, |x| > 4 (asymptotic region).
const P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
/// Cody erfc denominator coefficients, |x| > 4 (asymptotic region).
const Q: [f64; 5] = [
    2.568_520_192_289_822_42e0,
    1.872_952_849_923_460_47e0,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];

/// `1 / sqrt(pi)`.
const SQRPI: f64 = 5.641_895_835_477_562_87e-1;
/// Below this, `erf(x) ~ 2x/sqrt(pi)` to full precision.
const XSMALL: f64 = 1.11e-16;
/// Above this, `erfc(x)` underflows to 0 in double precision.
const XBIG: f64 = 26.543;

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Cody's three-region rational approximation; relative accuracy is a few
/// ulps everywhere, including the far tail where `erfc` underflows smoothly
/// to 0 (for `x > 26.543`).
pub(crate) fn erfc(x: f64) -> f64 {
    let y = x.abs();
    let result = if y <= 0.46875 {
        // Central region: compute erf via the degree-4/4 rational in y^2.
        let z = if y > XSMALL { y * y } else { 0.0 };
        let mut num = A[4] * z;
        let mut den = z;
        for i in 0..3 {
            num = (num + A[i]) * z;
            den = (den + B[i]) * z;
        }
        // erfc(x) = 1 - erf(x); x (not y) keeps the sign correct here.
        return 1.0 - x * (num + A[3]) / (den + B[3]);
    } else if y <= 4.0 {
        let mut num = C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + C[i]) * y;
            den = (den + D[i]) * y;
        }
        let r = (num + C[7]) / (den + D[7]);
        exp_neg_xsq(y) * r
    } else if y < XBIG {
        // Asymptotic region: erfc(y) = e^{-y^2}/y * (1/sqrt(pi) - R(1/y^2)).
        let ysq = 1.0 / (y * y);
        let mut num = P[5] * ysq;
        let mut den = ysq;
        for i in 0..4 {
            num = (num + P[i]) * ysq;
            den = (den + Q[i]) * ysq;
        }
        let r = ysq * (num + P[4]) / (den + Q[4]);
        exp_neg_xsq(y) * ((SQRPI - r) / y)
    } else {
        0.0
    };
    if x < 0.0 {
        2.0 - result
    } else {
        result
    }
}

/// `exp(-y^2)` computed as `exp(-hi^2) * exp(-(y-hi)(y+hi))` with `hi` a
/// 4-bit truncation of `y`; splitting the argument avoids the rounding-error
/// amplification of squaring `y` directly (Cody's trick).
fn exp_neg_xsq(y: f64) -> f64 {
    let hi = (y * 16.0).trunc() / 16.0;
    let del = (y - hi) * (y + hi);
    (-hi * hi).exp() * (-del).exp()
}

/// Standard normal CDF `N(x) = 0.5 * erfc(-x / sqrt(2))`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal density `phi(x) = exp(-x^2 / 2) / sqrt(2 pi)`.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_reference_values() {
        // Reference values from mpmath (50 digits, rounded to double).
        let cases = [
            (0.0, 1.0),
            (0.1, 0.887_537_083_981_715_2),
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 0.004_677_734_981_047_265),
            (3.0, 2.209_049_699_858_543_8e-5),
            (5.0, 1.537_459_794_428_035_1e-12),
            (10.0, 2.088_487_583_762_545e-45),
            (-1.0, 1.842_700_792_949_715),
            (-3.0, 1.999_977_909_503_001_5),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(1.0),
                "erfc({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn norm_cdf_symmetry_and_tails() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
        for x in [-3.0, -1.5, -0.3, 0.4, 2.2] {
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-15);
        }
        // N(1.959963984540054) = 0.975 (the 95% two-sided quantile).
        assert!((norm_cdf(1.959_963_984_540_054) - 0.975).abs() < 1e-12);
        // Deep tail keeps relative precision.
        let deep = norm_cdf(-8.0);
        assert!((deep - 6.220_960_574_271_819e-16).abs() < 1e-27);
    }

    #[test]
    fn norm_pdf_values() {
        assert!((norm_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-16);
        assert!((norm_pdf(1.0) - 0.241_970_724_519_143_37).abs() < 1e-16);
        assert!((norm_pdf(-1.0) - norm_pdf(1.0)).abs() == 0.0);
    }
}
