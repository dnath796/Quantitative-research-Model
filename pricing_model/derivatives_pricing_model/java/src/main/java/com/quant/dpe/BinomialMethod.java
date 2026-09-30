package com.quant.dpe;

/**
 * Binomial lattice parameterisation: {@link #CRR} (Cox-Ross-Rubinstein) or
 * {@link #JR} (Jarrow-Rudd equal-probability).
 *
 * <p>Canonical cross-language string spellings are {@code "crr"} and
 * {@code "jr"}; {@link #parse(String)} accepts them case-insensitively.</p>
 */
public enum BinomialMethod {
    /** Cox-Ross-Rubinstein: {@code u = exp(sigma sqrt(dt))}, {@code d = 1/u}. */
    CRR,
    /** Jarrow-Rudd: drift absorbed into node placement, {@code p = 1/2}. */
    JR;

    /**
     * Parses a lattice-method string case-insensitively.
     *
     * @param method {@code "crr"} or {@code "jr"} (any case)
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if the string is null or not a valid method
     */
    public static BinomialMethod parse(String method) {
        if (method != null) {
            String m = method.toLowerCase(java.util.Locale.ROOT);
            if (m.equals("crr")) {
                return CRR;
            }
            if (m.equals("jr")) {
                return JR;
            }
        }
        throw new IllegalArgumentException("method must be 'crr' or 'jr', got " + method);
    }
}
