package com.quant.dpe;

/**
 * Exercise style of a vanilla option: {@link #EUROPEAN} (exercise only at
 * expiry) or {@link #AMERICAN} (exercise at any time up to expiry).
 *
 * <p>Canonical cross-language string spellings are {@code "european"} and
 * {@code "american"}; {@link #parse(String)} accepts them case-insensitively.</p>
 */
public enum ExerciseStyle {
    /** Exercise only at expiry. */
    EUROPEAN,
    /** Exercise at any node up to and including expiry. */
    AMERICAN;

    /**
     * Parses an exercise-style string case-insensitively.
     *
     * @param style {@code "european"} or {@code "american"} (any case)
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if the string is null or not a valid style
     */
    public static ExerciseStyle parse(String style) {
        if (style != null) {
            String st = style.toLowerCase(java.util.Locale.ROOT);
            if (st.equals("european")) {
                return EUROPEAN;
            }
            if (st.equals("american")) {
                return AMERICAN;
            }
        }
        throw new IllegalArgumentException(
                "style must be 'european' or 'american', got " + style);
    }
}
