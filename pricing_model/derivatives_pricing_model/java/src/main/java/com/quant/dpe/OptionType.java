package com.quant.dpe;

/**
 * Vanilla option type: {@link #CALL} or {@link #PUT}.
 *
 * <p>The canonical cross-language string spellings are {@code "call"} and
 * {@code "put"}; {@link #parse(String)} accepts them case-insensitively and
 * rejects anything else, matching the contract of the Python reference
 * implementation.</p>
 */
public enum OptionType {
    /** Right to buy: payoff {@code max(S - K, 0)}. */
    CALL,
    /** Right to sell: payoff {@code max(K - S, 0)}. */
    PUT;

    /**
     * Parses an option-type string case-insensitively.
     *
     * @param optionType {@code "call"} or {@code "put"} (any case)
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if the string is null or not a valid type
     */
    public static OptionType parse(String optionType) {
        if (optionType != null) {
            String ot = optionType.toLowerCase(java.util.Locale.ROOT);
            if (ot.equals("call")) {
                return CALL;
            }
            if (ot.equals("put")) {
                return PUT;
            }
        }
        throw new IllegalArgumentException(
                "option_type must be 'call' or 'put', got " + optionType);
    }
}
