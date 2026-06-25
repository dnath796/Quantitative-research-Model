import numpy as np


def monte_carlo_call_price(S, K, T, r, sigma, num_simulations=100_000, seed=42):
    np.random.seed(seed)

    z = np.random.standard_normal(num_simulations)

    final_prices = S * np.exp(
        (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z
    )

    payoffs = np.maximum(final_prices - K, 0)

    discounted_price = np.exp(-r * T) * np.mean(payoffs)

    return discounted_price