from src.pricing.black_scholes import call_price, put_price, vega

def implied_volatility(market_price, S, K, T, r, option_type='call', initial_guess= 0.2, tolerance=1e-6, max_iterations=100):
    for _ in range(max_iterations):
        if option_type == 'call':
            model_price = call_price(S, K, T, r, initial_guess)
        elif option_type == 'put':
            model_price = put_price(S, K, T, r, initial_guess)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        price_diff = model_price - market_price

        if abs(price_diff) < tolerance:
            return initial_guess

        option_vega = vega(S, K, T, r, initial_guess)

        initial_guess = initial_guess - price_diff / option_vega

        if initial_guess <= 0:
            initial_guess = 1e-6
    raise ValueError("Implied volatility did not converge")