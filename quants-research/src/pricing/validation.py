from black_scholes import call_price, call_delta

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

h = 0.01

price_up = call_price(S + h, K, T, r, sigma)
price_down = call_price(S - h, K, T, r, sigma)
numerical_delta = (price_up - price_down) / (2 * h)  #central difference method for numerical approximation
analytic_delta = call_delta(S, K, T, r, sigma)

print(f"Numerical Delta: {numerical_delta:.6f}")
print(f"Analytic Delta: {analytic_delta:.6f}")