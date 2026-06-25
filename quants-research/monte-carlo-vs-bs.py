import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pricing.black_scholes import call_price
from src.pricing.monte_carlo import monte_carlo_call_price


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

bs_price = call_price(S, K, T, r, sigma)
mc_price = monte_carlo_call_price(S, K, T, r, sigma)

difference = mc_price - bs_price

print("\n" + "=" * 60)
print("BLACK-SCHOLES VS MONTE CARLO COMPARISON")
print("=" * 60)
print(f"Black-Scholes Price : {bs_price:.4f}")
print(f"Monte Carlo Price   : {mc_price:.4f}")
print(f"Difference          : {difference:.4f}")
print("=" * 60)