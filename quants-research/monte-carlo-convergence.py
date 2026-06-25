import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pricing.black_scholes import call_price

from src.pricing.monte_carlo import monte_carlo_call_price


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

simulation_counts = [
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    250_000,
]

bs_price = call_price(S, K, T, r, sigma)

mc_prices = []

for n in simulation_counts:
    price = monte_carlo_call_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        num_simulations=n,
        seed=42,
    )

    mc_prices.append(price)

plt.figure(figsize=(10, 6))
plt.plot(simulation_counts, mc_prices, marker="o", label="Monte Carlo Price")
plt.axhline(bs_price, linestyle="--", label="Black-Scholes Price")
plt.xscale("log")
plt.title("Monte Carlo Convergence to Black-Scholes Price")
plt.xlabel("Number of Simulations")
plt.ylabel("Call Option Price")
plt.legend()
plt.grid(True)

output_dir = Path("research/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "monte_carlo_convergence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Black-Scholes Price: {bs_price:.4f}")
print(f"Final Monte Carlo Price: {mc_prices[-1]:.4f}")
print(f"Final Difference: {mc_prices[-1] - bs_price:.4f}")