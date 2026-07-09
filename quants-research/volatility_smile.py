import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pricing.implied_volatility import (
    implied_volatility,
)

S = 100
T = 1
r = 0.05

market_data = [
    (80, 24.59),
    (90, 16.70),
    (100, 10.45),
    (110, 6.04),
    (120, 3.25),
]

strikes = []
ivs = []

for K, market_price in market_data:
    iv = implied_volatility(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type="call",
    )

    strikes.append(K)
    ivs.append(iv)

    print(
        f"Strike={K}, IV={iv:.4f}"
    )

plt.figure(figsize=(10, 6))

plt.plot(
    strikes,
    ivs,
    marker="o",
)

plt.title("Volatility Smile")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")

plt.grid(True)

output_dir = Path("research/figures")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    output_dir / "volatility_smile.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\n" + "=" * 50)
print("VOLATILITY SMILE RESEARCH SUMMARY")
print("=" * 50)

max_iv = max(ivs)
min_iv = min(ivs)

max_strike = strikes[ivs.index(max_iv)]
min_strike = strikes[ivs.index(min_iv)]

print(f"Underlying Price : {S}")
print(f"Highest IV       : {max_iv:.4f}")
print(f"Highest IV Strike: {max_strike}")
print(f"Lowest IV        : {min_iv:.4f}")
print(f"Lowest IV Strike : {min_strike}")

print("\nObservation:")

if max_strike != 100:
    print(
        "Implied volatility increases away from ATM, "
        "indicating a volatility smile."
    )
else:
    print(
        "Maximum implied volatility occurs near ATM."
    )

print("=" * 50)