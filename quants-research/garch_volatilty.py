import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize


ticker = "AAPL"

data = yf.download(
    ticker,
    period="2y",
    auto_adjust=True,
    progress=False,
)

close = data["Close"]

if hasattr(close, "columns"):
    close = close.iloc[:, 0]

returns = np.log(close / close.shift(1)).dropna()
returns = returns * 100  # scale returns for numerical stability


def garch_log_likelihood(params, returns):
    omega, alpha, beta = params

    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    n = len(returns)
    variance = np.zeros(n)
    variance[0] = np.var(returns)

    for t in range(1, n):
        variance[t] = (
            omega
            + alpha * returns.iloc[t - 1] ** 2
            + beta * variance[t - 1]
        )

    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi)
        + np.log(variance)
        + returns.values**2 / variance
    )

    return -log_likelihood


initial_guess = [0.1, 0.05, 0.9]

result = minimize(
    garch_log_likelihood,
    initial_guess,
    args=(returns,),
    method="Nelder-Mead",
)

omega, alpha, beta = result.x

n = len(returns)
variance = np.zeros(n)
variance[0] = np.var(returns)

for t in range(1, n):
    variance[t] = (
        omega
        + alpha * returns.iloc[t - 1] ** 2
        + beta * variance[t - 1]
    )

annualized_volatility = np.sqrt(variance) / 100 * np.sqrt(252)

print("\n" + "=" * 60)
print("GARCH(1,1) VOLATILITY FORECAST REPORT")
print("=" * 60)
print(f"Ticker : {ticker}")
print(f"Omega  : {omega:.6f}")
print(f"Alpha  : {alpha:.6f}")
print(f"Beta   : {beta:.6f}")
print(f"Alpha + Beta: {alpha + beta:.6f}")
print(f"Latest GARCH Volatility Forecast: {annualized_volatility[-1]:.2%}")
print("=" * 60)

plt.figure(figsize=(12, 6))
plt.plot(returns.index, annualized_volatility)
plt.title("GARCH(1,1) Estimated Volatility")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.grid(True)

output_dir = Path("research/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "garch_volatility_forecast.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()