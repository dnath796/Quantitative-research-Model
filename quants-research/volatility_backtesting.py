import yfinance as yf
import numpy as np
import pandas as pd


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

returns = np.log(close / close.shift(1))

realized_vol = returns.rolling(21).std() * np.sqrt(252)
forecast_vol = realized_vol.shift(1)

df = pd.DataFrame(
    {
        "Return": returns,
        "Realized Vol": realized_vol,
        "Forecast Vol": forecast_vol,
    }
).dropna()

df["Signal"] = np.where(
    df["Forecast Vol"] > df["Realized Vol"],
    "BUY VOL",
    "SELL VOL",
)

df["Next Day Abs Return"] = df["Return"].shift(-1).abs()

df["Median Abs Return"] = df["Next Day Abs Return"].rolling(63).median()

df["Correct"] = np.where(
    (
        (df["Signal"] == "BUY VOL")
        & (df["Next Day Abs Return"] > df["Median Abs Return"])
    )
    | (
        (df["Signal"] == "SELL VOL")
        & (df["Next Day Abs Return"] <= df["Median Abs Return"])
    ),
    1,
    0,
)

df = df.dropna()

win_rate = df["Correct"].mean()
num_trades = len(df)

print("\n" + "=" * 70)
print("VOLATILITY STRATEGY BACKTEST")
print("=" * 70)
print(f"Ticker     : {ticker}")
print(f"Trades     : {num_trades}")
print(f"Win Rate   : {win_rate:.2%}")
print("=" * 70)