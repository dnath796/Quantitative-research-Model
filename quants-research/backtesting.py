import yfinance as yf
import numpy as np
import pandas as pd


tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOG",
]

threshold = 0.10

all_returns = []

for ticker in tickers:

    try:

        data = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False,
        )

        close = data["Close"]

        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        returns = np.log(
            close / close.shift(1)
        )

        realized_vol = (
            returns.rolling(21).std()
            * np.sqrt(252)
        )

        forecast_vol = realized_vol.shift(1)

        iv_proxy = (
            realized_vol
            .rolling(63)
            .mean()
        )

        df = pd.DataFrame(
            {
                "Return": returns,
                "Forecast Vol": forecast_vol,
                "IV Proxy": iv_proxy,
            }
        ).dropna()

        df["Spread"] = (
            df["Forecast Vol"]
            - df["IV Proxy"]
        )

        df["Signal"] = np.where(
            df["Spread"] > threshold,
            1,
            np.where(
                df["Spread"] < -threshold,
                -1,
                0,
            ),
        )

        df["Strategy Return"] = (
            df["Signal"]
            * df["Return"].shift(-1)
        )

        df = df.dropna()

        all_returns.append(
            df["Strategy Return"]
            .reset_index(drop=True)
        )

        print(
            f"{ticker}: "
            f"{(df['Signal'] != 0).sum()} trades"
        )

    except Exception as e:

        print(
            f"Failed for {ticker}: {e}"
        )


portfolio = pd.concat(
    all_returns,
    axis=1,
)

portfolio.columns = tickers

portfolio["Portfolio Return"] = (
    portfolio.mean(axis=1)
)

portfolio_return = (
    portfolio["Portfolio Return"]
    .mean()
)

portfolio_vol = (
    portfolio["Portfolio Return"]
    .std()
)

sharpe = (
    portfolio_return
    / portfolio_vol
    * np.sqrt(252)
)

equity_curve = (
    1 + portfolio["Portfolio Return"]
).cumprod()

rolling_max = (
    equity_curve.cummax()
)

drawdown = (
    equity_curve
    - rolling_max
) / rolling_max

max_drawdown = (
    drawdown.min()
)

total_return = (
    equity_curve.iloc[-1] - 1
)

print("\n" + "=" * 80)
print("PORTFOLIO BACKTEST REPORT")
print("=" * 80)
print(f"Threshold       : {threshold:.0%}")
print(f"Total Return    : {total_return:.2%}")
print(f"Sharpe Ratio    : {sharpe:.2f}")
print(f"Max Drawdown    : {max_drawdown:.2%}")
print("=" * 80)   