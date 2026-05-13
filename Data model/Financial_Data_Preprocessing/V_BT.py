import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(ticker="GOOGL"):
    # 1. Download Data
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2026-01-01")
    df = data[['Close']].copy()

    # 2. Strategy Logic (Golden Cross)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # 3. Generate Signals (Vectorized)
    # 1 = Long, 0 = Cash
    df['Signal'] = np.where(df['SMA50'] > df['SMA200'], 1, 0)

    # 4. Shift Signals
    # IMPORTANT: We shift by 1 to trade on the NEXT day's open/close 
    # to avoid "Look-Ahead Bias" (using today's close to predict today's trade).
    df['Position'] = df['Signal'].shift(1)

    # 5. Calculate Returns
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Market_Returns'] * df['Position']

    # 6. Cumulative Growth
    df['Market_Cum'] = (1 + df['Market_Returns']).cumprod()
    df['Strategy_Cum'] = (1 + df['Strategy_Returns']).cumprod()

    # 7. Metrics
    total_ret = (df['Strategy_Cum'].iloc[-1] - 1) * 100
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    
    print(f"Total Strategy Return: {total_ret:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

    # 8. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['Market_Cum'], label='Buy & Hold GOOGL', alpha=0.7)
    plt.plot(df['Strategy_Cum'], label='SMA Strategy', linewidth=2)
    plt.title(f'Vectorized Backtest: {ticker} (2020 - 2026)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_backtest()