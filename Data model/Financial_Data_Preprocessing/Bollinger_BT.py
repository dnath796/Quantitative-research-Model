import yfinance as yf
import pandas as pd
import numpy as np

# 1. Download MSFT data
df = yf.download("MSFT", start="2017-12-01", end="2018-12-31")

# FIX: If yfinance returns a MultiIndex (Price/Ticker), flatten it
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Calculate Bollinger Bands
window = 20
k = 2
df['SMA'] = df['Close'].rolling(window=window).mean()
df['STD'] = df['Close'].rolling(window=window).std()
df['Upper'] = df['SMA'] + (k * df['STD'])
df['Lower'] = df['SMA'] - (k * df['STD'])

# Filter for 2018 and drop rows with NaN values from the moving average
df = df.loc['2018-01-01':].dropna().copy()

# 3. Strategy Logic
cash = 100000 
shares = 0
commission_rate = 0.001
portfolio_value = []

for i in range(len(df)):
    # We use .values[i] to get the raw numpy scalar, bypassing pandas index issues
    current_price = df['Close'].values[i]
    lower_band = df['Lower'].values[i]
    upper_band = df['Upper'].values[i]
    
    if i > 0:
        prev_price = df['Close'].values[i-1]
    else:
        prev_price = current_price

    # Strategy signals
    is_buy_signal = (prev_price < lower_band) and (current_price > lower_band)
    is_sell_signal = (prev_price > upper_band) and (current_price < upper_band)
    
    # Execution logic
    if is_buy_signal and shares == 0:
        shares_to_buy = cash // (current_price * (1 + commission_rate))
        cost = shares_to_buy * current_price * (1 + commission_rate)
        shares += shares_to_buy
        cash -= cost
        print(f"BUY at {current_price:.2f} on {df.index[i].date()}")

    elif is_sell_signal and shares > 0:
        revenue = shares * current_price * (1 - commission_rate)
        cash += revenue
        print(f"SELL at {current_price:.2f} on {df.index[i].date()}")
        shares = 0
    
    portfolio_value.append(cash + (shares * current_price))

df['Portfolio'] = portfolio_value
final_val = df['Portfolio'].iloc[-1]
final_return = ((final_val - 100000) / 100000) * 100

print(f"---")
print(f"Final Portfolio Value: ${final_val:,.2f}")
print(f"Final Strategy Return: {final_return:.2f}%")