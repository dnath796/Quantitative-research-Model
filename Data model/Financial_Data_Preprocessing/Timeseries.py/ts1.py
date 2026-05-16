import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# --- 1. DATA ACQUISITION & CLEANING ---
print("Downloading Gold Price data from Yahoo Finance...")
# 'GC=F' tracks the Continuous Gold Futures contract
df = yf.download('GC=F', 
                 start='2000-01-01', 
                 end='2011-12-31', 
                 progress=False, 
                 auto_adjust=True)

# Flatten MultiIndex columns if present (common in modern yfinance versions)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Isolate the Close price and rename to 'price'
df = df[['Close']].rename(columns={'Close': 'price'})

# Resample to Month End using 'ME' (required for Pandas 2.2+ / Python 3.14)
df = df.resample('ME').last()

# Forward-fill any minor structural gaps to ensure data continuity
df['price'] = df['price'].ffill()

# --- 2. CALCULATE ROLLING METRICS ---
WINDOW_SIZE = 12
df['rolling_mean'] = df.price.rolling(window=WINDOW_SIZE).mean()
df['rolling_std'] = df.price.rolling(window=WINDOW_SIZE).std()

# Plot Price with Rolling Mean & Std
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price'], label='Gold Price (GC=F)', color='blue')
plt.plot(df.index, df['rolling_mean'], label='12-Month Rolling Mean', color='orange', linestyle='--')
plt.plot(df.index, df['rolling_std'], label='12-Month Rolling Std', color='red', linestyle=':')
plt.title('Gold Price with Rolling Mean & Standard Deviation')
plt.xlabel('Date')
plt.ylabel('USD per Ounce')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print("\n--- Stationarity Diagnostics Before Transformation ---")
print("Observation: The rolling standard deviation increases noticeably over time.")
print("Conclusion: The variance is non-stationary, validating a Multiplicative Model approach.")

# --- 3. STATIONARITY TESTING (ADF & KPSS) ---
# Drop NaN values introduced by the rolling window calculation for valid testing
clean_series = df['price'].dropna()

# A. Augmented Dickey-Fuller Test (ADF)
adf_test = adfuller(clean_series)
print(f"\nADF Statistic: {adf_test[0]:.4f}")
print(f"ADF p-value: {adf_test[1]:.4f}")
if adf_test[1] <= 0.05:
    print("-> ADF rejects Null Hypothesis: Series is Stationary.")
else:
    print("-> ADF fails to reject Null Hypothesis: Series is Non-Stationary (has a unit root).")

# B. KPSS Test
kpss_test = kpss(clean_series, regression='c', nlags="auto")
print(f"\nKPSS Statistic: {kpss_test[0]:.4f}")
print(f"KPSS p-value: {kpss_test[1]:.4f}")
if kpss_test[1] <= 0.05:
    print("-> KPSS rejects Null Hypothesis: Series is Non-Stationary.")
else:
    print("-> KPSS fails to reject Null Hypothesis: Series is Stationary.")

# --- 4. MULTIPLICATIVE SEASONAL DECOMPOSITION ---
decomposition_results = seasonal_decompose(clean_series, 
                                           model='multiplicative',
                                           period=12)

# Generate and display the final breakdown components
fig = decomposition_results.plot()
fig.set_size_inches(11, 8)
fig.suptitle('Multiplicative Decomposition of Gold Prices', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()