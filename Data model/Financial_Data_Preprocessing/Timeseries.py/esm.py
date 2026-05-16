import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. FETCH AND CLEAN DATA ---
print("Downloading Gold Price data from Yahoo Finance...")
df = yf.download('GC=F', start='2000-01-01', end='2011-12-31', progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Isolate Close price, rename, resample to Month End ('ME'), and fill gaps
df = df[['Close']].rename(columns={'Close': 'price'}).resample('ME').last().ffill()

# --- 2. FIT THE HOLT-WINTERS MODEL ---
# Using additive trend and multiplicative seasonality due to widening variance
model = ExponentialSmoothing(
    df['price'], 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=12
)
fitted_model = model.fit()

# --- 3. FORECAST FUTURE HORIZONS ---
forecast_horizons = 24
forecast = fitted_model.forecast(steps=forecast_horizons)

# --- 4. PLOT THE RESULTS ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price'], label='Historical Actual Price', color='blue')

# FIXED: Changed df.fittedvalues to fitted_model.fittedvalues
plt.plot(fitted_model.fittedvalues.index, fitted_model.fittedvalues, label='In-Sample Fitted Model', color='orange', linestyle='--')

plt.plot(forecast.index, forecast, label='24-Month Forecast', color='red', linewidth=2)

plt.title('Gold Price Modeling with Holt-Winters Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('USD per Ounce')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Display Model Summary Parameters (Alpha, Beta, Gamma)
print(fitted_model.summary())