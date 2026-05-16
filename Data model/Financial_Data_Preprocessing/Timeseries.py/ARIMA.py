import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- 1. FETCH AND CLEAN DATA ---
print("Downloading Gold Price data from Yahoo Finance...")
df = yf.download('GC=F', start='2000-01-01', end='2011-12-31', progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Close']].rename(columns={'Close': 'price'}).resample('ME').last().ffill()

# --- 2. STATIONARITY & DIAGNOSTICS ---
# We already determined from the ADF test that raw Gold price is non-stationary.
# Let's perform 1st order differencing (d=1) to see the stationary data.
df['price_diff'] = df['price'].diff()

# Optional: You can plot ACF/PACF to inspect the patterns visually
# fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# plot_acf(df['price_diff'].dropna(), ax=ax[0])
# plot_pacf(df['price_diff'].dropna(), ax=ax[1])
# plt.show()

# --- 3. FIT THE ARIMA MODEL ---
# For demonstration, we will fit an ARIMA(1, 1, 1) model.
# (p=1: look 1 month back, d=1: apply 1 difference, q=1: look at 1 error lag)
model = SARIMAX(
    df['price'], 
    order=(1, 1, 1), 
    trend='c' # 'c' includes a constant/drift component to handle long-term upward trajectory
)
fitted_model = model.fit(disp=False)

# --- 4. FORECAST FUTURE HORIZONS ---
forecast_horizons = 24
forecast_output = fitted_model.get_forecast(steps=forecast_horizons)

# Extract the predicted mean and the confidence intervals
forecast_mean = forecast_output.predicted_mean
confidence_intervals = forecast_output.conf_int()

# --- 5. PLOT THE RESULTS ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price'], label='Historical Actual Price', color='blue')
plt.plot(fitted_model.fittedvalues.index[1:], fitted_model.fittedvalues[1:], label='In-Sample Fitted Model', color='orange', linestyle='--')

# Plot the Forecast Mean
plt.plot(forecast_mean.index, forecast_mean, label='24-Month ARIMA Forecast', color='red', linewidth=2)

# Plot Confidence Intervals (Shadowed area representing uncertainty)
plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:, 0],
    confidence_intervals.iloc[:, 1],
    color='pink',
    alpha=0.4,
    label='95% Confidence Interval'
)

plt.title('Gold Price Modeling with ARIMA(1, 1, 1)')
plt.xlabel('Date')
plt.ylabel('USD per Ounce')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Display Diagnostic Summary
print(fitted_model.summary())