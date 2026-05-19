import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima as pm_auto_arima

# Define the missing color palette variable
COLORS = sns.color_palette("Set2", 3)

# 1. Download full dataset split into Train and Test
# We need historical training data before 2019-01-01 to fit the models!
df_all = yf.download('GOOG', start='2017-01-01', end='2019-03-31', progress=False)

# Clean up column names for multi-index structural safety
if isinstance(df_all.columns, pd.MultiIndex):
    df_all.columns = df_all.columns.get_level_values(0)

# Resample weekly
df_weekly = df_all.resample('W').last().rename(columns={'Adj Close': 'adj_close'})['adj_close']

# Split train vs test
train = df_weekly.loc[:'2018-12-31']
test = df_weekly.loc['2019-01-01':'2019-03-31']
n_forecasts = len(test)

# --- FIX: Instantiate and fit the missing models ---
# Manual ARIMA(2,1,1) model
arima_model = ARIMA(train, order=(2, 1, 1))
arima = arima_model.fit()

# Auto ARIMA model (chooses optimal order automatically)
auto_arima = pm_auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, d=1, seasonal=False)


# 2. Obtain forecasts from the first model (Modern statsmodels API style)
arima_result = arima.get_forecast(steps=n_forecasts)
arima_pred = pd.DataFrame({
    'prediction': arima_result.predicted_mean.values,
    'ci_lower': arima_result.conf_int().iloc[:, 0].values,
    'ci_upper': arima_result.conf_int().iloc[:, 1].values
}, index=test.index)


# 3. Obtain forecasts from the second model (pmdarima)
auto_arima_output, auto_ci = auto_arima.predict(n_periods=n_forecasts, 
                                                return_conf_int=True, 
                                                alpha=0.05)
auto_arima_pred = pd.DataFrame({
    'prediction': auto_arima_output.values,
    'ci_lower': auto_ci[:, 0],
    'ci_upper': auto_ci[:, 1]
}, index=test.index)


# 4. Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual data
sns.lineplot(data=test, color=COLORS[0], label='Actual', ax=ax)

# Plot ARIMA(2,1,1)
ax.plot(arima_pred.index, arima_pred.prediction, c=COLORS[1], label='ARIMA(2,1,1)')
ax.fill_between(arima_pred.index,
                arima_pred.ci_lower,
                arima_pred.ci_upper,
                alpha=0.3, 
                facecolor=COLORS[1])

# Plot ARIMA(3,1,2)
ax.plot(auto_arima_pred.index, auto_arima_pred.prediction, c=COLORS[2], label='ARIMA(3,1,2)')
ax.fill_between(auto_arima_pred.index,
                auto_arima_pred.ci_lower,
                auto_arima_pred.ci_upper,
                alpha=0.2, 
                facecolor=COLORS[2])

ax.set(title="Google's stock price - actual vs. predicted", 
       xlabel='Date', 
       ylabel='Price ($)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()