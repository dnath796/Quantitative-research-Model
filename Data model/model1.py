import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

# Replace with your CSV file name
df = pd.read_csv("/Users/deepikanath/dnath796/Quantitative-research-Model/Nat_Gas.csv")

# Assume CSV columns:
# Date, Price

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')


# ---------------------------------------------------
# VISUALIZE DATA
# ---------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], marker='o')
plt.title("Natural Gas Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

# Numeric time index
df['TimeIndex'] = np.arange(len(df))

# Month for seasonality
df['Month'] = df['Date'].dt.month

# Average monthly seasonal component
monthly_avg = df.groupby('Month')['Price'].mean()

# ---------------------------------------------------
# TREND MODEL
# ---------------------------------------------------

X = df[['TimeIndex']]
y = df['Price']

trend_model = LinearRegression()
trend_model.fit(X, y)

# Trend prediction
df['Trend'] = trend_model.predict(X)

# Seasonal adjustment
df['Seasonality'] = df.apply(
    lambda row: monthly_avg[row['Month']] - monthly_avg.mean(),
    axis=1
)

# Final fitted price
df['FittedPrice'] = df['Trend'] + df['Seasonality']

# ---------------------------------------------------
# INTERPOLATION FOR HISTORICAL DATES
# ---------------------------------------------------

date_ordinal = df['Date'].map(datetime.toordinal)

interp_function = interp1d(
    date_ordinal,
    df['Price'],
    kind='linear',
    fill_value='extrapolate'
)

# ---------------------------------------------------
# PRICE ESTIMATION FUNCTION
# ---------------------------------------------------

def estimate_price(input_date):

    input_date = pd.to_datetime(input_date)

    # Historical interpolation
    if input_date <= df['Date'].max():

        estimated_price = float(
            interp_function(input_date.toordinal())
        )

    # Future extrapolation
    else:

        months_ahead = (
            (input_date.year - df['Date'].min().year) * 12
            + input_date.month
            - df['Date'].min().month
        )

        trend_price = trend_model.predict([[months_ahead]])[0]

        seasonal_component = (
            monthly_avg[input_date.month]
            - monthly_avg.mean()
        )

        estimated_price = trend_price + seasonal_component

    return round(float(estimated_price), 2)

# ---------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------

test_dates = [
    "2022-06-15",
    "2024-12-31",
    "2025-06-30"
]

for d in test_dates:
    print(f"{d} --> Estimated Price: {estimate_price(d)}")

# ---------------------------------------------------
# FUTURE FORECAST VISUALIZATION
# ---------------------------------------------------

future_dates = pd.date_range(
    start=df['Date'].max(),
    periods=12,
    freq='M'
)

future_prices = []

for future_date in future_dates:
    future_prices.append(
        estimate_price(future_date)
    )

plt.figure(figsize=(12, 6))

plt.plot(
    df['Date'],
    df['Price'],
    label='Historical Prices',
    marker='o'
)

plt.plot(
    future_dates,
    future_prices,
    label='Forecast Prices',
    linestyle='--',
    marker='x'
)

plt.title("Natural Gas Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.show()