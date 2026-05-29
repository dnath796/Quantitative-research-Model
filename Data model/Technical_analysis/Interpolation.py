import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# LOAD DATA

df = pd.read_csv("Nat_Gas.csv")

df.columns  = ['Date', 'Price']
df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y') 
df['Price'] = df['Price'].astype(float)
df = df.sort_values('Date').reset_index(drop=True)

# VISUALIZE RAW DATA

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], marker='o', color='steelblue', label='Observed Prices')
plt.title("Natural Gas Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($/MMBtu)")
plt.grid(True)
plt.legend()
plt.savefig("nat_gas_historical.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: nat_gas_historical.png")

# FEATURE ENGINEERING

df['TimeIndex'] = np.arange(len(df))    
df['Month']     = df['Date'].dt.month

# Average price per calendar month — captures seasonal pattern
monthly_avg = df.groupby('Month')['Price'].mean()


# TREND MODEL (Linear Regression)

X = df[['TimeIndex']]
y = df['Price']

trend_model = LinearRegression()
trend_model.fit(X, y)

df['Trend']       = trend_model.predict(X)
df['Seasonality'] = df['Month'].map(monthly_avg) - monthly_avg.mean()
df['FittedPrice'] = df['Trend'] + df['Seasonality']


# INTERPOLATION SETUP (for historical date queries)

# Convert dates to ordinal integers so interp1d can work on them
date_ordinals = df['Date'].map(lambda d: d.toordinal())

interp_function = interp1d(
    date_ordinals,
    df['Price'],
    kind='linear',
    fill_value='extrapolate'
)


# PRICE ESTIMATION FUNCTION
def estimate_price(input_date):
    """
    Returns an estimated natural gas purchase price for any given date.

    - Dates within the observed range (Oct 2020 – Sep 2024):
        Linear interpolation between the two nearest monthly data points.
    - Dates beyond the observed range:
        Linear trend + seasonal component extrapolation.

    Parameters
    ----------
    input_date : str | datetime
        Any parseable date string (e.g. "2025-06-30") or datetime object.

    Returns
    -------
    float
        Estimated price in $/MMBtu, rounded to 2 decimal places.
    """
    input_date = pd.to_datetime(input_date)

    if input_date <= df['Date'].max():
        # Historical: interpolate
        price = float(interp_function(input_date.toordinal()))

    else:
        # Future: trend + seasonality
        # months_from_start maps directly onto the TimeIndex scale (1 step = 1 month)
        months_from_start = (
            (input_date.year  - df['Date'].min().year)  * 12
            + input_date.month - df['Date'].min().month
        )
        trend_value       = trend_model.predict(
                                pd.DataFrame({'TimeIndex': [months_from_start]})
                            )[0]
        seasonal_value    = monthly_avg[input_date.month] - monthly_avg.mean()
        price             = trend_value + seasonal_value

    return round(float(price), 2)

# EXAMPLE USAGE

test_dates = [
    "2021-03-15",   # historical, mid-month
    "2022-06-15",   # historical, mid-month
    "2024-09-30",   # last observed data point
    "2024-12-31",   # near-future
    "2025-06-30",   # ~9 months ahead
    "2025-09-30",   # 1 full year ahead
]

print("\nPrice estimates:")
for d in test_dates:
    print(f"  {d}  -->  ${estimate_price(d):.2f}")

# FORECAST — extrapolate 1 full year beyond last data point

# Start one month after the last observed date to avoid overlap
future_dates  = pd.date_range(
    start=df['Date'].max() + pd.DateOffset(months=1),
    periods=12,
    freq='MS'      
)
future_prices = [estimate_price(d) for d in future_dates]

plt.figure(figsize=(12, 6))
plt.plot(df['Date'],    df['Price'],   marker='o',  color='steelblue', label='Historical Prices')
plt.plot(future_dates, future_prices, marker='x',  color='orange',
         linestyle='--', label='Forecast (+1 year)')
plt.axvline(df['Date'].max(), color='grey', linestyle=':', linewidth=1, label='Forecast start')
plt.title("Natural Gas Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price ($/MMBtu)")
plt.legend()
plt.grid(True)
plt.savefig("nat_gas_forecast.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: nat_gas_forecast.png")
