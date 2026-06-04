import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                          
import matplotlib.pyplot as plt
from datetime import date, timedelta

date_time = ["10-2020", "11-2020", "12-2020"]
date_time = pd.to_datetime(date_time, format='%m-%Y')
data = [1, 2, 3]

df = pd.read_csv('Nat_Gas.csv')
df.columns = ['Date', 'Price']
df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')   
prices = df['Price'].values
dates  = df['Date'].values

fig, ax = plt.subplots()
ax.plot(dates, prices, '-')                    
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('nat_gas_raw.png', dpi=150, bbox_inches='tight')   
plt.close()

# From the plot we can see the prices have a natural frequency of around a year but trend upwards.
# We do a linear regression to get the trend, then fit a sin function to the intra-year variation.

# Build monthly end-of-month dates in terms of days from start
start_date = date(2020, 10, 31)
end_date   = date(2024, 9,  30)
months = []
year   = start_date.year
month  = start_date.month + 1

while True:
    current = date(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1

days_from_start = [(d - start_date).days for d in months]

# ── Simple linear regression: y = Ax + B ─────────────────────────────────────
def simple_regression(x, y):
    xbar      = np.mean(x)
    ybar      = np.mean(y)
    slope     = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

time             = np.array(days_from_start)
slope, intercept = simple_regression(time, prices)

# Plot linear trend
plt.figure()
plt.plot(time, prices, label='Prices')
plt.plot(time, time * slope + intercept, label='Linear trend')
plt.xlabel('Days from start date')
plt.ylabel('Price')
plt.title('Linear Trend of Monthly Input Prices')
plt.legend()
plt.savefig('nat_gas_trend.png', dpi=150, bbox_inches='tight')  # FIX: save instead of show
plt.close()
print(f"Slope: {slope:.6f}   Intercept: {intercept:.4f}")

# ── Bilinear regression to fit seasonal sine curve ───────────────────────────
# Model: y = A*sin(kt + z) with k = 2π/365
# Rewrite as: y = A*cos(z)*sin(kt) + A*sin(z)*cos(kt)
# → bilinear regression solves for u = A*cos(z), w = A*sin(z)

sin_prices = prices - (time * slope + intercept)
sin_time   = np.sin(time * 2 * np.pi / 365)
cos_time   = np.cos(time * 2 * np.pi / 365)

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)

# Recover amplitude A and phase shift z
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift     = np.arctan2(slope2, slope1)

# Plot smoothed seasonal component
plt.figure()
plt.plot(time, amplitude * np.sin(time * 2 * np.pi / 365 + shift), label='Sine fit')
plt.plot(time, sin_prices, label='De-trended prices')
plt.title('Smoothed Estimate of Monthly Input Prices')
plt.xlabel('Days from start date')
plt.ylabel('Price residual')
plt.legend()
plt.savefig('nat_gas_seasonal.png', dpi=150, bbox_inches='tight')  # FIX: save instead of show
plt.close()

# ── Interpolation / extrapolation function ────────────────────────────────────
def interpolate(query_date):
    """
    Return estimated gas price for any date.

    - Exact month-end dates in the dataset: return the observed price directly.
    - All other dates: evaluate the fitted trend + seasonal sine model.

    Parameters
    ----------
    query_date : str | datetime | pd.Timestamp
        The date for which a price estimate is required.

    Returns
    -------
    float : estimated price ($/MMBtu)
    """
    query_date = pd.Timestamp(query_date)
    days       = (query_date - pd.Timestamp(start_date)).days

    if days in days_from_start:
        return prices[days_from_start.index(days)]
    else:
        return (amplitude * np.sin(days * 2 * np.pi / 365 + shift)
                + days * slope + intercept)

# ── Full visualisation: raw + smoothed + sine fit ────────────────────────────
continuous_dates = pd.date_range(
    start=pd.Timestamp(start_date),
    end=pd.Timestamp(end_date),
    freq='D'
)

fit_amplitude    = np.sqrt(slope1 ** 2 + slope2 ** 2)
fit_shift        = np.arctan2(slope2, slope1)
fit_slope, fit_intercept = simple_regression(
    time,
    prices - fit_amplitude * np.sin(time * 2 * np.pi / 365 + fit_shift)
)

plt.figure(figsize=(12, 5))
plt.plot(
    continuous_dates,
    [interpolate(d) for d in continuous_dates],
    label='Smoothed Estimate'
)
plt.plot(dates, prices, 'o', label='Monthly Input Prices')
plt.plot(
    continuous_dates,
    fit_amplitude * np.sin(
        (continuous_dates - pd.Timestamp(start_date)).days * 2 * np.pi / 365 + fit_shift
    ) + (continuous_dates - pd.Timestamp(start_date)).days * fit_slope + fit_intercept,
    label='Fit to Sine Curve'
)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices')
plt.legend()
plt.tight_layout()
plt.savefig('nat_gas_final.png', dpi=150, bbox_inches='tight')   # FIX: save instead of show
plt.close()

# ── Sample outputs ────────────────────────────────────────────────────────────
print("\nSample price estimates:")
for d in ["2022-06-15", "2024-09-30", "2025-03-31", "2025-09-30"]:
    print(f"  {d}  -->  ${interpolate(d):.2f}")

print("\nAll charts saved: nat_gas_raw.png, nat_gas_trend.png, nat_gas_seasonal.png, nat_gas_final.png")
