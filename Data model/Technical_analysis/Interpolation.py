import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from datetime import datetime

df = pd.read_csv("Nat_Gas.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], marker='o')
plt.title("Natural Gas Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

df['TimeIndex'] = np.arange(len(df))

df['Month'] = df['Date'].dt.month


monthly_avg = df.groupby('Month')['Price'].mean()

X = df[['TimeIndex']]
y = df['Price']

trend_model = LinearRegression()
trend_model.fit(X, y)

df['Trend'] = trend_model.predict(X)


df['Seasonality'] = df.apply(lambda row: monthly_avg[row['Month']] - monthly_avg.mean(),axis=1)
df['FittedPrice'] = df['Trend'] + df['Seasonality']

date_ordinal = df['Date'].map(datetime.toordinal)

interp_function = interp1d(
    date_ordinal,
    df['Price'],
    kind='linear',
    fill_value='extrapolate'
)

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