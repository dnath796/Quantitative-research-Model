import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf



df= yf.download('AAPL', start ='2000-01-01',end='2010-12-31',auto_adjust= False)
df.columns = df.columns.get_level_values(0)

# 3. Now this line will work perfectly
df= df[['Adj Close']]

df_adj = df.rename(columns={'Adj Close': 'adj_close'})

df['simple_rtn'] = df.pct_change()
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

# 1. Calculate rolling metrics
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()

# 2. Join and clean
df_outliers = df.join(df_rolling).dropna()

# 3. Improved outlier function
def identify_outliers(row, n_sigmas=3):
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    # Use the variable n_sigmas instead of hardcoded 3
    if (x > mu + n_sigmas * sigma) or (x < mu - n_sigmas * sigma):
        return 1
    return 0

# 4. Apply and extract
df_outliers['outlier'] = df_outliers.apply(identify_outliers, axis=1)
outliers = df_outliers.loc[df_outliers['outlier'] == 1]

# 5. Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_outliers.index, df_outliers.simple_rtn, color='gray', alpha=0.5, label='Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly', s=20)
ax.set_title("AAPL Returns: Anomaly Detection (3-Sigma Rolling Window)")
ax.legend(loc='lower right')
plt.show()




# 1. Calculate the bands
df_outliers['upper_band'] = df_outliers['mean'] + 3 * df_outliers['std']
df_outliers['lower_band'] = df_outliers['mean'] - 3 * df_outliers['std']

# 2. Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the returns and the rolling mean
ax.plot(df_outliers.index, df_outliers['simple_rtn'], color='gray', alpha=0.3, label='Daily Returns')
ax.plot(df_outliers.index, df_outliers['mean'], color='blue', label='Rolling Mean (21d)', alpha=0.8)

# Fill the area between the 3-sigma bands
ax.fill_between(df_outliers.index, df_outliers['lower_band'], df_outliers['upper_band'], 
                color='blue', alpha=0.1, label='3-Sigma Range')

# Highlight the outliers
ax.scatter(outliers.index, outliers['simple_rtn'], color='red', label='Anomaly', s=30, zorder=5)

ax.set_title("AAPL Returns with 3-Sigma Bollinger Bands")
ax.legend(loc='upper left')
plt.show()
