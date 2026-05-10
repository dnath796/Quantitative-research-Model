import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Download
df_raw = yf.download(['^GSPC', '^VIX'], start='1985-01-01', end='2018-12-31', progress=False)

# 2. THE ROBUST FIX: 
# This grabs the first 2 columns regardless of whether they are called 'Close' or 'Adj Close'
df = df_raw.iloc[:, :2].copy() 

# 3. Explicitly name them so the rest of your code works
# Note: yfinance downloads in alphabetical order, so VIX usually comes before GSPC
# Let's check the order or force it:
df.columns = ['sp500', 'vix'] if '^GSPC' in df_raw.columns.get_level_values(1)[0] else ['vix', 'sp500']

# 4. Now the calculations will work perfectly
df['log_rtn'] = np.log(df.sp500 / df.sp500.shift(1))
df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))

df.dropna(inplace=True)
print(df.head())


corr_coeff = df.log_rtn.corr(df.vol_rtn)

ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, 
                 line_kws={'color': 'red'})
ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})',
       ylabel='VIX log returns',
       xlabel='S&P 500 log returns')

# Create a joint plot with regression and hex bins (to see density)
# grid = sns.jointplot(x='log_rtn', y='vol_rtn', data=df, kind='reg', 
#                      joint_kws={'line_kws':{'color':'red'}},
#                      height=8, ratio=5, space=0.2)

# grid.fig.suptitle(f'S&P 500 vs. VIX Correlation: {corr_coeff:.2f}', y=1.02)
plt.show()

