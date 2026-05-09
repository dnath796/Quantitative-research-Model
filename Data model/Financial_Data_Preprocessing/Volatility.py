import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import cufflinks as cf
from plotly.offline import init_notebook_mode

# 1. Initialize the offline mode for Notebooks
init_notebook_mode(connected=True)

# 2. Configure Cufflinks to work offline
cf.go_offline()

# 3. Set the theme (optional but looks great)
cf.set_config_file(theme='pearl')


def realized_volatility(x):
    # Ensure we only sum the squared returns, skipping non-numeric columns
    return np.sqrt(np.sum(x**2))


df = yf.download('MSFT', start ='2000-01-01',end='2010-12-31',auto_adjust= False)
print(df)   
df.columns = df.columns.get_level_values(0)

# 3. Now this line will work perfectly
df= df[['Adj Close']]

df_adj = df.rename(columns={'Adj Close': 'adj_close'})

df['simple_rtn'] = df.pct_change()
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

print(df)   

# 1. Calculate RV and ensure it stays as a DataFrame
# 'ME' is the updated frequency for Month End in newer pandas versions
df_rv = df[['log_rtn']].groupby(pd.Grouper(freq='ME')).apply(realized_volatility)

# Use name= instead of columns=
df_rv = df_rv.rename("rv").to_frame() 

# 3. Annualize the monthly volatility (sqrt of 12 months)
df_rv['rv'] = df_rv['rv'] * np.sqrt(12)



# Plotting the log returns and the realized volatility

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot the raw log returns
ax[0].plot(df.index, df['log_rtn'], label='Log Returns', color='blue', alpha=0.7)
ax[0].set_title('Daily Log Returns')
ax[0].legend()

# Plot the monthly realized volatility
ax[1].plot(df_rv.index, df_rv['rv'], label='Annualized Volatility (Monthly)', color='red', marker='o')
ax[1].set_title('Monthly Realized Volatility (Annualized)')
ax[1].legend()

plt.tight_layout()
# plt.show()


#  Increase figsize for better visibility with 3 subplots

fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# 1. Price Time Series
df['Adj Close'].plot(ax=ax[0], color='tab:blue')
ax[0].set_title('MSFT Price & Returns Analysis', fontsize=16)
ax[0].set_ylabel('Stock Price ($)')

# 2. Simple Returns
df['simple_rtn'].plot(ax=ax[1], color='tab:orange', alpha=0.8)
ax[1].set_ylabel('Simple Returns (%)')

# 3. Log Returns
df['log_rtn'].plot(ax=ax[2], color='tab:green', alpha=0.8)
ax[2].set_ylabel('Log Returns (%)')
ax[2].set_xlabel('Date')

# Adjust layout to prevent label clipping
plt.tight_layout()
plt.show()


# Create the figure


import plotly.express as px
import plotly.io as pio

# 1. Create the figure
fig = px.line(
    df.reset_index(), 
    x='Date', 
    y=['Adj Close', 'simple_rtn', 'log_rtn'], 
    facet_row='variable', 
    title="MSFT Analysis"
)

# 2. THE CRITICAL FIX: Unlock the Y-axes
fig.update_yaxes(matches=None)

# 3. Optional: Make the returns plots clearer by centering them at zero
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')

pio.renderers.default = "browser"

fig.show()


df.log_rtn.plot(title='Daily MSFT returns-volatility clustering', figsize=(12, 6))
plt.show()