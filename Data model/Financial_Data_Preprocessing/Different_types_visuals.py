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


df = yf.download('AAPL', start ='2000-01-01',end='2010-12-31',auto_adjust= False)
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

# Melt the data so we have 'Date', 'Metric', and 'Value' columns
df_tidy = df.reset_index().melt(id_vars='Date', value_vars=['Adj Close', 'simple_rtn', 'log_rtn'], 
                                var_name='Metric', value_name='Value')

import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
for i, col in enumerate(['Adj Close', 'simple_rtn', 'log_rtn']):
    axes[i].plot(df.index, df[col])
    axes[i].set_ylabel(col)
axes[0].set_title('AAPL Analysis')
plt.tight_layout()
plt.savefig("matplotlib_AAPL_analysis.png")


# Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_tidy, x='Date', y='Value', hue='Metric') # Overlaid
# Or faceted:
sns.relplot(data=df_tidy, x='Date', y='Value', row='Metric', kind='line', facet_kws={'sharey': False})
plt.savefig("seaborn_AAPL_analysis.png")



import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Plotly Express (The modern way)
fig = px.line(df_tidy, x='Date', y='Value', facet_row='Metric', title="AAPL Analysis")
fig.update_yaxes(matches=None)
# pio.renderers.default = "browser"
# fig.show()
# fig.savefig("plotly_Express_msft_analysis.png")

# Plotly Graph Objects (The low-level way)
from plotly.subplots import make_subplots
fig = make_subplots(rows=3, cols=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['simple_rtn'], name='Simple Rtn'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['log_rtn'], name='Log Rtn'), row=3, col=1)
pio.renderers.default = "browser"
fig.show()


import altair as alt

alt.Chart(df_tidy).mark_line().encode(
    x='Date:T',
    y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
    row='Metric:N'
).properties(height=200, width=600).resolve_scale(y='independent')


from plotnine import ggplot, aes, geom_line, facet_wrap, theme_minimal

plot = (
    ggplot(df_tidy, aes(x='Date', y='Value')) 
    + geom_line(color='steelblue') 
    + facet_wrap('~Metric', ncol=1, scales='free_y')
    + theme_minimal()
)
plot.save("Plotnine_AAPL_analysis.png")
