import pandas as pd
import numpy as np
import scipy.stats as scs
import yfinance as yf
import matplotlib.pyplot as plt   
import seaborn as sns  

df = yf.download('AAPL', start ='2000-01-01',end='2010-12-31',auto_adjust= False)
print(df)   
df.columns = df.columns.get_level_values(0)

# 3. Now this line will work perfectly
df= df[['Adj Close']]

df_adj = df.rename(columns={'Adj Close': 'adj_close'})

df['simple_rtn'] = df.pct_change()
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
# Fact Check: Ensure NaNs are dropped for accurate math
clean_rtn = df['log_rtn'].dropna()

# 1. Normality Fact (Jarque-Bera)
# Tests if Skewness=0 and Kurtosis=3. P-value < 0.05 means NOT normal.
jb_stat, p_val = scs.jarque_bera(clean_rtn)

# 2. Symmetry Fact (Skewness)
# Negative = longer left tail (more frequent small losses).
skew = clean_rtn.skew()

# 3. Tail Fact (Excess Kurtosis)
# Value > 0 means "Fat Tails" (extreme events happen more than expected).
kurt = clean_rtn.kurtosis()

# 4. Volatility Fact (Annualized)
# Standard deviation scaled to a 252-day trading year.
ann_vol = clean_rtn.std() * np.sqrt(252)

# 5. Peak-to-Trough Fact (Max Drawdown)
# The largest cumulative loss from a peak to a following bottom.
cum_rtn = clean_rtn.cumsum()
peak = cum_rtn.cummax()
drawdown = (cum_rtn - peak).min()

# 6. Risk-Adjusted Fact (Sharpe Ratio)
# Ratio of return to risk (assuming 0% risk-free rate for simplicity).
sharpe = (clean_rtn.mean() / clean_rtn.std()) * np.sqrt(252)

# Displaying all facts
print(f"--- 6 Quantitative Facts ---")
print(f"1. Jarque-Bera P-Value: {p_val:.4f} (Is it Normal? {p_val > 0.05})")
print(f"2. Skewness:            {skew:.4f}")
print(f"3. Excess Kurtosis:     {kurt:.4f}")
print(f"4. Annual Volatility:   {ann_vol:.2%}")
print(f"5. Max Drawdown:        {drawdown:.2%}")
print(f"6. Annualized Sharpe:   {sharpe:.2f}")


