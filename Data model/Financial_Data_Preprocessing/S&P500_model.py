import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# 1. Download data
df= yf.download('^GSPC', start ='2000-01-01',end='2010-12-31',auto_adjust= False)
df.columns = df.columns.get_level_values(0)

df= df[['Adj Close']]

df_adj = df.rename(columns={'Adj Close': 'adj_close'})

df['simple_rtn'] = df.pct_change()
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
# 1. Setup the math
# Note: Ensure log_rtn has no NaNs
clean_returns = df['log_rtn'].dropna()
r_range = np.linspace(clean_returns.min(), clean_returns.max(), num=1000)
mu = clean_returns.mean()
sigma = clean_returns.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()

# 2. Join and clean
df_outliers = df.join(df_rolling).dropna()

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

print(f"Kurtosis: {df_outliers['simple_rtn'].kurtosis()}") 
# A value > 0 means your data has fatter tails than a Normal Distribution.
print(f"Skewness: {df_outliers['simple_rtn'].skew()}")
# A value > 0 means your data is skewed to the right (more extreme positive returns than negative ones).

# 2. Plotting
fig, ax = plt.subplots(1, 2, figsize=(16, 8))



# Histogram - Replacing deprecated distplot with histplot
sns.histplot(clean_returns, kde=False, stat="density", ax=ax[0], color='gray', alpha=0.3)
ax[0].plot(r_range, norm_pdf, 'g', lw=2, label=f'Normal N({mu:.4f}, {sigma**2:.4f})')
ax[0].set_title('Distribution of S&P 500 returns', fontsize=16)
ax[0].legend(loc='upper left')

# Q-Q plot - Using statsmodels
sm.qqplot(clean_returns, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize=16)


plt.tight_layout()
plt.show()

