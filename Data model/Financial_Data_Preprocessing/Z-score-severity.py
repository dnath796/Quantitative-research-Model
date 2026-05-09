import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


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



# 1. Calculate Z-Scores to rank severity
df_outliers['z_score'] = (df_outliers['simple_rtn'] - df_outliers['mean']) / df_outliers['std']

# 2. Setup Plot
plt.figure(figsize=(12, 6))

# Plot the actual distribution of returns
sns.histplot(df_outliers['simple_rtn'], kde=False, element="step", color="gray", 
             alpha=0.3, stat="density", label='Actual Distribution')

# 3. Overlay the theoretical Normal Distribution (Bell Curve)
mu = df_outliers['simple_rtn'].mean()
sigma = df_outliers['simple_rtn'].std()
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='blue', lw=2, label='Normal Distribution')

# 4. Highlight where the 3-Sigma outliers live
plt.axvline(mu + 3*sigma, color='red', linestyle='--', alpha=0.6, label='3-Sigma Threshold')
plt.axvline(mu - 3*sigma, color='red', linestyle='--', alpha=0.6)

plt.title("Distribution of MSFT Returns vs. Normal Distribution")
plt.xlabel("Simple Return")
plt.legend()
plt.show()

# 5. Print the "Most Severe" Outliers
print("Top 5 Most Severe Outliers (by Z-Score):")
print(df_outliers.sort_values(by='z_score', key=abs, ascending=False)[['simple_rtn', 'z_score']].head())


print(f"Kurtosis: {df_outliers['simple_rtn'].kurtosis()}") 
# A value > 0 means your data has fatter tails than a Normal Distribution.
print(f"Skewness: {df_outliers['simple_rtn'].skew()}")
# A value > 0 means your data is skewed to the right (more extreme positive returns). A value < 0 means it's skewed to the left (more extreme negative returns).



# Create the Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(df_outliers['simple_rtn'], dist="norm", plot=plt)

plt.title("Q-Q Plot: MSFT Returns vs. Normal Distribution")
plt.show()
