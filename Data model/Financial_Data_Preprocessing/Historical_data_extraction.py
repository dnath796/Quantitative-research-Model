import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
 

df_a= yf.download('AAPL', start ='2000-01-01',end='2010-12-31',progress = False) 
df = yf.download('MSFT', start ='2000-01-01',end='2010-12-31',auto_adjust= False)

# We can pass a list of multiple tickers, such as ['AAPL', 'MSFT'].
# We can set auto_adjust=True to download only the adjusted prices.
# We can additionally download dividends and stock splits by setting actions='inline'.
# Setting progress=False disables the progress bar.
df.columns = df.columns.get_level_values(0)

# 3. Now this line will work perfectly
df= df[['Adj Close']]

# df_adj = df_adj.rename(columns={'Adj Close': 'adj_close'})

df['simple_rtn'] = df.pct_change()
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

# df_adj['simple_rtn'] = df_adj.adj_close.pct_change()
# df_adj['log_rtn'] = np.log(df_adj/df_adj.shift(1))
# print(df)



# # 1. Download raw stock data
# df = yf.download("AAPL", start="2010-01-01", end="2023-12-31")
# df.columns = df.columns.get_level_values(0) # Flatten if multi-index

# # 2. Convert index to a format the cpi library understands
# df['Date'] = df.index.date

# # 3. Apply inflation adjustment (adjusting to current dollars)
# def adjust_price(row):
#     return cpi.inflate(row['Close'], row['Date'])

# df['Real_Close'] = df.apply(adjust_price, axis=1)
print(df)



