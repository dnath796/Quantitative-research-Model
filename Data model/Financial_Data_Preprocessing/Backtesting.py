import backtrader as bt
import yfinance as yf

# 1. Download the data manually using yfinance
data_df = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# 2. Feed that dataframe into Backtrader
data = bt.feeds.PandasData(dataname=data_df)

cerebro = bt.Cerebro()
cerebro.adddata(data)




# Import the libraries:
from datetime import datetime
import backtrader as bt
# Define a class representing the trading strategy:
class SmaSignal(bt.Signal):
    params = (('period', 20), )
    
    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)


# Download data from Yahoo Finance:

data_df = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

data_df.columns = data_df.columns.get_level_values(0) 

# Feed that dataframe into Backtrader
data = bt.feeds.PandasData(dataname=data_df)
# Set up the backtest:
cerebro = bt.Cerebro(stdstats = False)

cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
# Run the backtest:
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
# Plot the results:
cerebro.plot(iplot=True, volume=False)
