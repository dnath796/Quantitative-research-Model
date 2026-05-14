import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime

# 1. Define the Signal Strategy
class RsiSignalStrategy(bt.SignalStrategy):
    params = dict(rsi_periods=14, rsi_upper=70, 
                  rsi_lower=30, rsi_mid=50)

    def __init__(self):
        # Standard Backtrader RSI
        rsi = bt.indicators.RSI(self.data, 
                                period=self.p.rsi_periods,
                                upperband=self.p.rsi_upper,
                                lowerband=self.p.rsi_lower)

        # Signal 1: LONG when RSI crosses 30 upwards
        rsi_signal_long = bt.ind.CrossUp(rsi, self.p.rsi_lower, plot=False)
        self.signal_add(bt.SIGNAL_LONG, rsi_signal_long)
        
        # Signal 2: EXIT LONG when RSI > 50
        # We use -(rsi > 50) because a LONGEXIT signal should be negative (-1)
        self.signal_add(bt.SIGNAL_LONGEXIT, -(rsi > self.p.rsi_mid))

        # Signal 3: SHORT when RSI crosses 70 downwards
        # Short signals must be negative (-1)
        rsi_signal_short = -bt.ind.CrossDown(rsi, self.p.rsi_upper, plot=False)
        self.signal_add(bt.SIGNAL_SHORT, rsi_signal_short)
        
        # Signal 4: EXIT SHORT when RSI < 50
        # Short exits are triggered by positive values (1)
        self.signal_add(bt.SIGNAL_SHORTEXIT, rsi < self.p.rsi_mid)

# 2. Download Data (FB is now META)
# Using yfinance to avoid the 'incorrect header' zlib error
df = yf.download("META", start="2018-01-01", end="2018-12-31")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

data = bt.feeds.PandasData(dataname=df)

# 3. Setup and Run
cerebro = bt.Cerebro(stdstats=False)
cerebro.addstrategy(RsiSignalStrategy)
cerebro.adddata(data)

cerebro.broker.setcash(1000.0)
cerebro.broker.setcommission(commission=0.001)

# Add observers to see triangles on the plot
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 4. Plot
cerebro.plot(iplot=False, volume=False)