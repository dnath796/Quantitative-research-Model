import backtrader as bt
import pandas as pd
import yfinance as yf

# 1. Define the Strategy
class BBand_Strategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2.0),)

    def __init__(self):
        # Keep track of open and close prices
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        self.order = None

        # Add Bollinger Bands indicator
        self.b_band = bt.ind.BollingerBands(self.datas[0], period=self.p.period, devfactor=self.p.devfactor)
        
        # CrossOver signals: 1 (crossed up), -1 (crossed down)
        self.buy_signal = bt.ind.CrossOver(self.datas[0].close, self.b_band.lines.bot)
        self.sell_signal = bt.ind.CrossOver(self.datas[0].close, self.b_band.lines.top)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def next_open(self):
        if not self.position:
            if self.buy_signal > 0:
                size = int(self.broker.getcash() / self.data_open[0])
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}')
                self.buy(size=size)
        else: 
            if self.sell_signal < 0:
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)

# --- THE FIX: DATA LOADING ---
# Download data using yfinance instead of Backtrader's internal loader
df = yf.download("AAPL", start="2015-01-01", end="2025-12-31")

# If yfinance returns MultiIndex columns (happens with recent versions), flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Pass the clean Pandas DataFrame to Backtrader
data = bt.feeds.PandasData(dataname=df)

# --- SETUP CEREBRO ---
cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
cerebro.addstrategy(BBand_Strategy)
cerebro.adddata(data)

cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001) # 0.1%

# Observers and Analyzers
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

# Run
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest_result = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Optional: Print return stats
returns_dict = backtest_result[0].analyzers.returns.get_analysis()
print(f"Total Return: {returns_dict.get('rtot', 0):.4f}")

# --- PLOTTING ---
# volume=False keeps the chart clean
# iplot=False is used for standard pop-up windows; set to True if in a Jupyter Notebook
try:
    cerebro.plot(style='candlestick', volume=False, iplot=False)
except Exception as e:
    print(f"Plotting failed: {e}")
