from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
import yfinance as yf
import sys
from TrendFilteringModel.lstm.nStep import NStep
from TrendFilteringModel.lstm.DePrado import DePrado


def get_data(name, begin_date=None, end_date=None):
    df = yf.download(name, start=begin_date,
                     auto_adjust=True,  # only download adjusted data
                     end=end_date)
    # my convention: always lowercase
    df.columns = ['open', 'high', 'low',
                  'close', 'volume']

    return df


class MyStrategy(Strategy):

    def init(self):
        super().init()

    def next(self):
        if self.data.Out[-1] == 1 and self.data.Out[-2] == 0:
            self.buy()
        elif self.data.Out[-1] == 0 and self.data.Out[-2] == 1:
            self.sell()


aapl = get_data('AAPL', '2010-01-01', '2010-12-31')
#print(aapl)
nstep = NStep(aapl, 10)
aapl = nstep.get_labels()
aapl = aapl.dropna()
aapl = aapl.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume", "out": "Out"})
print(aapl)

bt = Backtest(aapl, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)

output = bt.run()
print(output)
#bt.plot()
