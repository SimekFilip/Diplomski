from backtesting import Strategy, Backtest
import pandas as pd


class MyStrategy(Strategy):

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Out) != 2:
            if self.data.Out[-1] == 1 and self.data.Out[-2] == 0:
                self.buy()
            elif self.data.Out[-1] == 0 and self.data.Out[-2] == 1:
                self.sell()
        else:
            if self.data.Out[-1] == 1:
                self.buy()


class Metrics:
    def __init__(self, data):
        self.data = data

    def get_metrics(self, prediction, measures=None, commission=0.01):
        self.data['Out'] = prediction
        bt = Backtest(self.data, MyStrategy, cash=10000, commission=commission, exclusive_orders=True)
        output = bt.run()
        if measures is None:
            return output
        else:
            return pd.Series(output)[measures]
