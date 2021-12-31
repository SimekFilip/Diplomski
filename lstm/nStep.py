import pandas as pd
import yfinance as yf


class NStep:
    def __init__(self, data, step):
        self.data = data
        self.step = step

    def get_labels(self):
        self.data['out'] = None
        for i in range(len(self.data.index)):
            start = self.data.index[i]
            if i+self.step >= len(self.data.index):
                break
            end = self.data.index[i+self.step]
            if pd.notna(end):
                price_initial = self.data.close[start]
                price_final = self.data.close[end]
                if price_final >= price_initial:
                    self.data['out'][i] = 1
                else:
                    self.data['out'][i] = 0
        return self.data


'''
aapl = get_data('AAPL', '2010-01-01', '2010-12-31')
nstep = NStep(aapl, 10)
aapl = nstep.get_labels()
aapl = aapl.dropna()
print(aapl)
'''