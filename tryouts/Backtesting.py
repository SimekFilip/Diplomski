import sys
import pandas as pd
from backtesting import Backtest
from TrendFilteringModel.tryouts.Trading import MyStrategy
from TrendFilteringModel.lstm.nStep import NStep
from TrendFilteringModel.tryouts.DataFactory import get_data
import warnings
warnings.filterwarnings("ignore")


aapl = get_data('AAPL', '2020-06-01', '2020-10-31')
nstep = NStep(aapl, 1)
aapl = nstep.get_labels()
aapl = aapl.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "out": "Out"})
# aapl.to_csv('C:/Filip/FER/5.GODINA/DIPLOMSKI_RAD/aapl.csv')
# aapl['Out'] = out.values
# aapl.to_csv('C:/Filip/FER/5.GODINA/DIPLOMSKI_RAD/aapl3.csv')
# print(aapl)
# sys.exit(0)

bt = Backtest(aapl, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)
output = bt.run()
print(output)
#print('Sharpe', round(pd.Series(output)['Sharpe Ratio'], 3))
# print('Trades', pd.Series(output)['# Trades'])
#print(list(output.values[-1].columns))
# (output.values[-1]).to_csv('C:/Filip/FER/5.GODINA/DIPLOMSKI_RAD/backtest_output3.csv')

output = output.values[-1].iloc[::2]
output = output['EntryTime'][output['PnL'] < 0]
print(output)
for row in output:
    date = row.date()
    loc = aapl.index.get_loc(date) - 1
    aapl.iloc[loc, -1] = 0

# print(len(output))

