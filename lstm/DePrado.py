import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
'''
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.family'] = 'serif'
'''


def get_data(name, begin_date=None, end_date=None):
    df = yf.download(name, start=begin_date,
                     auto_adjust=True,  # only download adjusted data
                     end=end_date)  # interval="1m"
    # my convention: always lowercase
    df.columns = ['open', 'high', 'low',
                  'close', 'volume']

    return df


# za daily podatke
def get_Daily_Volatility(close, span0=20):
    # simple percentage returns
    df0 = close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0 = df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0


# za minutne podatke
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0>0]
    a = df0 - 1  # using a variable to avoid the error message.
    df0 = pd.Series(close.index[a],
                  index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index]/close.loc[df0.values].values-1
    # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


def get_barriers(prices, daily_volatility, upper_lower_multipliers, t_final):
    # create a container
    barriers = pd.DataFrame(columns=['days_passed', 'price', 'vert_barrier',
                                     'top_barrier', 'bottom_barrier'], index=daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc[daily_volatility.index[0]:day])
        # set the vertical barrier
        if days_passed + t_final < len(daily_volatility.index) and t_final != 0:
            vert_barrier = daily_volatility.index[
                                days_passed + t_final]
        else:
            vert_barrier = np.nan
        # set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * \
                          upper_lower_multipliers[0] * vol
        else:
            # set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        # set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * \
                          upper_lower_multipliers[1] * vol
        else:
            # set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)

        barriers.loc[day, ['days_passed', 'price', 'vert_barrier', 'top_barrier', 'bottom_barrier']] = \
            days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier
    return barriers


class DePrado:
    def __init__(self, data, upper_boundary=2, lower_boundary=2, step_ahead=10):
        self.data = data
        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary
        self.t_final = step_ahead

    def get_labels(self):
        '''
        start: first day of the window
        end:last day of the window
        price_initial: first day stock price
        price_final:last day stock price
        top_barrier: profit taking limit
        bottom_barrier:stop loss limit
        condition_pt:top_barrier touching condition
        condition_sl:bottom_barrier touching condition
        '''

        price = self.data['close']
        # set the boundary of barriers, based on 20 days EWM
        daily_volatility = get_Daily_Volatility(price)
        upper_lower_multipliers = [self.upper_boundary, self.lower_boundary]
        prices = price[daily_volatility.index]
        barriers = get_barriers(prices, daily_volatility, upper_lower_multipliers, self.t_final)
        barriers['out'] = None

        for i in range(len(barriers.index)):
            start = barriers.index[i]
            end = barriers.vert_barrier[i]
            if pd.notna(end):
                # assign the initial and final price
                price_initial = barriers.price[start]
                price_final = barriers.price[end]
                # assign the top and bottom barriers
                top_barrier = barriers.top_barrier[i]
                bottom_barrier = barriers.bottom_barrier[i]
                # set the profit taking and stop loss conditions
                condition_pt = (barriers.price[start: end] >= top_barrier).any()
                condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
                # assign the labels
                if condition_pt:
                    barriers['out'][i] = 1
                elif condition_sl:
                    # barriers['out'][i] = -1
                    barriers['out'][i] = 0
                else:
                    # barriers['out'][i] = max(
                    #          [(price_final - price_initial) / (top_barrier - price_initial),
                    #           (price_final - price_initial) / (price_initial - bottom_barrier)], key=abs)
                    if price_final > price_initial:
                        barriers['out'][i] = 1
                    else:
                        barriers['out'][i] = 0
        barriers = barriers.dropna()
        return barriers


aapl = get_data('AAPL', '1980-01-01', '2020-12-31')
#print(aapl)
# Hold for: no more than 10 days
# Profit-taking boundary: 2 times of 20 days return EWM std
# Stop-loss boundary: 2 times of 20 days return EWM std
deprado = DePrado(aapl, 2, 2, 10)
barriers = deprado.get_labels()
output1 = pd.concat([aapl, barriers['out']], axis=1)
output1 = output1.dropna()
print(output1)
# print(aapl.out.value_counts())
#print(barriers)

'''
pd.plotting.register_matplotlib_converters()
fig, ax = plt.subplots()
ax.set(title='Apple stock price',
       xlabel='date', ylabel='price')
ax.plot(barriers.price[100: 200])
start = barriers.index[120]
end = barriers.vert_barrier[120]
upper_barrier = barriers.top_barrier[120]
lower_barrier = barriers.bottom_barrier[120]
ax.plot([start, end], [upper_barrier, upper_barrier], 'r--');
ax.plot([start, end], [lower_barrier, lower_barrier], 'r--');
ax.plot([start, end], [(lower_barrier + upper_barrier)*0.5, \
                       (lower_barrier + upper_barrier)*0.5], 'r--');
ax.plot([start, start], [lower_barrier, upper_barrier], 'r-');
ax.plot([end, end], [lower_barrier, upper_barrier], 'r-');
plt.show()
'''
'''
# dynamic graph
fig,ax = plt.subplots()
ax.set(title='Apple stock price',
       xlabel='date', ylabel='price')
ax.plot(barriers.price[100: 200])
start = barriers.index[120]
end = barriers.index[120+t_final]
upper_barrier = barriers.top_barrier[120]
lower_barrier = barriers.bottom_barrier[120]
ax.plot(barriers.index[120:120+t_final+1], barriers.top_barrier[start:end], 'r--');
ax.plot(barriers.index[120:120+t_final+1], barriers.bottom_barrier[start:end], 'r--');
ax.plot([start, end], [(lower_barrier + upper_barrier)*0.5, \
                       (lower_barrier + upper_barrier)*0.5], 'r--');
ax.plot([start, start], [lower_barrier, upper_barrier], 'r-');
ax.plot([end, end], [barriers.bottom_barrier[end], barriers.top_barrier[end]], 'r-');
'''


def triple_barrier_labeling(
        price: pd.DataFrame,
        volatility_span: float = 20,
        upper_barrier_scaler: float = 2,
        bottom_barrier_scaler: float = 2,
        n_step: int = 10
):
    labels = []

    dv = get_Daily_Volatility(price, volatility_span)
    price = price.loc[dv.index]

    for i in tqdm(range(len(price) - n_step)):
        starting_price = price.iloc[i]
        upper_barrier_price = starting_price * (1 + upper_barrier_scaler * dv.iloc[i])
        bottom_barrier_price = starting_price * (1 - bottom_barrier_scaler * dv.iloc[i])
        look_ahead_price_window = price.iloc[i:i + n_step]

        upper_barrier_hit = (look_ahead_price_window >= upper_barrier_price).any()
        bottom_barrier_hit = (look_ahead_price_window <= bottom_barrier_price).any()

        if upper_barrier_hit and bottom_barrier_hit:
            argwhereupper = np.argwhere(look_ahead_price_window.values >= upper_barrier_price)[0]
            argwherebottom = np.argwhere(look_ahead_price_window.values <= bottom_barrier_price)[0]
            if argwherebottom < argwhereupper:
                labels.append(1)
            else:
                labels.append(0)
        elif upper_barrier_hit:
            labels.append(1)
        elif bottom_barrier_hit:
            labels.append(0)
        else:
            rtn = (look_ahead_price_window.values[-1] - starting_price) / starting_price
            labels.append(1 if rtn >= 0 else 0)

    return pd.Series(labels, index=dv.index[:-n_step])


output1['out2'] = triple_barrier_labeling(aapl['close'])
print(output1)

print(len(output1[output1['out'] != output1['out2']]) / len(output1) * 100)
