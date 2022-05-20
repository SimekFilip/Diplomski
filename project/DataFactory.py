import pandas as pd
import numpy as np
import torch
import glob
import yfinance as yf
from backtesting import Backtest
from Trading import MyStrategy


def import_data():
    path = r'C:\Filip\FER\5.GODINA\DIPLOMSKI_RAD\clean_parts'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col='datetime', header=0)
        li.append(df)
    data = pd.concat(li, axis=0, ignore_index=False)
    return data


def get_data(name, begin_date=None, end_date=None):
    df = yf.download(name, start=begin_date,
                     auto_adjust=True,
                     end=end_date)
    df.columns = ['Open', 'High', 'Low',
                  'Close', 'Volume']
    return df


def scale_data(scaler, x_train, x_test):
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def get_tensors(x, y):
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    y = y.view(y.shape[0], 1)
    return x, y


def train_test_split(data, ratio):
    split_size = int(len(data) * ratio)
    train = data[:split_size].values
    test = data[split_size:].values
    return train, test


def train_test_split1(data, ratio):
    split_size = int(len(data) * ratio)
    X_train = data.loc[:, data.columns != 'Out'][:split_size].values
    X_test = data.loc[:, data.columns != 'Out'][split_size:].values
    y_train = data[:split_size]['Out'].values
    y_test = data[split_size:]['Out'].values
    return X_train, X_test, y_train, y_test


def train_test_split2(x, y, ratio):
    split_size = int(len(x) * ratio)
    X_train = x[:split_size]
    X_test = x[split_size:]
    y_train = y[:split_size]
    y_test = y[split_size:]
    return X_train, X_test, y_train, y_test


def price_expansion(data, n_past, feature_cols):
    features = data[feature_cols].values
    labels = data['Out'].values
    x = []
    for i in range(n_past, len(features) + 1):
        x.append(features[i - n_past:i])
    x = np.array(x)
    y = labels[n_past - 1:]
    return x, y


def generate_returns(data, price_col, periods):
    for i in range(len(periods)):
        col_name = 'return' + str(periods[i])
        data[col_name] = data[price_col].pct_change(periods[i])
    data = data.dropna()
    data = data.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume", "out": "Out"})
    return data


def returns_expansion(data, n_past, return_cols):
    features = data[return_cols].values
    returns = []
    for i in range(n_past, len(features) + 1):
        returns.append(features[i - n_past:i])
    returns = np.array(returns)
    return returns


def scale_prices(x):
    means = np.mean(x, axis=1).reshape(x.shape[0], 1, x.shape[2])
    x = (x - means) / means
    return x


def filter_false_positives(data, commission=0.00):
    data.index = pd.to_datetime(data.index)
    col_index = data.columns.get_loc("Out")
    fp_exist = True
    while fp_exist:
        bt = Backtest(data, MyStrategy, cash=10000, commission=commission,
                      exclusive_orders=True, trade_on_close=False)
        output = bt.run()
        output = output.values[-1].iloc[::2]
        output = output['EntryTime'][output['PnL'] < 0]
        for row in output:
            date = row.date()
            date = pd.to_datetime(date)
            row_index = data.index.get_loc(date) - 1
            data.iloc[row_index, col_index] = 0
        if len(output) == 0:
            fp_exist = False
        return data
        return x
