import pandas as pd
import numpy as np
import torch
import glob
import yfinance as yf


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


def train_test_split(data, ratio, train_cols, test_col):
    split_size = int(len(data) * ratio)
    X_train = data[:split_size][train_cols].values
    X_test = data[split_size:][train_cols].values
    y_train = data[:split_size][test_col].values
    y_test = data[split_size:][test_col].values
    return X_train, X_test, y_train, y_test
