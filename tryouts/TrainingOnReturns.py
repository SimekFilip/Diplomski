import numpy as np
import sys
import pandas as pd
from backtesting import Backtest
from sklearn.preprocessing import StandardScaler
from TrendFilteringModel.lstm.nStep import NStep
from TrendFilteringModel.tryouts.Model import *
from TrendFilteringModel.tryouts.Trading import MyStrategy
from TrendFilteringModel.tryouts.DataFactory import get_data, import_data
import warnings
warnings.filterwarnings("ignore")


aapl = get_data('AAPL', '2018-01-01', '2020-12-31')
nstep = NStep(aapl, 1)
aapl = nstep.get_labels()
aapl['return1'] = aapl['close'].pct_change(1)
aapl['return2'] = aapl['close'].pct_change(2)
aapl['return3'] = aapl['close'].pct_change(3)
aapl = aapl.dropna()
print(aapl)
backtesting_cols = ['open', 'high', 'low', 'close']
# training_cols = ['return1', 'return2', 'return3']  # dodati volumen
training_cols = ['open', 'high', 'low', 'close', 'volume']  # dodati volumen

N = len(aapl)
size1 = int(N*0.7)

X_train = aapl[0:size1][training_cols].values
X_test = aapl[size1:N][training_cols].values
X_test_original = aapl[size1:N][backtesting_cols]
X_test_original = X_test_original.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                                  "close": "Close", "volume": "Volume"})
y_train = aapl[0:size1]['out'].values
y_test = aapl[size1:N]['out'].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

n_samples, n_features = X_train.shape
model = LSTM(n_features)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)  # weight_decay=1e-4
num_epochs = 1000

best_test_acc = 0
for epoch in range(num_epochs):
    train_loss, train_acc = train(X_train, y_train, model, criterion, optimizer, threshold=0.5)
    test_prediction_class, test_loss, test_acc, f1 = evaluate(X_test, y_test, model, criterion, threshold=0.5)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_acc_prediction = test_prediction_class
        test_iter = epoch

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1},'
              f' train loss = {train_loss.item():.4f}, '
              f'train accuracy = {train_acc:.4f},'
              f' test loss = {test_loss.item():.4f},'
              f' test accuracy = {test_acc:.4f},'
              f' f1 = {f1:.4f}')
# print(best_test_acc_prediction)
print('iter:', test_iter)
print('best acc: ', best_test_acc)
print('Zeros: ', np.sum(best_test_acc_prediction.numpy() == 0))
