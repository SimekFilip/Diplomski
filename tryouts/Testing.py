import numpy as np
import sys
from skopt import BayesSearchCV
import pandas as pd
from backtesting import Backtest
from sklearn.preprocessing import StandardScaler
from TrendFilteringModel.tryouts.Model import *
from TrendFilteringModel.tryouts.Trading import MyStrategy
from TrendFilteringModel.tryouts.DataFactory import get_data
from TrendFilteringModel.tryouts.nStepLabeling import NStep
from TrendFilteringModel.tryouts.EarlyStopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


aapl = get_data('AAPL', '2018-01-01', '2020-12-31')
nstep = NStep(aapl, 1)
aapl = nstep.get_labels()
print(aapl)
sys.exit(0)

aapl['return1'] = aapl['close'].pct_change(1)
aapl['return2'] = aapl['close'].pct_change(2)
aapl['return3'] = aapl['close'].pct_change(3)
aapl['return4'] = aapl['close'].pct_change(5)
aapl['return5'] = aapl['close'].pct_change(10)
aapl['return6'] = aapl['close'].pct_change(15)
aapl = aapl.dropna()

cols = ['open', 'high', 'low', 'close', 'volume', 'return1', 'return2', 'return3', 'return4', 'return5', 'return6']
# cols = ['return1', 'return2', 'return3', 'return4', 'return5', 'return6']
backtesting_cols = ['open', 'high', 'low', 'close']
N = len(aapl)
size1 = int(N*0.7)

X_train = aapl[0:size1][cols].values
X_test = aapl[size1:N][cols].values
# X_test_original = aapl[size1:N][cols]
X_test_original = aapl[size1:N][backtesting_cols]
X_test_original = X_test_original.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
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
# model = LSTM(n_features)
criterion = nn.BCELoss()  # weight=normedWeights
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)  # weight_decay=1e-4
# early_stopping = EarlyStopping(patience=1000, verbose=False)

num_epochs = 2500

thresholds = np.arange(0.0, 1.05, 0.1)
# thresholds = [0.5]
test_sharpe = []
test_accuracies = []
test_iters = []
trades = []
for threshold in thresholds:
    print('Rounding threshold: ', threshold)
    model = LSTM(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)  # 1e-4, 1e-3 najbolja komba
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(X_train, y_train, model, criterion, optimizer, threshold)
        test_prediction_class, test_loss, test_acc, f1 = evaluate(X_test, y_test, model, criterion, threshold)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_acc_prediction = test_prediction_class
            test_iter = epoch
        '''
        if (epoch + 1) % 25 == 0:
            print(f'epoch: {epoch + 1},'
                  f' train loss = {train_loss.item():.4f}, '
                  f'train accuracy = {train_acc:.4f},'
                  f' test loss = {test_loss.item():.4f},'
                  f' test accuracy = {test_acc:.4f},'
                  f' f1 = {f1:.4f}')
        '''
        #early_stopping(test_acc, model)
        #if early_stopping.early_stop:
            # print("Early stopping", test_iter)
        #    break
        # model.load_state_dict(torch.load('checkpoint.pt'))

    print('Zeros: ', np.sum(best_test_acc_prediction.numpy() == 0))
    print('Test acc: ', best_test_acc.item(), ' in iter: ', test_iter)
    test_accuracies.append(round(best_test_acc.item(), 3))
    test_iters.append(test_iter)
    best_test_acc_prediction = best_test_acc_prediction.squeeze().numpy()
    # print(best_test_acc_prediction)

    X_test_original['Out'] = best_test_acc_prediction
    bt = Backtest(X_test_original, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)
    output = bt.run()
    print('Sharpe', round(pd.Series(output)['Sharpe Ratio'], 3))
    print('Trades', pd.Series(output)['# Trades'])
    test_sharpe.append(round(pd.Series(output)['Sharpe Ratio'], 3))
    trades.append(pd.Series(output)['# Trades'])

print('Sharpe: ', test_sharpe)
print('Best acc: ', test_accuracies)
print('Best iter: ', test_iters)
print('Trades: ', trades)
