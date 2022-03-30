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

aapl = get_data('AAPL', '2019-01-02', '2020-12-31')
nstep = NStep(aapl, 1)
daily_return = aapl['close'].pct_change(1)
return_thresholds = np.linspace(-daily_return.max(), daily_return.max(), num=10)

test_sharpe = []
test_accuracies = []
test_iters = []

for threshold in return_thresholds:
    print('Return threshold: ', threshold)
    aapl = nstep.get_labels(threshold)

    n_1 = np.sum(aapl['out'] == 1)
    n_0 = np.sum(aapl['out'] == 0)
    weight = n_1/(n_1+n_0)
    normedWeights = torch.FloatTensor([weight])

    cols = list(aapl)[0:5]
    N = len(aapl)
    size1 = int(N * 0.7)
    X_train = aapl[0:size1][cols].values
    X_test = aapl[size1:][cols].values
    X_test_original = aapl[size1:N][cols]
    X_test_original = X_test_original.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                                      "close": "Close", "volume": "Volume"})
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))

    n_samples, n_features = X_train.shape
    num_epochs = 5000
    criterion = nn.BCELoss(weight=normedWeights)

    y_train = aapl[0:size1]['out'].values
    y_test = aapl[size1:]['out'].values
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    model = LSTM(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_train_acc = 0
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(X_train, y_train, model, criterion, optimizer, threshold)
        test_prediction_class, test_loss, test_acc, f1 = evaluate(X_test, y_test, model, criterion, threshold)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_acc_prediction = test_prediction_class
            test_iter = epoch

    print('Zeros: ', np.sum(best_test_acc_prediction.numpy() == 0))
    print('Test acc: ', best_test_acc.item(), ' in iter: ', test_iter)
    test_accuracies.append(round(best_test_acc.item(), 2))
    test_iters.append(test_iter)
    best_test_acc_prediction = best_test_acc_prediction.squeeze().numpy()
    print(y_test.tolist())
    print(best_test_acc_prediction)

    X_test_original['Out'] = best_test_acc_prediction
    bt = Backtest(X_test_original, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)
    output = bt.run()
    print('Sharpe:', round(pd.Series(output)['Sharpe Ratio'], 3))
    print('Trades:', pd.Series(output)['# Trades'])
    test_sharpe.append(round(pd.Series(output)['Sharpe Ratio'], 2))

print(test_sharpe)
print(test_accuracies)
print(test_iters)
print(return_thresholds)

