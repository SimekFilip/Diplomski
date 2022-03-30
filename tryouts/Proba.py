import numpy as np
import sys
import pandas as pd
from backtesting import Backtest
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from TrendFilteringModel.tryouts.Model import *
from TrendFilteringModel.tryouts.Trading import MyStrategy
from TrendFilteringModel.tryouts.DataFactory import get_data
from TrendFilteringModel.tryouts.nStepLabeling import NStep
from TrendFilteringModel.tryouts.EarlyStopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


#aapl = get_data('AAPL', '2019-01-02', '2020-12-31')
#daily_return = aapl['close'].pct_change(1)
#rint(aapl.head())
#print(daily_return)
#rint(daily_return.max())

aapl = get_data('AAPL', '2018-01-01', '2020-12-31')
nstep = NStep(aapl, 1)
aapl = nstep.get_labels()
print(aapl)
sys.exit(0)

cols = ['Open', 'High', 'Low', 'Close']
backtesting_cols = ['Open', 'High', 'Low', 'Close']
N = len(aapl)
size1 = int(N*0.7)

X_train = aapl[:size1][cols].values
X_test = aapl[size1:][cols].values
data = aapl[size1:][cols]
y_train = aapl[:size1]['Out'].values
y_test = aapl[size1:]['Out'].values

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
criterion = nn.BCELoss()  # weight=normedWeights
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)  # weight_decay=1e-4
num_epochs = 100
threshold = 0.5
best_test_loss = sys.maxsize
for epoch in range(num_epochs):
    train_prediction, train_loss = train(X_train, y_train, model, criterion, optimizer, threshold)
    train_prediction[train_prediction < threshold] = 0
    train_prediction[train_prediction >= threshold] = 1
    train_acc = train_prediction.eq(y_train).sum() / float(y_train.shape[0])

    test_prediction, test_loss = evaluate(X_test, y_test, model, criterion, threshold)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_test_prediction = test_prediction
        test_iter = epoch
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1},'
            f' train loss = {train_loss.item():.4f}, '
            f' train accuracy = {train_acc:.4f},'
            f' test loss = {test_loss.item():.4f}')


print('Best test loss', best_test_loss.item(), ' in iter ', test_iter)
#print(list(best_test_prediction))
#print(list(best_test_prediction.round()))
lr_auc = metrics.roc_auc_score(y_test, best_test_prediction)
print('roc auc score ', lr_auc)
ns_probs = [0 for _ in range(len(best_test_prediction))]
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(np.array(y_test), best_test_prediction)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='LSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

best_test_prediction[best_test_prediction < threshold] = 0
best_test_prediction[best_test_prediction >= threshold] = 1
best_test_acc = best_test_prediction.eq(y_test).sum() / float(y_test.shape[0])
print('Test accuracy ', best_test_acc.item())

data['Out'] = np.array(best_test_prediction.squeeze())
bt = Backtest(data, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)
output = bt.run()
print(output)


