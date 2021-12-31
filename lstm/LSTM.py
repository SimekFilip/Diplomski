import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import torch.optim as optim
import numpy as np
from backtesting import Backtest, Strategy
import sys
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TrendFilteringModel.lstm.nStep import NStep
from TrendFilteringModel.lstm.DePrado import DePrado
import matplotlib.pyplot as plt


class MyStrategy(Strategy):

    def init(self):
        super().init()

    def next(self):
        if self.data.Out[-1] == 1 and self.data.Out[-2] == 0:
            self.buy()
        elif self.data.Out[-1] == 0 and self.data.Out[-2] == 1:
            self.sell()


def get_data(name, begin_date=None, end_date=None):
    df = yf.download(name, start=begin_date,
                     auto_adjust=True,  # only download adjusted data
                     end=end_date)
    # my convention: always lowercase
    df.columns = ['open', 'high', 'low',
                  'close', 'volume']

    return df


class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_hidden=50, n_layers=1, n_classes=1):
        super(LSTMPredictor, self).__init__()
        num_classes = n_classes
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(in_features=self.n_hidden, out_features=num_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = 1  # x.size(0)
        h_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, dtype=torch.float32)
        c_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, dtype=torch.float32)
        x = x.unsqueeze(0)
        out, _ = self.lstm1(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.linear(out)
        #out = self.sigmoid(out)
        return out


def train(model, train_input, train_target, optimizer, criterion):
    #posalji 10 vrijednosti u model i odredi output za njih  (treba 30)
    model.train()
    model.zero_grad()
    logits = torch.empty(1, 1)
    for k in range(10, len(train_input)):
        logits = torch.cat((logits, model(train_input[k-10:k])), 0)
    logits = torch.cat([logits[1:]])
    loss = criterion(logits, train_target[0:len(logits)])
    loss.backward()
    optimizer.step()
    #optimizer.zero_grad()
    return loss


def evaluate(model, test_input, test_target, criterion):
    model.eval()
    with torch.no_grad():
        logits = torch.empty(1, 1)
        for k in range(10, len(test_input)):
            logits = torch.cat((logits, model(test_input[k - 10:k])), 0)
        logits = torch.cat([logits[1:]])
        loss = criterion(logits, test_target[0:len(logits)])
        logits[logits < 0.5] = 0
        logits[logits >= 0.5] = 1
        acc = logits.eq(test_target[0:len(logits)]).sum() / float(test_target.shape[0])
        #acc = 0
    return loss, acc, logits # torch.sigmoid(logits)


aapl = get_data('AAPL', '2010-01-01', '2011-12-31')
# print(aapl)
nstep = NStep(aapl, 10)
aapl = nstep.get_labels()

#deprado = DePrado(aapl, 2, 2, 10)
#barriers = deprado.get_labels()
#aapl = pd.concat([aapl, barriers['out']], axis=1)
aapl = aapl.dropna()
#print(aapl)

'''
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
#'''
n_1 = np.sum(aapl['out'] == 1)
n_0 = np.sum(aapl['out'] == 0)
weight = n_1/(n_1+n_0)
#print(weight)
normedWeights = torch.FloatTensor([weight])

cols = list(aapl)[0:5]
# X_train, X_test, y_train, y_test = train_test_split(aapl[cols].values, aapl['out'].values,
#                                      test_size=0.3, shuffle=False)  # random_state=1234)
N = len(aapl)
size1 = int(N*0.7)

X_train = aapl[0:size1][cols].values
X_test = aapl[size1:N][cols].values
y_train = aapl[0:size1]['out'].values
y_test = aapl[size1:N]['out'].values

X2_test = aapl[size1:N][cols]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#'''
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

num_epochs = 100
n_hidden = 50
n_samples, n_features = X_train.shape
n_classes = 1
n_layers = 2

model = LSTMPredictor(n_features, n_hidden, n_layers, n_classes)
# criterion = nn.CrossEntropyLoss()
learning_rate = 0.005  # ili 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss(weight=normedWeights)
# criterion = nn.MSELoss()
# criterion = nn.NLLLoss()

best_test_loss = torch.tensor([[1.0]])
best_test_acc = 0
train_losses = [0 for i in range(num_epochs)]
test_losses = [0 for i in range(num_epochs)]
epochs = [i+1 for i in range(num_epochs)]
for epoch in range(num_epochs):
    train_loss = train(model, X_train, y_train, optimizer, criterion)
    # model.zero_grad()
    test_loss, test_acc, output = evaluate(model, X_test, y_test, criterion)
    # if test_loss < best_test_loss:
    if test_acc > best_test_acc:
    #    best_test_loss = test_loss
        best_test_acc = test_acc
        best_output = output
    if (epoch + 1) % 1 == 0:
        print(f'epoch: {epoch + 1}/{num_epochs}, train loss = {train_loss:.4f},'
              f' test loss = {test_loss:.4f}, test acc = {test_acc:.4f}')
    train_losses[epoch] = train_loss
    test_losses[epoch] = test_loss
    if epoch == num_epochs-1:
        print(best_test_loss.item())
        print(best_output)

#plt.plot(epochs, train_losses, label="train loss")
#plt.plot(epochs, test_losses, label="test loss")
#plt.legend()
#plt.show()

ns_probs = [0 for _ in range(len(best_output))]

# f1 trazi binarne oznake, a auc kontinuirane
ns_auc = metrics.roc_auc_score(y_test[0:len(best_output)], ns_probs)
lr_auc = metrics.roc_auc_score(y_test[0:len(best_output)], best_output.numpy())
print('no skill:', ns_auc)
print('lstm roc auc:', lr_auc)

ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test[0:len(best_output)], ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(np.array(y_test[0:len(best_output)]), best_output.numpy())
#plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#plt.plot(lr_fpr, lr_tpr, marker='.', label='LSTM')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend()
#plt.show()

best_output[best_output < 0.5] = 0
best_output[best_output >= 0.5] = 1
f1 = metrics.f1_score(y_test[0:len(best_output)], best_output.numpy())
print('f1 score', f1)

test_acc = best_output.eq(y_test[0:len(best_output)]).sum() / float(y_test.shape[0])
print('acc:', test_acc.item())

output = best_output.squeeze().numpy()
print(output)
X2_test["Out"] = np.nan
iter = 0
for index, row in X2_test.iterrows():
    X2_test.loc[index, 'Out'] = output[iter]
    iter += 1
    if iter >= len(output):
        break

X2_test = X2_test.dropna()
print(X2_test)

X2_test = X2_test.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})

aapl = aapl.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume", "out": "Out"})

bt = Backtest(X2_test, MyStrategy, cash=10000, commission=0.01, exclusive_orders=True)
output = bt.run()
print(output)
