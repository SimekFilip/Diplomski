import torch
import torch.nn as nn
import numpy as np
import sys
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0 prepare data
#1 model
#2 loss and optimizer
#3 training loop

# 0)
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
y_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1)


class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        prediction = torch.sigmoid(self.linear(x))
        # prediction = self.linear(x)  # ne radi baš
        return prediction


class LSTM(nn.Module):

    def __init__(self, n_features, n_hidden=16, n_layers=1, n_classes=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
            # dropout=0.75
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)  # -> sequence length, batch size, input size (features); batch_size=1
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        prediction = torch.sigmoid(self.linear(out))
        return prediction


class RNN(nn.Module):

    def __init__(self, n_features, n_hidden=16, n_layers=1, n_classes=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x = x.unsqueeze(1)  # -> sequence length, batch size, input size (features); batch_size=1
        _, hidden = self.rnn(x)
        out = hidden[-1]
        prediction = torch.sigmoid(self.linear(out))
        return prediction


class GRU(nn.Module):

    def __init__(self, n_features, n_hidden=16, n_layers=1, n_classes=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = x.unsqueeze(1)  # -> sequence length, batch size, input size (features); batch_size=1
        _, hidden = self.gru(x)
        out = hidden[-1]
        prediction = torch.sigmoid(self.linear(out))
        return prediction


# model = LogisticRegression(n_features)
model = LSTM(n_features)
# model = RNN(n_features)
# model = GRU(n_features)

# 2)
criterion = nn.BCELoss()  # radi samo za brojeve izmedu 0 i 1
# criterion = nn.BCEWithLogitsLoss()  # katastrofalne performanse
# criterion = nn.CrossEntropyLoss()  # ne radi iz nekog razloga
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # bolji od SGD u slucaju breast cancer dataseta

# 3)
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    # zero gradients  ili na kraju (iza optimizer.step)
    optimizer.zero_grad()  # ili model.zero_grad()
    # backward pass
    loss.backward()
    # updates
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    # print(y_predicted)
    y_predicted_cls = y_predicted.round()
    # print(y_predicted_cls)
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
