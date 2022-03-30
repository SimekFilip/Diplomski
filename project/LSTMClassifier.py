import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from TrendFilteringModel.project.DataFactory import get_tensors
# from TrendFilteringModel.project.EarlyStopping import EarlyStopping


class LSTMClassifier(nn.Module):

    def __init__(self, n_features=11, n_hidden=16, n_layers=1, n_classes=1,
                 max_iter=100, lr=1e-3, l2=1e-3, threshold=0.5, stopping_patience=25):
        super(LSTMClassifier, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes)

        self.num_epochs = max_iter
        self.lr = lr
        self.l2 = l2
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.criterion = nn.BCELoss()
        self.threshold = threshold
        # self.early_stopping = EarlyStopping(patience=stopping_patience, verbose=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        prediction = torch.sigmoid(self.linear(out))
        return prediction

    def get_params(self, deep=True):
        return {"lr": self.lr, "l2": self.l2, "n_hidden": self.n_hidden,
                "n_layers": self.n_layers}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x, y):
        x, y = get_tensors(x, y)
        for epoch in range(self.num_epochs):
            prediction = self(x)
            loss = self.criterion(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # score = metrics.roc_auc_score(y, prediction.detach())
            # if (epoch + 1) % 10 == 0:
            #    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}, roc_auc = {score:.4f}')

    def predict(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        prediction = self(x)
        prediction[prediction < self.threshold] = 0
        prediction[prediction >= self.threshold] = 1
        return np.array(prediction.detach().squeeze()).astype(int)

    def predict_continuous(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        return np.array(self(x).detach().squeeze())

    def score(self, x, y):
        x, y = get_tensors(x, y)
        with torch.no_grad():
            prediction = self(x)
            score = metrics.roc_auc_score(y, prediction)
        return score

