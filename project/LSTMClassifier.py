import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from DataFactory import get_tensors
import warnings
warnings.filterwarnings("ignore")
from TrendFilteringModel.project.DataFactory import get_tensors


class LSTMClassifier(nn.Module):

    def __init__(self, n_features=11, n_hidden=16, n_layers=1, n_classes=1,
                 num_epochs=100, lr=1e-3, l2=1e-3, threshold=0.5):
                 max_iter=100, lr=1e-3, l2=1e-3, threshold=0.5):
        super(LSTMClassifier, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        torch.manual_seed(0)
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")

        torch.manual_seed(0)

        # self.device = torch.device(﻿"cuda:0" if torch.cuda.is_available(﻿) else "cpu"﻿)

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
        ).to(self.device)
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes).to(self.device)

        ).to("cpu")
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_classes).to("cpu")


        self.num_epochs = num_epochs
        self.lr = lr
        self.l2 = l2
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.criterion = nn.BCELoss()
        self.threshold = threshold

    def forward(self, x):
        self.lstm.flatten_parameters()

        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(1).to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        prediction = torch.sigmoid(self.linear(out)).to(self.device)
        return prediction

    def get_params(self, deep=True):
        return {"lr": self.lr, "l2": self.l2, "n_hidden": self.n_hidden,
                "n_layers": self.n_layers, "num_epochs": self.num_epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x, y):
        x, y = get_tensors(x, y)
        x = x.to(self.device)
        y = y.to(self.device)
        for epoch in range(self.num_epochs):
            prediction = self(x)
            loss = self.criterion(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #score = metrics.roc_auc_score(y, prediction.detach())
            #if (epoch + 1) % 10 == 0:
            #   print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}, roc_auc = {score:.4f}')
        score = metrics.roc_auc_score(y.cpu(), prediction.detach().cpu())
        print(self.get_params(), score) 

            # print(np.array(prediction.detach()).reshape(-1))
            # print(metrics.roc_auc_score(y, prediction.detach()))

        return self

    def predict(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        prediction = self(x)
        prediction[prediction < self.threshold] = 0
        prediction[prediction >= self.threshold] = 1
        return np.array(prediction.detach().squeeze().cpu()).astype(int)

    def predict_proba(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        return np.array(self(x).detach().squeeze().cpu())

    def score(self, x, y):
        x, y = get_tensors(x, y)
        x = x.to(self.device)
        with torch.no_grad():
            prediction = self(x)
            score = metrics.roc_auc_score(y, prediction.cpu())
            #prediction = np.round(prediction.cpu())
            #prediction[prediction < self.threshold] = 0
            #prediction[prediction >= self.threshold] = 1
            #score = metrics.f1_score(y, prediction)
            #score = metrics.accuracy_score(y, prediction)
        return score

    def accuracy(self, x, y):
        x, y = get_tensors(x, y)
        with torch.no_grad():
            prediction = self(x)
            prediction[prediction < self.threshold] = 0
            prediction[prediction >= self.threshold] = 1
            score = metrics.accuracy_score(y, prediction)
        return score

