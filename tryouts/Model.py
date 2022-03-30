import torch
import torch.nn as nn
import sklearn.metrics as metrics


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


def train(X, y, model, criterion, optimizer, threshold=0.5):
    prediction = model(X)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # prediction[prediction < threshold] = 0
    # prediction[prediction >= threshold] = 1
    # acc = prediction.eq(y).sum() / float(y.shape[0])
    return prediction, loss


def evaluate(X, y, model, criterion, threshold=0.5):
    # model.eval()
    with torch.no_grad():
        # model.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        # prediction[prediction < threshold] = 0
        # prediction[prediction >= threshold] = 1
        # acc = prediction.eq(y).sum() / float(y.shape[0])
        # f1 = metrics.f1_score(y, prediction)
        return prediction, loss
