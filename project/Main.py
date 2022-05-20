import numpy as np
import sys
import pandas as pd
import torch
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from nStepLabeling import NStep
from Trading import Metrics
from LSTMClassifier import LSTMClassifier
from DataFactory import get_data, train_test_split, scale_data

aapl = get_data('AAPL', '2000-01-01', '2020-12-31')
nstep = NStep(aapl, 1)

nstep = nStepLabeling.NStep(aapl, 1)
aapl = nstep.get_labels()

aapl['return1'] = aapl['Close'].pct_change(1)
aapl['return2'] = aapl['Close'].pct_change(2)
aapl['return3'] = aapl['Close'].pct_change(3)
aapl['return4'] = aapl['Close'].pct_change(5)
aapl['return5'] = aapl['Close'].pct_change(10)
aapl['return6'] = aapl['Close'].pct_change(15)
aapl = aapl.dropna()
aapl = aapl.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume", "out": "Out"})
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return1',
        'return2', 'return3', 'return4', 'return5', 'return6']

X_train, X_test, y_train, y_test = train_test_split(aapl, 0.9, cols, 'Out')
X_train, X_test = scale_data(StandardScaler(), X_train, X_test)

opt = BayesSearchCV(
     LSTMClassifier(),
     {
         'lr': Real(1e-5, 1e-2, prior='uniform'),
         'l2': Real(1e-5, 1e-2, prior='uniform'),
         'n_layers': Integer(1, 2),
         'n_hidden': Integer(4, 16),
         'num_epochs': Integer(100, 500)
     },
     cv=TimeSeriesSplit(n_splits=3, max_train_size=5000),
     n_iter=100,
         'lr': Real(1e-4, 1e-2, prior='uniform'),
         'l2': Real(1e-4, 1e-2, prior='uniform')
         # 'n_layers': Integer(1, 2),
         # 'n_hidden': Integer(8, 64)
     },
     cv=TimeSeriesSplit(n_splits=3, max_train_size=1000),
     n_iter=10,
     random_state=0,
     refit=True
)
_ = opt.fit(X_train, y_train)


df = pd.DataFrame.from_dict(opt.cv_results_)
#df.to_csv('out.csv')
np.save('./predicted_labels', np.array(opt.predict(X_test)))

measures = ['Start', 'End', 'Sharpe Ratio', 'Equity Final [$]',
'Equity Peak [$]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', '# Trades']
c = 0.0
split_size = int(len(aapl)*0.9)
backtest_data = aapl[split_size:][cols]
metrics = Metrics(backtest_data)
print(metrics.get_metrics(opt.predict(X_test), measures, c))
print('test score:', opt.score(X_test, y_test))
print('train score:', opt.score(X_train, y_train))
print('best score:', opt.best_score_)
print('best params:', opt.best_params_)


print(opt.score(X_test, y_test))
print(opt.score(X_train, y_train))
df = pd.DataFrame.from_dict(opt.cv_results_)
# pd.set_option('display.max_columns', None)
# df.to_csv('C:/Filip/FER/5.GODINA/DIPLOMSKI_RAD/out.csv')
print(df)

