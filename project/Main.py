import sys
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from TrendFilteringModel.project import nStepLabeling
from TrendFilteringModel.project.LSTMClassifier import LSTMClassifier
from TrendFilteringModel.project.DataFactory import get_data, train_test_split, scale_data


aapl = get_data('AAPL', '2016-01-01', '2020-12-31')
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

X_train, X_test, y_train, y_test = train_test_split(aapl, 0.7, cols, 'Out')
X_train, X_test = scale_data(StandardScaler(), X_train, X_test)

opt = BayesSearchCV(
     LSTMClassifier(),
     {
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

print(opt.score(X_test, y_test))
print(opt.score(X_train, y_train))
df = pd.DataFrame.from_dict(opt.cv_results_)
# pd.set_option('display.max_columns', None)
# df.to_csv('C:/Filip/FER/5.GODINA/DIPLOMSKI_RAD/out.csv')
print(df)
