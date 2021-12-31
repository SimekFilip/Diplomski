import sklearn.metrics as metrics
import matplotlib.pyplot as plt


a = [0,0,1]
b = [1,0,1]
print(metrics.roc_curve(a,b))
print(metrics.roc_auc_score(a,b))
#metrics.plot_roc_curve(a,b)

fpr, tpr, _ = metrics.roc_curve(a, b)
plt.plot(fpr, tpr, marker='.', )
plt.show()


'''
data = np.loadtxt(r"C:\Filip\FER\3.GODINA\6.SEMESTAR\ZAVRŠNI RAD\MeanVarPortofolio\data\Google.csv",
                  delimiter=',', dtype=np.float32, skiprows=1, usecols=(1, 2, 3, 4, 5))
X_train = data[0:1000, :]
X_test = data[1000:, :]
n_samples, n_features = X_train.shape
arr = np.diff(X_train[:, 0]) / X_train[:-1, 0]
arr = np.insert(arr, 0, 0, axis=0)
arr = arr > 0
y_train = arr.astype(int)
arr = np.diff(X_test[:, 0]) / X_test[:-1, 0]
arr = np.insert(arr, 0, 0, axis=0)
arr = arr > 0
y_test = arr.astype(int)
#print(y_test)
#print(np.sum(y_test)/len(y_test))

'''

'''
trainX = []
trainY = []
n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print(trainX)
print(trainY)
'''

'''
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''


'''
# self.lstm1 = nn.LSTMCell(input_size=n_features, hidden_size=n_hidden)  # num_layers=n_layers
# self.lstm2 = nn.LSTMCell(input_size=n_features, hidden_size=self.n_hidden2)
# h_1, c_1 = self.lstm1(x, (h_0, c_0))
# out, _ = self.lstm2(x, (h_1, c_1))
'''