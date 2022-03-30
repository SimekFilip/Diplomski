from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


X, y = load_iris(return_X_y=True)
print(X, y)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf)
print(clf.predict(X[:1, :]))
print(X[:2, :])
print(X[0])
print(y[0])
print(clf.predict(X[:1, :]))
print(clf.score(X, y))
