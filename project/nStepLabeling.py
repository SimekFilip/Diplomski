import numpy as np


class NStep:
    def __init__(self, data, step):
        self.data = data
        self.step = step

    def get_labels(self, label_col='Close', threshold=0):
        self.data['return'] = self.data[label_col].pct_change(self.step).shift(-self.step)  # dodan shift
        self.data['Out'] = np.where(self.data['return'] >= threshold, 1, 0)
        self.data = self.data.drop(['return'], axis=1)
        return self.data.dropna()
