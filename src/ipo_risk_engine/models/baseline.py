import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self):
       pass

    def fit(self, X, y):
       self.mean_ = np.mean(y)
       return self
    # y(x) = E[y]

    def predict(self, X):
       return np.full(shape=(len(X),), fill_value = self.mean_)


class BaseRidge():
    def __init__(self, alpha = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha= alpha)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        return self.model.fit(X,y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)


class FiveKNN():
    def __init__(self, n_neighbors = 5):
        self.model = KNeighborsRegressor(n_neighbors= n_neighbors)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        return self.model.fit(X,y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
















