import numpy as np
from scipy.stats import mode

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import accuracy_score, r2_score
from MachineLearning.exceptions import *


class KNN(BaseEstimator):
    """
        KNN (Base Class: BaseEstimator)
            It is the base class of K Nearest Neighbor models.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            distance:
                -type: string
                -about: define the metric used to calcualte distance
                        between points. Available options are: ['euclidean', 'manhattan']
                -default: 'euclidean'
            
    """
    def __init__(self, k = 5, distance = 'euclidean'):
        if not isinstance(k, int):
            raise ValueError('k value should be an integer.')
        if k<=0:
            raise ValueError(f'Expected k>0. Got{k}')
        self.k = k

        if distance not in ['euclidean', 'manhattan']:
            raise ValueError('`distance` must be either one of these [euclidean, manhattan]')
        self.distance = distance.lower()

    def __euclidean_distance(self, Xtr, Xpr):
        return np.sqrt(np.sum(np.square(np.expand_dims(Xpr, axis = 1) - Xtr), axis = -1))

    def __manhattan_distance(self, Xtr, Xpr):
        return np.sqrt(np.sum(np.abs(np.expand_dims(Xpr, axis = 1) - Xtr), axis = -1))

    def _get_neighbors(self, X):

        if self.distance == 'euclidean':
            dist = self.__euclidean_distance(self.X, X)
        elif self.distance == 'manhattan':
            dist = self.__euclidean_distance(self.X, X)
            
        self.dist = dist

        args = np.argsort(dist, axis = 1)[:, :self.k]
        neighbors = np.squeeze(self.y[args])
        return neighbors

    def _predict(self, X, type_):
        self._check_predict_input(X)
        neighbors = self._get_neighbors(X)
        if type_ == 'reg':
            out = np.mean(neighbors, axis = 1, keepdims = True)
        elif type_ == 'clf':
            out = mode(neighbors, axis = 1)[0]
        return self._format_output(out)


class KNNClassifier(KNN):
    """
        KNNClassifier (Base Class: KNN)
            It is the K Nearest Neighbor model for Classification.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            distance:
                -type: string
                -about: define the metric used to calcualte distance
                        between points. Available options are: ['euclidean', 'manhattan']
                -default: 'euclidean'

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, k = 5, distance = 'euclidean'):
        super().__init__(k, distance)
    def predict(self, X): return self._predict(X, 'clf')
    def score(self, X, y): return self._score(X, y, 'clf')

class KNNRegressor(KNN):
    """
        KNNRegressor (Base Class: KNN)
            It is the K Nearest Neighbor model for Regression.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            distance:
                -type: string
                -about: define the metric used to calcualte distance
                        between points. Available options are: ['euclidean', 'manhattan']
                -default: 'euclidean'

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, k = 5, distance = 'euclidean'):
        super().__init__(k, distance)
    def predict(self, X): return self._predict(X, 'reg')
    def score(self, X, y): return self._score(X, y, 'reg')
