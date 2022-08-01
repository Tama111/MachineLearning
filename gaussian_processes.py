import numpy as np

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.kernels import *
from MachineLearning.metrics import r2_score


class GaussianProcesses(BaseEstimator):
    """
        GaussianProcesses (Base Class: BaseEstimator)
            It is GaussianProcesses, mainly used for regression task.

        Parameters:
            kernel:
                -type: string
                -about: defines the kernel that need to used.
                -default: 'rbf'

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: False

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional 

    """
    def __init__(self, kernel='rbf', power = 2.0, sigma = 3.0, increase_dim = False):
        
        if kernel == 'linear':
            self.kernel = LinearKernel(increase_dim)
        elif kernel == 'poly':
            self.kernel = PolynomialKernel(power, increase_dim)
        elif kernel == 'rbf':
            self.kernel = RadialBasisFunctionKernel(sigma, increase_dim)
        else:
            self.kernel = kernel
    
    def fit(self, X, y):
        super().fit(X, y)
        self.__K_BB_inv = np.linalg.inv(self.kernel(self.X, self.X))
        
    def predict(self, X, return_cov_sigma = False):
        self._check_predict_input(X)
        K_AB = self.kernel(X, self.X)
        sol = K_AB@self.__K_BB_inv
        
        mu = (sol@self.y).reshape(X.shape[0], 1)
        mu = self._format_output(mu)
        
        sigma = self.kernel(X, X) - (sol@K_AB.T)
        std = np.sqrt(np.diag(sigma))
        
        if return_cov_sigma:
            return mu, std, sigma
        return mu, std

    def score(self, X, y):
        return super()._score(X, y, 'reg')
