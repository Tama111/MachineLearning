import numpy as np
from scipy.stats import mode

from MachineLearning.kernels import LinearKernel
from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import r2_score, accuracy_score
from MachineLearning.exceptions import *


# TODO: Check if it is the right implementation
# Note: If kernelized models are used, then careful adjustment 
#       of parameters should be done for better result.

class KernelizedModel(BaseEstimator):
    """
        KernelizedModel (Base Class: BaseEstimator)
            It is a base class for kernelized models.

        Parameters:
            kernel:
                -type: string
                -about: defines the type of kernel to be used.

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.

            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
    """
    def __init__(self, kernel, power, sigma, increase_dim):
        self.increase_dim = increase_dim
        if kernel == 'linear':
            self.kernel = LinearKernel(increase_dim)
        elif kernel == 'poly':
            self.kernel = PolynomialKernel(power, increase_dim)
        elif kernel == 'rbf':
            self.kernel = RadialBasisFunctionKernel(sigma, increase_dim)
        else:
            self.kernel = kernel


class KernelizedLinearRegression(KernelizedModel):
    """
        KernelizedModel (Base Class: BaseEstimator)
            It is kernelized Linear Regression.

        Parameters:
            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: True

            kernel:
                -type: string
                -about: defines the type of kernel to be used.
                -default: 'linear'

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0

            max_iter:
                -type: int
                -about: defines the number of iterations to run at maximum to find 
                        optimal solution.
                -default: 1000

            lr:
                -type: float
                -about: defines the learning rate.
                -default: 0.00001

            tol:
                -type: float
                -about: it is used for early stopping, ie; if the model's error is
                        less than `tol` then, the loop will stop.
                -default: 0.00001

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, fit_intercept = True, increase_dim = True,
                 kernel = 'linear', power = 2.0, sigma = 3.0, 
                 max_iters = 1000, lr = 0.00001, tol = 0.00001):
        super().__init__(kernel, power, sigma, increase_dim)
        self.initialize_ = 'normal'
        self.fit_intercept = fit_intercept
        self.max_iters = max_iters
        self.lr = lr
        self.tol = tol

    def __initialize_alpha(self, a_shp):
        
        if self.initialize_ == 'zeros':
            self.alpha = np.zeros(shape = (a_shp, 1))
        elif self.initialize_ == 'normal':
            self.alpha = np.random.normal(size = (a_shp, 1))
        elif self.initialize_ == 'uniform':
            self.alpha = np.random.uniform(size = (a_shp, 1))
        else:
            raise Exception('Available initializing of alpha are [zeros, normal, uniform]')

    
    def fit(self, X, y):
        super().fit(X, y)
        
        if (self.fit_intercept & (not self.increase_dim)):
            X = np.concatenate([self.X, np.ones((self.X.shape[0], 1))], axis = 1)
            
        K = self.kernel(X, X)
        self.__initialize_alpha(self.X.shape[0])
        for _ in range(self.max_iters):
            if (1-self.score(self.X, y))<=self.tol:
                break
            z = K.T@self.alpha
            self.alpha -= self.lr * 2 * -(self.y - z)

    def predict(self, X):
        self._check_predict_input(X)
        if (self.fit_intercept & (not self.increase_dim)):
            Xtr = np.concatenate([self.X, np.ones((self.X.shape[0], 1))], axis = 1)
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis = 1)
        else:
            Xtr = self.X

        K = self.kernel(Xtr, X)
        pred = (K.T@self.alpha).reshape(len(X), 1)
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'reg')
            


# KernelizedKNN is not adviced to use. It does not work well as compared to KNN.
# Higher dimensions affect the model negatively.
class KernelizedKNN(KernelizedModel):
    """
        KernelizedKNN (Base Class: KernelizedModel)
            It is base class for kernelized models of K Nearest Neighbors.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            kernel:
                -type: string
                -about: defines the type of kernel to be used.
                -default: 'linear'


            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0
            

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: True
            
    """
    def __init__(self, k = 5, kernel = 'linear', sigma = 3.0,
                 power = 2, increase_dim = True):
        super().__init__(kernel, power, sigma, increase_dim)
        self.k = k

    def fit(self, X, y):
        super().fit(X, y)
        self._K = self.kernel(self.X, self.X)
        
    def _predict(self, X):
        self._check_predict_input(X)
        dist = (np.diag(self.kernel(X, X)).reshape(-1, 1) - 
                2*self.kernel(X, self.X) + 
                np.diag(self._K).reshape(-1,)
               )
        
        neighbors = np.argsort(dist, axis = 1)[:, :self.k]
        return neighbors


class KernelizedKNNClassifier(KernelizedKNN):
    """
        KernelizedKNNClassifier (Base Class: KernelizedKNN)
            It is kernelized KNN classifier.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            kernel:
                -type: string
                -about: defines the type of kernel to be used.
                -default: 'linear'

            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: True

        Input:
            X: numpy array 2-Dimensional
            y: numpy array (1 or 2)-Dimensional
            
    """
    def __init__(self, k = 5, kernel = 'linear', sigma = 3.0, 
        power = 2.0, increase_dim = True):
        super().__init__(k, kernel, sigma, power, increase_dim)

    def predict(self, X):
        neighbors = self._predict(X)
        pred = mode(np.squeeze(self.y[neighbors]), axis = 1)[0]
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'clf')


class KernelizedKNNRegressor(KernelizedKNN):
    """
        KernelizedKNNRegressor (Base Class: KernelizedKNN)
            It is kernelized KNN regressor.

        Parameters:
            k:
                -type: int
                -about: define the number of nearest neighbors.
                -default: 5

            kernel:
                -type: string
                -about: defines the type of kernel to be used.
                -default: 'linear'


            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0
            

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: True

        Input:
            X: numpy array 2-Dimensional
            y: numpy array (1 or 2)-Dimensional
            
    """

    def __init__(self, k = 5, kernel = 'linear', sigma = 3.0, 
        power = 2.0, increase_dim = True):
        super().__init__(k, kernel, sigma, power, increase_dim)

    def predict(self, X):
        neighbors = self._predict(X)
        pred = np.mean(args, axis = 1, keepdims = True)
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'reg')
