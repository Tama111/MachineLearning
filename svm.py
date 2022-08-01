import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import accuracy
from MachineLearning.kernels import *
from MachineLearning.linear_model import LinearModel
from MachineLearning.exceptions import *


class SimpleSVC(LinearModel):
    """
        SimpleSVC (Base Class: LinearModel)
            It is the Simple Linear Support Vector Machine for classification.

        Parameters:
            C:
                -type: float
                -about: regularizer parameter for l2 penalty.
                -default: 0.001

            max_iter:
                -type: int
                -about: defines the number of iterations to run at maximum to find 
                        optimal solution.
                -default: 1000

            lr:
                -type: float
                -about: defines the learning rate.
                -default: 0.0001

            tol:
                -type: float
                -about: it is used for early stopping, ie; if the model's error is
                        less than `tol` then, the loop will stop.
                -default: 0.00001

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, C = 0.001, max_iters = 1000, lr = 0.0001, tol = 0.00001):
        super().__init__(True, max_iters, lr, tol)
        if C<=0:
            raise ValueError('`C` value must be greater than 0.')
        self.C = C

    def fit(self, X, y):
        super().fit(X, y)

        self.classes_ = np.sort(np.unique(self.y))
        if len(self.classes_)!=2:
            raise Exception(f'`{self.__class__.__name__}` is suitable for binary classification.')
        yt = np.where(self.y==self.classes_[0], -1, 1)

        self._initialize_params(X.shape)

        for _ in range(self.max_iters):
            if (1-self.score(X, y))<=self.tol: break

            z = X@self.W + self.B
            hl = np.maximum(1 - yt*z, 0)
            
            deriv_hl = -yt * np.where(hl==0, 0, 1)

            deriv_W = self.C * 2 * self.W + np.sum((deriv_hl*X).T, axis = 1, keepdims = True)
            deriv_B = np.sum(deriv_hl).reshape(1, 1)
            self.W -= self.lr * deriv_W
            self.B -= self.lr * deriv_B
            
    def predict(self, X):
        self._check_predict_input(X)
        pred = X@self.W + self.B
        pred = np.where(pred<0, self.classes_[0], self.classes_[1])
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'clf')



# Below code for optimization of svm with kernels with `cvxopt` heavily borrowed from:
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/support_vector_machine.py
class SVC(BaseEstimator):
    def __init__(self, kernel = 'rbf', sigma = 3.0, power = 2.0, C = 1.0,
                 increase_dim = False):
        """
        SimpleSVC (Base Class: BaseEstimator)
            It is the Support Vector Machine for classification with kernels.

        Parameters:
            kernel:
                -type: string
                -about: defines the type of kernel to be used.
                -default: 'rbf'

            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0

            power:
                -type: float
                -about: defines the degree for polynomial kernel. It is
                        used only when kernel is set to `poly`.
                -default: 2.0

            C:
                -type: float
                -about: regularizer parameter for l2 penalty.
                -default: 1.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: False

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
        if C<=0:
            raise ValueError('`C` value must be greater than 0.')
        self.C = C
        
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

        self.classes_ = np.sort(np.unique(self.y))
        if len(self.classes_)!=2:
            raise Exception(f'`{self.__class__.__name__}` is suitable for binary classification.')
        y = np.where(self.y==self.classes_[0], -1, 1)

        n_samples = X.shape[0]
        K = self.kernel(X, X)
        
        P = cvxopt.matrix(np.outer(y, y)*K, tc = 'd')
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1), tc = 'd')
        b = cvxopt.matrix(0, tc = 'd')
        
        if self.C:
            G_max = -np.identity(n_samples)
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples)*self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
            
        else:
            G = cvxopt.matrix(-np.identity(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
            
        self.minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        lagr_mult = np.ravel(self.minimization['x'])
        idx = lagr_mult>1e-7
        
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        
        self.intercept = self.support_vector_labels[0]
        self.intercept -= self.kernel(self.support_vectors, self.support_vectors[0].reshape(1, -1)).T@(
                            self.support_vector_labels.reshape(-1, 1)*self.lagr_multipliers.reshape(-1, 1))

        
    def predict(self, X):
        self._check_predict_input(X)
        pred = self.kernel(self.support_vectors, X).T@(self.lagr_multipliers.reshape(-1, 1)
                *self.support_vector_labels.reshape(-1, 1))
                
        pred += self.intercept
        pred = np.where(np.sign(pred)<0, self.classes_[0], self.classes_[1])
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'clf')

