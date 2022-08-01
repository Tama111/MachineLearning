import numpy as np

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import r2_score, accuracy_score
from MachineLearning.exceptions import *

class LinearModel(BaseEstimator):
    """
        LinearModel (Base Class: BaseEstimator)
            It is a base class for Linear Models.

        Parameters:
            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
            
    """
    def __init__(self, fit_intercept = True, max_iters = 1000, 
        lr = 0.0001, tol = 0.00001):
        self.fit_intercept = fit_intercept
        self.max_iters = max_iters
        self.lr = lr
        self.tol = tol
        self.initialize_ = 'normal'

    def _initialize_params(self, X_shape, dissolve_intercept = False):
        X_shp = X_shape[1] + 1 if dissolve_intercept else X_shape[1]
        
        if self.initialize_ == 'zeros':
            self.W = np.zeros(shape = (X_shp, 1))
        elif self.initialize_ == 'normal':
            self.W = np.random.normal(size = (X_shp, 1))
        elif self.initialize_ == 'uniform':
            self.W = np.random.uniform(size = (X_shp, 1))
        else:
            raise Exception('Available initializing of weights are [zeros, normal, uniform]')

        if self.fit_intercept & (not dissolve_intercept):
            self.B = np.zeros(shape = (1, 1))


class LinearRegression_(LinearModel):
    """
        LinearRegression_ (Base Class: LinearModel)
            It is the base class for model that are based on Linear Regression.

        Parameters:
            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
            
    """
    def __init__(self, fit_intercept = True, max_iters = 1000, 
        lr = 0.0001, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)

    def _closed_form_sol(self, X, y):
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape = (X.shape[0], 1)), X], axis = 1)
            
        try:
            self.W = np.linalg.inv(X.T @ X) @ X.T @ self.y
        except np.linalg.LinAlgError:
            try:
                x = X.T @ X
                x_inv = np.linalg.inv(x.T@x)@x.T # moore-penrose pseudoinverse
                self.W = x_inv @ X.T @ self.y
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError('X is a singular matrix. Use `estimator_type` as `gradient_descent`.')


    def _gradient_descent_sol(self, X, y, regularizer = None):
        self._initialize_params(X.shape)
            
        y_ = np.squeeze(y) if self._y_sqz else y
        for _ in range(self.max_iters):
            if (1-self.score(X, y_))<=self.tol: break

            z = X @ self.W
            if self.fit_intercept:
                z += self.B
            
            deriv = -2 * (y - z)

            d_W = (X.T @ deriv).reshape(self.W.shape)
            if regularizer is not None:
                s_W = np.sum(self.W)
                if regularizer.lower() in ['ridge', 'l2']:
                    d_W += self.alpha * 2 * s_W
                elif regularizer.lower() in ['lasso', 'l1']:
                    d_W += self.alpha * (s_W/np.abs(s_W))
                elif regularizer.lower() == 'elastic_net':
                    d_W += self.alpha * self.l1_ratio * (s_W/np.abs(s_W))
                    d_W += 0.5 * self.alpha * (1 - self.l1_ratio) * 2 * s_W
                else:
                    raise Exception(f'Invalid Regularizer: `{regularizer}`')
            
            self.W -= (self.lr * d_W)
            if self.fit_intercept:
                d_B = np.sum(deriv, axis = 0, keepdims = True)
                self.B -= (self.lr * d_B)


    def _closed_form_predict(self, X):
        self._check_predict_input(X)
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape = (X.shape[0], 1)), X], axis = 1)
        return self._format_output(X @ self.W)

    def _gradient_descent_predict(self, X):
        self._check_predict_input(X)
        pred = X @ self.W
        if self.fit_intercept:
            pred += self.B
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'reg')


class LinearRegression(LinearRegression_):
    """
        LinearRegression (Base Class: LinearRegression_)
            It is the Linear Regression model.

        Parameters:
            estimate_type:
                -type: str
                -about: define what method to use while solving.
                        Available options are: ['gradient_descent', 'closed_form']
                        `closed_form` is suitable if data points are less, because it is
                        related to the computation power of the system.
                -default: 'gradient_descent'

            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
    def __init__(self, estimate_type = 'gradient_descent', fit_intercept = True, 
        max_iters = 1000, lr = 0.0001, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)
        if estimate_type not in ['gradient_descent', 'closed_form']:
            raise ValueError('`estimate_type` must belong to these: [`closed_form`, `gradient_descent`]')
        self.estimate_type = estimate_type.lower()

    def fit(self, X, y):
        super().fit(X, y)

        if self.estimate_type == 'closed_form':
            self._closed_form_sol(self.X, y)
            
        elif self.estimate_type == 'gradient_descent':
            self._gradient_descent_sol(self.X, self.y)
                
                
    def predict(self, X):
        if self.estimate_type == 'closed_form':
            return self._closed_form_predict(X)

        elif self.estimate_type == 'gradient_descent':
            return self._gradient_descent_predict(X)


class RidgeRegression(LinearRegression_):
    """
        RidgeRegression (Base Class: LinearRegression_)
            It is the Linear Regression model with L2 Regularizer.

        Parameters:
            alpha:
                -type: float
                -about: weight assigned for regularizer.
                -default: 1.0

            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
    def __init__(self, alpha = 1.0, fit_intercept = True, max_iters = 1000, 
        lr = 0.0001, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)
        self.alpha = alpha

    def fit(self, X, y):
        super().fit(X, y)
        self._gradient_descent_sol(self.X, self.y, regularizer = 'ridge')
                
    def predict(self, X):
        return self._gradient_descent_predict(X)
    

class LassoRegression(LinearRegression_):
    """
        LassoRegression (Base Class: LinearRegression_)
            It is the Linear Regression model with L1 Regularizer.

        Parameters:
            alpha:
                -type: float
                -about: weight assigned for regularizer.
                -default: 1.0

            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
    def __init__(self, alpha = 1.0, fit_intercept = True, max_iters = 1000, 
        lr = 0.0001, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)
        self.alpha = alpha

    def fit(self, X, y):
        super().fit(X, y)
        self._gradient_descent_sol(self.X, self.y, regularizer = 'lasso')
                
    def predict(self, X):
        return self._gradient_descent_predict(X)


class ElasticNetRegression(LinearRegression_):
    """
        ElasticNetRegression (Base Class: LinearRegression_)
            It is the Linear Regression model with both L1 & L2 Regularizer.

        Parameters:
            alpha:
                -type: float
                -about: weight assigned for regularizer.
                -default: 1.0

            l1_ratio:
                -type: float
                -about: weight assigned for l1 regularizer. And, then
                        (1-l1_ratio) weight will be assigned to l2 regularizer.
                -default: 0.5

            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
    def __init__(self, alpha = 1.0, l1_ratio = 0.5, fit_intercept = True, 
        max_iters = 1000, lr = 0.0001, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        super().fit(X, y)
        self._gradient_descent_sol(self.X, self.y, regularizer = 'elastic_net')
                
    def predict(self, X):
        return self._gradient_descent_predict(X)


class LinearClassifier_(LinearModel):
    """
        LinearClassifier_ (Base Class: LinearModel)
            It is the base class for model that are based on Linear Classifier.

        Parameters:
            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.

            max_iter:
                -type: int
                -about: defines the number of iterations to run at maximum to find 
                        optimal solution.

            lr:
                -type: float
                -about: defines the learning rate.

            tol:
                -type: float
                -about: it is used for early stopping, ie; if the model's error is
                        less than `tol` then, the loop will stop.
            
    """
    def __init__(self, fit_intercept, max_iters, lr, tol):
        super().__init__(fit_intercept, max_iters, lr, tol)

    def _predict(self, X, thresh):
        self._check_predict_input(X)
        pred = np.where(self.predict_vals(X)<thresh, self.classes_[0], self.classes_[1])
        return pred

    def score(self, X, y):
        return self._score(X, y, 'clf')



class LogisticRegression(LinearClassifier_):
    """
        LogisticRegression (Base Class: LinearClassifier_)
            It is the Logistic Regression model with Regularizer.

        Parameters:
            penalty:
                -type: string
                -about: define the penalty that to be used.
                        Available options are: ['l1', 'l2', 'elastic_net', 'ridge', 'lasso']
                -default: 'l2'

            alpha:
                -type: float
                -about: weight assigned for regularizer.
                -default: 0.0001

            l1_ratio:
                -type: float
                -about: weight assigned for l1 regularizer. And, then
                        (1-l1_ratio) weight will be assigned to l2 regularizer.
                        It is only used if penalty is set to `elastic_net`
                -default: None

            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

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
    def __init__(self, penalty = 'l2', alpha = 0.0001, l1_ratio = None,
                 fit_intercept = True, max_iters = 1000, lr = 0.0001, 
                 tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)
        if penalty is not None: 
            if penalty not in ['l1', 'l2', 'elastic_net', 'ridge', 'lasso']:
                raise ValueError('penalty should be either of [none, l1, l2, elastic_net, ridge, lasso]')
            self.penalty = penalty
        else:
            self.penalty = penalty

        self.alpha = alpha

        if l1_ratio is not None:
            if (l1_ratio < 0) & (l1_ratio > 1):
                raise ValueError('alpha value must lie within [0, 1]')
        self.l1_ratio = l1_ratio


    def __sigmoid(self, z):
        # to avoid overflow
        z = np.where(z<-500, -500, z)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        super().fit(X, y)

        self.classes_ = np.sort(np.unique(self.y))
        if len(self.classes_)!=2:
            raise Exception(f'`{self.__class__.__name__}` is suitable for binary classification.')
        self.y = np.where(self.y==self.classes_[0], -1, 1)

        self._initialize_params(self.X.shape)

        for _ in range(self.max_iters):
            if (1-self.score(self.X, y))<=self.tol: break

            z = self.X @ self.W
            if self.fit_intercept:
                z += self.B
            
            # negative log likelihood
            # argmin log(1 + exp(-ywx)); where y = {-1, 1}
            deriv = self.y * (self.__sigmoid(self.y * z) - 1)
            
            # regularizer
            if self.penalty is not None:
                s_W = np.sum(self.W)
                if self.penalty in ['l1', 'lasso']:
                     deriv += self.alpha * (s_W/np.abs(s_W))
                elif self.penalty in ['l2', 'ridge']:
                    deriv += self.alpha * 2 * s_W
                elif self.penalty == 'elastic_net':
                    deriv += self.alpha * self.l1_ratio * (s_W/np.abs(s_W))
                    deriv += 0.5 * self.alpha * (1 - self.l1_ratio) * 2 * s_W
                    

            d_W = self.X.T @ deriv
            self.W -= (self.lr * d_W).reshape(self.W.shape)

            if self.fit_intercept:
                d_B = np.sum(deriv, axis = 0, keepdims = True)
                self.B -= (self.lr * d_B).reshape(self.B.shape)

    def predict_vals(self, X):
        pred = X @ self.W
        if self.fit_intercept:
            pred += self.B
        out = self.__sigmoid(pred)
        return self._format_output(out)

    def predict(self, X):
        return self._predict(X, 0.5)


class Perceptron(LinearClassifier_):
    """
        Perceptron (Base Class: LinearClassifier_)
            It is the Perceptron model.
            It is suitable for binary classification task with linearly
            seperable data. 
            
            Note: It is not a linear layer with single as used in neural_networks. 

        Parameters:
            fit_intercept:
                -type: boolean
                -about: whether to involve intercept or not.
                -default: True

            max_iter:
                -type: int
                -about: defines the number of iterations to run at maximum to find 
                        optimal solution.
                -default: 10000

            lr:
                -type: float
                -about: defines the learning rate.
                -default: 1.0

            tol:
                -type: float
                -about: it is used for early stopping, ie; if the model's error is
                        less than `tol` then, the loop will stop.
                -default: 0.00001

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, fit_intercept = True, max_iters = 10000, 
        lr = 1.0, tol = 0.00001):
        super().__init__(fit_intercept, max_iters, lr, tol)  

    def fit(self, X, y):
        super().fit(X, y)

        self.classes_ = np.sort(np.unique(self.y))
        if len(self.classes_)!=2:
            raise Exception(f'`{self.__class__.__name__}` is suitable for binary classification.')
        yt = np.where(self.y==self.classes_[0], -1, 1)

        self._initialize_params(X.shape, dissolve_intercept = True)

        if self.fit_intercept:
            X = np.concatenate([self.X, np.ones((self.X.shape[0], 1))], axis = -1)
            
        for _ in range(self.max_iters):
            if (1-self.score(self.X, y))<=self.tol: break

            z = yt * (X @ self.W)

            check = np.where(z >= 0, 0, 1)
            
            if np.sum(check) > 0:
                self.W += self.lr * ((X * check).T @ yt)
            else:
                break

    def predict_vals(self, X):
        if self.fit_intercept:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis = -1)
        return self._format_output(X @ self.W)

    def predict(self, X):
        return self._predict(X, 0)
        
