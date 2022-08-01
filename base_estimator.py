from MachineLearning.exceptions import *
from MachineLearning.metrics import r2_score, accuracy_score

class BaseEstimator(object):
    """
        BaseEstimator
            It is a Base Class for all the estimators.
    """

    def _check_if_fitted(self):
        if not hasattr(self, '_fitted'):
            raise NotFittedError(self.__class__.__name__)

    def _check_input_dim(self, X):
        if len(X.shape)!=2:
            err_msg = 'X should be two dimensional. Input\'s X shape is {X.shape}'
            raise InvalidShape(err_msg=err_msg)

    def _check_input_len(self, X, y):
        if X.shape[0] != y.shape[0]:
            err_msg = ('Unequal length of X & y.'+ 
                            'Length of X is {X.shape[0]} and '+
                            'length of y is {y.shape[0]}')
            raise InvalidShape(err_msg=err_msg)

    def _check_input_ftr(self, X):
        if X.shape[1] != self.X.shape[1]:
            raise InvalidShape(self.X.shape[1], X.shape[1], True)

    def _check_input(self, X, y):
        self._check_input_len(X, y)
        self._check_input_dim(X)

    def _check_predict_input(self, X):
        self._check_if_fitted()
        self._check_input_dim(X)
        self._check_input_ftr(X)

    def _format_output(self, out):
        out = out.reshape(-1, 1)
        return np.squeeze(out) if self._y_sqz else out

    def fit(self, X, y):
        self._fitted = True
        self.X = X
        if y is not None:
            self._check_input(X, y)
            self._y_sqz = len(y.shape)==1
            self.y = y.reshape(-1, 1)
        else:
            self._check_input_dim(X)

    def _score(self, X, y, type_):
        if y is None:
            raise ValueError('`y` cannot be `None` for calculating score.')

        self._check_if_fitted()
        self._check_input(X, y)

        if X.shape[1]!=self.X.shape[1]:
            raise InvalidShape(self.X.shape[1], X.shape[1], True)

        pred = self.predict(X)
        if (isinstance(pred, tuple) or isinstance(pred, list)):
            pred = pred[0]

        if pred.shape != y.shape:
            raise InvalidShape(pred.shape, y.shape)

        if type_ == 'reg':
            return r2_score(y, pred)
        elif type_ == 'clf':
            return accuracy_score(y, pred)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
