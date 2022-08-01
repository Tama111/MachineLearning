import numpy as np
from copy import copy
from scipy.stats import mode

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.tree import DecisionTreeClassifier, DecisionTreeRegressor
from MachineLearning.metrics import r2_score, accuracy_score
from MachineLearning.exceptions import *


class Bagging(BaseEstimator):
    """
        Bagging (Base Class: BaseEstimator)
            It is a Base Class for Bagging Estimators.

        Parameters:
            n_estimators:
                -type: int
                -about: the number of estimators in the bag.
                -default: 100

            max_features:
                -type: string
                -about: the method to select the number of features
                        from the data. Available options are: [None, 'sqrt', 'auto', 'log2'].
                -default: 'sqrt'

            bootstrap:
                -type: boolean
                -about: whether samples are drawn with replacement or not.
                        If `False` whole data is passed to each estimator with defined
                        number of features.
                -default: True

            max_samples:
                -type: int/float/None
                -about: define how many samples to be used. If `None`, same number of 
                        samples used as in input `X`. If `int`, then that amount of samples
                        are used. If `float`, then smaples will be `max_samples*X.shape[0]`.
                -default: None

    """
    def __init__(self, n_estimators = 100, max_features = 'sqrt', bootstrap = True, 
                 max_samples = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        
    def __get_total_samples(self, xlen):
        if self.max_samples is None:
            return xlen
        else:
            if isinstance(self.max_samples, int):
                return self.max_samples
            elif isinstance(self.max_samples, float):
                if ((self.max_samples<=1)&(self.max_samples>0)):
                    return np.ceil(self.max_features * xlen)
                else:
                    raise Exception(f'`{self.max_samples}` - max_samples is not supported.') 
            else:
                raise Exception(f'`{self.max_samples}`- max_samples is not available.')
                
    def __get_num_features(self, ftr_len):
        if self.max_features is None:
            return ftr_len
        else:
            if self.max_features in ['sqrt', 'auto']:
                return int(np.ceil(np.sqrt(ftr_len)))
            elif self.max_features == 'log2':
                return int(np.ceil(np.log2(ftr_len)))
            else:
                raise Exception(f'`{self.max_features}`- max_features is not available.')

    def _get_sub_data_idx(self, X, y):
        self._num_features = self.__get_num_features(X.shape[1])
        sub_datas_idx = []
        for _ in range(self.n_estimators):
            ftrs = np.random.choice(np.arange(X.shape[1]), size=self._num_features, replace=False)
            if self.bootstrap:
                tot_samples = self.__get_total_samples(X.shape[0])
                samples_idx = np.random.choice(np.arange(X.shape[0]), size=tot_samples, replace = True)
                sub_datas_idx.append((samples_idx, ftrs))
            else:
                sub_datas_idx.append((np.arange(X.shape[0]), ftrs))
        return sub_datas_idx

    def fit(self, X, y, root_ftr_split = True):
        super().fit(X, y)

        sub_datas = self._get_sub_data_idx(self.X, self.y)
        self._estimators = []
        for samples_idx, ftrs in sub_datas:
            est = self._estimator()
            
            if root_ftr_split:
                est.fit(self.X[samples_idx, :][:, ftrs], self.y[samples_idx, :])
                self._estimators.append((est, ftrs))
            else:
                # mainly for random_forest
                est.fit(self.X[samples_idx, :], self.y[samples_idx, :], ftrs_split = self._num_features)
                self._estimators.append(est)

    def _predict(self, X, type_):
        self._check_predict_input(X)
        preds = []
        for est in self._estimators:
            if isinstance(est, tuple):
                preds.append(est[0].predict(X[:, est[1]]))
            else:
                preds.append(est.predict(X))
        pred = np.squeeze(np.array(preds)).T
        if type_ == 'reg':
            out = np.mean(pred, axis = 1)
        elif type_ == 'clf':
            out = mode(pred, axis = 1)[0]
        return self._format_output(out)


class BaggingClassifier(Bagging):
    """
        BaggingClassifier (Base Class: Bagging)
            It is a Bagging Classifier Estimator.It uses multiple
            classifier and calculate the most common output.

        Parameters:
            base_estimator:
                -type: object
                -about: define the estimator to be used for estimating.
                        If `None`, `DecisionTreeClassifier()` is used.
                -default: None 

            n_estimators:
                -type: int
                -about: the number of estimators in the bag.
                -default: 100

            max_features:
                -type: string
                -about: the method to select the number of features
                        from the data. Available options are: [None, 'sqrt', 'auto', 'log2'].
                -default: 'sqrt'

            bootstrap:
                -type: boolean
                -about: whether samples are drawn with replacement or not.
                        If `False` whole data is passed to each estimator with defined
                        number of features.
                -default: True

            max_samples:
                -type: int/float/None
                -about: define how many samples to be used. If `None`, same number of 
                        samples used as in input `X`. If `int`, then that amount of samples
                        are used. If `float`, then smaples will be `max_samples*X.shape[0]`.
                -default: None


        Input:
            X: numpy array of 2-Dimension
            y: numpy array of (1 or 2)-Dimension

    """
    def __init__(self, base_estimator = None, n_estimators = 100, max_features = 'sqrt', bootstrap = True, 
                 max_samples = None):
        super().__init__(n_estimators, max_features, bootstrap, max_samples)
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        self._estimator = lambda : copy(base_estimator)
        
    def predict(self, X): return self._predict(X, 'clf')

    def score(self, X, y): return self._score(X, y, 'clf')
    
    
class BaggingRegressor(Bagging):
    """
        BaggingRegressor (Base Class: Bagging)
            It is a Bagging Regressor Estimator. It uses multiple
            regressor and calculate the average output.

        Parameters:
            base_estimator:
                -type: object
                -about: define the estimator to be used for estimating.
                        If `None`, `DecisionTreeRegressor()` is used.
                -default: None 

            n_estimators:
                -type: int
                -about: the number of estimators in the bag.
                -default: 100

            max_features:
                -type: string
                -about: the method to select the number of features
                        from the data. Available options are: [None, 'sqrt', 'auto', 'log2'].
                -default: 'sqrt'

            bootstrap:
                -type: boolean
                -about: whether samples are drawn with replacement or not.
                        If `False` whole data is passed to each estimator with defined
                        number of features.
                -default: True

            max_samples:
                -type: int/float/None
                -about: define how many samples to be used. If `None`, same number of 
                        samples used as in input `X`. If `int`, then that amount of samples
                        are used. If `float`, then smaples will be `max_samples*X.shape[0]`.
                -default: None


        Input:
            X: numpy array of 2-Dimension
            y: numpy array of (1 or 2)-Dimension

    """
    def __init__(self, base_estimator = None, n_estimators = 100, max_features = 'sqrt', bootstrap = True, 
                 max_samples = None):
        super().__init__(n_estimators, max_features, bootstrap, max_samples)
        if base_estimator is None:
            base_estimator = DecisionTreeRegressor()
        self._estimator = lambda : copy(base_estimator)

    def predict(self, X): return self._predict(X, 'reg')
    
    def score(self, X, y): return self._score(X, y, 'reg')
    
    
class RandomForestClassifier(BaggingClassifier):
    """
        RandomForestClassifier (Base Class: BaggingClassifier)
            It is a Random Forest Classifier Estimator. It uses multiple
            `DecisionTreeClassifier` and calculate the most common output.

        Parameters:
            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 100

            criterion:
                -type: string
                -about: define the impurity type to be used to calculate the split.
                        Available options are: ['entropy', 'gini']
                -default: 'entropy'

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: None

            min_samples_split:
                -type: int
                -about: defines the minimum number of samples to be available in the node
                        to get split. It should be atleast 2.
                -default: 2

            min_samples_leaf:
                -type: int
                -about: defines the minimum number of samples to be available in both
                        left and right node after splitting. It should be atleast 1.
                -default: 1

            max_features:
                -type: string
                -about: define the method to select the number of features
                        from the data. Available options are: [None, 'sqrt', 'auto', 'log2'].
                -default: 'sqrt'

            bootstrap:
                -type: boolean
                -about: define whether samples are drawn with replacement or not.
                        If `False` whole data is passed to each estimator with defined
                        number of features.
                -default: True

            max_samples:
                -type: int/float/None
                -about: define how many samples to be used. If `None`, same number of 
                        samples used as in input `X`. If `int`, then that amount of samples
                        are used. If `float`, then smaples will be `max_samples*X.shape[0]`.
                -default: None


        Input:
            X: numpy array of 2-Dimension
            y: numpy array of (1 or 2)-Dimension

    """
    def __init__(self, n_estimators = 100, criterion = 'entropy', 
                 max_depth = None, min_samples_split = 2, 
                 min_samples_leaf = 1, max_features = 'sqrt', 
                 bootstrap = True, max_samples = None):
        super().__init__(DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, 
                                                     min_samples_split = min_samples_split, 
                                                     min_samples_leaf = min_samples_leaf), 
                         n_estimators, max_features, bootstrap, max_samples)
        
    def fit(self, X, y):
        super().fit(X, y, False)
    
    
class RandomForestRegressor(BaggingRegressor):
    """
        RandomForestRegressor (Base Class: BaggingRegressor)
            It is a Random Forest Regressor Estimator. It uses multiple
            `DecisionTreeRegressor` and calculate the mean of the outputs.

        Parameters:
            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 100

            criterion:
                -type: string
                -about: define the impurity type to be used to calculate the split.
                        Available options are: ['mse', 'mae']
                -default: 'mse'

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: None

            min_samples_split:
                -type: int
                -about: defines the minimum number of samples to be available in the node
                        to get split. It should be atleast 2.
                -default: 2

            min_samples_leaf:
                -type: int
                -about: defines the minimum number of samples to be available in both
                        left and right node after splitting. It should be atleast 1.
                -default: 1

            max_features:
                -type: string
                -about: define the method to select the number of features
                        from the data. Available options are: [None, 'sqrt', 'auto', 'log2'].
                -default: 'sqrt'

            bootstrap:
                -type: boolean
                -about: define whether samples are drawn with replacement or not.
                        If `False` whole data is passed to each estimator with defined
                        number of features.
                -default: True

            max_samples:
                -type: int/float/None
                -about: define how many samples to be used. If `None`, same number of 
                        samples used as in input `X`. If `int`, then that amount of samples
                        are used. If `float`, then smaples will be `max_samples*X.shape[0]`.
                -default: None


        Input:
            X: numpy array of 2-Dimension
            y: numpy array of (1 or 2)-Dimension

    """
    def __init__(self, n_estimators = 100, criterion = 'mse', 
                 max_depth = None, min_samples_split = 2, 
                 min_samples_leaf = 1, max_features = 'sqrt', 
                 bootstrap = True, max_samples = None):
        super().__init__(DecisionTreeRegressor(criterion = criterion, max_depth = max_depth, 
                                                     min_samples_split = min_samples_split, 
                                                     min_samples_leaf = min_samples_leaf), 
                         n_estimators, max_features, bootstrap, max_samples)
        
    def fit(self, X, y):
        super().fit(X, y, False)
        

