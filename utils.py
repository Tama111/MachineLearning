import numpy as np
from scipy.stats import mode
from copy import copy
from itertools import permutations

from MachineLearning.exceptions import *


def train_test_split(*arrays, train_size=None, test_size=None, shuffle = True):
    """
        train_test_split
            It splits the data for training and testing phase, based on
            `train_size` or `test_size`

        Parameters:
            *arrays:
                -type: sequence of numpy arrays
                -about: list of numpy arrays of same length
                -default: None

            train_size:
                -type: float
                -about: define what percent of data to be used for training.
                        Its value should should in range (0, 1). If both `train_size`
                        & `test_size` set to `None`, then, it will be 0.8
                -default: None

            test_size:
                -type: float
                -about: define what percent of data to be used for testing.
                        Its value should should in range (0, 1). If both `train_size`
                        & `test_size` set to `None`, then, it will be 0.2
                -default: None

            shuffle:
                -type: boolean
                -about: whether to shuffle data before splitting data for training 
                        & testing.
                -default: True
                
    """
    if (train_size is None) and (test_size is None):
        train_size = 0.8
    elif (train_size is None) and (test_size is not None):
        if (test_size<=0)&(test_size>=1):
            raise ValueError('`test_size` must be between (0, 1).')
        train_size = 1-test_size
    elif (train_size is not None) and (test_size is None):
        if (train_size<=0)&(train_size>=1):
            raise ValueError('`train_size` must be between (0, 1).')
    elif (train_size is not None) and (test_size is not None):
        if (train_size+test_size)!=1:
            raise ValueError('Sum of train_size & test_size must be equal to 1.')

    for i in arrays:
        if len(i.shape)!=2:
            raise Exception('Input Array\'s must be two dimensional.')

    inp_len = len(arrays)
    arr = np.concatenate(arrays, axis = 1)
    np.random.shuffle(arr)
        
    train_len = int(arr.shape[0]*train_size)
    train_arr = arr[:train_len, :]
    test_arr = arr[train_len:, :]
    
    k, train, test = 0, [], []
    for ar in arrays:
        train.append(train_arr[:, k:k+ar.shape[1]])
        test.append(test_arr[:, k:k+ar.shape[1]])
        k += ar.shape[1]
    
    return train, test


class SimpleImputer(object):
    """
        SimpleImputer
            It is used to `nan` values using basic techniques.

        Parameters:
            strategy:
                -type: string
                -about: define technique to use to impute nan values.
                        Available options are: ['min', 'max', 'mean', 'median', 'mode', 
                        'constant', 'random', 'std_random']
                -default: 'mean'

            fill_value:
                -type: string or numerical value
                -about: used when strategy is set to `constant`, then all 
                        the nan values are filled with this value.
                -default: None

            sigma:
                -type: float
                -about: define the amount of standard deviation to be used.
                -default: 1.0

            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, strategy = 'mean', fill_value = None, sigma = 1.0, copy = True):
        valid_strategies = ['min', 'max', 'mean', 'median', 'mode', 'constant', 'random', 'std_random']
        if strategy not in valid_strategies:
            raise ValueError(f'`strategy should be in {valid_strategies}.')
        self.strategy = strategy
        self.fill_value = fill_value # used only when strategy = 'constant'
        self.sigma = sigma # used only when strategy = `std_random`
        self.copy = True
        
    def fit(self, X):
        self.X = X
        if self.strategy == 'min':
            self._fill_val = np.nanmin(X, axis = 0, keepdims = True)[0]
        elif self.strategy == 'max':
            self._fill_val = np.nanmax(X, axis = 0, keepdims = True)[0]
        elif self.strategy == 'mean':
            self._fill_val = np.nanmean(X, axis = 0, keepdims = True)[0]
        elif self.strategy == 'median':
            self._fill_val = np.nanmedian(X, axis = 0, keepdims = True)[0]
        elif self.strategy == 'mode':
            self._fill_val = mode(X, axis = 0)[0][0]
        elif self.strategy == 'constant':
            if self.fill_value is None:
                raise ValueError('value should be assigned to `fill_value` when `strategy` is `constant`.')
            self._fill_val = self.fill_value
        elif self.strategy == 'std_random':
            self._mean = np.nanmean(X, axis = 0, keepdims = True)[0]
            self._std = np.nanstd(X, axis = 0, keepdims = True)[0]
        
    def transform(self, X):
        if not hasattr(self, 'X'):
            raise NotFittedError(self.__class__.__name__)
        
        x = X.copy() if self.copy else X
        if not np.any(np.isnan(x)):
            raise Exception('No `nan` value to impute.')
                    
        for i in range(x.shape[1]):
            get_nan = np.isnan(x[:, i])
            if np.any(get_nan):
                
                if self.strategy == 'random':
                    gn = np.isnan(self.X[:, i])
                    no_nan = list(np.where(gn==True, 0, self.X[:, i]))
                    if not np.any(np.where(self.X[:, i]==0, True, False)):
                        while True:
                            no_nan.remove(0)
                            if 0 not in no_nan:
                                break
                    val = np.random.choice(no_nan, size=len(x[:, i]))
                    
                elif self.strategy == 'std_random':
                    val = np.random.normal(loc = self._mean[i], scale = self._std[i]*self.sigma, size = x.shape[0])
                    
                elif self.strategy in ['mean', 'median', 'max', 'min', 'mode']:
                    val = self._fill_val[i]
                
                elif self.strategy == 'constant':
                    val = self._fill_val
                        
                x[:, i] = get_nan*val + np.where(get_nan==True, 0, x[:, i])

        return x
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return f'{self.__class__.__name__}()'




class CrossValidation(object):
    """
        CrossValidation
            It is used for cross validation of estimator.

        Parameters:
            estimator:
                -type: object
                -about: define the estimator class (object).

            cv:
                -type: int
                -about: define the number of cross validations.
                -default: 5

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """
    def __init__(self, estimator, cv = 5):
        self.get_estimator = lambda: copy(estimator)
        self.cv = cv
        self._y_sqz = False
        
    def __call__(self, X, y):
        xy = np.concatenate([X, y.reshape(-1, 1)], axis = 1)
        np.random.shuffle(xy)

        remainder = xy.shape[0]%self.cv
        main_xy = xy[:xy.shape[0]-remainder, :]
        xy_splits = np.split(main_xy, self.cv)
        xy_splits[-1] = np.concatenate([xy_splits[-1], xy[xy.shape[0]-remainder:, :]], axis = 0)

        self.estimators = []
        self.scores = []
        for i in range(self.cv):
            tr_idx = list(np.arange(self.cv))
            tr_idx.remove(i)
            
            train = np.concatenate([xy_splits[idx] for idx in tr_idx], axis = 0)
            test = xy_splits[i]
            
            train_x, train_y = train[:, :-1], train[:, -1]
            test_x, test_y = test[:, :-1], test[:, -1]
            if not self._y_sqz:
                train_y = train_y[:, np.newaxis]
                test_y = test_y[:, np.newaxis]

            self.estimators.append(self.get_estimator())
            self.estimators[-1].fit(train_x, train_y)
            self.scores.append(self.estimators[-1].score(test_x, test_y))

    def __repr__(self):
        return f'{self.__class__.__name__}()'
            

class ParameterSearchCV(object):
    """
        ParameterSearchCV
            It is used for hyperparameter optimization.

        Parameters:
            estimator:
                -type: object
                -about: define the estimator class (object).

            param_grid:
                -type: dict
                -about: define the keys as parameters and values as the list of 
                        values for the parameters to try.

            cv:
                -type: int
                -about: define the number of cross validations.
                -default: 5

            search_type:
                -type: string
                -about: select the type of searching best hyperparameter.
                        Available options are: ['grid', 'random']
                -default: 'grid'

            n_iter:
                -type: int
                -about: define the maximum number of iterations to run, when 
                        search_type is set to `random`.
                -default: 10

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """
    def __init__(self, estimator, param_grid, cv = 10, search_type = 'grid', n_iter = 10):
        self.get_estimator = lambda: copy(estimator)
        self.param_grid = param_grid
        self.cv = cv
        
        if search_type not in ['grid', 'random']:
            raise ValueError(f'{search_type} is not suitable `search_type`. Available search types are `random` & `grid`.')
        self.search_type = search_type
        
        self.n_iter = n_iter # used only when `search_type`='random'
        
        self._y_sqz = False
        
    def fit(self, X, y):
        params = self.param_grid.keys()
        values = self.param_grid.values()
        
        # est_ = self.get_estimator()
        # for p in params:
        #     if not hasattr(est_, p):
        #         raise AttributeError(f'No attribute named `{p}` in {est_.__class__.__name__}.')

        n_params = len(params)
        param_vals = list(values)
        params_idx = []
        
        tot_models = 1
        for pv in param_vals:
            params_idx += list(np.arange(len(pv)))
            tot_models*=len(pv)
        params_idx = sorted(params_idx)
        
        if self.search_type == 'random':
            tot_models = np.minimum(tot_models, self.n_iter)
        
        permutations_idx = []
        for perm in permutations(params_idx, n_params):
            m=0
            if perm not in permutations_idx:
                m += 1
            for j in range(n_params):
                if perm[j]<len(param_vals[j]):
                    m+=1
                    
            if m==(n_params+1):
                permutations_idx.append(perm)
                
        if self.search_type == 'random':
            np.random.shuffle(permutations_idx)
            permutations_idx = permutations_idx[:tot_models]
                
                
        best_score = -1
        self.best_estimator_ = None
        self.best_params_ = None
        for e, perm_idx in enumerate(permutations_idx):
            all_params=[]
            for param, vals, idx in zip(params, values, perm_idx):
                all_params.append((param, vals[idx]))
                
            est = self.get_estimator()
            
            for param, val in all_params:
                setattr(est, param, val)
                
            try:
                cross_val = CrossValidation(est, cv = self.cv)
                cross_val.y_sqz = self._y_sqz
                cross_val(X, y)
                score = np.mean(cross_val.scores)
                
                if best_score<score:
                    self.best_estimator_ = est
                    self.best_params_ = dict(all_params)
                    best_score = score
                
                print(f'Model :{e+1}/{tot_models} \nParams: {dict(all_params)} \nScore: {score}\n')
            except:
                print(f'Model :{e+1}/{tot_models} \nParams: {dict(all_params)} \n(Unsuitable Combination)\n')

    def __repr__(self):
        return f'{self.__class__.__name__}()'
                
