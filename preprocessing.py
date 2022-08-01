import numpy as np
from MachineLearning.exceptions import *

class Preprocessing(object):
    """
        Preprocessing
            This is a base class for different preprocessing technique.
    """
    def _check_ftr_shp(self, exp, recv=None):
        if recv is None:
            recv = self._inpXshp
        if exp != recv:
            raise InvalidShape(exp, recv, True)
            
    def _check_dim(self, X):
        if len(X.shape) != 2:
            raise InvalidShape(err_msg='Input should be 2-Dimensional.')
            
    def _check_fit(self, attr):
        if not hasattr(self, attr):
            raise NotFittedError(self.__class__.__name__)
            
    def _check_input(self, attr, X, **kwargs):
        self._check_fit(attr)
        self._check_ftr_shp(X.shape[1], kwargs.get('recv', self._inpXshp[1]))

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class StandardScaler(Preprocessing):
    """
        StandardScaler (Base Class: Preprocessing)
            It is used to scale the data by centering the mean
            and scaling the standard deviation to 1.

        Parameters:
            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

            with_mean:
                -type: boolean
                -about: whether to center the mean to zero.
                -default: True

            with_std:
                -type: boolean
                -about: whether to scale the data with standard deviation of one.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, copy = True, with_mean = True, with_std = True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.__get_mean_ = lambda x: np.mean(x, axis = 0, keepdims = True) if self.with_mean else None
        self.__get_var_ = lambda x: np.var(x, axis = 0, keepdims = True) if self.with_std else None
        
    def fit(self, X):
        self._inpXshp = X.shape
        self.mean_ = self.__get_mean_(X)
        self.var_ = self.__get_var_(X)
        
    def transform(self, X):
        self._check_input('mean_', X)
            
        out = X.copy() if self.copy else X
        if self.mean_ is not None:
            out -= self.mean_
        if self.var_ is not None:
            out /= np.sqrt(np.where(self.var_==0, 1e-9, self.var_))
        return out
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        self._check_input('mean_', X)
            
        out = X.copy() if self.copy else X
        if self.var_ is not None:
            out *= np.sqrt(self.var_ + 1e-9)
        if self.mean_ is not None:
            out += self.mean_
        return out


class MinMaxScaler(Preprocessing):
    """
        MinMaxScaler (Base Class: Preprocessing)
            It is used to scale the data between a range.

        Parameters:
            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

            feature_range:
                -type: tuple
                -about: define the range between which data should be scaled.
                -default: (0, 1)

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, feature_range = (0, 1), copy = True):
        self.feature_range = feature_range
        self.copy = copy
        
    def fit(self, X):
        self._inpXshp = X.shape
        self.min_ = np.min(X, axis = 0, keepdims = True)
        self.max_ = np.max(X, axis = 0, keepdims = True)
        
    def transform(self, X):
        self._check_input('min_', X)
        
        out = X.copy() if self.copy else X
        out = ((out - self.min_)/(self.max_ - self.min_)) * self.feature_range[-1]
        return out
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        self._check_input('min_', X)
            
        out = X.copy() if self.copy else X
        out = (out/self.feature_range[-1]) * (self.max_ - self.min_) + self.min_
        return out


class Binarizer(Preprocessing):
    """
        Binarizer (Base Class: Preprocessing)
            It is used to convert the data into 0 and 1, 
            supported by some threshold.

        Parameters:
            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

            threshold:
                -type: float
                -about: convert the data to 0.0, where the data is less than
                        threshold, otherwise convert it to 1.0. 
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, threshold = 0.0, copy = True):
        self.threshold = threshold
        self.copy = copy
        
    def fit(self, X):
        # does nothing
        pass
    
    def transform(self, X):
        out = X.copy() if self.copy else X
        out = np.where(X<=threshold, 0.0, 1.0)
        return out
    
    def fit_transform(self, X):
        #self.fit(X)
        return self.transform(X)



class Normalizer(Preprocessing):
    """
        Normalizer (Base Class: Preprocessing)
            It is used to normalize the data.

        Parameters:
            norm:
                -type: string
                -about: define which method to use. Available options 
                        are: ['l1', 'l2']
                -default: 'l2'

            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, norm = 'l2', copy = True):
        if norm not in ['l1', 'l2']:
            raise Exception(f'{norm} is not supported. Only `l1` & `l2` is available.')
        self.norm = norm
        self.copy = copy
        
    def fit(self, X):
        self._inpXshp = X.shape
        if self.norm == 'l2':
            self.norm_ = np.sqrt(np.sum(X**2, axis = 0, keepdims = True))
        elif self.norm == 'l1':
            self.norm_ = np.sum(np.abs(X), axis = 0, keepdims = True)
            
    def transform(self, X):
        self._check_input('norm_', X)
        out = X.copy() if self.copy else X
        out /= self.norm_
        return out
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        self._check_input('norm_', X)
        out = X.copy() if self.copy else X
        out *= self.norm_
        return out


class RobustScaler(Preprocessing):
    """
        RobustScaler (Base Class: Preprocessing)
            It is used to scale the data by centering the median
            and dividing it by IQR (Inter Quantile Range). It is suitable
            for data with high outliers.

        Parameters:
            with_centering:
                -type: boolean
                -about: whether to center the median.
                -default: True

            with_scaling:
                -type: boolean
                -about: whether to scale the data with iqr.
                -default: True

            quantile_range:
                -type: tuple
                -about: defines the range for quantile range.
                -default: (0.25, 0.75)

            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, with_centering = True, with_scaling = True, 
                 quantile_range = (0.25, 0.75), copy = True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        
    def fit(self, X):
        self._inpXshp = X.shape
        self.median_, self.iqr_ = None, None
        if self.with_centering:
            self.median_ = np.median(X, axis = 0, keepdims = True)
        
        if self.with_scaling:
            q1 = np.quantile(X, self.quantile_range[0], axis = 0, keepdims = True)
            q3 = np.quantile(X, self.quantile_range[1], axis = 0, keepdims = True)
            self.iqr_ = q3 - q1
        
    def transform(self, X):
        self._check_input('median_', X)
        out = X.copy() if self.copy else X
        if self.with_centering:
            out -= self.median_
        if self.with_scaling:
            out /= (self.iqr_ + 1e-9)
        return out
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        self._check_input('median_', X)
        out = X.copy() if self.copy else X
        if self.with_scaling:
            out *= (self.iqr_+1e-9)
        if self.with_centering:
            out += self.median_
        return out



class OrdinalEncoder(Preprocessing):
    """
        OrdinalEncoder (Base Class: Preprocessing)
            It encodes categorical features into numpy array

        Parameters:
            dtype:
                -type: number type
                -about: convert the output into this data type.
                -default: np.float64

            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, dtype=np.float64, copy = True):
        self.dtype = dtype
        self.copy = copy
        
    def fit(self, X):
        self._check_dim(X)
        
        self._inpXshp = X.shape
        self._inp_dtype = X.dtype
        self.categories_ = []
        for i in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, i]))
            
    def transform(self, X):
        self._check_input('categories_', X)
        out = X.copy() if self.copy else X
        for i in range(X.shape[1]):
            for e, cat in enumerate(self.categories_[i]):
                out[:, i] = np.where(out[:, i]==cat, e, out[:, i])
            
        return out.astype(self.dtype)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        self._check_input('categories_', X)
        out = X.copy() if self.copy else X
        out = out.astype(str)
        for i in range(X.shape[1]):
            for e, cat in enumerate(self.categories_[i]):
                e = str(self.dtype(e))
                out[:, i] = np.where(out[:, i]==e, cat, out[:, i])
        return out.astype(self._inp_dtype)
    


class LabelEncoder(OrdinalEncoder):
    """
        LabelEncoder (Base Class: OrdinalEncoder)
            It encodes single categorical features into numpy array

        Parameters:
            dtype:
                -type: number type
                -about: convert the output into this data type.
                -default: np.float64

            copy:
                -type: boolean
                -about: whether to use copy of input for scaling or just
                        use the original input.
                -default: True

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, dtype = np.float64, copy = True):
        super().__init__(dtype, copy)
        
    def fit(self, X):
        self._y_sqz = False
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            self._y_sqz = True
        if X.shape[1]!=1:
            raise InvalidShape(X.shape[1], 1, True)
        super().fit(X)

        if isinstance(self.categories_, list):
            self.labels_ = self.categories_[0]
        else:
            self.labels_ = self.categories_
        
    def fit_transform(self, X):
        self.fit(X)
        if self._y_sqz:
            X = X.reshape(-1, 1)
        out = self.transform(X)
        return np.squeeze(out) if self._y_sqz else out
    


class OneHotEncoder(Preprocessing):
    """
        OneHotEncoder (Base Class: Preprocessing)
            Encode categorical features as a one-hot numeric array.

        Parameters:
            drop:
                -type: string
                -about: whether to drop any feature of one_hot representation.
                        Available options are: ['if_binary', 'first']
                -default: None

            dtype:
                -type: number type
                -about: convert the output into this data type.
                -default: np.float64

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, drop = None, dtype = np.float64):
        if (drop is not None) and (drop not in ['if_binary', 'first']):
            raise ValueError('`drop` value must be either `first` or `if_binary`.')
        
        self.drop = drop
        self.dtype = dtype
        
        
    def fit(self, X):
        self._inpXshp = X.shape
        self._inp_dtype = X.dtype
        self.labels_ = []
        self.drop_idx_ = np.full((X.shape[1], ), None) if self.drop is not None else None
        self._out_ftrs = 0
        for i in range(X.shape[1]):
            self.labels_.append(np.unique(X[:, i]))
            if len(self.labels_[-1])==1:
                raise Exception(f'Single unique value in feature: `{i}`')
                
                
            self._out_ftrs += len(self.labels_[-1])
            if (
                (self.drop is not None) and (
                    (self.drop=='first') or (
                        (self.drop=='if_binary') and (len(self.labels_[-1])==2)
                    )
                )
               ): 
                self.drop_idx_[i] = 0
                self._out_ftrs -= 1
                        
    def transform(self, X):
        self._check_input('labels_', X)
        out = []
        idx = 0 if self.drop_idx_ is None else None
        for i, lbls in enumerate(self.labels_):
            ftr_vals = []
            if self.drop_idx_ is not None:
                idx = (self.drop_idx_[i]+1) if self.drop_idx_[i] is not None else 0
            for lbl in lbls[idx:]:
                ftr_vals.append(np.where(X[:, i]==lbl, 1, 0).reshape(-1, 1))
            out.append(np.concatenate(ftr_vals, axis = 1))
        outs = np.concatenate(out, axis=1).astype(self.dtype)
        assert self._out_ftrs == outs.shape[1]
        return outs
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        self._check_input('labels_', X, recv=self._out_ftrs)

        if self.drop_idx_ is None:
            d = np.full((X.shape[1], ), -1)
        else:
            d = np.where(self.drop_idx_ == None, -1, self.drop_idx_)
            
        out, k = [], 0
        for e, lbls in enumerate(self.labels_):
            idx = d[e]+1
            x_ = X[:, k:k+len(lbls)-idx]
            k += len(lbls)-idx
            
            
            if x_.shape[1]>1:
                if (self.drop is None) or (self.drop == 'if_binary'):
                    f=lbls[np.argmax(x_, axis = 1)].reshape(-1, 1)
                else:
                    if self.drop == 'first':
                        f=lbls[idx:][np.argmax(x_, axis = 1)].reshape(-1, 1)
                        for i in np.argwhere(np.sum(x_, axis=1, keepdims=True)==0):
                            f[i[0], i[1]]=lbls[0]
                
            else:
                f=lbls[x_[:, 0].astype(int)].reshape(-1, 1)
                
            out.append(f)
        return np.concatenate(out, axis = 1)
    
    
