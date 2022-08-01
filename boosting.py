import numpy as np
from warnings import warn
from copy import copy

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import r2_score, accuracy_score
from MachineLearning.tree import Tree, DecisionTreeRegressor, DecisionTreeClassifier
from MachineLearning.exceptions import *


class XgbTree(Tree):
    """
        XgbTree (Base Class: Tree)
            It is a Tree for Extreme Gradient Boosting

        parameters:
            max_depth:
                -type: int
                -about: define the depth of the Tree.
                -default: 6

            gamma:
                -type: int
                -about: helps in pruning tree.
                -default: 0

            lambda_:
                -type: int
                -about: used to as a kind of regularizer for claculating similarity
                -default: 0

    """
    def __init__(self, max_depth = 6, gamma = 0, lambda_ = 0):
        super().__init__('xgb', max_depth)
        self.gamma = gamma
        self.lambda_ = lambda_
        
        self.iter_prunes = 100000
        
    def _get_vals(self, y):
        return list(y[:, 0])
    
    def _prune_tree(self):
        def prune_tree_nodes(node):
            endl, endr = False, False
            if (node.left is not None)&(node.right is not None):
                diff = node.gain-self.gamma
                
                if ((node.left.left is None)&(node.left.right is None)&(diff<0)):
                    endl = True
                else:
                    node.left = prune_tree_nodes(node.left)
                    
                if ((node.right.right is None)&(node.right.left is None)&(diff<0)):
                    endr = True
                else:
                    node.right = prune_tree_nodes(node.right)

            if endl&endr:
                node.left = None
                node.right = None
                node.leaf = True
                self.__pruned=True
            return node
        
        i = 0
        while i<self.iter_prunes:
            self.__pruned = False
            self.tree_ = prune_tree_nodes(self.tree_)
            if not self.__pruned:
                break
            i += 1
            
        self.dead_tree_ = ((self.tree_.left is None)&(self.tree_.right is None))
        
        if i==self.iter_prunes:
            warn('Tree is quite big. Set `iter_prunes` greater than {self.iter_prunes}.')
    
    
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self._prune_tree()
    
    def predict_vals(self, X):
        return super().predict_vals(X) if not self.dead_tree_ else None
    
    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs) if not self.dead_tree_ else None
        
        

class XgbTreeRegressor(XgbTree):
    """
        XgbTreeRegressor (Base Class: XgbTree)
            It is a Tree for Extreme Gradient Boosting for Regression.
            It will be used by XgbRegression for creating trees.

        parameters:
            max_depth:
                -type: int
                -about: define the depth of the Tree.
                -default: 6

            gamma:
                -type: int
                -about: helps in pruning tree.
                -default: 0

            lambda_:
                -type: int
                -about: used to as a kind of regularizer for claculating similarity
                -default: 0

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            **kwargs: Optional

    """
    def __init__(self, max_depth = 6, gamma = 0, lambda_ = 0):
        super().__init__(max_depth, gamma, lambda_)
        
    def _get_similarity(self, res, **kwargs):
        return (np.sum(res)**2)/(len(res)+self.lambda_)
    
    def _calc_pred(self, vals, **kwargs):
        return np.sum(vals)/(len(vals) + self.lambda_)
        
    
    
class XgbTreeClassifier(XgbTree):
    """
        XgbTreeClassifier (Base Class: XgbTree)
            It is a Tree for Extreme Gradient Boosting for Classification.
            It will be used by XgbClassifier for creating trees.

        parameters:
            max_depth:
                -type: int
                -about: define the depth of the Tree.
                -default: 6

            gamma:
                -type: int
                -about: helps in pruning tree.
                -default: 0

            lambda_:
                -type: int
                -about: used to as a kind of regularizer for claculating similarity
                -default: 0

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            kwargs: Optional

    """
    def __init__(self, max_depth = 6, gamma = 0, lambda_ = 0):
        super().__init__(max_depth, gamma, lambda_)
        
    def _get_similarity(self, res, **kwargs):
        prev_prob = kwargs['prev_prob']
        return (np.sum(res)**2)/(np.sum(prev_prob*(1-prev_prob)) + self.lambda_)



class Boosting(BaseEstimator):
    """
        Boosting (Base Class: BaseEstimator)
            It is a Base Class for Boosting.
    """



class GradientBoosting(Boosting):
    """
        GradientBoosting (Base Class: Boosting)
            It is a Base Class for GradientBoosting of both `Classifier` & `Regressor`.
            It uses `DecisionTreeRegressor` as estimators.

        Parameters:
            loss:
                -type: string
                -about: define the type to loss to be used.

            learning_rate:
                -type: float
                -about: define the learning rate of the estimator
                -default: 0.1

            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 100

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: None

            criterion:
                -type: string
                -about: define the impurity type to be used to calculate the split.
                -default: None

            min_samples_split:
                -type: int
                -about: defines the minimum number of samples to be available in the node
                        to get split. It should be atleast 2.
                -default: None

            min_samples_leaf:
                -type: int
                -about: defines the minimum number of samples to be available in both
                        left and right node after splitting. It should be atleast 1.
                -default: None

    """
    def __init__(self, loss, learning_rate = 0.1, n_estimators = 100, 
                 max_depth = None, criterion = None, min_samples_split = None, 
                 min_samples_leaf = None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        self._estimator = lambda : DecisionTreeRegressor(criterion, max_depth, 
                                                         min_samples_split, min_samples_leaf)
    
    def _calc_residual(self, y, target):
        neg_diff = target - y # -(y - target)
        if self.loss in ['ls', 'log']:
            return neg_diff
        elif self.loss == 'lad':
            return neg_diff/np.abs(neg_diff)
        elif self.loss == 'huber':
            check = np.abs(neg_diff)<=self.alpha
            return neg_diff*check + np.bitwise_not(check)*self.alpha*neg_diff/np.abs(neg_diff)


class GradientBoostingRegressor(GradientBoosting):
    """
        GradientBoostingRegressor (Base Class: GradientBoosting)
            It is the Gradient Boosting of Regression.
            It uses `DecisionTreeRegressor` as estimators.

        Parameters:
            loss:
                -type: string
                -about: define the type to loss to be used.
                        Available options are: ['ls', 'lad', 'huber']
                -default: 'ls'

            learning_rate:
                -type: float
                -about: define the learning rate of the estimator
                -default: 0.1

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
                -default: 3

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

            alpha:
                -type: float
                -about: it is used only if loss is set to `huber`.
                -default: 0.9

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional

    """
    def __init__(self, loss = 'ls', learning_rate = 0.1, 
                 n_estimators = 100, criterion = 'mse', max_depth = 3, 
                 min_samples_split = 2, min_samples_leaf = 1, 
                 alpha = 0.9):
        if loss not in ['ls', 'lad', 'huber']:
            err_msg = f'`{loss}` is not supported.\nSupported loss functions are `ls`, `lad` & `huber`.'
            raise ValueError(err_msg)
        super().__init__(loss, learning_rate, n_estimators, max_depth, 
                         criterion, min_samples_split, min_samples_leaf)
        self.alpha = alpha

    def fit(self, X, y):
        super().fit(X, y)
        target, f = 0, np.ones_like(self.y)*np.mean(self.y)
        target += f
        
        self._estimators = []
        for i in range(self.n_estimators):
            est = self._estimator()
            est.fit(self.X, target)
            self._estimators.append(est)
            pred = est.predict(self.X)
            
            # for xgb
            if pred is None:
                pred = target
            
            f += self.learning_rate*pred if i>0 else 0
            target = -self._calc_residual(self.y, f)
            
    def predict(self, X):
        self._check_predict_input(X)
        pred = np.full((X.shape[0], 1), np.mean(self.y))
        for i, est in enumerate(self._estimators):
            pred_ = est.predict(X)
            
            # for xgb
            if pred_ is None:
                if i>0:
                    pred_ = self._estimators[i-1].predict(X) 
                if pred_ is None:
                    pred_ = np.ones((X.shape[0], 1))*np.mean(self.y)
            
            pred += pred_*self.learning_rate if i>0 else 0
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'reg')


class GradientBoostingClassifier(GradientBoosting):
    """
        GradientBoostingClassifier (Base Class: GradientBoosting)
            It is the Gradient Boosting of Classification.
            It uses `DecisionTreeRegressor` as estimators.

        Parameters:
            loss:
                -type: string
                -about: define the type to loss to be used.
                        Available options are: ['log']
                -default: 'log'

            learning_rate:
                -type: float
                -about: define the learning rate of the estimator
                -default: 0.1

            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 100

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: 3

            criterion:
                -type: string
                -about: define the impurity type to be used to calculate the split.
                        Available options are: ['mse', 'mae']
                -default: 'mse'

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

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """
    def __init__(self, loss = 'log', learning_rate = 0.1, 
                 n_estimators = 100, max_depth = 3, criterion = 'mse', 
                 min_samples_split = 2, min_samples_leaf = 1):
        if loss not in ['log']:
            err_msg = f'`{self.loss}` is not supported.\nSupported loss functions are `log`.'
            raise ValueError(err_msg)
        super().__init__(loss, learning_rate, n_estimators, max_depth, 
                         criterion, min_samples_split, min_samples_leaf)
        
    def _sigmoid(self, z):
        # to avoid overflow
        z = np.where(z<-500, -500, z)
        return 1/(1+np.exp(-z))

    def _fit_binary(self, x, y):
        target, f = 0, np.ones_like(y)*np.log(len(y[y==1])/len(y[y==0]))
        target += self._sigmoid(f)
        
        estimators_ = []
        for i in range(self.n_estimators):
            prev_prob = self._sigmoid(f)
            prob_res = np.concatenate([prev_prob, target], axis = 1)
            
            est = self._estimator()
            est.fit(x, target, prev_prob = prev_prob)
            estimators_.append((est, prob_res))
            pred_vals = est.predict_vals(x)
            
            if pred_vals is not None:
                pred_v = []
                for pv in pred_vals:
                    deno = 0
                    for val in pv:
                        pp = prev_prob[target==val][0]
                        deno += pp*(1-pp)
                    pred_v.append(np.sum(pv)/deno)
                pred_v = np.array(pred_v).reshape(-1, 1)
            elif i==0:
                pred_v = f
            
            f += self.learning_rate*pred_v if i>0 else 0
            target = -self._calc_residual(y, self._sigmoid(f))
        return estimators_
    
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(self.y)
        self.is_multi_ = len(self.classes_)>2
        
        if self.is_multi_:
            self._estimators = {}
            for cls in self.classes_:
                self._estimators[cls] = self._fit_binary(X, np.where(self.y == cls, 1, 0))
        else:
            self._estimators = self._fit_binary(X, y)
            
    def _predict_binary(self, x, y, estimators):
        pred = np.full((x.shape[0], 1), np.log(len(y[y==1])/len(y[y==0])))
        pred_v_ = [pred]
        for i, (est, prob_res) in enumerate(estimators):
            pred_vals = est.predict_vals(x)
            
            pred_v = None
            if pred_vals is not None:
                pred_v = []
                for pv in pred_vals:
                    deno = 0
                    for val in pv:
                        pp_ = prob_res[prob_res[:, -1]==val][0, 0]
                        deno += pp_*(1-pp_)
                    if hasattr(self, 'lambda_'):
                        deno += self.lambda_
                    pred_v.append(np.sum(pv)/deno)
                    
                pred_v = np.array(pred_v).reshape(-1, 1)
                pred_v_.append(pred_v)
                
            else:
                if i>0:
                    pred_v = pred_v_[-1]
                if pred_v is None:
                    pred_v = pred_v_[0]
                
            pred += self.learning_rate * pred_v if i>0 else 0
        
        del pred_v_
        return np.where(self._sigmoid(pred)>=0.5, 1, 0).reshape(-1, 1)
    
    def predict(self, X):
        self._check_predict_input(X)
        if self.is_multi_:
            cls_preds = []
            for cls in self.classes_:
                cls_preds.append(self._predict_binary(X, np.where(self.y==cls, 1, 0), self._estimators[cls]))
            preds = np.argmax(np.concatenate(cls_preds, axis = 1), axis = 1)
        else:
            preds = self._predict_binary(X, self.y, self._estimators)
        return self._format_output(preds)

    def score(self, X, y):
        return self._score(X, y, 'clf')



class XgbRegressor(GradientBoostingRegressor):
    """
        XgbRegressor (Base Class: GradientBoostingRegressor)
            It is the XgbRegressor. It uses XgbTree for Regression.

        Parameters:
            loss:
                -type: string
                -about: define the type to loss to be used.
                        Available options are: ['ls', 'lad', 'huber']
                -default: 'ls'

            learning_rate:
                -type: float
                -about: define the learning rate of the estimator
                -default: 0.3

            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 10

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: 6

            gamma:
                -type: int
                -about: helps in pruning tree.
                -default: 0

            lambda_:
                -type: int
                -about: used to as a kind of regularizer for claculating similarity
                -default: 0

            alpha:
                -type: float
                -about: it is used only when loss is set to `huber`.
                -default: 0.9

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """

    def __init__(self, loss = 'ls', learning_rate = 0.3, 
                 n_estimators = 10, max_depth = 6, gamma = 0, 
                 lambda_ = 0, alpha = 0.9):
        super().__init__(loss, learning_rate, n_estimators, max_depth)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha # only used for huber loss
        
        self._estimator = lambda : XgbTreeRegressor(max_depth, gamma, lambda_)

        
class XgbClassifier(GradientBoostingClassifier):
    """
        XgbClassifier (Base Class: GradientBoostingClassifier)
            It is the XgbClassifier. It uses XgbTree for classification.

        Parameters:
            loss:
                -type: string
                -about: define the type to loss to be used.
                        Available options are: ['log']
                -default: 'log'

            learning_rate:
                -type: float
                -about: define the learning rate of the estimator
                -default: 0.3

            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 10

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
                -default: 6

            gamma:
                -type: int
                -about: helps in pruning tree.
                -default: 0

            lambda_:
                -type: int
                -about: used to as a kind of regularizer for claculating similarity
                -default: 0

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """
    def __init__(self, loss = 'log', learning_rate = 0.3,
                 n_estimators = 10, max_depth = 6, gamma = 0, 
                 lambda_ = 0):
        super().__init__(loss, learning_rate, n_estimators, max_depth)
        self.gamma = gamma
        self.lambda_ = lambda_
        
        self._estimator = lambda : XgbTreeClassifier(max_depth, gamma, lambda_)



class AdaBoostClassifier(Boosting):
    """
        AdaBoostClassifier (Base Class: Boosting)
            It is the AdaBoostClassifier. It used for classification.

        Parameters:
            base_estimator:
                -type: object
                -about: defines the estimator. If set to `None`,
                        then uses `DecisionTreeClassifier` as default. 
                -default: None

            n_estimators:
                -type: int
                -about: define the number of estimators in the bag.
                -default: 10

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
    """
    def __init__(self, base_estimator = None, n_estimators = 10):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(criterion = 'gini', max_depth = 1)
        self._estimator = lambda : copy(base_estimator)
        self.n_estimators = n_estimators
        
    def _performance_say(self, y_true, preds):
        tot_err = np.sum(preds != y_true)/y_true.shape[0]
        return 0.5*np.log((1-tot_err+1e-9)/(tot_err+1e-9))
    
    def _update_weights(self, wgts, y_true, preds, performance_say):
        errs = np.where(preds==y_true, -1, 1)
        new_wgts = wgts * np.exp(errs*performance_say)
        return new_wgts/np.sum(new_wgts)
    
    def _get_new_data(self, z, new_wgts):
        sample_z = np.concatenate([z, np.cumsum(new_wgts, axis=0)], axis = 1)
        rnd_choice = np.random.uniform(low=0.0, high=1.0, size=(1, z.shape[0]))
        smp_grps = np.tile(sample_z[:, -1].reshape(-1, 1), [1, z.shape[0]])>=rnd_choice
        idxs_ = np.argwhere(np.cumsum(smp_grps, axis = 0)==1)[:, 0]
        return z[idxs_, :]
        
    def _fit_binary(self, X, y):
        z = np.concatenate([X, y], axis = 1)
        sample_wgts = np.ones_like(y)/z.shape[0]
        estimators = []
        for _ in range(self.n_estimators):
            est = self._estimator()
            est.fit(z[:, :-1], z[:, -1].reshape(-1, 1))
            preds = est.predict(z[:, :-1])
            
            performance_say = self._performance_say(z[:, -1].reshape(-1, 1), preds)
            new_wgts = self._update_weights(sample_wgts, z[:, -1].reshape(-1, 1), preds, performance_say)
            z = self._get_new_data(z, new_wgts)
            
            estimators.append((est, performance_say))
            
        return estimators
        
        
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.sort(np.unique(self.y))
        self.is_multi_ = len(self.classes_) > 2
        
        if self.is_multi_:
            self._estimators = {}
            for cls in self.classes_:
                self._estimators[cls] = self._fit_binary(X, np.where(self.y==cls, 1, -1))    
        else:
            self._estimators = self._fit_binary(X, np.where(self.y==1, 1, -1))
            
            
    def _predict_binary(self, X, estimators):
        pred = 0
        for est, ps in estimators:
            pred += est.predict(X)*ps
        return pred.reshape(-1, 1)
            
    
    def predict(self, X):
        self._check_predict_input(X)
        if self.is_multi_:
            preds_cls = []
            for cls in self.classes_:
                preds_cls.append(self._predict_binary(X, self._estimators[cls]))
            pred = np.argmax(np.concatenate(preds_cls, axis = 1), axis = 1).reshape(-1, 1)
                
        else:
            pred = np.sign(self._predict_binary(X, self._estimators))
            if list(self.classes_) == [0, 1]:
                pred = np.where(pred>=0, 1, 0)
                
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'clf')
