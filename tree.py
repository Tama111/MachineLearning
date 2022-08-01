import numpy as np

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import r2_score, accuracy_score
from MachineLearning.exceptions import *


class Node(object):
    """
        Node
            It is used to create node in a tree.

        Parameters:
            vals:
                -type: list or dict
                -about: contains the values that are present in the node

            col:
                -type: int
                -about: define by which feature, the node should be splitted.
                -default: None

            thresh:
                -type: int or float
                -about: define by which threshold, the column should be splitted.
                -default: None

            gain:
                -type: float
                -about: define the gain of this node.
                -default: None

            leaf:
                -type: boolean
                -about: Whether this node is a leaf (terminal node)
                -default: False 

    """
    def __init__(self, vals, col = None, thresh = None, gain = None, leaf = False):
        self.vals = vals
        
        self.col = col
        self.thresh = thresh
        self.gain = gain
        
        self.leaf = leaf
        
        self.left = None
        self.right = None



class Tree(BaseEstimator):
    """
        Tree (Base Class: BaseEstimator)
            It is used to create trees.

        Parameters:
            tree_type:
                -type: string
                -about: define the tree type. Available options are:    
                        ['decision', 'xgb']

            max_depth:
                -type: int
                -about: define the maximum depth of the tree. If set to `None`, 
                        then the tree will grow until it has pure leafs.
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
    def __init__(self, tree_type, max_depth = None, 
                 min_samples_split = None, min_samples_leaf = None):

        if tree_type not in ['decision', 'xgb']:
            raise Exception('`Tree` statisfies needs for tree of type `decision` and `xgb`')
        self.tree_type = tree_type
        
        if (max_depth is not None) and (max_depth<=0):
            raise ValueError('`max_depth` must be greater than zero.')
        self.max_depth = max_depth
        
        if (min_samples_split is not None) and (min_samples_split<2):
            raise ValueError('`min_samples_split` must be atleast 2.')
        self.min_samples_split = min_samples_split
        
        if (min_samples_leaf is not None) and (min_samples_leaf<1):
            raise ValueError('`min_samples_split` must be atleast 1.')
        self.min_samples_leaf = min_samples_leaf
        
        self.depth = 1
    
    def _get_split(self, x, y, **kwargs):
        prev_prob = kwargs.get('prev_prob')
        vals = self._get_vals(y)
        z = np.concatenate([x, y], axis = 1)
        
        if self.tree_type == 'decision':
            root_gain = self._get_impurity(y[:, 0])
        elif self.tree_type == 'xgb':
            root_gain = self._get_similarity(y, prev_prob = prev_prob)
        
        ftrs_gain = {}
        
        if kwargs.get('ftrs_idx', None) is not None:
            ftrs_idx = kwargs.get('ftrs_idx')
        else:
            ftrs_idx = np.arange(x.shape[-1])
            
        for ftr in ftrs_idx:
            z_ = z[:, [ftr, -1]]
            if np.all(z_[:, 0] == z_[0, 0]):
                continue
                
            ftrs_gain[ftr] = {}
            ftrs_gain[ftr]['gain'] = {}
            
            unique_lbls = np.sort(np.unique(z_[:, 0]))
            for lbl_idx in range(len(unique_lbls)):
                if lbl_idx < len(unique_lbls)-1:
                    lbl = 0.5*(unique_lbls[lbl_idx]+unique_lbls[lbl_idx+1])
                else:
                    continue
                    
                zl = z_[z_[:, 0] <= lbl]
                zr = z_[z_[:, 0] > lbl]
                
                if (zl.shape[0] == 0) | (zr.shape[0] == 0):
                    continue
                    
                if self.tree_type == 'decision':
                    lbl_gain = (root_gain - len(zl)/len(z_) * self._get_impurity(zl[:, -1]) - 
                    len(zr)/len(z_) * self._get_impurity(zr[:, -1]))
                elif self.tree_type == 'xgb':
                    lbl_gain = (self._get_similarity(zl[:, -1], prev_prob=prev_prob) + 
                                self._get_similarity(zr[:, -1], prev_prob=prev_prob) - root_gain)
                    
                ftrs_gain[ftr]['gain'][lbl] = lbl_gain
            
        split_ftr, thresh, gain, gain_changed = None, None, -1, False
        for ftr, gain_info in ftrs_gain.items():
            for lbl, val in gain_info['gain'].items():
                if gain<val:
                    split_ftr, thresh, gain, gain_changed = ftr, lbl, val, True
                    
        if (gain_changed==False)&(gain_changed==-1): gain = None
            
        xleft, yleft, xright, yright, prev_prob_l, prev_prob_r = None, None, None, None, None, None
        if (split_ftr is not None) & (thresh is not None):
            lcond, rcond = x[:, split_ftr]<=thresh, x[:, split_ftr]>thresh
            xleft, xright = x[lcond], x[rcond]
            yleft, yright = y[lcond], y[rcond]
            
            if prev_prob is not None:
                prev_prob_l = prev_prob[lcond]
                prev_prob_r = prev_prob[rcond]
            
        return split_ftr, thresh, gain, xleft, yleft, xright, yright, vals, prev_prob_l, prev_prob_r
                
        
    def _build_tree(self, x, y, depth = 1, **kwargs):
        prev_prob = kwargs.get('prev_prob', None)
        
        ftrs_idx = None
        if kwargs.get('ftrs_split', None) is not None:
            ftrs_idx = np.random.choice(np.arange(x.shape[1]), size = kwargs.get('ftrs_split'), replace = False)
            
        split_ftr, thresh, gain, xl, yl, xr, yr, vals, ppl, ppr = self._get_split(x, y, **kwargs)
        node = Node(vals, split_ftr, thresh, gain)
        
        if self.tree_type == 'decision':
            if (x.shape[0] < self.min_samples_split):
                node.leaf, node.left, node.right = True, None, None
                return node

            end_node_ = False
            if (xl is None) or (xl.shape[0]<self.min_samples_leaf):
                node.leaf, node.left = True, None
                end_node_ = True

            if (xr is None) or (xr.shape[0]<self.min_samples_leaf):
                node.leaf, node.right = True, None
                end_node_ = True

            if end_node_: return node
            
        if ((split_ftr is not None) & (thresh is not None)):
            max_depth_ = ((self.max_depth is not None) and (self.max_depth <= depth))
            
            if ftrs_idx is None:
                sm_vals_l = np.all(xl==xl[0, :])
                sm_vals_r = np.all(xr==xr[0, :])
            else:
                sm_vals_l = np.all(xl[:, ftrs_idx]==xl[:, ftrs_idx][0, :])
                sm_vals_r = np.all(xr[:, ftrs_idx]==xr[:, ftrs_idx][0, :])
            
            kwargs['prev_prob'] = ppl
            node.left = Node(self._get_vals(yl), leaf = True)\
                        if ((len(np.unique(yl))==1) or max_depth_ or sm_vals_l)\
                        else self._build_tree(x = xl, y = yl, depth = depth+1, **kwargs)

            kwargs['prev_prob'] = ppr
            node.right = Node(self._get_vals(yr), leaf = True)\
                        if ((len(np.unique(yr))==1) or max_depth_ or sm_vals_r)\
                        else self._build_tree(x = xr, y = yr, depth = depth+1, **kwargs)
            
        else:
            node.leaf = True
            
        if self.depth < depth:
            self.depth = depth
        return node
    
    def fit(self, X, y, **kwargs):
        super().fit(X, y)
        self.tree_ = self._build_tree(X, self.y, **kwargs)
        
    def _get_prediction(self, x, node, return_vals = False):
        if node.leaf:
            return node.vals if return_vals else self._calc_pred(node.vals)
        
        if x[node.col] <= node.thresh:
            return self._get_prediction(x[x[node.col]<=node.thresh][0], node.left, return_vals)
        else:
            return self._get_prediction(x[x[node.col]>node.thresh][0], node.right, return_vals)
        
    def predict_vals(self, X):
        preds = []
        for e, i in enumerate(X):
            preds.append(self._get_prediction(i, self.tree_, return_vals = True))
        return preds
    
    def predict(self, X, **kwargs):
        self._check_predict_input(X)
        preds = []
        for i in X:
            pred_vals = self._get_prediction(i, self.tree_, return_vals = True)
            preds.append(self._calc_pred(pred_vals, **kwargs))
        return self._format_output(np.array(preds))
        


class DecisionTree(Tree):
    """
        DecisionTree (Base Class: Tree)
            It is a base class for Decision Tree for Regression & Classification.

        Parameters:
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

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional

    """
    def __init__(self, max_depth = None, min_samples_split = 2, 
                 min_samples_leaf = 1):
        super().__init__('decision', max_depth, min_samples_split, 
                         min_samples_leaf)


class DecisionTreeClassifier(DecisionTree):
    """
        DecisionTreeClassifier (Base Class: DecisionTree)
            It is the Decision Tree for classification.

        Parameters:
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

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional

    """
    def __init__(self, criterion = 'entropy', max_depth = None, 
                 min_samples_split = 2, min_samples_leaf = 1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        
        if criterion == 'entropy':
            self.__impurity_func = self.__entropy
        elif criterion == 'gini':
            self.__impurity_func = self.__gini
        else:
            raise Exception(f'Impurity function `{criterion}` not available.\n'\
                            f'Try using `entropy` or `gini`')
        
    def _get_vals(self, y):
        vals = {}
        for cls in self.classes_:
            vals[cls] = len(y[y==cls])
        return vals
    
    def __entropy(self, p):
        p = np.where(p==0, 1, p)
        return np.sum(-p*np.log2(p))
    
    def __gini(self, p):
        return 1-np.sum(p**2)
    
    def _get_impurity(self, y):
        classes = np.unique(y)
        probs = []
        for cls in classes:
            y_ = y[y==cls]
            probs.append(len(y_)/len(y))
        return self.__impurity_func(np.array(probs))
    
    def fit(self, X, y, **kwargs):
        self.classes_ = np.sort(np.unique(y))
        super().fit(X, y, **kwargs)
        
    def _calc_pred(self, vals, **kwargs):
        return self.classes_[np.argmax(np.array(list(vals.values())))]

    def score(self, X, y):
        return self._score(X, y, 'clf')

    
class DecisionTreeRegressor(DecisionTree):
    """
        DecisionTreeRegressor (Base Class: DecisionTree)
            It is the Decision Tree for regression.

        Parameters:
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

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional

    """
    def __init__(self, criterion = 'mse', max_depth = None, min_samples_split = 2,
                 min_samples_leaf = 1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        
        if criterion == 'mse':
            self.__impurity_func = self.__mse
        elif criterion == 'mae':
            self.__impurity_func = self.__mae
        else:
            raise Exception(f'Impurity function `{criterion}` not available.\n'\
                            f'Try using `mse` or `mae`')
            
    def _get_vals(self, y):
        return list(y[:, 0])
    
    def __mse(self, y):
        return np.mean((y - np.mean(y))**2)
    
    def __mae(self, y):
        return np.mean(np.abs(y - np.mean(y)))
    
    def _get_impurity(self, y):
        return self.__impurity_func(y)
    
    def _calc_pred(self, vals, **kwargs):
        return np.mean(vals)
    
    def score(self, X, y):
        return self._score(X, y, 'reg')

