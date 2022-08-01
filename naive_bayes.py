import numpy as np

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.metrics import accuracy_score
from MachineLearning.exceptions import *

class NaiveBayes(BaseEstimator):
    """
        NaiveBayes (Base Class: BaseEstimator)
            It is the base class for different types of naive bayes.
    """
    def __get_priors(self, y):
        self.classes_ = np.sort(np.unique(y))
        self.priors = {}
        for c in self.classes_:
            self.priors[c] = len(y[y==c])/len(y)

    def _fit(self, X, y):
        super().fit(X, y)
        self.__get_priors(self.y)
        Z = np.concatenate([self.X, self.y], axis = 1)
        return Z, self.y

    def __map_classes(self, x):
        for i, v in enumerate(self.classes_):
            if x == i:
                return v
        raise Exception(f'Unknown label {x}. Not in {self.classes_}')

    def predict_probs(self, X):
        predictions = {}
        for c in self.classes_:
            predictions[c] = np.empty(X.shape[0])
            for d in range(X.shape[0]):
                predictions[c][d] = self.predict_prob(X[d], c)

        preds = np.array([predictions[i] for i in list(predictions.keys())]).T
        return preds, np.array(predictions.keys())

    def predict(self, X):
        self._check_predict_input(X)
        pred_probs, _ = self.predict_probs(X)
        pred = np.argmax(pred_probs, axis = 1)
        pred = np.array(list(map(self.__map_classes, pred)))
        return self._format_output(pred)

    def score(self, X, y):
        return self._score(X, y, 'clf')


class CategoricalNB(NaiveBayes):
    """
        CategoricalNB (Base Class: NaiveBayes)
            It is the Categorical Naive Bayes.

        Parameters:
            alpha_smppthing:
                -type: int
                -about: it could be seen as some kind of regularizer or 
                        adding prior knowledge to the model.
                -default: 1 

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, alpha_smoothing = 1):
        self.alpha_smoothing = alpha_smoothing

    def fit(self, X, y):
        Z, y = super()._fit(X, y)

        self.likelihoods = {}
        for f in range(self.X.shape[1]):
            self.likelihoods[f] = {}
            z1 = Z[:, [f, -1]]
            for c_ in self.classes_:
                self.likelihoods[f][c_] = {}
                z2 = z1[z1[:, -1] == c_]
                for fc in np.unique(Z[:, f]):
                    z3 = z2[z2[:, 0] == fc]
                    self.likelihoods[f][c_][fc] = (len(z3) + self.alpha_smoothing)/(len(z2) + self.alpha_smoothing * len(self.classes_))

    def predict_prob(self, x, cls):
        out = np.log(self.priors[cls])
        for i, f in enumerate(x):
            # out *= self.likelihoods[i][cls][f]
            out += np.log(self.likelihoods[i][cls][f])
        return out



class MultinomialNB(NaiveBayes):
    """
        MultinomialNB (Base Class: NaiveBayes)
            It is the Multinomial Naive Bayes.

        Parameters:
            alpha_smppthing:
                -type: int
                -about: it could be seen as some kind of regularizer or 
                        adding prior knowledge to the model.
                -default: 1 

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self, alpha_smoothing = 1):
        self.alpha_smoothing = alpha_smoothing

    def fit(self, X, y):
        Z, y = super()._fit(X, y)

        self.likelihoods = {}
        for c_ in self.classes_:
            self.likelihoods[c_] = {}
            z1 = Z[Z[:, -1]==c_]
            for f in range(self.X.shape[1]):
                z2 = z1[:, f]
                self.likelihoods[c_][f] = (np.sum(z2) + self.alpha_smoothing)/(np.sum(z1[:, :-1]) + self.alpha_smoothing * self.X.shape[1])


    def predict_prob(self, x, cls):
        out = np.log(self.priors[cls])
        for i, f in enumerate(x):
            out += np.log(self.likelihoods[cls][i])*f
            # out *= self.likelihoods[cls][i]**f
        return out


class GaussianNB(NaiveBayes):
    """
        GaussianNB (Base Class: NaiveBayes)
            It is the Gaussian Naive Bayes.

        Input:
            X: numpy array of 2-Dimensional
            y: numpy array of (1 or 2)-Dimensional
            
    """
    def __init__(self):
        pass

    def __gaussian(self, x, mean, std):
        return np.exp(-0.5*((x - mean)/std)**2)/std #1/(2*pi)**0.5

    def fit(self, X, y):
        Z, y = super()._fit(X, y)

        self.likelihoods = {}
        for f in range(self.X.shape[1]):
            self.likelihoods[f] = {}
            z1 = Z[:, [f, -1]]
            for c_ in self.classes_:
                z2 = z1[z1[:, -1] == c_]
                self.likelihoods[f][c_] = {}
                self.likelihoods[f][c_]['mean'] = np.mean(z2[:, 0])
                self.likelihoods[f][c_]['std'] = np.std(z2[:, 0])

    def predict_prob(self, x, cls):
        out = np.log(self.priors[cls])
        for f, v in enumerate(x):
            mean = self.likelihoods[f][cls]['mean']
            std = self.likelihoods[f][cls]['std']
            out += np.log(self.__gaussian(v, mean, std))
            # out *= self.__gaussian(v, mean, std)
        return out

    
