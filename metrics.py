import numpy as np
from MachineLearning.exceptions import InvalidShape


class CheckMetricInput(object):
    def __init__(self, metric):
        self.metric = metric
        
    def __call__(self, y_true, y_pred, *args, **kwargs):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        if y_true.shape != y_pred.shape:
            raise InvalidShape(err_msg = f'Unequal Shape. y_true->{y_true.shape} != y_pred->{y_pred.shape}')
        return self.metric(y_true, y_pred, *args, **kwargs)

############# Regression performance metrics ###################

# r2 score
@CheckMetricInput
def r2_score(y_true, y_pred):
    return 1 - (np.sum(np.square(y_true - y_pred))/np.sum(np.square(y_true - np.mean(y_true))))

# mean squared error
@CheckMetricInput
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# mean absolute error
@CheckMetricInput
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# root mean squared error
@CheckMetricInput
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


############# Classification performance metrics ###################

# classification accuracy score
@CheckMetricInput
def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

# binary cross entropy
@CheckMetricInput
def binary_cross_entropy(y_true, y_pred):
    return -(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))

# cross entropy
@CheckMetricInput
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true*np.log(y_pred))

# confusion matrix
@CheckMetricInput
def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    n_labels = len(labels)
    cm = np.empty((n_labels, n_labels), dtype=int)
    
    for t, yt in enumerate(labels):
        for p, yp in enumerate(labels):
            tot = np.sum(np.where(y_true==yt, 1, 0)==np.where(y_pred==yp, 1, -1))
            cm[t, p] = tot
    return cm

# cohen kappa
@CheckMetricInput
def cohen_kappa_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tot = np.sum(cm)
    
    pe, po = 0, accuracy_score(y_true, y_pred) # np.sum(np.diag(cm))/tot
    for i in range(cm.shape[0]):
        pe += (np.sum(cm[i, :])/tot)*(np.sum(cm[:, i])/tot)
    return (po-pe)/(1-pe)

# precision
@CheckMetricInput
def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    out = []
    for i in range(cm.shape[1]):
        out.append(cm[i, i]/np.sum(cm[i, :]))
    return np.array(out)

# recall
@CheckMetricInput
def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    out = []
    for i in range(cm.shape[1]):
        out.append(cm[i, i]/np.sum(cm[:, i]))
    return np.array(out)

# f-beta score
@CheckMetricInput
def f_beta_score(y_true, y_pred, beta):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    out = []
    for p, r in zip(precision_, recall_):
        out.append((1 + beta**2)*((p*r)/(((beta**2)*p)+r)))
    return out

# f1 score
@CheckMetricInput
def f1_score(y_true, y_pred):
    return f_beta_score(y_true, y_pred, 1)
