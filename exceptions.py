
class NotFittedError(Exception):
    """
        NotFittedError (Base Class: Exception)
            It is Custom Error, build to raise when an estimator is not fitted.
    """
    def __init__(self, estimator_name=None, err_msg=None):
        self.estimator_name = estimator_name
        self.err_msg = err_msg
        
    def __str__(self):
        if self.err_msg is None:
            return f'This {self.estimator_name} instance is not fitted yet. Call `fit` with appropriate arguments.'
        return self.err_msg
        
    
class InvalidShape(Exception):
    """
        InvalidError (Base Class: Exception)
            It is Custom Error, build to raise the shape of two arrays don't match.
    """
    def __init__(self, exp=None, recv=None, ftr_msg=False, err_msg=None):
        self.exp = exp
        self.recv = recv
        self.err_msg = err_msg
        self.ftr_msg = ftr_msg
        
    def __str__(self):
        if self.err_msg is None:
            if self.ftr_msg:
                return f'Expected no. of features is {self.exp}. Received no. of features {self.recv}.'
            else:
                return f'Expected shape is {self.exp}. Received shape is {self.recv}.'
        return self.err_msg
