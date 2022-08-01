import numpy as np
from itertools import chain, combinations

class Kernels(object):
    """
        Kernels 
            It is a base class for different types of kernels.
    """
    def __init__(self, increase_dim = False):
        self.increase_dim = increase_dim
    
    
class LinearKernel(Kernels):
    """
        LinearKernel (Base Class: Kernels)
            It is the Linear Kernel.

        Parameters:
            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: False

        Input:
            X: numpy array of 2-Dimensional
            Z: numpy array of 2-Dimensional
    """
    def __init__(self, increase_dim = False):
        super().__init__(increase_dim)
    
    def __call__(self, X, Z):
        
        if self.increase_dim:
            nx = X.shape[0]
            nz = Z.shape[0]
            K = np.ones((nx, nz))
            if nz==1:
                for d in range(len(Z[0])):
                    K[:, 0] *= (1 + X[:, d]*Z[:, d])
            else:
                for i in range(nz//2):
                    for d in range(len(Z[i])):
                        K[:, i] *= (1 + X[:, d]*Z[i, d])
                        K[:, nz-i-1] *= (1 + X[:, d]*Z[nz-i-1, d])
                        if (nz%2==1) & (i==(nz//2)-1):
                            K[:, nz//2] *= (1 + X[:, d] * Z[nz//2, d])        
            return K
        
        else:
            return X@Z.T
    
    
class PolynomialKernel(Kernels):
    """
        PolynomialKernel (Base Class: Kernels)
            It is the Polynomial Kernel.

        Parameters:
            power:
                -type: float
                -about: defines the degree for polynomial kernel.
                -default: 2.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: False

        Input:
            X: numpy array of 2-Dimensional
            Z: numpy array of 2-Dimensional
    """
    def __init__(self, power = 2.0, increase_dim = False):
        super().__init__(increase_dim)
        self.power = power
        self.linear_kernel = LinearKernel(increase_dim)
    
    def __call__(self, X, Z):
        return (1 + self.linear_kernel(X, Z)) ** self.power

    
    
class RadialBasisFunctionKernel(Kernels):
    """
        RadialBasisFunctionKernel (Base Class: Kernels)
            It is the Radial Basis Function Kernel (or rbf).

        Parameters:
            sigma:
                -type: float
                -about: it is used when kernel is set to `rbf`.
                -default: 3.0

            increase_dim:
                -type: boolean
                -about: whether to increase dimension of the vector
                        while calculating kernel.
                -default: False

        Input:
            X: numpy array of 2-Dimensional
            Z: numpy array of 2-Dimensional
    """    
    def __init__(self, sigma = 3.0, increase_dim = False):
        super().__init__(increase_dim)
        self.sigma = sigma

    def __powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def __convert_to_high_dim(self, V):
        KV = np.empty((V.shape[0], 2**V.shape[1]))
        for e, l in enumerate(V):
            k_ = [1]
            for i in self.__powerset(l):
                if len(i) != 0:
                    k_.append(np.prod(i))
            KV[e, :] = k_
        return KV
    
    def __call__(self, X, Z):
        KX = self.__convert_to_high_dim(X) if self.increase_dim else X
        KZ = self.__convert_to_high_dim(Z) if self.increase_dim else Z


        # Sometimes, while experimenting with some data.
        # `euclidean` tends to work well as compared to `squared euclidean` in rbf kernel.
        # for ex; in gaussian processes.
        # TODO: Check if its True for most of the time with most of the models or not.
        K = np.sum(np.square(np.expand_dims(KX, axis = 1) - KZ), axis = -1) / np.square(self.sigma + 1e-8)
        return np.exp(-K)

    
