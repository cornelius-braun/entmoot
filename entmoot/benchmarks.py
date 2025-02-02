# TODO: credits
"""Different benchmark problems for testing"""

import numpy as np
from entmoot.utils import is_2Dlistlike
from abc import ABC, abstractmethod
class BenchmarkFunction(ABC):

    def __init__(self, func_config={}):
        self.name = 'benchmark_function'
        self.func_config = func_config
        self.y_opt = 0.0

    def get_bounds(self, n_dim=2):
        pass

    def get_X_opt(self, n_dim=2):
        pass

    def __call__(self, X):
        # check if multiple points are given
        if not is_2Dlistlike(X):
            X = [X]

        res = []

        for x in X:
            res.append(self._eval_point(x))
        
        if len(res) == 1:
            res = res[0]

        return res
    
    @abstractmethod
    def _eval_point(self,x):
        pass

class Rosenbrock(BenchmarkFunction):

    def __init__(self, func_config={}):
        self.name = 'rosenbrock'
        self.func_config = func_config

    def get_bounds(self, n_dim=2):
        return [(-2.048,2.048) for _ in range(n_dim)]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X)
        X0 = X[:-1]
        X1 = X[1:]

        add1 = sum( (1.0 - X0)**2.0 )
        add2 = 100.0 * sum( (X1 - X0**2.0)**2.0 )
        return add1 + add2

class Parabola(BenchmarkFunction):
    def __init__(self, func_config={}):
        self.name = 'parabola'
        self.func_config = func_config

    def get_bounds(self, n_dim=1):
        return [(-2.048, 2.048) for _ in range(n_dim)]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X)
        return np.sum(X**2)

class SimpleCat(BenchmarkFunction):

    def __init__(self, func_config={}):
        from skopt.space import Categorical
        self.cat_dims = [
            Categorical(['mult6','pow2'])
        ]
        self.name = 'benchmark_function'
        self.func_config = func_config

    def get_bounds(self, n_dim=2):
        temp_bounds = [(-2.0,2.0) for _ in range(n_dim)]
        temp_bounds.extend(self.cat_dims)
        return temp_bounds

    def _eval_point(self, X):
        cat = X[-1]
        X = np.asarray_chkfinite(X[:-1])
        X0 = X[:-1]
        X1 = X[1:]

        add1 = X0[0]
        add2 = X1[0]
        
        if cat == 'mult6':
            return 6*(add1 + add2)
        elif cat == 'pow2':
            return (add1 + add2)**2
        else:
            raise ValueError("Please pick a category from '['mult2','pow2']' for X[-1].")

class FonzecaFleming(BenchmarkFunction):
    # test function from: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    # code based on: https://github.com/scwolof/HBMOO/blob/master/test_problems.py

    def __init__ (self, n=2):
        self.name = 'FonzecaFleming'
        self.n = n

    @property
    def invn(self):
        return 1. / np.sqrt(self.n)

    @property
    def r (self):
        return np.array([[0.,1.],[0.,1.]])

    def get_bounds(self):
        return np.array([[-4.,4.]]*self.n)

    def f1 (self,x):
        return self.f(x,1)
    def f2 (self,x):
        return self.f(x,2)
    def f (self,x,i):
        x = np.asarray(x)
        xx = (x + (2*i-3)*self.invn)**2
        sx = np.sum(xx, axis = None if x.ndim == 1 else 1)
        return 1. - np.exp(-sx)

    def _eval_point(self, X):
        return self.f1(X), self.f2(X)


class RosenbrockMulti(BenchmarkFunction):

    def __init__(self, func_config={}):
        self.name = 'rosenbrock_multi'
        self.func_config = func_config

    def get_bounds(self, n_dim=2):
        return [(-2.048,2.048) for _ in range(n_dim)]

    def get_X_opt(self, n_dim=2):
        return [[ 1.0 for _ in range(n_dim) ]]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X)
        X0 = X[:-1]
        X1 = X[1:]

        add1 = sum( (1.0 - X0)**2.0 )
        add2 = 100.0 * sum( (X1 - X0**2.0)**2.0 )
        return add1 + add2, add2*5/add1


class Periodic(BenchmarkFunction):
    def __init__(self, func_config={}):
        self.name = 'periodic'
        self.func_config = func_config

    def get_bounds(self, n_dim=1):
        return [(-20.48, 20.48) for _ in range(n_dim)]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X).sum()
        return np.sin(X) + np.sin((10. / 3.) * X)


class Townsend(BenchmarkFunction):
    """
    Townsend function: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    inspired by https://gpflowopt.readthedocs.io/en/latest/notebooks/constrained_bo.html
    """
    def __init__(self, n_dim=2):
        self.name = "Townsend"
        self.n_dim = n_dim
        if self.n_dim != 2:
            raise ValueError("Townsend only defined for 2D")

    def get_bounds(self, n_dim=2):
        return [(-2.25, 2.5), (-2.5, 1.75)]

    def _eval_point(self, x):
        a = np.cos((x[0] - 0.1) * x[1]) ** 2
        b = x[0] * np.sin(3 * x[0] + x[1])
        return -(a + b)

class Branin(BenchmarkFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        if self.n_dim != 2:
            raise ValueError("Branin only defined for 2D")
        self.name = 'Branin'

    def get_bounds(self, n_dim=2):
        return [(-5.0, 10.0), (0.0, 15.0)]

    def _eval_point(self, x):
        b = 5.1 / (4 * np.pi ** 2)
        c = 5. / np.pi
        s = 10
        t = 1. / (8 * np.pi)
        return (x[1] - b * x[0] ** 2 + c * x[0] - 6) ** 2 + \
               s * (1 - t) * np.cos(x[0]) + s