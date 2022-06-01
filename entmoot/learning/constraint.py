import numpy as np

class BlackBoxConstraint:
    """
    Base class to formulate back box constraints.
    Every class object specifies a constraint of the shape 'evaluator' <= 'rhs'.

    Parameters
    ----------
    n_dim : int,
        number of dimensions that we operate in.

    evaluator : function, -> float
        the left-hand side of the constraint. Can be any function that maps to R.

    rhs : float,
        right-hand side of a less-than constraint.
    """
    def __init__(self,
                 n_dim: int,
                 evaluator,
                 rhs: int
                 ):
        self.n_dim = n_dim
        self.evaluator = evaluator
        self.rhs = rhs

    def evaluate(self, X):
        X0 = np.reshape(X, (-1, self.n_dim))
        return float(self.evaluator(X0))

class UnknownConstraintModel:
    """Predict value of unknown constraint surrogate and store the RHS of the constraint

        Parameters
        ----------
        base_estimator : Regressor model instance.
        RHS : float
            Right-hand side of a constraint of the shape <= RHS.
        """

    def __init__(self,
                 base_estimator,
                 rhs: int):
        self.model = base_estimator
        self.rhs = rhs

    def evaluate(self, X, return_std=True):
        return self.model.predict(X, return_std=return_std)

    def fit(self, X, y):
        return self.model.fit(X, y)