import copy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from skgarden import MondrianForestRegressor as _sk_MondrianForestRegressor
from joblib import delayed, Parallel
from lightgbm import LGBMRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

# New experiment
class LGBMBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    """Predict several quantiles with one estimator.

        This is a wrapper around `GradientBoostingRegressor`'s quantile
        regression that allows you to predict several `quantiles` in
        one go.

        Parameters
        ----------
        quantiles : array-like
            Quantiles to predict. By default the 16, 50 and 84%
            quantiles are predicted.

        base_estimator : GradientBoostingRegressor instance or None (default)
            Quantile regressor used to make predictions. Only instances
            of `GradientBoostingRegressor` are supported. Use this to change
            the hyper-parameters of the estimator.

        n_jobs : int, default=1
            The number of jobs to run in parallel for `fit`.
            If -1, then the number of jobs is set to the number of cores.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        """

    def __init__(self, quantiles=[0.16, 0.5, 0.84], base_estimator=None, n_jobs=1, random_state=None):
        self.quantiles = quantiles
        self.random_state = random_state

        # define base estimator
        self.base_estimator = base_estimator
        if self.base_estimator is None:
            self.base_estimator = LGBMRegressor(**self.params)
        else:
            if not isinstance(self.base_estimator, LGBMRegressor):
                raise ValueError('base_estimator has to be of type LGBMRegressor.')

        self.n_jobs = n_jobs
        self.params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'min_child_samples': 2,
            'boosting_type': 'gbdt'
        }

    def fit(self, X, y):
        """Fit one regressor for each quantile.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape=(n_samples,)
            Target values (real numbers in regression)
        """
        rng = check_random_state(self.random_state)

        # The predictions for different quantiles should be sorted.
        # Therefore each of the regressors need the same seed.
        self.base_estimator.set_params(random_state=rng)
        regressors = []
        for q in self.quantiles:
            regressor = clone(self.base_estimator)
            regressor.set_params(alpha=q)

            regressors.append(regressor)

        self.regressors_ = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._parallel_fit)(regressor, X, y)
            for regressor in regressors)

        return self

    def _parallel_fit(self, regressor, X, y):
        return regressor.fit(X, y)

    def predict(self, X, return_std=False, return_quantiles=False):
        """Predict.

        Predict `X` at every quantile if `return_std` is set to False.
        If `return_std` is set to True, then return the mean
        and the predicted standard deviation, which is approximated as
        the (0.84th quantile - 0.16th quantile) divided by 2.0

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        predicted_quantiles = np.asarray(
            [rgr.predict(X) for rgr in self.regressors_])
        if return_quantiles:
            return predicted_quantiles.T

        elif return_std:
            std_quantiles = [0.16, 0.5, 0.84]
            is_present_mask = np.in1d(std_quantiles, self.quantiles)
            if not np.all(is_present_mask):
                raise ValueError(
                    "return_std works only if the quantiles during "
                    "instantiation include 0.16, 0.5 and 0.84")
            low = self.regressors_[self.quantiles.index(0.16)].predict(X)
            high = self.regressors_[self.quantiles.index(0.84)].predict(X)
            mean = self.regressors_[self.quantiles.index(0.5)].predict(X)
            return mean, ((high - low) / 2.0)

        # return the mean
        return self.regressors_[self.quantiles.index(0.5)].predict(X)

    def copy(self):
        return copy.copy(self)

    def set_params(self, **params):
        self.base_estimator.set_params(**params)


# TODO: docstring
class MondrianForestRegressor(_sk_MondrianForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 std_type="default"
                 ):
        super(MondrianForestRegressor, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        self.std_type = std_type


    def predict(self, X, return_std=False):
        """Predict continuous output for X.

        Parameters
        ----------
        X : array of shape = (n_samples, n_features)
            Input data.

        return_std : boolean
            Whether or not to return the standard deviation.

        Returns
        -------
        predictions : array-like of shape = (n_samples,)
            Predicted values for X. If criterion is set to "mse",
            then `predictions[i] ~= mean(y | X[i])`.

        std : array-like of shape=(n_samples,)
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.

        """
        if self.std_type == "default":
            ret = super(MondrianForestRegressor, self).predict(X, return_std=return_std)
        elif self.std_type == "ensembling":
            ret = self.sample_ensembles(X, return_std=return_std)

        if return_std:
            mu, std = ret
            return mu, std
        return ret

    def copy(self):
        return copy.copy(self)

    def set_params(self, **params):
        # delete non-existing keys for MF
        if "min_child_samples" in params.keys():
            del params["min_child_samples"]

        if "verbose" in params.keys():
            del params["verbose"]

        self.base_estimator.set_params(**params)

    def sample_ensembles(self, X, dropout=0.1, n_iter=10, return_std=False):
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model has to be fit before prediction.")
        min_trees = np.ceil(len(self.estimators_) / 2)
        ensemble_mean = np.zeros((n_iter, X.shape[0]))

        # sample i random sub-ensembles
        n_trees = max(int(len(self.estimators_) * (1 - dropout)), min_trees)
        for i in range(n_iter):
            predictions = np.zeros(X.shape[0])
            estimators = np.random.choice(self.estimators_, n_trees)
            for est in estimators:
                predictions += est.predict(X, return_std=False)
            ensemble_mean[i] = predictions / n_trees

        mean = np.mean(ensemble_mean, axis=0)
        if not return_std:
            return mean
        std = np.std(ensemble_mean, axis=0) * 2
        return mean, std