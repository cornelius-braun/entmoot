import numpy as np
from joblib import delayed, Parallel
from lightgbm import LGBMRegressor
from sklearn.base import clone, BaseEstimator, RegressorMixin
from skgarden import MondrianForestRegressor as _sk_MondrianForestRegressor
import copy

from sklearn.utils import check_random_state


class EntingRegressor:
    """Predict with LightGBM tree model and include model uncertainty 
    defined by distance-based standard estimator.


    Parameters
    ----------
    base_estimator : LGBMRegressor instance, EntingRegressor instance or 
        None (default). If LGBMRegressor instance of None is given: new 
        EntingRegressor is defined. If EntingRegressor is given: base_estimator
        and std_estimator are given to new instance.
    std_estimator : DistanceBasedStd instance,
        Determines which measure is used to capture uncertainty.
    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.
    """

    def __init__(self,
                space,
                base_estimator,
                std_estimator,
                random_state=None,
                cat_idx=None):

        if cat_idx is None:
            cat_idx = []

        np.random.seed(random_state)

        self.random_state = random_state
        self.space = space
        self.base_estimator = base_estimator
        self.std_estimator = std_estimator
        self.cat_idx = cat_idx
        self.num_obj = len(self.base_estimator)

        # check if base_estimator is EntingRegressor
        if isinstance(base_estimator[0], EntingRegressor):
            self.base_estimator = base_estimator.base_estimator
            self.std_estimator = base_estimator.std_estimator

    def fit(self, X, y):
        """Fit model and standard estimator to observations.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape=(n_samples,)
            Target values (real numbers in regression)

        Returns
        -------
        -
        """
        self.regressor_ = []
        y = np.asarray(y)
        self._y = y

        for i,est in enumerate(self.base_estimator):
            # suppress lgbm output
            est.set_params(
                random_state=self.random_state,
                verbose=-1
            )

            # clone base_estimator (only supported for sklearn estimators)
            self.regressor_.append(clone(est))

            # update std estimator
            self.std_estimator.update(X, y, cat_column=self.cat_idx)

            # update tree model regressor
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.num_obj > 1:
                    if self.cat_idx:
                        self.regressor_[-1].fit(X, y[:,i], categorical_feature=self.cat_idx)
                    else:
                        self.regressor_[-1].fit(X, y[:,i])
                else:
                    if self.cat_idx:
                        self.regressor_[-1].fit(X, y, categorical_feature=self.cat_idx)
                    else:
                        self.regressor_[-1].fit(X, y)

    def set_params(self, **params):
        """Sets parameters related to tree model estimator. All parameter 
        options for LightGBM are given here: 
            https://lightgbm.readthedocs.io/en/latest/Parameters.html

        Parameters
        ----------
        kwargs : dict
            Additional arguments to be passed to the tree model

        Returns
        -------
        -
        """
        if not "min_child_samples" in params.keys():
            params.update({"min_child_samples":2})

        for i in range(len(self.base_estimator)):
            self.base_estimator[i].set_params(**params)

    def predict(self, X, return_std=False):
        """Predict.

        If `return_std` is set to False, only tree model prediction is returned.
        Return mean and predicted standard deviation, which is approximated 
        based on standard estimator specified, when `return_std` is set to True. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        mean or (mean, std): np.array, shape (n_rows, n_dims) 
            or tuple(np.array, np.array), depending on value of `return_std`.
        """

        if self.num_obj == 1:
            mean = self.regressor_[0].predict(X)
        else:
            mean = []
            for tree in self.regressor_:
                mean.append(tree.predict(X))

        if return_std:
            std = self.std_estimator.predict(X)
            return mean, std

        # return the mean
        return mean

    def get_gbm_model(self):
        """
        Returns `GbmModel` instance of the tree model.

        Parameters
        ----------
        -

        Returns
        -------
        gbm_model : `GbmModel`
            ENTMOOT native tree model format to formulate optimization model
        """

        from entmoot.learning.lgbm_processing import order_tree_model_dict
        from entmoot.learning.gbm_model import GbmModel
        gbm_model_dict = {}
        for i, tree in enumerate(self.regressor_):
            original_tree_model_dict = tree._Booster.dump_model()

            # import json
            # with open('tree_dict_next.json', 'w') as f:
            #     json.dump(
            #         original_tree_model_dict,
            #         f,
            #         indent=4,
            #         sort_keys=False
            #     )

            ordered_tree_model_dict = \
                order_tree_model_dict(
                    original_tree_model_dict,
                    cat_column=self.cat_idx
                )

            gbm_model_dict[i] = GbmModel(ordered_tree_model_dict)
        return gbm_model_dict

    def get_global_next_x(self,
                          acq_func,
                          acq_func_kwargs,
                          acq_optimizer_kwargs,
                          add_model_core,
                          weight,
                          verbose,
                          gurobi_env,
                          gurobi_timelimit,
                          has_unc=True):
        from entmoot.optimizer.gurobi_utils import \
            get_core_gurobi_model, add_gbm_to_gurobi_model, \
            add_std_to_gurobi_model, add_acq_to_gurobi_model, \
            set_gurobi_init_to_ref, get_gbm_obj_from_model, get_gbm_multi_obj_from_model

        # suppress  output to command window
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        # start building model
        gurobi_model = \
            get_core_gurobi_model(
                self.space, add_model_core, env=gurobi_env
            )

        if verbose == 2:
            print("")
            print("")
            print("")
            print("SOLVER: *** start gurobi solve ***")
            gurobi_model.Params.LogToConsole = 1
        else:
            gurobi_model.Params.LogToConsole = 0

        # convert into gbm_model format
        # and add to gurobi model
        gbm_model_dict = self.get_gbm_model()
        add_gbm_to_gurobi_model(
            self.space, gbm_model_dict, gurobi_model
        )

        # add std estimator to gurobi model
        if has_unc:
            add_std_to_gurobi_model(self, gurobi_model)

        # collect different objective function contributions
        from entmoot.optimizer.gurobi_utils import get_gbm_model_multi_obj_mu, get_gbm_model_mu

        if self.num_obj > 1:
            model_mu = get_gbm_model_multi_obj_mu(gurobi_model, self._y)

            if has_unc:
                model_unc = self.std_estimator.get_gurobi_obj(gurobi_model)
            else: model_unc = None
        else:
            model_mu = get_gbm_model_mu(gurobi_model, self._y, norm=False)
            model_unc = self.std_estimator.get_gurobi_obj(gurobi_model)

        # add obj to gurobi model
        from opti.sampling.simplex import sample

        weight = sample(self.num_obj,1)[0] if weight is None else weight

        add_acq_to_gurobi_model(gurobi_model, model_mu, model_unc,
                                self,
                                weights=weight,
                                num_obj=self.num_obj,
                                acq_func=acq_func,
                                acq_func_kwargs=acq_func_kwargs)

        # set initial gurobi model vars to std_est reference points
        if has_unc:
            if self.std_estimator.std_type == 'distance':
                set_gurobi_init_to_ref(self, gurobi_model)

        # set gurobi time limit
        if gurobi_timelimit is not None:
            gurobi_model.Params.TimeLimit = gurobi_timelimit
        gurobi_model.Params.OutputFlag = 1

        gurobi_model.update()
        gurobi_model.optimize()

        assert gurobi_model.SolCount >= 1, "gurobi couldn't find a feasible " + \
                                           "solution. Try increasing the timelimit if specified. " + \
                                           "In case you specify your own 'add_model_core' " + \
                                           "please check that the model is feasible."

        # store optimality gap of gurobi computation
        gurobi_mipgap = None
        if acq_func not in ["HLCB"]:
            gurobi_mipgap = gurobi_model.mipgap

        # for i in range(2): REMOVEX
        #     gurobi_model.params.ObjNumber = i
        #     # Query the o-th objective value
        #     print(f"{i}:")
        #     print(gurobi_model.ObjNVal)

        # output next_x
        next_x = np.empty(len(self.space.dimensions))

        # cont features
        for i in gurobi_model._cont_var_dict.keys():
            next_x[i] = gurobi_model._cont_var_dict[i].x

        # cat features
        for i in gurobi_model._cat_var_dict.keys():
            cat = \
                [
                    key
                    for key in gurobi_model._cat_var_dict[i].keys()
                    if int(
                    round(gurobi_model._cat_var_dict[i][key].x, 1)
                ) == 1
                ]

            next_x[i] = cat[0]

        model_mu = get_gbm_multi_obj_from_model(gurobi_model)
        if has_unc:
            model_std = gurobi_model._alpha.x
        else:
            model_std = None

        return next_x, model_mu, model_std, gurobi_mipgap

    def copy(self):
        return copy.copy(self)


class MisicRegressor(EntingRegressor):

    def __init__(self,
                space,
                base_estimator=None,
                std_estimator=None,
                random_state=None,
                cat_idx=None):

        if cat_idx is None:
            cat_idx = []

        self.random_state = random_state
        self.space = space
        self.base_estimator = base_estimator
        self.std_estimator = std_estimator
        self.cat_idx = cat_idx
        self.num_obj = len(self.base_estimator)

        # check if base_estimator is EntingRegressor
        if isinstance(base_estimator, MisicRegressor):
            self.base_estimator = base_estimator.base_estimator
            self.std_estimator = base_estimator.std_estimator

    def fit(self, X, y):
        """Fit model and standard estimator to observations.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape=(n_samples,)
            Target values (real numbers in regression)

        Returns
        -------
        -
        """
        self.regressor_ = []
        y = np.asarray(y)
        self._y = y

        for i,est in enumerate(self.base_estimator):
            # suppress lgbm output
            est.set_params(
                random_state=self.random_state,
                verbose=-1
            )

            # clone base_estimator (only supported for sklearn estimators)
            self.regressor_.append(clone(est))

            # update tree model regressor
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.num_obj > 1:
                    if self.cat_idx:
                        self.regressor_[-1].fit(X, y[:,i], categorical_feature=self.cat_idx)
                    else:
                        self.regressor_[-1].fit(X, y[:,i])
                else:
                    if self.cat_idx:
                        self.regressor_[-1].fit(X, y, categorical_feature=self.cat_idx)
                    else:
                        self.regressor_[-1].fit(X, y)

        gbm_model = self.get_gbm_model()[0]

        # update std estimator
        self.std_estimator.update(X, y, gbm_model, cat_column=self.cat_idx)


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

    def __init__(self, quantiles=[0.16, 0.5, 0.84], base_estimator=None,
                 n_jobs=1, random_state=None):
        self.quantiles = quantiles
        self.random_state = random_state
        self.base_estimator = base_estimator
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

        if self.base_estimator is None:
            base_estimator = LGBMRegressor(**self.params)
        else:
            base_estimator = self.base_estimator

            if not isinstance(base_estimator, LGBMRegressor):
                raise ValueError('base_estimator has to be of type'
                                 ' LGBMRegressor.')

        # The predictions for different quantiles should be sorted.
        # Therefore each of the regressors need the same seed.
        base_estimator.set_params(random_state=rng)
        regressors = []
        for q in self.quantiles:
            regressor = clone(base_estimator)
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


class MondrianForestRegressor(_sk_MondrianForestRegressor):
    def __init__(self):
        super(MondrianForestRegressor, self).__init__()


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
        ret = super(MondrianForestRegressor, self).predict(X, return_std=return_std)

        if return_std:
            mu, std = ret # _return_std(X, self.estimators_, mean, self.min_variance)
            return mu, std
        return ret

    def copy(self):
        return copy.copy(self)