"""
Copyright (c) 2016-2020 The scikit-optimize developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

NOTE: Changes were made to the scikit-optimize source code included here. 
For the most recent version of scikit-optimize we refer to:
https://github.com/scikit-optimize/scikit-optimize/

Copyright (c) 2019-2020 Alexander Thebelt.
"""

import numpy as np
import warnings
from scipy.stats import norm
from typing import Optional
from scipy.optimize import minimize

def _gaussian_acquisition(X,
                          model,
                          y_opt=None,
                          constraint_pof: Optional[list] = None,
                          num_obj=1,
                          acq_func="LCB",
                          acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # not available for multi-obj prediction
    assert num_obj == 1, f"acquisition predict is not available for 'num_obj > 1'"

    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X is {}-dimensional, however,"
                         " it must be 2-dimensional.".format(X.ndim))

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()

    kappa = acq_func_kwargs.get("kappa", 1.96)

    # Evaluate acquisition function
    if acq_func == "LCB":
        acq_vals = gaussian_lcb(X, model, kappa, acq_func_kwargs=acq_func_kwargs)
    elif acq_func == "EI":
        if y_opt is None:
            raise ValueError("y_opt cannot needs to have a value!")
        print("best y:", y_opt)
        acq_vals = expected_improvement(X, model, y_opt=y_opt)
    elif acq_func == "CWEI":
        if y_opt is None:
            raise ValueError("y_opt needs to have a value!")
        if constraint_pof is None:
            raise ValueError("constraint_pof needs to be defined!")
        acq_vals = cw_ei(X, obj_model=model, obj_y_opt=y_opt, pof=constraint_pof)
    else:
        raise ValueError("Acquisition function not implemented.")

    return acq_vals

def gaussian_lcb(X, model, kappa=1.96, return_grad=False, acq_func_kwargs=None):
    """
    Use the lower confidence bound to estimate the acquisition
    values.

    The trade-off between exploitation and exploration is left to
    be controlled by the user through the parameter ``kappa``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    kappa : float, default 1.96 or 'inf'
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        If set to 'inf', the acquisition function will only use the variance
        which is useful in a pure exploration setting.
        Useless if ``method`` is not set to "LCB".

    Returns
    -------
    values : array-like, shape (X.shape[0],)
        Acquisition function values computed at X.

    """
    # Compute posterior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mu, std = model.predict(X, return_std=True)

        if kappa == "inf":
            return -std

        return mu - kappa * std


def expected_improvement(X, model, y_opt):
    """
    Returns the expected improvement acquisition function

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    y_opt: array-like, shape (1, n_features)
        The best function value that was found so far.

    Returns
    -------
    values : array-like, shape (X.shape[0],)
        Acquisition function values computed at X.

    """
    # Compute posterior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mu, std = model.predict(X, return_std=True)
        gamma = get_gamma(mu, y_opt, std)

        return std * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

def cw_ei(X, obj_y_opt, obj_model, pof=None):
    """
    Use the constraint weighted expected improvement to
    return the constraint weighted expected improvement

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Values where the acquisition function should be computed.
    obj_y_opt : array-like, shape (1, n_features)
        The best function value that was found so far.
    obj_model : sklearn estimator of the objective function that implements predict with ``return_std``
        The fit estimator that approximates the objective function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.
    constraint_model : sklearn estimator that implements predict
        The fit estimator that approximates the constraint function through the method ``predict``.

    Returns
    -------
    values : array-like, shape (X.shape[0],)
        Acquisition function values computed at X.
    """
    #if pof is None:
    #    raise ValueError("Constraint probabilities are not defined")
    ei = expected_improvement(X, obj_model, obj_y_opt)
    pof = prob_of_feasibility(X, pof) if pof is not None else 1.  # check if constraints are satisfied
    return ei * pof

def prob_of_feasibility(X, models):
    # idea: here we evaluate how likely it is that the constraint is met
    # i.e. given some data we check prob. of observing the desired value or smaller
    # only works for inequality constraints as of now (for equality it would just be pdf)

    # loop over all models and predict the constraint surrogate
    pof = 1.
    for model in models:
        mu, std = model.evaluate(X, return_std=True)
        normal = norm(loc=mu, scale=std)
        pof *= normal.cdf(model.rhs)    # multiply the cdf values
    return pof

def get_gamma(X, y_opt, model_uncertainty=None):
    if model_uncertainty is not None:
        gamma = (X - y_opt) / model_uncertainty
    else:
        gamma = X - y_opt
    return gamma

def bfgs_max_acq(X_tries,
                 X_seeds,
                 model,
                 y_opt=None,
                 constraint_pof=None,
                 num_obj=1,
                 acq_func="LCB",
                 space=None,
                 acq_func_kwargs=None
                 ):

    # define proxy for acquisition
    acquisition_fct = lambda X: _gaussian_acquisition(X=X, model=model, y_opt=y_opt, constraint_pof=constraint_pof,
                                                      num_obj=num_obj, acq_func=acq_func,
                                                      acq_func_kwargs=acq_func_kwargs)

    # Warm up with random points
    ys = acquisition_fct(X_tries)  # acquisitions[acq_func](X_tries, model)
    x_max = X_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    for x_try in X_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -acquisition_fct(np.array(x).reshape(1, -1)),
                       x_try.reshape(1, -1),
                       bounds=space.bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    # return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    model_mu, model_std = model.predict(np.asarray(x_max).reshape(1, -1), return_std=True)

    return x_max, model_mu, model_std
