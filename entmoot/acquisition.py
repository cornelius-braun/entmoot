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
#import sys
from scipy.stats import norm

def _gaussian_acquisition(X, model, y_opt=None,
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
            raise ValueError("y_opt cannot needs to have a value!")
        print("best y:", y_opt)
        acq_vals = cw_ei(X, obj_model=model, obj_y_opt=y_opt, constraint_model=None)
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
        gamma = (mu - y_opt) / std

        return std * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

def cw_ei(X, obj_y_opt, obj_model, constraint_model=None):
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
    ei = expected_improvement(X, obj_model, obj_y_opt)
    pof = constraint_model.predict(X) if constraint_model is not None else 1.  # check if constraints are satisfied
    return ei * pof
