import numbers
from entmoot import Optimizer
import numpy as np
from entmoot.acquisition import _gaussian_acquisition
from scipy.optimize import minimize

from entmoot.learning import EntingRegressor

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


# TODO: remove this eventually
def bfgs_optimize(
        func,
        dimensions,
        n_calls=60,
        batch_strategy="cl_mean",
        base_estimator="GBRT",
        n_initial_points=50,
        initial_point_generator="random",
        acq_func="LCB",
        acq_optimizer="global",
        x0=None,
        y0=None,
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        base_estimator_kwargs=None,
        model_queue_size=None,
        verbose=False,
        plot=False
):
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]

    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    required_calls = n_initial_points + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
    # calculate the total number of initial points
    n_initial_points = n_initial_points + len(x0)

    # Build optimizer
    # create optimizer class
    optimizer = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=random_state,
        acq_func_kwargs=acq_func_kwargs,
        acq_optimizer_kwargs=acq_optimizer_kwargs,
        base_estimator_kwargs=base_estimator_kwargs,
        model_queue_size=model_queue_size,
        verbose=verbose
    )

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        result = optimizer.tell(x0, y0)

    # Handle solver output
    if not isinstance(verbose, (int, type(None))):
        raise TypeError("verbose should be an int of [0,1,2] or bool, "
                        "got {}".format(type(verbose)))

    if isinstance(verbose, bool):
        if verbose:
            verbose = 1
        else:
            verbose = 0
    elif isinstance(verbose, int):
        if verbose not in [0, 1, 2]:
            raise TypeError("if verbose is int, it should in [0,1,2], "
                            "got {}".format(verbose))

    # Optimize
    _n_calls = n_calls

    itr = 1

    if verbose > 0:
        print("")
        print("SOLVER: start solution process...")
        print("")
        print(f"SOLVER: generate \033[1m {n_initial_points}\033[0m initial points...")

    while _n_calls > 0:

        # get next points
        _n_calls -= 1
        next_x = optimizer.ask(strategy=batch_strategy)

        next_y = func(next_x)

        # first iteration uses next_y as best point instead of min of next_y
        if itr == 1:
            best_fun = next_y

        # handle output print at every iteration
        if verbose > 0:
            print("")
            print(f"\033[1m itr_{itr}\033[0m")

            if isinstance(next_y, Iterable):
                # in case of batch optimization, print all new proposals and
                # mark improvements of objectives with (*)
                print_str = []
                for y in next_y:
                    if y <= best_fun:
                        print_str.append(f"\033[1m{round(y, 5)}\033[0m (*)")
                    else:
                        print_str.append(str(round(y, 5)))
                print(f"   new points obj.: {print_str[0]}")
                for y_str in print_str[1:]:
                    print(f"                    {y_str}")
            else:
                # in case of single point sequential optimization, print new
                # point proposal
                print(f"   new point obj.: {round(next_y, 5)}")

        # print best obj until (not including) current iteration
        print(f"   best obj.:       {round(best_fun, 5)}")

        itr += 1

        optimizer.tell(
            next_x, next_y,
            fit=not _n_calls <= 0
        )

        result = optimizer.get_result()

        best_fun = result.fun

    # print end of solve once convergence criteria is met
    if verbose > 0:
        print("")
        print("SOLVER: finished run!")
        print(f"SOLVER: best obj.: {round(result.fun, 5)}")
        print("")

    return result


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
    # Check inputs
    # assert acq_func in ["LCB", "EI", "CWEI"]

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

class BlackBoxConstraint:
    def __init__(self,
                 n_dim,
                 evaluator,
                 rhs):
        self.n_dim = n_dim
        self.evaluator = evaluator
        self.rhs = rhs

    def evaluate(self, X):
        X0 = np.reshape(X, (-1, self.n_dim))
        return self.evaluator(X0)

