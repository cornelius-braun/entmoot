import numpy as np
import copy
import inspect
import numbers
from entmoot.optimizer.optimizer import Optimizer
from entmoot.plot import plotfx_1d, plotfx_2d
from entmoot.utils import get_verbosity
from typing import List

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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

def bb_constraints_minimize(
    func,
    dimensions,
    bb_constraints: List[BlackBoxConstraint] = None,
    n_calls=60,
    batch_size=None,
    batch_strategy="cl_mean",
    n_points=10000,
    base_estimator="ENTING",
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
    # TODO: docstring
    if bb_constraints is None:
        raise ValueError("No black box constraints have been specified. Please specify constraints or use 'entmoot_minimize'")

    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    if acq_optimizer_kwargs is None:
        acq_optimizer_kwargs = {}

    acq_optimizer_kwargs["n_points"] = n_points

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
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

    # get the constraint r
    rhs_list = []
    for constraint in bb_constraints:
        rhs_list.append(constraint.rhs)

    # check dims
    if plot and len(dimensions) > 2:
        raise ValueError(f"can only plot up to 2D objectives, your dimensionality is {len(dimensions)}")

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        bb_constraints=rhs_list,
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
        # FIXME: this needs testing!!!
        y_feas = [constraint.evaluate(x0) for constraint in bb_constraints]
        result = optimizer.tell(x0, y0, const_y=y_feas)
        result.specs = specs

    # Handle solver output
    verbose = get_verbosity(verbose)

    # Optimize
    _n_calls = n_calls

    itr = 1

    if verbose > 0:
        print("")
        print("SOLVER: start solution process...")
        print("")
        print(f"SOLVER: generate \033[1m {n_initial_points}\033[0m initial points...")

    while _n_calls > 0:

        # Ask
        # check if optimization is performed in batches
        if batch_size is not None:
            _batch_size = min([_n_calls, batch_size])
            _n_calls -= _batch_size
            next_x = optimizer.ask(_batch_size, strategy=batch_strategy)
        else:
            _n_calls -= 1
            next_x = optimizer.ask(strategy=batch_strategy)

        # get next value of objective and constraint surrogates
        next_y = func(next_x)
        next_const = [constraint.evaluate(next_x) for constraint in bb_constraints] # check this for speed: https://stackoverflow.com/questions/11736407/apply-list-of-functions-on-an-object-in-python

        # first iteration uses next_y as best point instead of min of next_y
        if itr == 1:
            if batch_size is None:
                best_fun = next_y
            else:
                best_fun = min(next_y)

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

        # plot objective function
        if plot and optimizer.num_obj == 1:
            n_dim = len(dimensions)
            if n_dim == 1:
                est = optimizer.base_estimator_
                est.fit(optimizer.space.transform(optimizer.Xi), optimizer.yi)
                plotfx_1d(obj_f=func, surrogate_f=est, evaluated_points=optimizer.Xi, next_x=next_x)
            elif n_dim == 2:
                plotfx_2d(obj_f=func, evaluated_points=optimizer.Xi, next_x=next_x)


        # tell next
        optimizer.tell(
            next_x, next_y, next_const,
            fit=batch_size is None and not _n_calls <= 0
        )

        result = optimizer.get_result()
        best_fun = result.fun
        result.specs = specs
        itr += 1

    # print end of solve once convergence criteria is met
    if verbose > 0:
        print("")
        print("SOLVER: finished run!")
        print(f"SOLVER: best obj.: {round(result.fun, 5)}")
        print("")

    return result