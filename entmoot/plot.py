import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from entmoot.learning.constraint import UnknownConstraintModel, BlackBoxConstraint
from entmoot.utils import predict_trained_est

def plotfx_1d(obj_f, surrogate_f, n_init, evaluated_points=None, next_x=None, const_f=None):
    # get data
    X, Z, Z_surrogate, std = get_plot_data(obj_f, n_dim=1, const_f=const_f, surrogate_f=surrogate_f)

    f, axes = plt.subplots(1, 1, figsize=(7, 5))
    axes.plot(X, Z)
    axes.plot(X, Z_surrogate, c="orange")
    axes.fill_between(X.flatten(), Z_surrogate-std, Z_surrogate+std, alpha=0.3, facecolor="orange")
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    # plot evaluations
    if evaluated_points is not None and len(evaluated_points) > 0:
        X = np.array(evaluated_points).reshape(-1, 1)
        axes.scatter(X[:n_init], obj_f(X[:n_init]), marker="x", color="grey")
        axes.scatter(X[n_init:], obj_f(X[n_init:]), color="blue", marker="x")
        axes.axvline(next_x, color="red")

    f.suptitle("Objective function")
    f.show()


# Plot
def plotfx_2d(obj_f, n_init, evaluated_points=None, next_x=None, const_f=None):
    # get data
    bounds = obj_f.get_bounds(2)
    X, Z = get_plot_data(obj_f, n_dim=2, const_f=const_f)

    f, axes = plt.subplots(1, 1, figsize=(7, 5))
    im = axes.contourf(X[0].flatten(), X[1].flatten(), Z.reshape(100, 100))
    axes.set_xlabel('x0')
    axes.set_ylabel('x1')
    axes.set_xlim([bounds[0][0], bounds[0][1]])
    axes.set_ylim([bounds[1][0], bounds[1][1]])

    # plot evaluations
    if evaluated_points is not None and len(evaluated_points) > 0:
        X = np.asarray(evaluated_points).reshape(-1, 2)
        next = np.asarray(next_x).reshape(-1, 2)
        axes.scatter(X[:n_init, 0], X[:n_init, 1], marker="+", color="grey")
        axes.scatter(X[n_init:, 0], X[n_init:, 1], marker="+", color="yellow")
        axes.scatter(next[:, 0], next[:, 1], marker="x", color="red")

    title = get_2d_plot_title(constraint_f=const_f)
    f.suptitle(title)
    f.colorbar(im)
    f.show()

def get_meshgrid(list_number_points_per_axis, dataset_bounds):
    list_grid_points = []
    for index_axis, (x_min, x_max) in enumerate(dataset_bounds):
        list_grid_points.append(np.linspace(x_min, x_max, list_number_points_per_axis[index_axis]))
    return np.asarray(np.meshgrid(*list_grid_points, sparse=True))

def get_plot_data(obj_f, n_dim=1, const_f=None, surrogate_f=None):
    if n_dim == 1:
        # get data
        bounds = obj_f.get_bounds(n_dim=1)
        X = get_meshgrid([100] * len(bounds), bounds).reshape(-1, 1)

        # evaluate
        Z = obj_f(X)
        if const_f is not None:
            Zc = const_f.evaluate(X)
            mask = Zc >= 0
            Zc[mask] = np.nan
            Zc[np.logical_not(mask)] = 1
            Z *= Zc

        if surrogate_f is None:
            return X, Z
        else:
            Z_surrogate, std_surrogate = predict_trained_est(surrogate_f, X)
            return X, Z, Z_surrogate, std_surrogate

    elif n_dim == 2:
        # get data
        bounds = obj_f.get_bounds(n_dim=2)
        X = get_meshgrid([100] * len(bounds), bounds)
        X0 = np.dstack(np.meshgrid(*X)).reshape(-1, 2)

        if const_f is None:
            Z = np.asarray(obj_f(X0))

        # determine feasible region
        # essentially this is the pof that is plotted
        elif isinstance(const_f, UnknownConstraintModel):
            Zc_mu, Zc_std = const_f.evaluate(X0)
            normal = norm(loc=Zc_mu, scale=Zc_std)
            # Z = pof
            Z = normal.cdf(const_f.rhs)

        elif isinstance(const_f, BlackBoxConstraint):
            Z = obj_f(X)
            X0 = np.dstack(np.meshgrid(*X)).reshape(-1, 2)
            Zc = const_f.evaluate(X0)

            mask = Zc >= const_f.rhs
            Zc[mask] = np.nan
            Zc[np.logical_not(mask)] = 1
            Z = Z.flatten() * Zc

        else:
            raise ValueError(f"Wrong type of constraint, "
                             f"must be in [UnknownConstraintModel, BlackBoxConstraint], but is {type(const_f)}")

        return X, Z

    else:
        raise ValueError("n dim must be in [1, 2] for plotting!")


def get_2d_plot_title(constraint_f):
    if constraint_f is None:
        return "Objective function"
    if isinstance(constraint_f, UnknownConstraintModel):
        return "Approximated probability of feasibility"
    if isinstance(constraint_f, BlackBoxConstraint):
        return "Feasible region"