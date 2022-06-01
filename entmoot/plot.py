import numpy as np
import matplotlib.pyplot as plt
from entmoot.utils import predict_trained_est

# Plot
def plotfx_1d(obj_f, surrogate_f, evaluated_points=None, next_x=None, const_f=None):
    # get data
    bounds = obj_f.get_bounds(n_dim=1)
    X = get_meshgrid([100], bounds).reshape(-1, 1)


    # evaluate
    Z = obj_f(X)
    Z_surrogate = predict_trained_est(surrogate_f, X, return_std=False) # TODO: use map here
    if const_f is not None:
        Zc = const_f(X)
        mask = Zc >= 0
        Zc[mask] = np.nan
        Zc[np.logical_not(mask)] = 1
        Z *= Zc

    f, axes = plt.subplots(1, 1, figsize=(7, 5))
    axes.plot(X.flatten(), Z)
    axes.plot(X.flatten(), Z_surrogate)
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    # plot evaluations
    if evaluated_points is not None and len(evaluated_points) > 0:
        axes.axvline(next_x, color="red")

    f.suptitle("Objective function")
    f.show()


# Plot
def plotfx_2d(obj_f, evaluated_points=None, next_x=None, const_f=None):
    # get data
    bounds = obj_f.get_bounds(n_dim=2)
    X = get_meshgrid([100]*2, bounds)

    # evaluate
    Z = obj_f(X)
    if const_f is not None:
        Zc = const_f(X)
        mask = Zc >= 0
        Zc[mask] = np.nan
        Zc[np.logical_not(mask)] = 1
        Z *= Zc

    f, axes = plt.subplots(1, 1, figsize=(7, 5))
    axes.contourf(X[0].flatten(), X[1].flatten(), Z)
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.set_xlim([bounds[0][0], bounds[0][1]])
    axes.set_ylim([bounds[1][0], bounds[1][1]])

    # plot evaluations
    if evaluated_points is not None and len(evaluated_points) > 0:
        X = np.array(evaluated_points).T
        axes.scatter(*X, marker="+", color="blue")
        axes.scatter(*next_x, marker="x", color="red")

    f.suptitle("Objective function")
    f.show()

def get_meshgrid(list_number_points_per_axis, dataset_bounds):
    list_grid_points = []
    for index_axis, (x_min, x_max) in enumerate(dataset_bounds):
        list_grid_points.append(np.linspace(x_min, x_max, list_number_points_per_axis[index_axis]))
    return np.asarray(np.meshgrid(*list_grid_points, sparse=True))