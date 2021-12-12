import os

import lightgbm as lgb
import opti
import pandas as pd

from entmoot.optimizer import EntmootOpti
from gurobipy import Env


def get_gurobi_env():
    """Return a Gurobi CloudEnv if environment variables are set, else None."""
    if "GRB_CLOUDPOOL" in os.environ:
        return Env.CloudEnv(
            logfilename="gurobi.log",
            accessID=os.environ["GRB_CLOUDACCESSID"],
            secretKey=os.environ["GRB_CLOUDKEY"],
            pool=os.environ["GRB_CLOUDPOOL"],
        )
    return None


def test_api():
    # Definition of test problem
    test_problem = opti.problems.Zakharov_Categorical(n_inputs=3)
    n_init = 15
    test_problem.create_initial_data(n_init)

    base_estimator_params = {
        "lgbm_params": {"min_child_samples": 2},
        "unc_metric": 'exploration',
        "unc_scaling": "standard",
        "dist_metric": "manhattan"
    }

    entmoot = EntmootOpti(
        problem=test_problem,
        base_est_params=base_estimator_params,
        gurobi_env=get_gurobi_env,
    )

    X_pred = pd.DataFrame(
        [
            {"x0": 5, "x1": 5, "expon_switch": "one"},
            {"x0": 0, "x1": 0, "expon_switch": "two"},
        ]
    )

    # Prediction based on surrogate model
    y_pred = entmoot.predict(X_pred)
    assert len(y_pred) == 2

    # Optimize acquisition function
    X_next: pd.DataFrame = entmoot.propose(n_proposals=10)
    assert len(X_next) == 10

    # Run Bayesian Optimization loop
    n_steps = 3
    n_proposals = 7
    entmoot.run(n_steps=3, n_proposals=7)
    assert len(entmoot.problem.data) == n_init + n_steps * n_proposals


def test_mixed_constraints():
    # single objective, linear-equality + n-choose-k constraints
    problem = opti.problems.Photodegradation()
    entmoot = EntmootOpti(problem=problem, gurobi_env=get_gurobi_env)

    X_pred = problem.data[problem.inputs.names]
    y_pred = entmoot.predict(X_pred)
    assert len(y_pred) == len(X_pred)

    X_next = entmoot.propose(n_proposals=2)
    assert len(X_next) == 2


def test_no_initial_data():
    # Using Entmoot on a problem without data should raise an informative error.
    problem = opti.problems.Zakharov_Categorical(n_inputs=3)

    try:
        EntmootOpti(problem)
    except ValueError:
        assert True
    else:
        assert False


def test_biobjective():
    # opti.problems.ReizmanSuzuki -> bi-objective, cat + cont variables
    problem = opti.problems.ReizmanSuzuki()

    entmoot = EntmootOpti(problem=problem, gurobi_env=get_gurobi_env)

    X_pred = problem.data[problem.inputs.names]
    y_pred = entmoot.predict(X_pred)

    assert len(y_pred) == len(X_pred)

    X_next = entmoot.propose(n_proposals=2)

    assert len(X_next) == 2


def test_with_missing_data():
    # In the multi-objective case we the model should handle missing data
    pass