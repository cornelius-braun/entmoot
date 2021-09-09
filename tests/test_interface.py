import opti
import pandas as pd
from lightgbm import Booster

from entmoot.optimizer import EntmootOpti


def test_api():
    # Definition of test problem
    test_problem = opti.problems.Zakharov_categorical(n_inputs=3)
    test_problem.create_initial_data(100)

    # Declaration of entmoot instanceTrain surrogate model
    surrogat_params = {"verbose": -1}
    entmoot = EntmootOpti(problem=test_problem, surrogat_params=surrogat_params)

    assert entmoot.model is None

    # Train surrogate model
    entmoot._fit_model()

    assert type(entmoot.model) == Booster

    X_pred = pd.DataFrame([
        {"x0": 5, "x1": 5, "expon_switch": "one"},
        {"x0": 0, "x1": 0, "expon_switch": "two"}
    ])

    # Prediction based on surrogate model
    y_pred = entmoot.predict(X_pred)

    assert len(y_pred) == 2

    # Optimize acquisition function
    n_proposals = 1
    X_next = entmoot.propose(n_proposals=n_proposals)

    assert len(X_next) == n_proposals

    # Run Bayesian Optimization loop
    # entmoot.run()
