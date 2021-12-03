from typing import Callable, Dict, List, Optional

import os
import sys

from boss.boss_trial import boss_trial


def experiment_manager(
    problem: str,
    algo: str,
    algo_params: Dict,
    restart: bool,
    first_trial: int,
    last_trial: int,
    objective_function: Callable,
    cost_function: Callable,
    input_dim: int,
    n_init_evals: int,
    budget: float,
    n_max_iter: int = 300,
    ignore_failures: bool = False,
) -> None:

    for trial in range(first_trial, last_trial + 1):
        boss_trial(
            problem=problem,
            algo=algo,
            algo_params=algo_params,
            trial=trial,
            restart=restart,
            objective_function=objective_function,
            cost_function=cost_function,
            input_dim=input_dim,
            n_init_evals=n_init_evals,
            budget=budget,
            n_max_iter=n_max_iter,
            ignore_failures=ignore_failures,
        )
