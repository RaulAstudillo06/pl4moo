from typing import Callable, Dict, List, Optional

import os
import sys

from src.pbo_trial import pbo_trial


def experiment_manager(
    problem: str, # problem id
    attribute_func: Callable, # attribute function (x -> h(x))
    utility_func: Callable, # utility function (y=h(x) -> g(y))
    input_dim: int, # input dimension
    num_attributes: int, # number of attributes (i.e., output dimension of h)
    comp_noise_type: str, # type of comparison noise ("probit" and "logit" noise are supported)
    comp_noise: float, # scalar determining the magnitude of the comparison noise
    algo: str, # algo or acquisition function id ("qEUBO", "qTS", and "Random" are supported)
    batch_size: int, # number of items in the query (e.g., 2 for standard binary queries)
    num_init_queries: int, # number of initial queries (these are selected uniformly at random over the input space)
    num_algo_iter: int, # number of queries selected by the algorithm
    first_trial: int, # first trial id to be ran
    last_trial: int, # last trial id to be ran
    restart: bool, # if True, this will try to restart the experiment from existing data
    model_type: str, # type of model ("Standard" and "Composite" are supported)
    ignore_failures: bool = False, # ignore this for now
    algo_params: Optional[Dict] = None, # ignore this for now 
) -> None:
    
    # `trial` determines the random seed of each trial
    for trial in range(first_trial, last_trial + 1):
        pbo_trial(
            problem=problem,
            attribute_func=attribute_func,
            utility_func=utility_func,
            input_dim=input_dim,
            num_attributes=num_attributes,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            algo=algo,
            algo_params=algo_params,
            batch_size=batch_size,
            num_init_queries=num_init_queries,
            num_algo_iter=num_algo_iter,
            trial=trial,
            restart=restart,
            model_type=model_type,
            ignore_failures=ignore_failures,
        )
