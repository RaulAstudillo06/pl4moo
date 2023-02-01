#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.emov import (
    ExpectedMaxObjectiveValue,
    qExpectedMaxObjectiveValue,
)
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_attribute_and_utility_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
    compute_posterior_mean_maximizer
)


# this function runs a single trial of a given problem
# see more details about the arguments in experiment_manager.py 
def pbo_trial(
    problem: str,
    attribute_func: Callable,
    utility_func: Callable,
    input_dim: int,
    num_attributes: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    trial: int,
    restart: bool,
    model_type: str,
    ignore_failures: bool,
    algo_params: Optional[Dict] = None,
) -> None:

    algo_id = algo + "_" + model_type

    # get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # check if training data is available
        try:
            # current available iterations
            queries = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries.shape[0], batch_size, int(queries.shape[1] / batch_size)
            )
            queries = torch.tensor(queries)
            attribute_vals = torch.tensor(
                np.loadtxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt")
            )
            attribute_vals = attribute_vals.reshape(
                attribute_vals.shape[0], batch_size, int(attribute_vals.shape[1] / batch_size)
            )
            utility_vals = torch.tensor(
                np.loadtxt(
                    results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt"
                )
            )
            responses = torch.tensor(
                np.loadtxt(
                    results_folder + "responses/responses_" + str(trial) + ".txt"
                )
            )
            # historical max utility values within queries
            max_utility_vals_within_queries = list(
                np.loadtxt(
                    results_folder
                    + "max_utility_vals_within_queries_"
                    + str(trial)
                    + ".txt"
                )
            )
            # historical utility values at the maximum of the posterior mean
            utility_vals_at_max_post_mean = list(
                np.loadtxt(
                    results_folder
                    + "utility_vals_at_max_post_mean_"
                    + str(trial)
                    + ".txt"
                )
            )
            # historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_utility_vals_within_queries) - 1
            print("Restarting experiment from available data.")

        except:
            # initial data
            queries, attribute_vals, utility_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                batch_size=batch_size,
                input_dim=input_dim,
                attribute_func=attribute_func,
                utility_func=utility_func,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                seed=trial,
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # historical utility values at the maximum of the posterior mean
            posterior_mean_maximizer = compute_posterior_mean_maximizer(model=model, model_type=model_type, input_dim=input_dim)
            utility_val_at_max_post_mean = utility_func(
            attribute_func(posterior_mean_maximizer)
            ).item()
            utility_vals_at_max_post_mean = [utility_val_at_max_post_mean]

            # historical max utility values within queries and runtimes
            max_utility_val_within_queries = utility_vals.max().item()
            max_utility_vals_within_queries = [max_utility_val_within_queries]

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # initial data
        queries, attribute_vals, utility_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            attribute_func=attribute_func,
            utility_func=utility_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            seed=trial,
        )

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # historical utility values at the maximum of the posterior mean
        posterior_mean_maximizer = compute_posterior_mean_maximizer(model=model, model_type=model_type, input_dim=input_dim)
        utility_val_at_max_post_mean = utility_func(
        attribute_func(posterior_mean_maximizer)
        ).item()
        utility_vals_at_max_post_mean = [utility_val_at_max_post_mean]

        # historical max utility values within queries and runtimes
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries = [max_utility_val_within_queries]

        # historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_algo_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # new suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # get response at new query
        (
            new_attribute_vals,
            new_utility_vals,
        ) = get_attribute_and_utility_vals(new_query, attribute_func, utility_func)
        new_responses = generate_responses(
            new_utility_vals,
            noise_type=comp_noise_type,
            noise_level=comp_noise,
        )

        # update training data
        queries = torch.cat((queries, new_query))
        attribute_vals = torch.cat([attribute_vals, new_attribute_vals], 0)
        utility_vals = torch.cat([utility_vals, new_utility_vals], 0)
        responses = torch.cat((responses, new_responses))

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # compute and append current utility value at the maximum of the posterior mean
        posterior_mean_maximizer = compute_posterior_mean_maximizer(model=model, model_type=model_type, input_dim=input_dim)
        utility_val_at_max_post_mean = utility_func(
        attribute_func(posterior_mean_maximizer)
        ).item()
        utility_vals_at_max_post_mean.append(utility_val_at_max_post_mean)
        print(
            "Utility value at the maximum of the posterior mean: "
            + str(utility_val_at_max_post_mean)
        )

        # append current max utility val within queries
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries.append(max_utility_val_within_queries)
        print(
            "Max utility value within queries: " + str(max_utility_val_within_queries)
        )

        # save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "attribute_vals/"):
            os.makedirs(results_folder + "attribute_vals/")
        if not os.path.exists(results_folder + "utility_vals/"):
            os.makedirs(results_folder + "utility_vals/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        attribute_vals_reshaped = attribute_vals.numpy().reshape(attribute_vals.shape[0], -1)
        np.savetxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt", attribute_vals_reshaped)
        np.savetxt(
            results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt",
            utility_vals.numpy(),
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )
        np.savetxt(
            results_folder + "utility_vals_at_max_post_mean_" + str(trial) + ".txt",
            np.atleast_1d(utility_vals_at_max_post_mean),
        )
        np.savetxt(
            results_folder + "max_utility_vals_within_queries_" + str(trial) + ".txt",
            np.atleast_1d(max_utility_vals_within_queries),
        )

def get_new_suggested_query(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 2 * input_dim * batch_size
    raw_samples = 60 * input_dim * batch_size
    batch_initial_conditions = None

    if algo == "Random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "ANALYTIC_EUBO":
        acquisition_function = ExpectedMaxObjectiveValue(model=model)
    elif algo == "EUBO":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        acquisition_function = qExpectedMaxObjectiveValue(model=model, sampler=sampler)
    elif algo == "TS":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_query(
            model, batch_size, standard_bounds, 2 * input_dim, 60 * input_dim
        )

    new_query = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
    )

    new_query = new_query.unsqueeze(0)
    return new_query
