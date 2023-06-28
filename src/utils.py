import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, PosteriorMean
from botorch.generation.gen import get_best_candidates
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.likelihoods import PairwiseLogitLikelihood, PairwiseProbitLikelihood
from botorch.optim.optimize import optimize_acqf
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel
from typing import Optional, Callable
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.models.composite_model import CompositePreferentialGP
from src.pymoo_problem import PymooProblem

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize


def fit_model(
    queries: Tensor,
    responses: Tensor,
    attribute_func: Callable,
    model_type: str,
    likelihood: Optional[str] = "logit",
):
    if model_type == "pairwise_gp":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)

        if likelihood == "probit":
            likelihood_func = PairwiseProbitLikelihood()
        else:
            likelihood_func = PairwiseLogitLikelihood()
        model = PairwiseGP(
            datapoints,
            comparisons,
            likelihood=likelihood_func,
            jitter=1e-4,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=model.likelihood, model=model
        )
        fit_gpytorch_mll(mll)
        model = model.to(device=queries.device, dtype=queries.dtype)
    elif model_type == "pairwise_kernel_variational_gp":
        model = PairwiseKernelVariationalGP(queries, responses)
    elif model_type == "composite_preferential_gp":
        model = CompositePreferentialGP(queries, responses, attribute_func)
    return model


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    attribute_func,
    utility_func,
    comp_noise_type,
    comp_noise,
    seed: int = None,
):
    # generates initial data
    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    attribute_vals, utility_vals = get_attribute_and_utility_vals(
        queries, attribute_func, utility_func
    )
    responses = generate_responses(utility_vals, comp_noise_type, comp_noise)
    return queries, attribute_vals, utility_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generate `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def get_attribute_and_utility_vals(queries, attribute_func, utility_func):
    attribute_vals = attribute_func(queries)
    utility_vals = utility_func(attribute_vals)
    return attribute_vals, utility_vals


def generate_responses(utility_vals, noise_type, noise_level):
    # generate simulated comparisons based on true underlying objective
    corrupted_utility_vals = corrupt_vals(utility_vals, noise_type, noise_level)
    responses = torch.argmax(corrupted_utility_vals, dim=-1)
    return responses


def corrupt_vals(vals, noise_type, noise_level):
    # corrupts (attribute or utility) values to simulate noise in the DM's responses
    if noise_type == "noiseless":
        corrupted_vals = vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=vals.shape)
        corrupted_vals = vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape)
        corrupted_vals = vals + noise
    elif noise_type == "constant":
        corrupted_vals = vals.clone()
        n = vals.shape[0]
        for i in range(n):
            coin_toss = Bernoulli(noise_level).sample().item()
            if coin_toss == 1.0:
                corrupted_vals[i, 0] = vals[i, 1]
                corrupted_vals[i, 1] = vals[i, 0]
    return corrupted_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    batch_size = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = batch_size * i + responses[i]
        comparison = [best_item_id]
        for j in range(batch_size):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparison.append(batch_size * i + j)
        comparisons.append(torch.tensor(comparison).unsqueeze(0))

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 2,
    init_batch_limit: Optional[int] = 30,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    aug_bounds = torch.cat([bounds for _ in range(batch_size)], dim=-1)
    problem = PymooProblem(
        var=aug_bounds.shape[-1], bounds=aug_bounds.numpy(), acq_function=acq_func
    )
    algorithm = GA(pop_size=5 * aug_bounds.shape[-1], eliminate_duplicates=True)

    res = pymoo_minimize(problem, algorithm, verbose=True)
    new_x = res.X
    new_x = torch.from_numpy(new_x)
    if batch_size > 1:
        new_x = new_x.reshape(
            2,
            int(new_x.shape[-1] / 2),
        )
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # candidates, acq_values = optimize_acqf(
    #     acq_function=acq_func,
    #     bounds=bounds,
    #     q=batch_size,
    #     num_restarts=num_restarts,
    #     raw_samples=raw_samples,
    #     batch_initial_conditions=batch_initial_conditions,
    #     options={
    #         "batch_limit": batch_limit,
    #         "init_batch_limit": init_batch_limit,
    #         "maxiter": 100,
    #         "nonnegative": False,
    #         "method": "L-BFGS-B",
    #     },
    #     return_best_only=False,
    # )
    #
    # candidates = candidates.detach()
    # # acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    # # print("Acquisition values:")
    # # print(acq_values_sorted)
    # # print("Candidates:")
    # # print(candidates[indices].squeeze())
    # # print(candidates.squeeze())
    # # print(candidates.shape)
    # # print(acq_values.shape)
    # new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    # print(new_x)

    return new_x


def compute_posterior_mean_maximizer(
    model: Model,
    model_type,
    input_dim: int,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim

    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return max_post_mean_func
