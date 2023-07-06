import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ2

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 5
num_attributes = 4
attribute_func = DTLZ2(dim=input_dim, num_objectives=num_attributes, negate=True)
reference_vector = attribute_func(torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))


def utility_func(Y):
    return -torch.square(Y - reference_vector).sum(dim=-1)


# Algos
algo = "EUBO"
model_type = "composite_preferential_gp"

# estimate noise level
comp_noise_type = "logit"
noise_level = 0.0001

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="dtlz2",
    attribute_func=attribute_func,
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=50,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
