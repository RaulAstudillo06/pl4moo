from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


class CompositePreferentialGP(Model):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        attribute_func: Tensor,
    ) -> None:
        super().__init__()
        self.queries = queries
        self.responses = responses
        self.attribute_func = attribute_func
        self.utility_model = PairwiseKernelVariationalGP(
            attribute_func(queries), responses
        )

    def posterior(self, X: Tensor, posterior_transform=None) -> MultivariateNormal:
        Y = self.attribute_func(X)
        return self.utility_model.posterior(Y)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
