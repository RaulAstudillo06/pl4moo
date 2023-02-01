import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import lazify


class PairwiseKernel(Kernel):
    """
    Wrapper to convert a kernel K on R^k to a kernel K' on R^{2k}, modeling
    functions of the form g(a, b) = f(a) - f(b), where f ~ GP(mu, K).

    Since g is a linear combination of Gaussians, it follows that g ~ GP(0, K')
    where K'((a,b), (c,d)) = K(a,c) - K(a, d) - K(b, c) + K(b, d).

    """

    def __init__(self, latent_kernel, input_dim, is_partial_obs=False, **kwargs):
        super(PairwiseKernel, self).__init__(**kwargs)

        self.latent_kernel = latent_kernel
        self.input_dim = input_dim
        self.is_partial_obs = is_partial_obs

    def forward(self, x1, x2, diag=False, **params):
        r"""
        TODO: make last_batch_dim work properly

        d must be 2*k for integer k, k is the dimension of the latent space
        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `diag`: `n` or `b x n`
        """
        if not diag:
            return (
                lazify(
                    self.latent_kernel(
                        x1[..., : self.input_dim],
                        x2[..., : self.input_dim],
                        diag=diag,
                        **params
                    )
                )
                + lazify(
                    self.latent_kernel(
                        x1[..., self.input_dim :],
                        x2[..., self.input_dim :],
                        diag=diag,
                        **params
                    )
                )
                - lazify(
                    self.latent_kernel(
                        x1[..., self.input_dim :],
                        x2[..., : self.input_dim],
                        diag=diag,
                        **params
                    )
                )
                - lazify(
                    self.latent_kernel(
                        x1[..., : self.input_dim],
                        x2[..., self.input_dim :],
                        diag=diag,
                        **params
                    )
                )
            )
        else:
            return (
                self.latent_kernel(
                    x1[..., : self.input_dim],
                    x2[..., : self.input_dim],
                    diag=diag,
                    **params
                )
                + self.latent_kernel(
                    x1[..., self.input_dim :],
                    x2[..., self.input_dim :],
                    diag=diag,
                    **params
                )
                - self.latent_kernel(
                    x1[..., self.input_dim :],
                    x2[..., : self.input_dim],
                    diag=diag,
                    **params
                )
                - self.latent_kernel(
                    x1[..., : self.input_dim],
                    x2[..., self.input_dim :],
                    diag=diag,
                    **params
                )
            )
