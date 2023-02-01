import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import lazify


class BatchMultioutputKernel(Kernel):
    """
    Wrapper to convert a kernel K .

    """

    def __init__(self, latent_kernel, batch_size, **kwargs):
        super(BatchMultioutputKernel, self).__init__(**kwargs)

        self.latent_kernel = latent_kernel
        self.batch_size = batch_size

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


        d = x1.shape[-1]

        assert d == x2.shape[-1], "tensors not the same dimension"
        assert d % self.batch_size == 0, "dimension must be even"

        x1_reshaped = x1.reshape(x1.shape[:-1] + (self.batch_size, int(d / self.batch_size)))
        x2_reshaped = x2.reshape(x2.shape[:-1] + (self.batch_size, int(d / self.batch_size)))
        self.latent_kernel(x1_reshaped, x2_reshaped, diag=diag, **params)
