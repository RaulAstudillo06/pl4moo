import torch

from kumaraswamy_distribution import Kumaraswamy


class KumaraswamyCDF(torch.nn.Module):
    def __init__(self, concentration1, concentration2, Y_bounds):
        super().__init__()
        self.register_buffer("concentration1", concentration1)
        self.register_buffer("concentration2", concentration2)
        self.register_buffer("Y_bounds", Y_bounds)
        self.kdist = Kumaraswamy(concentration1, concentration2)

    def calc_raw_util_per_dim(self, Y):
        Y_bounds = self.Y_bounds

        Y = (Y - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        eps = 1e-6
        Y = torch.clamp(Y, min=eps, max=1 - eps)

        util_val = self.kdist.cdf(Y)

        return util_val

    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = util_val[..., ::2] * util_val[..., 1::2]
        util_val = util_val.sum(dim=-1)

        return util_val


class KumaraswamyCDFProduct(KumaraswamyCDF):
    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = torch.prod(util_val, dim=-1)

        return util_val