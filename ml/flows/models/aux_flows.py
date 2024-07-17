import numpy as np
import scipy.linalg as sp_linalg
import torch
import torch.nn as nn
from torch.nn import functional as F


class Flow(nn.Module):
    def __init__(self):
        """Base class for all flows.

        Forward and inverse must also return log of Jacobian determinant summed over feature dimension.

        Basic idea is taken from [1] (note that original code is very buggy).

        Note
        ----
        forward - generative direction
        inverse - normalzing direction (assumed when training)

        References
        ----------
        [1] - Normalizing flows with PyTorch: https://github.com/acids-ircam/pytorch_flows

        """
        super().__init__()
        # True if uses masking and implements set_mask
        self.mask = False
        # if False forward is proper nn.Module forward else use inverse in NormalizingFlow base class
        self.normalizing_direction = True

    def forward(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError


class SimpleBatchNormFlow(Flow):
    def __init__(self, input_size, momentum=0.9, eps=1e-5, full_sum=True):
        """Batch normalization flow. See [1, 2].

        Parameters
        ----------
        input_size : int
            Input dimension.
        rho : float, optional
            Momentum, by default 0.9.
        eps : float, optional
            Additive constant, by default 1e-5.
        full_sum: bool
            If True sums over all dimensions. Else sums only feature dimension and leaves batch dimension.

        References
        ----------
        [1] - Normalizing Flows for Probabilistic Modeling and Inference (section 3.4): https://arxiv.org/abs/1912.02762
        [2] - Density estimation using Real NVP (appendix E): https://arxiv.org/abs/1605.08803
        [3] - https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py#L162

        """
        super().__init__()
        self.normalizing_direction = False
        self.momentum = momentum
        self.eps = eps
        self.full_sum = full_sum

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0, unbiased=False)

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)

        if self.full_sum:
            sum_log_det = log_det.sum() * x.new_ones(x.shape[0], 1)
        else:
            sum_log_det = log_det.expand_as(x).sum(dim=-1, keepdim=True)

        return y, sum_log_det

    def inverse(self, x, **kwargs):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_det = 0.5 * torch.log(var + self.eps) - self.log_gamma

        if self.full_sum:
            sum_log_det = log_det.sum() * x.new_ones(x.shape[0], 1)
        else:
            sum_log_det = log_det.expand_as(x).sum(dim=-1, keepdim=True)

        return x, sum_log_det


class BatchNormFlow(Flow):
    def __init__(self, features, eps=1e-5, momentum=0.1):
        """Transform that performs batch normalization.

        References
        ----------
        [1] - https://github.com/bayesiains/nflows/blob/master/nflows/transforms/normalization.py

        """
        super().__init__()
        self.normalizing_direction = False
        self.momentum = momentum
        self.eps = eps
        constant = np.log(np.exp(1 - eps) - 1)
        self.unconstrained_weight = nn.Parameter(constant * torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.zeros(features))

    @property
    def weight(self):
        return F.softplus(self.unconstrained_weight) + self.eps

    def forward(self, inputs):
        if inputs.dim() != 2:
            raise ValueError("Expected 2-dim inputs, got inputs of shape: {}".format(inputs.shape))

        if self.training:
            mean, var = inputs.mean(0), inputs.var(0)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var

        outputs = self.weight * ((inputs - mean) / torch.sqrt((var + self.eps))) + self.bias

        logabsdet_ = torch.log(self.weight) - 0.5 * torch.log(var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0], 1)

        return outputs, logabsdet

    def inverse(self, inputs):
        # Batch norm inverse should only be used in eval mode, not in training mode.
        outputs = torch.sqrt(self.running_var + self.eps) * ((inputs - self.bias) / self.weight) + self.running_mean

        logabsdet_ = -torch.log(self.weight) + 0.5 * torch.log(self.running_var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0], 1)

        return outputs, logabsdet


class ReverseFlow(Flow):
    def __init__(self, forward_dim, inverse_dim=None, repeats=1):
        super().__init__()
        self.normalizing_direction = False

        self.permute = torch.arange(forward_dim - 1, -1, -1).repeat(1, repeats).squeeze()

        if inverse_dim is None:
            self.inverse_permute = torch.argsort(self.permute)
        else:
            permute_ = torch.arange(inverse_dim - 1, -1, -1)
            self.inverse_permute = torch.argsort(permute_)

    def forward(self, z):
        return z[:, self.permute], torch.zeros_like(z).sum(dim=-1, keepdim=True)

    def inverse(self, z):
        return z[:, self.inverse_permute], torch.zeros_like(z).sum(dim=-1, keepdim=True)


class ShuffleFlow(Flow):
    def __init__(self, input_dim, groups, shuffle_first_last=True):
        """Shuffling (permutations) flow. Uses channel shuffling from [1].

        Parameters
        ----------
        input_dim : int
            Input dimension.
        groups : list
            See figure 1 in [1].
        shuffle_first_last : bool, optional
            Interchanges first and last element, by default True.

        References
        ----------
        [1] - ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices: https://arxiv.org/abs/1707.01083
        [2] - https://github.com/jaxony/ShuffleNet/blob/master/model.py

        """
        super().__init__()
        assert input_dim % 2 == 0
        assert input_dim > 2
        splits = input_dim // groups

        self.normalizing_direction = False

        self.permute = torch.arange(input_dim)
        self.permute = self.permute.view(groups, splits)
        self.permute = self.permute.T
        self.permute = torch.flatten(self.permute)

        if shuffle_first_last:
            t1, t2 = self.permute[0].item(), self.permute[-1].item()
            self.permute[0], self.permute[-1] = t2, t1

        self.inverse_permute = torch.argsort(self.permute)

    def forward(self, z):
        return z[:, self.permute], torch.zeros_like(z).sum(dim=-1, keepdim=True)

    def inverse(self, z):
        return z[:, self.inverse_permute], torch.zeros_like(z).sum(dim=-1, keepdim=True)


class Conv1x1PLU(nn.Module):
    def __init__(self, dim, device="cpu"):
        """Invertible 1x1 convolution using LU decomposition.

        References
        ----------
        [1] - https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py

        """
        super().__init__()
        self.normalizing_direction = False
        self.device = device
        self.dim = dim

        W, _ = sp_linalg.qr(np.random.randn(dim, dim))
        P, L, U = sp_linalg.lu(W)

        self.P = torch.tensor(P, dtype=torch.float, device=self.device)
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float, device=self.device))
        self.S = nn.Parameter(torch.tensor(np.diag(U), dtype=torch.float, device=self.device))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float, device=self.device), diagonal=1))

    def forward(self, x):
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))
        U = torch.triu(self.U, diagonal=1)

        W = self.P @ L @ (U + torch.diag(self.S))
        z = x @ W

        log_det = torch.sum(torch.log(torch.abs(self.S)))

        return z, log_det * x.new_ones(x.shape[0], 1)

    def inverse(self, z):
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))
        U = torch.triu(self.U, diagonal=1)

        W = self.P @ L @ (U + torch.diag(self.S))
        W_inv = torch.inverse(W)
        x = z @ W_inv

        log_det = -torch.sum(torch.log(torch.abs(self.S)))

        return x, log_det * z.new_ones(z.shape[0], 1)


class Conv1x1(nn.Module):
    def __init__(self, dim, device="cpu"):
        """Invertible 1x1 convolution using convolutions (i.e. matrix multiplication for 1x1 channels).

        References
        ----------
        [1] - Glow: Generative Flow with Invertible 1x1 Convolutions: https://arxiv.org/abs/1807.03039

        """
        super().__init__()
        self.normalizing_direction = False
        self.device = device
        self.dim = dim

        W, _ = sp_linalg.qr(np.random.randn(dim, dim))

        self.W = nn.Parameter(torch.Tensor(W.astype(np.float32)).to(device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x):
        z = F.conv2d(x.view(*x.shape, 1, 1), self.W.view(*self.W.shape, 1, 1), bias=self.bias)

        det = torch.sum(torch.log(torch.abs(torch.linalg.det(self.W))))

        return z.squeeze(), det * x.new_ones(x.shape[0], 1)

    def inverse(self, z):
        W_inv = torch.inverse(self.W)
        x = F.conv2d(z.view(*z.shape, 1, 1), W_inv.view(*W_inv.shape, 1, 1), bias=self.bias)

        det = -torch.sum(torch.log(torch.abs(torch.linalg.det(W_inv))))

        return x.squeeze(), det * z.new_ones(z.shape[0], 1)
