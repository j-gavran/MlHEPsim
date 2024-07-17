import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# inheritance chart:

# BaseMADE (uses MaskedLinear and create_masks)
# MADE (handles last mask) <- BaseMADE
# GaussianMADE (Gaussian factorization) <- MADE <- BaseMADE

# BaseResMADE (uses ResMaskedLinear and create_masks)
# ResMADE (handles last mask) <- BaseResMADE
# GaussianResMADE (Gaussian factorization) <- ResMADE <- BaseResMADE


def create_masks(D_in, L, K, D_out, random_degree=False):
    """MADE masks.

    l ... layer, k ... neuron, L ... last layer, D ... input dimension, m ... assigned degree

    A unit may only be connected to units in the previous layer whose degrees strictly do not exceed its own.

    Parameters
    ----------
    D_in : int
        Input dimension.
    L : int
        Number of hidden layers (not counting input layer).
    K : int
        Number of hidden neurons in hidden layers.
    D_out : int
        Output dimension
    random_degree : bool, optional
        Assign degrees at random, by default False (very poor performance if True).

    References
    ----------
    [1] - MADE: Masked Autoencoder for Distribution Estimation: https://arxiv.org/abs/1502.03509

    Returns
    -------
    list of weights and list of degrees

    """
    assert D_in == D_out

    degrees = [torch.arange(1, D_in + 1)]

    # assign degrees
    for l in range(L):
        layer_degrees = []
        for k in range(K):
            if random_degree:
                layer_degrees.append(torch.randint(torch.min(degrees[l]).item(), D_in, (1,)))
            else:
                layer_degrees.append(k % (D_in - 1) + 1)
        degrees.append(torch.Tensor(layer_degrees).to(torch.int64))

    degrees.append(torch.arange(1, D_out + 1))

    # input and hidden layers
    Ms = []
    for l in range(L):
        k1_degrees, k2_degrees = degrees[l], degrees[l + 1]

        M = torch.zeros(len(k2_degrees), len(k1_degrees))

        for k2, k2_degree in enumerate(k2_degrees):
            for k1, k1_degree in enumerate(k1_degrees):
                if k2_degree >= k1_degree:
                    M[k2, k1] = 1
        Ms.append(M)

    # last layer
    d_degrees, k_degrees = degrees[-1], degrees[-2]

    M = torch.zeros(len(d_degrees), len(k_degrees))
    for d, d_degree in enumerate(d_degrees):
        for k, k_degree in enumerate(k_degrees):
            if d_degree > k_degree:
                M[d, k] = 1

    Ms.append(M)

    return Ms, degrees


class MaskedLinear(nn.Linear):
    """Torch linear block with masking in forward."""

    def __init__(self, input_size, n_outputs, mask):
        super().__init__(input_size, n_outputs)
        self.register_buffer("mask", mask)

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        return out


class BaseMADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        activation="ReLU",
    ):
        """Masked Autoencoder for Distribution Estimation. Modified from [2].

        Parameters
        ----------
        input_size : int
            Input dimension.
        hidden_size : int
            Number of neurons per each layer (assume same for all layers).
        n_hidden : int
            Number of hidden layers (not counting first and last layer).
        activation : str, optional
            Activation function, by default "relu".

        Note
        ----
        This is a general autoregressive model.

        References
        ----------
        [1] - Masked Autoregressive Flow for Density Estimation: https://arxiv.org/abs/1705.07057
        [2] - https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

        """
        super().__init__()
        self.normalizing_direction = False
        self.input_size, self.hidden_size = input_size, hidden_size
        activation_fn = getattr(nn, activation)()

        output_size = input_size  # different output sizes are handled by mask duplication in the last layer

        self.masks, self.degrees = create_masks(input_size, n_hidden, hidden_size, output_size)

        net_input = MaskedLinear(input_size, hidden_size, self.masks[0])
        self.net = [net_input]

        for m in self.masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]

        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, output_size, self.masks[-1]),
        ]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class MADE(BaseMADE):
    def __init__(self, input_size, *args, out_mask_multiplier=None, **kwargs):
        super().__init__(input_size, *args, **kwargs)
        self.out_mask_multiplier = out_mask_multiplier

        if self.out_mask_multiplier > 2:
            deg = self.degrees[-2]
            deg_sort = deg.sort()[0]
            deg_rep = deg_sort.repeat(out_mask_multiplier * self.input_size // len(deg) + 1)
            idx = deg_rep[: out_mask_multiplier * self.input_size].argsort()
            new_deg = deg_rep[idx]

            new_mask = (new_deg[:, None] > deg).float()

            self.net = nn.Sequential(
                self.net[:-1],
                MaskedLinear(
                    self.hidden_size,
                    out_mask_multiplier * self.input_size,
                    new_mask,
                ),
            )
        elif self.out_mask_multiplier == 2:
            self.net = nn.Sequential(
                self.net[:-1], MaskedLinear(self.hidden_size, 2 * self.input_size, self.masks[-1].repeat(2, 1))
            )
        else:
            logging.warning("Not using output mask multiplier in MADE!")


class GaussianMADE(MADE):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(input_size, *args, out_mask_multiplier=2, **kwargs)

    def forward(self, x):
        # MAF eq 4, return mean and log std
        mu, alpha = self.net(x).chunk(chunks=2, dim=1)

        u = (x - mu) * torch.exp(-alpha)

        # MAF eq 5
        log_abs_det_jacobian = torch.sum(-alpha, dim=-1, keepdim=True)

        return u, log_abs_det_jacobian

    def inverse(self, u):
        # MAF eq 3
        x = torch.zeros_like(u)

        # run through reverse model
        for i in range(self.input_size):
            mu, alpha = self.net(x).chunk(chunks=2, dim=1)

            x[:, i] = u[:, i] * torch.exp(alpha[:, i]) + mu[:, i]

        log_abs_det_jacobian = alpha.sum(dim=-1, keepdim=True)

        return x, log_abs_det_jacobian


class ResMaskedLinear(nn.Module):
    def __init__(self, k, mask, activation, n=2, zero_initialization=True, batchnorm=False, act_first=False):
        """
        References
        ----------
        [1] - Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027

        """
        super().__init__()
        self.n = n
        layers = []

        for _ in range(n):
            if batchnorm and act_first:
                layers.append(activation)
                layers.append(nn.BatchNorm1d(k))
            elif batchnorm and not act_first:
                layers.append(nn.BatchNorm1d(k))
                layers.append(activation)
            else:
                layers.append(activation)

            layers.append(MaskedLinear(k, k, mask))

        self.layers = nn.Sequential(*layers)

        if zero_initialization:
            init.zeros_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        temp = x

        for layer in self.layers:
            temp = layer(temp)

        return x + temp


class BaseResMADE(BaseMADE):
    def __init__(self, input_size, k, n_blocks, l=2, activation="ReLU", res_batchnorm=False, act_first=False, **kwargs):
        """MADE with residual connections.

        Note
        ----
        Input order is assumed sequential. Batch normalization usually does not work well with this model.

        Parameters
        ----------
        input_size : int
            Dimension of input layer.
        k : int
            Number of neurons in each layer.
        n_blocks : int, optional
            Number of repeated residual blocks.
        l : int, optional
            Number of layers in each residual block, by default 2.
        activation : str, optional
            Activation in residual blocks, by default "ReLU".
        res_batchnorm : bool, optional
            Batch normalization in residual blocks, by default False.
        act_first : bool, optional
            Activation before batch normalization, by default False.

        References
        ----------
        [1] - Autoregressive Energy Machines, https://arxiv.org/abs/1904.05626

        """
        super().__init__(input_size, k, l, **kwargs)
        activation_fn = getattr(nn, activation)()
        self.blocks = []

        output_size = input_size

        self.masks, _ = create_masks(input_size, 2, k, output_size)

        input_layer_mask = self.masks[0]
        hidden_layer_mask = self.masks[1]
        self.output_layer_mask = self.masks[2]

        net_input = MaskedLinear(input_size, k, input_layer_mask)
        self.blocks.append(net_input)

        for _ in range(n_blocks):
            self.blocks.append(
                ResMaskedLinear(
                    k,
                    hidden_layer_mask,
                    activation_fn,
                    l,
                    batchnorm=res_batchnorm,
                    act_first=act_first,
                )
            )

        self.blocks.append(activation_fn)
        self.blocks.append(MaskedLinear(k, output_size, self.output_layer_mask))
        self.net = nn.Sequential(*self.blocks)


class ResMADE(BaseResMADE):
    def __init__(self, input_size, k, n_blocks, l=2, out_mask_multiplier=None, **kwargs):
        super().__init__(input_size, k, n_blocks, l, **kwargs)
        self.out_mask_multiplier = out_mask_multiplier

        if self.out_mask_multiplier > 2:
            deg = self.degrees[-2]
            deg_sort = deg.sort()[0]
            deg_rep = deg_sort.repeat(out_mask_multiplier * self.input_size // len(deg) + 1)
            idx = deg_rep[: out_mask_multiplier * self.input_size].argsort()
            new_deg = deg_rep[idx]

            new_mask = (new_deg[:, None] > deg).float()

            self.net = nn.Sequential(
                self.net[:-1],
                MaskedLinear(
                    self.hidden_size,
                    out_mask_multiplier * self.input_size,
                    new_mask,
                ),
            )
        elif self.out_mask_multiplier == 2:
            self.net = nn.Sequential(
                self.net[:-1], MaskedLinear(self.hidden_size, 2 * self.input_size, self.masks[-1].repeat(2, 1))
            )
        else:
            logging.warning("Not using output mask multiplier in ResMADE!")


class GaussianResMADE(ResMADE):
    def __init__(self, input_size, k, n_blocks, l=2, **kwargs):
        """ResMADE with Gaussian factorization.

        Parameters
        ----------
        input_size : int
            Dimension of input layer.
        k : int
            Number of neurons in each layer (hidden_layer_dim).
        n_blocks : int
            Number of repeated residual blocks (num_hidden_layers). Each block of size l has a residual connection.
        l : int, optional
            Number of layers in each residual block, by default 2 (res_layers_in_block).
        kwargs: dict
            Keyword arguments for nn.

        """
        super().__init__(input_size, k, n_blocks, l, out_mask_multiplier=2, **kwargs)

    def forward(self, x):
        # MAF eq 4, return mean and log std
        mu, alpha = self.net(x).chunk(chunks=2, dim=1)
        u = (x - mu) * torch.exp(-alpha)

        # MAF eq 5
        log_abs_det_jacobian = -alpha.sum(dim=-1, keepdim=True)

        return u, log_abs_det_jacobian

    def inverse(self, u):
        # MAF eq 3
        x = torch.zeros_like(u)

        # run through reverse model
        for i in range(self.input_size):
            mu, alpha = self.net(x).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(alpha[:, i]) + mu[:, i]

        log_abs_det_jacobian = alpha.sum(dim=-1, keepdim=True)

        return x, log_abs_det_jacobian


if __name__ == "__main__":
    from ml.common.nn.netron_vis import visualize_model

    dummy = torch.randn((1024, 18))
    model = GaussianResMADE(
        input_size=18,
        k=128,
        n_blocks=3,
        l=2,
    )

    print(model)

    visualize_model(model, dummy)
