import torch
from torch import nn
from torch.nn import init


class MuSigmaLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mu_layer = nn.Linear(in_channels, out_channels)
        self.sigma_layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)
        return mu, sigma


class PreActLinearResNet(nn.Module):
    def __init__(self, k, l, activation, zero_initialization=True, batchnorm=True, eps=1e-3):
        """Linear residual network.

        Parameters
        ----------
        k : int
            Number of neurons per layer.
        l : int
            Number of layers per residual block.
        activation : function
            Activation in residual blocks.
        zero_initialization : bool, optional
            Initialize first layer to 0, by default True.
        batchnorm : bool
            Use batch normalization in residual network, by default True.
        eps: float
            BN parameter.

        Note
        ----
        Uses ReLU and BN as pre-activation of the weight layers.

        References
        ----------
        [1] - Identity Mappings in Deep Residual Networks: https://arxiv.org/abs/1603.05027

        """
        super().__init__()
        self.k, self.l = k, l

        layers = []
        for _ in range(l):
            if batchnorm:
                layers.append(nn.BatchNorm1d(k, eps=eps))

            layers.append(activation)
            layers.append(nn.Linear(k, k))

        self.layers = nn.Sequential(*layers)

        if zero_initialization:
            init.zeros_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        temp = x

        for layer in self.layers:
            temp = layer(temp)

        return x + temp


class PreActResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        k,
        l,
        n,
        activation="ReLU",
        output_activation="Identity",
        output_dim=None,
        double_output=False,
        **resnet_kwargs
    ):
        """_summary_

        Parameters
        ----------
        input_dim : int
            Input dimension.
        k, l : int
            See LinearResNet.
        n : int
            Number of repeated residual blocks.
        output_dim : None, optional
            If not given same as input_dim.
        double_output: bool
            If end network as autoencoder (with double networks).
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        blocks = []

        activation_fn = getattr(nn, activation)()
        output_activation_fn = getattr(nn, output_activation)()

        blocks.append(nn.Linear(input_dim, k))

        for _ in range(n):
            blocks.append(PreActLinearResNet(k, l, activation_fn, **resnet_kwargs))

        if double_output:
            blocks.append(MuSigmaLayer(k, output_dim))
        else:
            blocks.append(nn.Linear(k, output_dim))
            blocks.append(output_activation_fn)

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class PreActResNetAutoencoder:
    def __init__(
        self, input_dim, k, l, n, latent_dim, activation="ReLU", output_activation="Identity", **resnet_kwargs
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = PreActResNet(
            input_dim,
            k,
            l,
            n,
            activation,
            output_activation,
            output_dim=latent_dim,
            double_output=True,
            **resnet_kwargs
        )
        self.decoder = PreActResNet(
            latent_dim,
            k,
            l,
            n,
            activation,
            output_activation,
            output_dim=input_dim,
            double_output=False,
            **resnet_kwargs
        )


if __name__ == "__main__":
    from ml_hep_sim.ml_utils import netron_model

    rn = PreActResNet(18, k=128, l=2, n=2, double_output=True, output_dim=10)
    dummy = torch.randn((1024, 18))
    netron_model(rn, dummy)
