import torch
from torch import nn


def build_linear_blocks(input_dim, hidden_layers, activation, output_activation, batchnorm, layer_obj, pre_act):
    activation = getattr(nn, activation)
    output_activation = getattr(nn, output_activation)

    blocks = []
    blocks.append(layer_obj(input_dim, hidden_layers[0]))

    for i in range(len(hidden_layers) - 1):
        if batchnorm and not pre_act:
            blocks.append(nn.BatchNorm1d(hidden_layers[i]))

        blocks.append(activation())

        if batchnorm and pre_act:
            blocks.append(nn.BatchNorm1d(hidden_layers[i]))

        blocks.append(layer_obj(hidden_layers[i], hidden_layers[i + 1]))

    blocks.append(output_activation())

    return blocks


class BasicLinearNetwork(nn.Module):
    def __init__(
        self, input_dim, hidden_layers, activation="ReLU", output_activation="Identity", batchnorm=False, pre_act=False
    ):
        """Standard multi layer perceptron network.

        Parameters
        ----------
        input_dim : int
            Dimension of input data.
        hidden_layers : list of ints
            Number of neurons in each layer.
        activation : str, optional
            Activation function, by default "ReLU".
        output_activation : str, optional
            Output activation function, by default "Identity".
        batchnorm : bool, optional
            Use batch normalization, by default False.
        pre_act : bool, optional
            Where to put bathnorm (after of before activation function), by default False.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation

        blocks = build_linear_blocks(
            input_dim, hidden_layers, activation, output_activation, batchnorm, nn.Linear, pre_act=pre_act
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


if __name__ == "__main__":
    from ml_hep_sim.ml_utils import netron_model

    dummy = torch.ones((1024, 18))
    net = BasicLinearNetwork(18, [128, 128, 128, 128, 128, 128], "ReLU", batchnorm=False, pre_act=False)
    net(dummy)
    netron_model(net, dummy)
