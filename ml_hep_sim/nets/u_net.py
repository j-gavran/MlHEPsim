import torch
from torch import nn


def build_no_act_linear_blocks(input_dim, hidden_layers, layer_obj=nn.Linear):
    blocks = []
    blocks.append(layer_obj(input_dim, hidden_layers[0]))

    for i in range(len(hidden_layers) - 1):
        blocks.append(layer_obj(hidden_layers[i], hidden_layers[i + 1]))

    return blocks


class UNet(nn.Module):
    def __init__(
        self,
        input_dim,
        first_layer_dim,
        div,
        output_dim=None,
        activation="ReLU",
        output_activation="Identity",
    ):
        super().__init__()
        self.input_dim = input_dim

        if output_dim is None:
            self.output_dim = input_dim
            self.cut_output = True
        else:
            self.output_dim = output_dim
            self.cut_output = False

        self.first_layer_dim = first_layer_dim
        self.div = div

        self.activation = getattr(nn, activation)
        self.output_activation = getattr(nn, output_activation)

        self.hidden_layers = []
        for d in range(div):
            self.hidden_layers.append(first_layer_dim // 2**d)

        self.hidden_layers += self.hidden_layers[::-1]
        self.hidden_layers.append(self.output_dim)

        blocks = build_no_act_linear_blocks(input_dim, self.hidden_layers)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        temps = []

        mid_idx = len(self.hidden_layers) // 2
        last_idx = len(self.blocks) - 1
        half_idx = 0

        if self.cut_output:
            iter_blocks = self.blocks[:-1]
        else:
            iter_blocks = self.blocks

        for i, block in enumerate(iter_blocks):
            # 2nd half
            if i > mid_idx and i != last_idx:
                x = x + temps[::-1][half_idx]
                half_idx += 1

            if i != 0:
                x = self.activation()(x)

            x = block(x)

            # 1st half
            if i < mid_idx and i != 0:
                temps.append(x.clone())

        x = self.output_activation()(x)

        return x


if __name__ == "__main__":
    u_net = UNet(18, 32, 5, output_dim=18)
    print(u_net)
    dummy = torch.randn((1024, 18))
    u_net(dummy)
    from ml_hep_sim.ml_utils import netron_model

    netron_model(u_net, dummy)
