import torch
import torch.nn as nn

from ml.common.nn.mlp import BasicMLP
from ml.flows.models.aux_flows import BatchNormFlow, Flow, ReverseFlow
from ml.flows.models.base_flows import BaseFlowModel, NormalizingFlow
from ml.flows.models.real_nvp import MaskedNormalizingFlow


class ScaleNormalizingFlow(NormalizingFlow):
    def __init__(self, *args):
        """Normalizing flow for additive coupling layers described in [1].

        References
        ----------
        [1] - NICE: Non-linear Independent Components Estimation: https://arxiv.org/abs/1410.8516

        """
        super().__init__(*args)
        self.s = nn.Parameter(torch.randn(self.dim), requires_grad=True)

    def forward(self, z):
        self.log_det = []

        for bijector in self.bijectors:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.inverse(z)
            else:
                z, log_abs_det = bijector.forward(z)

            self.log_det.append(log_abs_det)

        z = z * torch.exp(self.s)

        log_abs_det = torch.sum(self.s)
        self.log_det.append(log_abs_det)

        return z, self.log_det

    def inverse(self, z):
        self.log_det = []

        for bijector in self.bijectors[::-1]:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.forward(z)
            else:
                z, log_abs_det = bijector.inverse(z)

            self.log_det.append(log_abs_det)

        z = z / torch.exp(self.s)

        log_abs_det = torch.sum(self.s)
        self.log_det.append(log_abs_det)

        return z, self.log_det


class MaskedScaleNormalizingFlow(MaskedNormalizingFlow):
    def __init__(self, *args, mask_type="checkerboard", mask_device="cpu"):
        super().__init__(*args, mask_type=mask_type, mask_device=mask_device)
        self.mask = True
        self.s = nn.Parameter(torch.randn(self.dim), requires_grad=True)

    def forward(self, z):
        self.log_det, c = [], 0

        for bijector in self.bijectors:
            if bijector.mask is not False:
                if c % 2 == 0:
                    mask = self.masks[0]
                else:
                    mask = self.masks[1]

                bijector.set_mask(mask)
                c += 1

            if bijector.normalizing_direction:
                z, log_abs_det = bijector.inverse(z)
            else:
                z, log_abs_det = bijector.forward(z)

            self.log_det.append(log_abs_det)

        z = z * torch.exp(self.s)

        log_abs_det = torch.sum(self.s)
        self.log_det.append(log_abs_det)

        return z, self.log_det

    def inverse(self, z):
        self.log_det, c = [], 0

        for bijector in self.bijectors[::-1]:
            if bijector.mask is not False:
                if c % 2 != 0:
                    mask = self.masks[0]
                else:
                    mask = self.masks[1]

                bijector.set_mask(mask)
                c += 1

            if bijector.normalizing_direction:
                z, log_abs_det = bijector.forward(z)
            else:
                z, log_abs_det = bijector.inverse(z)

            self.log_det.append(log_abs_det)

        z = z / torch.exp(self.s)

        log_abs_det = torch.sum(self.s)
        self.log_det.append(log_abs_det)

        return z, self.log_det


class AdditiveFlow(Flow):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", batchnorm=False, act_first=False):
        """Additive coupling layers. See [1].

        References
        ----------
        [1] - NICE: Non-linear Independent Components Estimation (page 7): https://arxiv.org/abs/1410.8516
        [2] - https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/NICE_Non_linear_Independent_Components_Estimation

        """
        super().__init__()
        self.input_dim = input_dim
        self.d = self.input_dim // 2
        self.net = self._build_network(hidden_layers, activation, None, batchnorm, act_first)

    def _build_network(self, hidden_layers, activation, output_activation, batchnorm, act_first):
        hidden_layers[-1] = self.d
        mlp = BasicMLP(
            [self.d] + hidden_layers,
            act=activation,
            act_out=output_activation,
            batchnorm=batchnorm,
            act_first=act_first,
        )
        return mlp

    def forward(self, x):
        x1 = x[:, : self.d]
        x2 = x[:, self.d :]

        h1 = x1
        h2 = x2 + self.net(x1)

        return torch.cat((h1, h2), dim=1), torch.zeros_like(x).sum(dim=-1, keepdim=True)

    def inverse(self, x):
        h1 = x[:, : self.d]
        h2 = x[:, self.d :]

        x1 = h1
        x2 = h2 - self.net(x1)

        return torch.cat((x1, x2), dim=1), torch.zeros_like(x).sum(dim=-1, keepdim=True)


class MaskedAdditiveFlow(AdditiveFlow):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", batchnorm=False, act_first=False):
        super().__init__(input_dim, hidden_layers, activation, batchnorm, act_first)
        self.d = self.input_dim
        self.mask = True

        self.net = self._build_network(hidden_layers, activation, None, batchnorm, act_first)

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        z1 = self.mask * x
        z2 = (1 - self.mask) * x
        z_add = z2 + (1 - self.mask) * self.net(z1)
        return z1 + z_add, torch.zeros_like(x).sum(dim=-1, keepdim=True)

    def inverse(self, z):
        x1 = self.mask * z
        x2 = (1 - self.mask) * z
        x_sub = x2 - (1 - self.mask) * self.net(x1)
        return x1 + x_sub, torch.zeros_like(z).sum(dim=-1, keepdim=True)


class NICE(BaseFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = self.model_conf["activation"]
        self.batchnorm = self.model_conf.get("batchnorm")
        self.act_first = self.model_conf.get("act_first")
        self.num_flows = self.model_conf["num_flows"]
        self.num_hidden_layers = self.model_conf["num_hidden_layers"]
        self.hidden_layer_dim = self.model_conf["hidden_layer_dim"]
        self.batchnorm_flow = self.model_conf.get("batchnorm_flow")
        self.use_masks = self.model_conf.get("use_masks")

        self.hidden_layer = []
        for _ in range(self.num_hidden_layers):
            self.hidden_layer.append(self.hidden_layer_dim)
        self.hidden_layer.append(self.input_dim)

        blocks = []

        if self.use_masks:
            for _ in range(self.num_flows):
                blocks.append(
                    MaskedAdditiveFlow(
                        self.input_dim,
                        self.hidden_layer,
                        activation=self.activation,
                        batchnorm=self.batchnorm,
                        act_first=self.act_first,
                    )
                )
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.model = MaskedScaleNormalizingFlow(
                self.input_dim, blocks, self.base_distribution, mask_device=self.base_distribution.loc.device
            )
        else:
            for _ in range(self.num_flows):
                blocks.append(
                    AdditiveFlow(
                        self.input_dim,
                        self.hidden_layer,
                        activation=self.activation,
                        batchnorm=self.batchnorm,
                        act_first=self.act_first,
                    )
                )
                blocks.append(ReverseFlow(self.input_dim))
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.model = ScaleNormalizingFlow(self.input_dim, blocks, self.base_distribution)
