import logging
from itertools import cycle

import torch

from ml.common.nn.mlp import BasicMLP
from ml.flows.models.aux_flows import BatchNormFlow, Flow, ReverseFlow, ShuffleFlow
from ml.flows.models.base_flows import BaseFlowModel, NormalizingFlow


class AffineFlow(Flow):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", batchnorm=False, act_first=False):
        """Autoregressive affine flow. Implements [2].

        Parameters
        ----------
        See nets.mlp for parameters.

        WARNING: input_dim % 2 == 0 must be True!

        References
        ----------
        [1] - Normalizing Flows for Probabilistic Modeling and Inference (section 3.1.2): https://arxiv.org/abs/1912.02762
        [2] - Density estimation using Real NVP (eq. 6, 7 and 8): https://arxiv.org/abs/1605.08803
        [3] - https://github.com/xqding/RealNVP
        [4] - https://github.com/senya-ashukha/real-nvp-pytorch

        """
        super().__init__()
        self.input_dim = input_dim
        self.d = self.input_dim // 2

        if self.input_dim % 2 != 0:
            logging.warning("Input dimension is not even ... use with masking only!")

        self.translate_net = self._build_network(hidden_layers, activation, None, batchnorm, act_first)
        self.scale_net = self._build_network(hidden_layers, "Tanh", None, batchnorm, act_first)

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

    def inverse(self, z):
        x1, x2 = z[:, : self.d], z[:, self.d :]
        x_affine_inv = (x2 - self.translate_net(x1)) * torch.exp(-self.scale_net(x1))

        log_det = -torch.sum(self.scale_net(x1), dim=-1, keepdim=True)

        return torch.cat((x1, x_affine_inv), dim=1), log_det

    def forward(self, x):
        z1, z2 = x[:, : self.d], x[:, self.d :]
        z_affine = z2 * torch.exp(self.scale_net(z1)) + self.translate_net(z1)

        log_det = torch.sum(self.scale_net(z1), dim=-1, keepdim=True)

        return torch.cat((z1, z_affine), dim=1), log_det


class MaskedNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, mask_type="checkerboard", mask_device="cpu"):
        """Masked normalizing flow. Shifts between odd and even masks at the begining of every block. This is a special
        case of a flow and is only used for the realNVP model. It is not a general purpose flow.
        """
        super().__init__(*args)
        self.mask_type, self.mask_device = mask_type, mask_device
        self._validate_blocks()
        self._mask_setup()

    def _checkerboard_mask(self):
        mask1 = torch.arange(0, self.dim, 1, dtype=torch.float) % 2
        mask2 = 1 - mask1
        return [mask1.to(self.mask_device), mask2.to(self.mask_device)]

    def _halfhalf_mask(self):
        mask_zeros = torch.zeros(self.dim // 2)
        mask_ones = torch.ones(self.dim // 2)

        mask1 = torch.cat((mask_zeros, mask_ones))
        mask2 = torch.cat((mask_ones, mask_zeros))
        return [mask1.to(self.mask_device), mask2.to(self.mask_device)]

    def _mask_setup(self):
        if self.mask_type == "checkerboard":
            self.masks = self._checkerboard_mask()
        elif self.mask_type == "halfhalf":
            self.masks = self._halfhalf_mask()
        else:
            raise NotImplementedError

    def _validate_blocks(self):
        c = 0
        for bijector in self.bijectors:
            if bijector.mask is not False:
                c += 1

        if c % 2 != 0:
            raise ValueError("Number of masked layers must be even!")

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

        return z, self.log_det


class MaskedAffineFlow(AffineFlow):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", batchnorm=False, act_first=False):
        super().__init__(input_dim, hidden_layers, activation, batchnorm, act_first)
        self.d = self.input_dim
        self.mask = True

        self.translate_net = self._build_network(hidden_layers, activation, None, batchnorm, act_first)
        self.scale_net = self._build_network(hidden_layers, "Tanh", None, batchnorm, act_first)

    def set_mask(self, mask):
        self.mask = mask

    def inverse(self, y):
        y1 = self.mask * y
        x = y1 + (1 - self.mask) * (y - self.translate_net(y1)) * torch.exp(-self.scale_net(y1))

        log_det = -torch.sum(self.scale_net(y1) * (1 - self.mask), dim=-1, keepdim=True)

        return x, log_det

    def forward(self, x):
        x1 = self.mask * x
        y = x1 + (1 - self.mask) * (x * torch.exp(self.scale_net(x1)) + self.translate_net(x1))

        log_det = torch.sum(self.scale_net(x1) * (1 - self.mask), dim=-1, keepdim=True)

        return y, log_det


class RealNVP(BaseFlowModel):
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
        self.groups = self.model_conf.get("groups")

        if self.groups:
            self.groups = cycle(self.groups)

        self.hidden_layer = []
        for _ in range(self.num_hidden_layers):
            self.hidden_layer.append(self.hidden_layer_dim)
        self.hidden_layer.append(self.input_dim)

        blocks = []
        if self.use_masks:  # realNVP with binary masks
            for _ in range(self.num_flows):
                blocks.append(
                    MaskedAffineFlow(
                        self.input_dim,
                        self.hidden_layer,
                        activation=self.activation,
                        batchnorm=self.batchnorm,
                        act_first=self.act_first,
                    )
                )

                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.model = MaskedNormalizingFlow(
                self.input_dim,
                blocks,
                self.base_distribution,
                mask_device=self.base_distribution.loc.device,
            )

        else:  # realNVP with inverse flow or permutation flows
            for i in range(self.num_flows):
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

                if self.groups:
                    shuffle_first_last = True if i % 2 == 0 else False
                    blocks.append(ShuffleFlow(self.input_dim, next(self.groups), shuffle_first_last))
                else:
                    blocks.append(ReverseFlow(self.input_dim))

                blocks.append(
                    AffineFlow(
                        self.input_dim,
                        self.hidden_layer,
                        activation=self.activation,
                        batchnorm=self.batchnorm,
                        act_first=self.act_first,
                    )
                )

            self.model = NormalizingFlow(
                self.input_dim,
                blocks,
                self.base_distribution,
            )
