import hydra
import torch
import torch.nn as nn
from torch.distributions import Normal

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.nets.mlp import BasicLinearNetwork
from ml_hep_sim.normalizing_flows.base_flows import (
    MaskedNormalizingFlow,
    NormalizingFlow,
)
from ml_hep_sim.normalizing_flows.flows import BatchNormFlow, ReverseFlow
from ml_hep_sim.normalizing_flows.real_nvp import AffineFlow, RealNVPFlowModel


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


class AdditiveFlow(AffineFlow):
    def __init__(self, *args, activation="ReLU", **kwargs):
        """Additive coupling layers. See [1].

        References
        ----------
        [1] - NICE: Non-linear Independent Components Estimation (page 7): https://arxiv.org/abs/1410.8516
        [2] - https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/NICE_Non_linear_Independent_Components_Estimation

        """
        super().__init__(*args, activation, **kwargs)
        self.scale_net, self.translate_net = None, None
        self.d = self.input_dim // 2
        self.net = self._build_network(*args, activation=activation, output_activation="Identity", **kwargs)

    def _build_network(self, dim, hidden_layers, activation, output_activation, **kwargs):
        hidden_layers[-1] = self.d
        mlp = BasicLinearNetwork(self.d, hidden_layers, activation, output_activation, **kwargs)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = self.input_dim
        self.net = self._build_network(*args, output_activation="Identity", **kwargs)

        self.mask = True

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


class NICEFlowModel(RealNVPFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        blocks = []

        if self.use_masks:
            for _ in range(self.num_flows):
                blocks.append(
                    MaskedAdditiveFlow(
                        self.input_dim, self.hidden_layer, activation=self.activation, batchnorm=self.batchnorm
                    )
                )
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.flow = MaskedScaleNormalizingFlow(
                self.input_dim, blocks, self.base_distribution, mask_device=self.base_distribution.loc.device
            )
        else:
            for _ in range(self.num_flows):
                blocks.append(
                    AdditiveFlow(
                        self.input_dim, self.hidden_layer, activation=self.activation, batchnorm=self.batchnorm
                    )
                )
                blocks.append(ReverseFlow(self.input_dim))
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.flow = ScaleNormalizingFlow(self.input_dim, blocks, self.base_distribution)

    def on_validation_epoch_end(self):
        self.log("scale", self.flow.s.mean())


@hydra.main(config_path="../conf", config_name="realnvp_config", version_base=None)
def train_nice(config):
    device = "cuda" if config["trainer_config"]["gpus"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    config["logger_config"]["run_name"] = "Higgs_NICE"

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=NICEFlowModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_nice()
