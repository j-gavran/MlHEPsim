import logging
from itertools import cycle

import hydra
import torch

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.nets.mlp import BasicLinearNetwork
from ml_hep_sim.normalizing_flows.base_flows import (
    MaskedNormalizingFlow,
    NormalizingFlow,
    PlFlowModel,
)
from ml_hep_sim.normalizing_flows.flow_utils import plot_flow_result_on_event
from ml_hep_sim.normalizing_flows.flows import (
    BatchNormFlow,
    Flow,
    ReverseFlow,
    ShuffleFlow,
)


class AffineFlow(Flow):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", **kwargs):
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

        self.translate_net = self._build_network(input_dim, hidden_layers, activation, "Identity", **kwargs)
        self.scale_net = self._build_network(input_dim, hidden_layers, "Tanh", "Identity", **kwargs)

    def _build_network(self, dim, hidden_layers, activation, output_activation, **kwargs):
        hidden_layers[-1] = self.d
        mlp = BasicLinearNetwork(self.d, hidden_layers, activation, output_activation, **kwargs)
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


class MaskedAffineFlow(AffineFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = self.input_dim
        self.mask = True

        self.translate_net = self._build_network(*args, output_activation="Identity", **kwargs)
        self.scale_net = self._build_network(*args, output_activation="Tanh", **kwargs)

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


class RealNVPFlowModel(PlFlowModel):
    def __init__(self, config, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = config.get("learning_rate")
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.activation = config["activation"]
        self.batchnorm = config.get("batchnorm")
        self.num_flows = config["num_flows"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.hidden_layer_dim = config["hidden_layer_dim"]
        self.batchnorm_flow = config["batchnorm_flow"]
        self.use_masks = config.get("use_masks")
        self.save_hyperparameters()

        if groups:
            groups = cycle(groups)

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
                    )
                )

                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

            self.flow = MaskedNormalizingFlow(
                self.input_dim,
                blocks,
                self.base_distribution,
                mask_device=self.base_distribution.loc.device,
            )

        else:  # realNVP with inverse flow or permutation flows
            for i in range(self.num_flows):
                if self.batchnorm_flow:
                    blocks.append(BatchNormFlow(self.input_dim))

                if groups:
                    shuffle_first_last = True if i % 2 == 0 else False
                    blocks.append(ShuffleFlow(self.input_dim, next(groups), shuffle_first_last))
                else:
                    blocks.append(ReverseFlow(self.input_dim))

                blocks.append(
                    AffineFlow(
                        self.input_dim,
                        self.hidden_layer,
                        activation=self.activation,
                        batchnorm=self.batchnorm,
                    )
                )

            self.flow = NormalizingFlow(
                self.input_dim,
                blocks,
                self.base_distribution,
            )

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            plot_flow_result_on_event(
                self.data_name,
                self.base_distribution,
                self,
                self.logger,
                self._trainer.datamodule.test,
                idx=self.current_epoch,
            )


@hydra.main(config_path="../conf", config_name="realnvp_config", version_base=None)
def train_realnvp(config):
    device = "cuda" if config["trainer_config"]["gpus"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=RealNVPFlowModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_realnvp()
