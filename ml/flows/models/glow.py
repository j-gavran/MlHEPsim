from ml.flows.models.aux_flows import BatchNormFlow, Conv1x1PLU
from ml.flows.models.base_flows import NormalizingFlow
from ml.flows.models.real_nvp import AffineFlow, RealNVP


class Glow(RealNVP):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)

        blocks = []
        for _ in range(self.num_flows):
            blocks.append(BatchNormFlow(self.input_dim))
            blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
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
