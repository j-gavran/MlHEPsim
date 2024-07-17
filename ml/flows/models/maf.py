from ml.flows.models.aux_flows import BatchNormFlow, Conv1x1PLU, ReverseFlow
from ml.flows.models.base_flows import (
    AutoregressiveNormalizingFlow,
    BaseFlowModel,
    NormalizingFlow,
)
from ml.flows.models.made import GaussianMADE, GaussianResMADE
from ml.flows.models.made_mog import MADEMOG


class MAF(BaseFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = self.model_conf["activation"]
        self.num_flows = self.model_conf["num_flows"]
        self.num_hidden_layers = self.model_conf["num_hidden_layers"]
        self.hidden_layer_dim = self.model_conf["hidden_layer_dim"]
        self.made_residuals = self.model_conf["made_residuals"]
        self.batchnorm_flow = self.model_conf["batchnorm_flow"]
        self.conv1x1 = self.model_conf["conv1x1"]
        self.res_layers_in_block = self.model_conf["res_layers_in_block"]
        self.normalization_out = self.model_conf["normalization_out"]

        blocks = []
        for _ in range(self.num_flows):
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

            if self.made_residuals:
                blocks.append(
                    GaussianResMADE(
                        self.input_dim,
                        k=self.hidden_layer_dim,
                        n_blocks=self.num_hidden_layers,
                        l=self.res_layers_in_block,
                        activation=self.activation,
                    )
                )
            else:
                blocks.append(
                    GaussianMADE(
                        self.input_dim,
                        self.hidden_layer_dim,
                        self.num_hidden_layers,
                        activation=self.activation,
                    )
                )

        if self.normalization_out:
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

        self.model = NormalizingFlow(self.input_dim, blocks, self.base_distribution)


class MAFMADEMOG(MADEMOG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_flows = self.model_conf["num_flows"]
        self.num_hidden_layers = self.model_conf["num_hidden_layers"]
        self.hidden_layer_dim = self.model_conf["hidden_layer_dim"]
        self.batchnorm_flow = self.model_conf["batchnorm_flow"]
        self.conv1x1 = self.model_conf["conv1x1"]
        self.maf_residuals = self.model_conf["maf_residuals"]
        self.res_layers_in_block = self.model_conf["res_layers_in_block"]
        self.normalization_out = self.model_conf["normalization_out"]

        blocks = []
        for _ in range(self.num_flows):
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

            if self.maf_residuals:
                blocks.append(
                    GaussianResMADE(
                        self.input_dim,
                        k=self.hidden_layer_dim,
                        n_blocks=self.num_hidden_layers,
                        l=self.res_layers_in_block,
                        activation=self.activation,
                    )
                )
            else:
                blocks.append(
                    GaussianMADE(
                        self.input_dim,
                        self.hidden_layer_dim,
                        self.num_hidden_layers,
                        activation=self.activation,
                    )
                )

        if self.normalization_out:
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

        blocks += self.blocks

        self.model = AutoregressiveNormalizingFlow(self.input_dim, blocks, device=self.device)
