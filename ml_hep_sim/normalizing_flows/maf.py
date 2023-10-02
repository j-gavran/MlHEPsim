import hydra
import torch

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.normalizing_flows.base_flows import NormalizingFlow, PlFlowModel
from ml_hep_sim.normalizing_flows.flow_utils import plot_flow_result_on_event
from ml_hep_sim.normalizing_flows.flows import BatchNormFlow, Conv1x1PLU, ReverseFlow
from ml_hep_sim.normalizing_flows.made import GaussianMADE, GaussianResMADE
from ml_hep_sim.normalizing_flows.made_mog import MADEMOGFlowModel, MOGNormalizingFlow


class MAFMADEFlowModel(PlFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = config["learning_rate"]
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.activation = config["activation"]
        self.num_flows = config["num_flows"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.hidden_layer_dim = config["hidden_layer_dim"]
        self.maf_residuals = config["maf_residuals"]
        self.batchnorm_flow = config["batchnorm_flow"]
        self.conv1x1 = config.get("conv1x1")
        self.save_hyperparameters()

        if self.maf_residuals:
            made = GaussianResMADE
        else:
            made = GaussianMADE

        blocks = []
        for _ in range(self.num_flows):
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

            blocks.append(
                made(
                    self.input_dim,
                    self.hidden_layer_dim,
                    self.num_hidden_layers,
                    activation=self.activation,
                )
            )

        if self.batchnorm_flow:
            blocks.append(BatchNormFlow(self.input_dim))

        if self.conv1x1:
            blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
        else:
            blocks.append(ReverseFlow(self.input_dim))

        self.flow = NormalizingFlow(self.input_dim, blocks, self.base_distribution)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # z: (batch, feature), log_jac: list of (batch, 1) for all flows
        z, log_jac = self.flow(x)

        # (batch, 1)
        sum_of_log_det_jacobian = sum(log_jac)

        # (batch, feature) -> (batch, 1)
        nll = self.base_distribution.log_prob(z).sum(-1, keepdim=True)

        # (batch, 1) + (batch, 1) -> (1,)
        loss = -torch.mean(sum_of_log_det_jacobian + nll)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        z, log_jac = self.flow(x)

        cutoff = 40
        nan_mask = ~torch.any(z.isnan(), dim=1)
        inf_mask = ~torch.any(z.isinf(), dim=1)
        overflow_mask = torch.all(torch.abs(z) < cutoff, dim=1)
        mask = nan_mask & inf_mask & overflow_mask

        z = z[mask]

        sum_of_log_det_jacobian = sum(log_jac)[mask]

        nll = self.base_distribution.log_prob(z).sum(-1, keepdim=True)
        loss = -torch.mean(sum_of_log_det_jacobian + nll)

        self.log("val_loss", loss)
        self.log("sum_log_det_jac", torch.mean(sum_of_log_det_jacobian))
        self.log("val_nll", torch.mean(nll))

        return {"val_loss": loss, "sum_log_det_jac": sum_of_log_det_jacobian, "val_nll": nll}

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


class MAFMADEMOGFlowModel(MADEMOGFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_flows = config["num_flows"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.maf_residuals = config["maf_residuals"]
        self.batchnorm_flow = config["batchnorm_flow"]
        self.conv1x1 = config.get("conv1x1")
        self.save_hyperparameters()

        if self.maf_residuals:
            made = GaussianResMADE
        else:
            made = GaussianMADE

        blocks = []
        for _ in range(self.num_flows):
            if self.batchnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

            blocks.append(
                made(
                    self.input_dim,
                    self.hidden_layer_dim,
                    self.num_hidden_layers,
                    activation=self.activation,
                )
            )

        if self.batchnorm_flow:
            blocks.append(BatchNormFlow(self.input_dim))

        if self.conv1x1:
            blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
        else:
            blocks.append(ReverseFlow(self.input_dim))

        blocks += self.blocks

        self.flow = MOGNormalizingFlow(self.input_dim, blocks, density=self.base_distribution)


@hydra.main(config_path="../conf", config_name="maf_config", version_base=None)
def train_maf(config):
    device = "cuda" if config["datasets"]["data_params"]["to_gpu"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    if config["model_config"]["use_mog"]:
        pl_model = MAFMADEMOGFlowModel
    else:
        pl_model = MAFMADEFlowModel

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=pl_model,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_maf()
