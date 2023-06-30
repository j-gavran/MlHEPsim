import hydra

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.normalizing_flows.base_flows import NormalizingFlow
from ml_hep_sim.normalizing_flows.flows import BatchNormFlow, Conv1x1PLU
from ml_hep_sim.normalizing_flows.real_nvp import AffineFlow, RealNVPFlowModel


class GlowFlowModel(RealNVPFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        blocks = []
        for _ in range(self.num_flows):
            blocks.append(BatchNormFlow(self.input_dim))
            blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
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

    # snapshots of 2d densities for animations
    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     if self.current_step % 3 == 0:
    #         density = get_2d_density(self.eval(), make_2d_mesh(-4, 4, 200))
    #         np.save(f"./data/realnvp_{self.data_name}/density_{self.current_step}.npy", density)
    #         self.train()


@hydra.main(config_path="../conf", config_name="glow_config", version_base=None)
def train_glow(config):
    device = "cuda" if config["datasets"]["data_params"]["to_gpu"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=GlowFlowModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    # from ml_hep_sim.normalizing_flows.flow_utils import get_2d_density, make_2d_mesh
    train_glow()
