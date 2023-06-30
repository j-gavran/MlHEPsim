import hydra
import torch
from ml_hep_sim.ml_utils import mlf_loader, train_wrapper
from ml_hep_sim.vaes.beta_vae import BetaVAEModel
from ml_hep_sim.vaes.vae import VAEModel


class StageTwoVAEModelBase:
    """
    References
    ----------
    [1] - https://github.com/probml/pyprobml/tree/master/vae

    """

    def training_step(self, batch, batch_idx):
        x, _ = batch

        mu_z, logvar_z = self.stage_one_vae.encode(x)
        z = self.stage_one_vae.reparameterize(mu_z, logvar_z)
        z = z.detach()

        recon_batch, mu, logvar = self(z)
        loss = self.vae.loss_function(recon_batch, z, mu, logvar)
        train_loss = loss[0] + loss[1]

        self.log("train_loss", train_loss)
        self.log("train_rec_loss", loss[0])
        self.log("train_kl_loss", loss[1])

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu_z, logvar_z = self.stage_one_vae.encode(x)
        z = self.stage_one_vae.reparameterize(mu_z, logvar_z)
        z = z.detach()

        recon_batch, mu, logvar = self(z)
        loss = self.vae.loss_function(recon_batch, z, mu, logvar)
        val_loss = loss[0] + loss[1]

        self.log("val_loss", val_loss)
        self.log("val_rec_loss", loss[0])
        self.log("val_kl_loss", loss[1])

        return {"val_loss": val_loss}

    def sample(self, n_samples, device="cpu"):
        noise = torch.randn(n_samples, self.vae.latent_dim, device=device)
        stage_2_out = self.vae.decode(noise)
        stage_1_out = self.stage_one_vae.decode(stage_2_out)
        return stage_1_out


class StageTwoVAEModel(StageTwoVAEModelBase, VAEModel):
    def __init__(self, config, input_dim, stage_one_vae, *args, **kwargs):
        super().__init__(config, input_dim, *args, **kwargs)
        self.stage_one_vae = stage_one_vae


class StageTwoBetaVAEModel(StageTwoVAEModelBase, BetaVAEModel):
    def __init__(self, config, input_dim, stage_one_vae, *args, **kwargs):
        super().__init__(config, input_dim, *args, **kwargs)
        self.stage_one_vae = stage_one_vae


@hydra.main(config_path="../conf", config_name="two_stage_vae_config", version_base=None)
def train_two_stage_vae(config):
    stage_one_model = mlf_loader(config["model_config"]["stage_one_vae"])
    stage_one_vae = stage_one_model.vae

    return train_wrapper(
        config["model_config"],
        input_dim=config["model_config"]["input_dim"],
        pl_model=StageTwoBetaVAEModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
        stage_one_vae=stage_one_vae,
    )


if __name__ == "__main__":
    train_two_stage_vae()
