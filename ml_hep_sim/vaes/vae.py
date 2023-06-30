import hydra
import torch
from torch.nn import functional as F

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.nets.autoencoder import Autoencoder
from ml_hep_sim.nets.resnet import PreActResNetAutoencoder
from ml_hep_sim.vaes.base_vae import BaseVAE, PlVAEModel
from ml_hep_sim.vaes.vae_utils import plot_vae_result_on_event


class VAE(BaseVAE):
    def __init__(self, ae):
        """
        References
        ----------
        [1] - https://github.com/pytorch/examples/blob/master/vae/main.py

        """
        super().__init__(ae)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def loss_function(self, recon_x, x, mu, logvar):
        # sum loss
        # mse = F.mse_loss(recon_x, x, reduction="sum")
        # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # mean loss
        mse = F.mse_loss(recon_x, x, reduction="mean") * x.shape[1]
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        return mse, kld

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEModel(PlVAEModel):
    def __init__(self, config, input_dim, *args, **kwargs):
        super().__init__(input_dim, *args, **kwargs)

        self.activation = config["activation"]
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.output_activation = config["output_activation"]
        self.latent_dim = config["latent_dim"]
        self.hidden_layers = config["hidden_layers"]

        self.use_resnet = config.get("use_resnet")

        if self.use_resnet:
            self.ae = PreActResNetAutoencoder(
                self.input_dim,
                k=self.hidden_layers[0],
                l=2,
                n=len(self.hidden_layers),
                latent_dim=self.latent_dim,
                activation=self.activation,
                output_activation=self.output_activation,
            )
        else:
            self.ae = Autoencoder(
                input_dim,
                self.hidden_layers,
                self.latent_dim,
                self.activation,
                self.output_activation,
            )

        self.save_hyperparameters()

        self.vae = VAE(self.ae)

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            plot_vae_result_on_event(
                self.data_name, self, self.logger, self._trainer.datamodule.test, idx=self.current_epoch
            )


@hydra.main(config_path="../conf", config_name="vae_config", version_base=None)
def train_vae(config):
    return train_wrapper(
        config["model_config"],
        input_dim=config["datasets"]["input_dim"],
        pl_model=VAEModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_vae()
