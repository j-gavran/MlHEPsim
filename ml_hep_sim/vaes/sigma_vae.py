import hydra
import numpy as np
import torch
import torch.nn.functional as F

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.vaes.beta_vae import BetaVAE, BetaVAEModel
from ml_hep_sim.vaes.vae_utils import plot_vae_result_on_event


class SigmaVAE(BetaVAE):
    def __init__(self, ae, beta=1.0, vec_sigma=None, mse_sigma=None, **kwargs):
        """VAE with learnable KL scaling (beta) parameter.

        By default learns :math:`\sigma^2` as described in [2].

        Parameters
        ----------
        vec_sigma : None, optional
            Vector of :math:`\sigma^2` (dimension of latent space) instead of a number, by default False.
        mse_sigma : None, optional
            Use mse :math:`\sigma^2` estimation from [1, 3], by default False.
        beta : float, optional
            Fixed KL scaling parameter, by default 1.0.

        Note
        ----
        :math:`\log(\sigma^2)` is used for stability reasons.

        References
        ----------
        [1] - https://github.com/orybkin/sigma-vae-pytorch
        [2] - https://github.com/daib13/TwoStageVAE
        [3] - https://github.com/asperti/BalancingVAE

        """
        super().__init__(ae, beta, **kwargs)

        self.beta = beta
        self.vec_sigma = vec_sigma
        self.mse_sigma = mse_sigma

        if self.mse_sigma:
            self.log_sigma = 0.0
        elif self.vec_sigma:
            self.log_sigma = torch.nn.Parameter(torch.full((ae.input_dim,), 0.0), requires_grad=True)
        else:
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)

        self._half_log_two_pi = 0.5 * np.log(2 * np.pi)

    @staticmethod
    def _softclip(tensor, mn=-6):
        return mn + F.softplus(tensor - mn)

    def loss_function(self, recon_x, x, mu, logvar):
        if self.mse_sigma:
            self.log_sigma = self._softclip(((x - recon_x) ** 2).mean().sqrt().log())

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        gen_loss = 0.5 * torch.sum(
            ((x - recon_x) / self._softclip(self.log_sigma).exp()) ** 2
            + self._softclip(self.log_sigma)
            + self._half_log_two_pi
        )

        self.beta = self.beta_scheduler(self.current_epoch)

        return gen_loss, self.beta * kl_loss


class SigmaVAEModel(BetaVAEModel):
    def __init__(self, config, input_dim, beta=1.0, *args, **kwargs):
        super().__init__(config, input_dim, *args, beta=beta, **kwargs)

        vec_sigma = config.get("vec_sigma")
        mse_sigma = config.get("mse_sigma")

        self.vae = SigmaVAE(
            self.ae,
            self.beta,
            vec_sigma=vec_sigma,
            mse_sigma=mse_sigma,
            anneal_type=self.anneal_type,
            anneal_kwargs=self.anneal_kwargs,
        )

    def on_train_epoch_end(self):
        self.log("beta", self.vae.beta)
        self.log("log_sigma", self.vae.log_sigma.mean())
        self.vae.current_epoch = self.current_epoch

        if self.current_epoch % 10 == 0:
            plot_vae_result_on_event(
                self.data_name, self, self.logger, self._trainer.datamodule.test, idx=self.current_epoch
            )


@hydra.main(config_path="../conf", config_name="sigma_vae_config", version_base=None)
def train_sigma_vae(config):
    return train_wrapper(
        config["model_config"],
        input_dim=config["datasets"]["input_dim"],
        pl_model=SigmaVAEModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_sigma_vae()
