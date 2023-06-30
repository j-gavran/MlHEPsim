import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.vaes.vae import VAE, VAEModel
from ml_hep_sim.vaes.vae_utils import plot_vae_result_on_event


class BetaAnnealingScheduler:
    def __init__(self, *, n_epoch, anneal_type="linear", ramp_y_step=0.1):
        """
        References
        ----------
        [1] - https://github.com/haofuml/cyclical_annealing

        """
        self.n_epoch = n_epoch

        self.start = 0.0
        self.stop = 1.0
        self.c = 1.0

        if anneal_type == "linear":
            self.L = self.frange_cycle_linear()
        elif anneal_type == "sigmoid":
            self.L = self.frange_cycle_sigmoid()
        elif anneal_type == "cosine":
            self.L = self.frange_cycle_cosine()
        elif anneal_type == "ramp":
            self.L = self.frange(ramp_y_step)
        else:
            raise NameError

    def frange_cycle_linear(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_epoch):
                L[int(i + c * period)] = v
                v += step
                i += 1

        self.L = L * self.c
        return self.L

    def frange_cycle_sigmoid(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # step is in [0,1]

        for c in range(n_cycle):
            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
                v += step
                i += 1

        self.L = L * self.c
        return self.L

    def frange_cycle_cosine(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # step is in [0,1]

        for c in range(n_cycle):
            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 0.5 - 0.5 * np.cos(v * np.pi)
                v += step
                i += 1

        self.L = L * self.c
        return self.L

    def frange(self, step):
        L = np.ones(self.n_epoch)
        v, i = self.start, 0
        while v <= self.stop:
            L[i] = v
            v += step
            i += 1

        self.L = L * self.c
        return self.L

    def test_plot(self):
        plt.plot(range(self.n_epoch), self.L)
        plt.scatter(range(self.n_epoch), self.L, s=10)
        plt.xlabel("epochs")
        plt.ylabel(r"$\beta$")

    def __call__(self, i, *args, **kwargs):
        return self.L[i]


class BetaVAE(VAE):
    def __init__(self, ae, beta=1.0, anneal_type=None, anneal_kwargs=None):
        super().__init__(ae)
        self.beta = beta
        self.current_epoch = 0

        if anneal_type:
            self.beta_scheduler = BetaAnnealingScheduler(anneal_type=anneal_type, **anneal_kwargs)
        else:
            self.beta_scheduler = lambda step: beta

    def loss_function(self, recon_x, x, mu, logvar):
        mse = F.mse_loss(recon_x, x, reduction="mean") * x.shape[1]
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        self.beta = self.beta_scheduler(self.current_epoch)

        return mse, self.beta * kld


class BetaVAEModel(VAEModel):
    def __init__(self, config, input_dim, *args, beta=1.0, **kwargs):
        super().__init__(config, input_dim, *args, **kwargs)
        self.beta = config.get("beta")

        if self.beta is None:
            self.beta = beta

        self.anneal_type = config.get("anneal_type")
        self.anneal_kwargs = config.get("anneal_kwargs")

        self.save_hyperparameters()

        self.vae = BetaVAE(self.ae, self.beta, anneal_type=self.anneal_type, anneal_kwargs=self.anneal_kwargs)

    def on_train_epoch_end(self):
        self.log("beta", self.vae.beta)
        self.vae.current_epoch = self.current_epoch

        if self.current_epoch % 10 == 0:
            plot_vae_result_on_event(
                self.data_name, self, self.logger, self._trainer.datamodule.test, idx=self.current_epoch
            )


@hydra.main(config_path="../conf", config_name="beta_vae_config", version_base=None)
def train_beta_vae(config):
    return train_wrapper(
        config["model_config"],
        input_dim=config["datasets"]["input_dim"],
        pl_model=BetaVAEModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    # annealing test plot
    #     anneal = BetaAnnealingScheduler(n_epoch=201, anneal_type="ramp", ramp_y_step=0.02)
    #     anneal.test_plot()
    #     plt.tight_layout()
    #     plt.savefig("1.pdf")
    #     plt.close()
    #
    #     anneal = BetaAnnealingScheduler(n_epoch=201, anneal_type="linear")
    #     anneal.test_plot()
    #     plt.tight_layout()
    #     plt.savefig("2.pdf")
    #     plt.close()
    #
    #     anneal = BetaAnnealingScheduler(n_epoch=201, anneal_type="cosine")
    #     anneal.test_plot()
    #     plt.tight_layout()
    #     plt.savefig("3.pdf")
    #     plt.close()

    train_beta_vae()
