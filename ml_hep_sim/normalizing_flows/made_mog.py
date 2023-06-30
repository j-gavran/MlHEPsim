import hydra
import torch
import torch.distributions as D
import torch.nn as nn

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.normalizing_flows.base_flows import NormalizingFlow, PlFlowModel
from ml_hep_sim.normalizing_flows.flow_utils import plot_flow_result_on_event
from ml_hep_sim.normalizing_flows.made import (
    GaussianMADE,
    GaussianResMADE,
    MaskedLinear,
)
from ml_hep_sim.normalizing_flows.mogs import MixtureNet


class MADEMOG(GaussianMADE):
    def __init__(self, *args, n_mixtures=1, **kwargs):
        """
        References
        ----------
        [1] - https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

        """
        super().__init__(*args, **kwargs)
        self.normalizing_direction = False

        self.register_buffer("base_distribution_mu", torch.zeros(self.input_size))
        self.register_buffer("base_distribution_var", torch.ones(self.input_size))

        self.n_mixtures = n_mixtures
        self.log_prob = None

        self.net[-1] = MaskedLinear(
            self.hidden_size, 3 * self.input_size * n_mixtures, self.masks[-1].repeat(3 * n_mixtures, 1)
        )
        self.net = MixtureNet(self.input_size, n_mixtures, self.net)

        self.naninf_mask = False

    @property
    def base_distribution(self):
        return D.Normal(self.base_distribution_mu, self.base_distribution_var)

    def forward(self, x, log_prob=True):
        mu, log_std, log_weights = self.net(x)

        N, C, L = mu.shape
        x = x.repeat(1, C).view(N, C, L)

        u = (x - mu) * torch.exp(-log_std)
        log_abs_det_jacobian = -log_std

        if log_prob:
            if self.naninf_mask:
                # A fix so we can get validation to work. The reason for invalid values in MADEs validation is probably
                # different behaviour of batchnorm when model in eval mode. Used only in validation_step.
                # https://stackoverflow.com/questions/64594493/filter-out-nan-values-from-a-pytorch-n-dimensional-tensor
                v = mu + log_std + log_weights

                shape = v.shape
                v_reshaped = v.reshape(shape[0], -1)

                cutoff = 1e3
                nan_mask = ~torch.any(v_reshaped.isnan(), dim=1)
                inf_mask = ~torch.any(v_reshaped.isinf(), dim=1)
                overflow_mask = torch.all(torch.abs(v_reshaped) < cutoff, dim=1)

                self.naninf_mask = nan_mask & inf_mask & overflow_mask

                log_weights_reshaped = log_weights.reshape(shape[0], -1)
                u_reshaped = u.reshape(shape[0], -1)
                log_abs_det_jacobian_reshaped = log_abs_det_jacobian.reshape(shape[0], -1)

                log_weights_reshaped = log_weights_reshaped[self.naninf_mask]
                u_reshaped = u_reshaped[self.naninf_mask]
                log_abs_det_jacobian_reshaped = log_abs_det_jacobian_reshaped[self.naninf_mask]

                log_weights = log_weights_reshaped.reshape(log_weights_reshaped.shape[0], *shape[1:])
                u = u_reshaped.reshape(u_reshaped.shape[0], *shape[1:])
                log_abs_det_jacobian = log_abs_det_jacobian_reshaped.reshape(
                    log_abs_det_jacobian_reshaped.shape[0], *shape[1:]
                )

            self.log_prob = torch.sum(
                torch.logsumexp(log_weights + self.base_distribution.log_prob(u) + log_abs_det_jacobian, dim=1),
                dim=-1,
                keepdim=True,
            )  # N x C x L -> N x L -> N x 1

        return u.view(u.shape[0], C * L), log_abs_det_jacobian

    def inverse(self, u):
        N, L = u.shape
        x = torch.zeros(N, L, device=u.device)

        for i in range(self.input_size):
            mu, log_std, log_weights = self.net(x)  # N x C x L

            mu_x = mu[:, :, i].unsqueeze(-1)  # N x C x 1
            std_x = torch.exp(log_std[:, :, i].unsqueeze(-1))  # N x C x 1
            log_weights_x = log_weights[:, :, i].unsqueeze(-1)  # N x C x 1

            x[:, i] = self.net.sample(log_weights_x, std_x, mu_x)

        log_abs_det_jacobian = log_std
        return x, log_abs_det_jacobian


class ResMADEMOG(MADEMOG, GaussianResMADE):
    def __init__(self, *args, n_mixtures=1, **kwargs):
        super().__init__(*args, n_mixtures=n_mixtures, **kwargs)


class MOGNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def sample(self, num_samples):
        dummy = torch.zeros(num_samples, self.dim, device=self.base_distribution.loc.device)
        xs, _ = self.inverse(dummy)
        return xs


class MADEMOGFlowModel(PlFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = config["learning_rate"]
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.activation = config["activation"]
        self.num_hidden_layers_mog_net = config["num_hidden_layers_mog_net"]
        self.hidden_layer_dim = config["hidden_layer_dim"]
        self.n_mixtures = config["n_mixtures"]
        self.residuals = config["residuals"]
        self.save_hyperparameters()

        if self.residuals:
            mademog = ResMADEMOG
        else:
            mademog = MADEMOG

        self.blocks = [
            mademog(
                self.input_dim,
                self.hidden_layer_dim,
                self.num_hidden_layers_mog_net,
                activation=self.activation,
                n_mixtures=self.n_mixtures,
            )
        ]

        self.pop_idx = -1

        self.flow = MOGNormalizingFlow(self.input_dim, self.blocks, density=self.base_distribution)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # _: (N, L), log_jac: list of [(N, 1),...,(N, C, L)]
        _, log_jac = self.flow(x)

        log_jac.pop(self.pop_idx)  # accounted for in logsumexp

        mog_nll = self.flow.bijectors[self.pop_idx].log_prob  # (N, 1)

        sum_of_log_det_jacobian = sum(log_jac)  # (N, 1)

        loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)  # (N, 1) + (N, 1) -> (1,)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        self.flow.bijectors[self.pop_idx].naninf_mask = True
        _, log_jac = self.flow(x)
        naninf_mask = self.flow.bijectors[self.pop_idx].naninf_mask
        self.flow.bijectors[self.pop_idx].naninf_mask = False

        log_jac.pop(self.pop_idx)

        mog_nll = self.flow.bijectors[self.pop_idx].log_prob

        if len(log_jac) != 0:
            sum_of_log_det_jacobian = sum(log_jac)[naninf_mask]
            loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)

            self.log("val_loss", loss)
            self.log("sum_log_det_jac", torch.mean(sum_of_log_det_jacobian))
            self.log("val_nll", torch.mean(mog_nll))

            return {"val_loss": loss, "sum_log_det_jac": sum_of_log_det_jacobian, "val_nll": mog_nll}
        else:
            loss = -torch.mean(mog_nll)

            self.log("val_loss", loss)
            self.log("val_nll", torch.mean(mog_nll))

            return {"val_loss": loss, "val_nll": mog_nll}

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


@hydra.main(config_path="../conf", config_name="made_mog_config", version_base=None)
def train_mademog(config):
    device = "cuda" if config["datasets"]["data_params"]["to_gpu"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=MADEMOGFlowModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_mademog()
