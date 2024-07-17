import numpy as np
import torch
import torch.distributions as D

from ml.flows.models.base_flows import (
    AutoregressiveNormalizingFlow,
    BaseFlowModel,
    FlowModel,
)
from ml.flows.models.made import GaussianMADE, GaussianResMADE, MaskedLinear
from ml.flows.models.mogs import MixtureNet


class MADEMOGModel(GaussianMADE):
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

    @property
    def base_distribution(self):
        return D.Normal(self.base_distribution_mu, self.base_distribution_var, validate_args=False)

    def forward(self, x, log_prob=True):
        mu, log_std, log_weights = self.net(x)

        N, C, L = mu.shape
        x = x.repeat(1, C).view(N, C, L)

        u = (x - mu) * torch.exp(-log_std)
        log_abs_det_jacobian = -log_std

        if log_prob:
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


class ResMADEMOGModel(MADEMOGModel, GaussianResMADE):
    def __init__(self, *args, n_mixtures=1, **kwargs):
        super().__init__(*args, n_mixtures=n_mixtures, **kwargs)


class MADEMOG(BaseFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = self.model_conf["activation"]
        self.num_hidden_layers_mog_net = self.model_conf["num_hidden_layers_mog_net"]
        self.hidden_layer_mog_dim = self.model_conf["hidden_layer_mog_dim"]
        self.n_mixtures = self.model_conf["n_mixtures"]
        self.mog_residuals = self.model_conf["mog_residuals"]
        self.res_layers_in_mog_block = self.model_conf["res_layers_in_mog_block"]
        self.res_batchnorm = self.model_conf.get("res_batchnorm", False)
        self.act_first = self.model_conf.get("act_first", False)

        self.pop_idx = -1

        if self.mog_residuals:
            self.blocks = [
                ResMADEMOGModel(
                    self.input_dim,
                    k=self.hidden_layer_mog_dim,
                    n_blocks=self.num_hidden_layers_mog_net,
                    l=self.res_layers_in_mog_block,
                    activation=self.activation,
                    n_mixtures=self.n_mixtures,
                    res_batchnorm=self.res_batchnorm,
                    act_first=self.act_first,
                )
            ]
        else:
            self.blocks = [
                MADEMOGModel(
                    self.input_dim,
                    self.hidden_layer_mog_dim,
                    self.num_hidden_layers_mog_net,
                    activation=self.activation,
                    n_mixtures=self.n_mixtures,
                )
            ]

        self.model = AutoregressiveNormalizingFlow(self.input_dim, self.blocks, density=None, device=self.device)

    def estimate_density(self, data_points, exp=True, mean=True):
        if self.model.training:
            raise ValueError("Model must be in eval mode!")

        if not torch.is_tensor(data_points):
            data_points = torch.from_numpy(data_points.astype(np.float32)).to(self.device)

        with torch.no_grad():
            _, log_jac = self.forward(data_points)
            log_jac.pop(self.pop_idx)
            mog_nll = self.model.bijectors[self.pop_idx].log_prob
            sum_of_log_det_jacobian = sum(log_jac)

        if mean:
            log_density = -torch.mean(mog_nll + sum_of_log_det_jacobian)
        else:
            log_density = -(mog_nll + sum_of_log_det_jacobian)

        if exp:
            return torch.exp(log_density).cpu().numpy()
        else:
            return log_density.cpu().numpy()


class MOGFlowModel(FlowModel):
    def __init__(self, model_conf, training_conf, data_conf, model=None, loss_func=None, tracker=None):
        super().__init__(model_conf, training_conf, data_conf, model, loss_func, tracker)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        _, log_jac = self.model(x)  # _: (N, L), log_jac: list of [(N, 1),...,(N, C, L)]

        log_jac.pop(self.model.pop_idx)  # accounted for in logsumexp

        mog_nll = self.model.model.bijectors[self.model.pop_idx].log_prob  # (N, 1)

        sum_of_log_det_jacobian = sum(log_jac)  # (N, 1)

        loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)  # (N, 1) + (N, 1) -> (1,)

        self.log("train_loss", loss)
        self.current_step += 1

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        _, log_jac = self.model(x)

        log_jac.pop(self.model.pop_idx)

        mog_nll = self.model.model.bijectors[self.model.pop_idx].log_prob

        if len(log_jac) != 0:
            sum_of_log_det_jacobian = sum(log_jac)

            mask_nll, mask_jac = self._get_invalid_mask(mog_nll), self._get_invalid_mask(sum_of_log_det_jacobian)
            mask = torch.logical_and(mask_nll, mask_jac)

            sum_of_log_det_jacobian, mog_nll = sum_of_log_det_jacobian[mask], mog_nll[mask]

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
