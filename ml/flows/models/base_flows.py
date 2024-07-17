import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm

from ml.common.nn.modules import Module
from ml.flows.models.flow_utils import LogisitcDistribution
from ml.flows.models.mogs import MOGPrior


class NormalizingFlow(nn.Module):
    def __init__(self, dim, blocks, density=None):
        """Implements base class for chaining together flow layers (bijectors).

        Note
        ----
        Training is implemented within the forward method that calls inverse of a flow because we are interested in
        the generative process and need to train in the normalizing direction (defined as inverse in flows). Also note
        that some flows only support forward in the sense of this class (e.g. batchnorm).

        Parameters
        ----------
        dim: int
            Input dimension.
        blocks : list of flow objects
            Chained together flow of nn modules.
        density : PyTorch distribution
            Base/prior distribution. Must subclass distributions.Distribution or implement log_prob and sample methods.
        """
        super().__init__()
        self.dim = dim
        self.bijectors = nn.ModuleList(blocks)
        self.base_distribution = density
        self.log_det = None

    def forward(self, z):
        self.log_det = []

        for bijector in self.bijectors:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.inverse(z)
            else:
                z, log_abs_det = bijector.forward(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det

    def inverse(self, z):
        self.log_det = []

        for bijector in self.bijectors[::-1]:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.forward(z)
            else:
                z, log_abs_det = bijector.inverse(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det

    def sample(self, num_samples):
        z = self.base_distribution.sample((num_samples,))
        xs, _ = self.inverse(z)
        return xs


class AutoregressiveNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def sample(self, num_samples):
        dummy = torch.zeros(num_samples, self.dim, device=self.device)
        xs, _ = self.inverse(dummy)
        return xs


class BaseFlowModel(nn.Module):
    def __init__(self, model_conf, data_conf, experiment_conf):
        super().__init__()
        self.model_conf = model_conf
        self.input_dim = data_conf["input_dim"]
        self.base_distribution_name = data_conf["base_distribution"]
        self.device = experiment_conf["device"]
        self.model_name = model_conf["model_name"]

        if self.base_distribution_name.lower() == "normal":
            self.base_distribution = Normal(
                torch.zeros(self.input_dim).to(self.device),
                torch.ones(self.input_dim).to(self.device),
            )

        elif self.base_distribution_name.lower() == "logistic":
            self.base_distribution = LogisitcDistribution(self.input_dim, device=self.device)

        elif self.base_distribution_name.lower() == "mog":
            n_mixtures = data_conf.get("n_mixtures")
            if n_mixtures is not None:
                n_mixtures = self.input_dim

            self.base_distribution = MOGPrior(self.input_dim, n_mixtures=n_mixtures, device=self.device)

        else:
            raise ValueError(f"Base distribution {self.base_distribution_name} not implemented!")

        self.data_name = data_conf["data_name"]

    def forward(self, x):
        z, log_det = self.model.forward(x)
        return z, log_det

    def inverse(self, z):
        x, log_det = self.model.inverse(z)
        return x, log_det

    def estimate_density(self, data_points, exp=True, mean=True, chunks=10):
        if self.model.training:
            raise ValueError("Model must be in eval mode!")

        if not torch.is_tensor(data_points):
            data_points = torch.from_numpy(data_points.astype(np.float32)).to(self.device)

        density_data = []
        N = data_points.shape[0]
        chunks_lst = chunks * [N // chunks]

        if N % chunks != 0:
            chunks_lst += [N % chunks]

        with torch.no_grad():
            for i, chunk in tqdm(enumerate(chunks_lst), desc=f"Estimating density for {N} examples", leave=False):
                z, log_jac = self.forward(data_points[i * chunk : (i + 1) * chunk])
                sum_log_jac = sum(log_jac)

                try:
                    nll = self.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
                except ValueError:
                    logging.warning("Density estimation failed for some data points!")
                    continue

                if mean:
                    log_density = -torch.mean(nll + sum_log_jac)
                else:
                    log_density = -(nll + sum_log_jac)

                density_data.append(log_density.squeeze().cpu().numpy())

        density_data = np.concatenate(density_data)

        if exp:
            return np.exp(density_data)
        else:
            return density_data

    def sample(self, N, chunks=10):
        if self.model.training:
            raise ValueError("Model must be in eval mode!")

        generated_data = []
        chunks_lst = chunks * [N // chunks]

        if N % chunks != 0:
            chunks_lst += [N % chunks]

        with torch.no_grad():
            for chunk in tqdm(chunks_lst, desc=f"Generating {N} examples", leave=False):
                try:
                    flow_sample = self.model.sample(chunk).cpu().numpy()
                except ValueError:
                    logging.warning("Sampling failed for some data points!")
                    continue

                generated_data.append(flow_sample)

        if len(generated_data) != 0:
            generated_data = np.vstack(generated_data)

        return generated_data


class FlowModel(Module):
    def __init__(self, model_conf, training_conf, data_conf, model=None, loss_func=None, tracker=None):
        super().__init__(model_conf, training_conf, model, loss_func, tracker)
        self.data_conf = data_conf
        self.current_step = 0
        self.save_hyperparameters(ignore=["loss_func", "tracker", "model"])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, log_jac = self.model(x)

        jac_loss = sum(log_jac)
        nll = self.model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
        loss = -torch.mean(jac_loss + nll)

        self.log("train_loss", loss)
        self.current_step += 1

        return {"loss": loss}

    @staticmethod
    def _get_invalid_mask(z):
        nan_mask = ~torch.any(z.isnan(), dim=1)
        inf_mask = ~torch.any(z.isinf(), dim=1)
        return torch.logical_and(nan_mask, inf_mask)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z, log_jac = self.model(x)

        jac_loss = sum(log_jac)

        # remove invalid values for validation (need to be careful with this!)
        mask_z, mask_jac = self._get_invalid_mask(z), self._get_invalid_mask(jac_loss)
        mask = torch.logical_and(mask_z, mask_jac)

        z, jac_loss = z[mask], jac_loss[mask]

        nll = self.model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
        loss = -torch.mean(jac_loss + nll)

        self.log("val_loss", loss)
        self.log("sum_log_det_jac", torch.mean(jac_loss))
        self.log("val_nll", torch.mean(nll))

        return {"val_loss": loss, "sum_log_det_jac": jac_loss, "val_nll": nll}
