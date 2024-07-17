import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.flows.models.glow import Glow
from ml.flows.models.real_nvp import RealNVP
from ml.flows.models.rq_splines import RqSplineFlow
from ml.vae.diststributions import log_normal_diag, log_standard_normal


class StandardPrior(nn.Module):
    def __init__(self, L, device):
        super().__init__()
        self.L = L
        self.device = device

        # params weights
        self.means = torch.zeros(1, L, device=self.device)
        self.logvars = torch.zeros(1, L, device=self.device)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.L, device=self.device)

    def log_prob(self, z):
        return log_standard_normal(z)


class MOGPrior(nn.Module):
    def __init__(self, L, num_components, device):
        super().__init__()

        self.L = L
        self.num_components = num_components
        self.device = device

        # params
        self.means = nn.Parameter(torch.randn(num_components, self.L, device=self.device))
        self.logvars = nn.Parameter(torch.randn(num_components, self.L, device=self.device))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L, device=self.device)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0)  # 1 x B x L
        means = means.unsqueeze(1)  # K x 1 x L
        logvars = logvars.unsqueeze(1)  # K x 1 x L

        log_p = log_normal_diag(z, means, logvars) + torch.log(w)  # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False)  # B x L

        return log_prob


class FlowPrior(nn.Module):
    def __init__(self, model_conf, flow_model_conf, data_conf, experiment_conf):
        super().__init__()
        # change to dict because of omegaconf
        self.model_conf = dict(model_conf)
        self.flow_model_conf = dict(flow_model_conf)
        self.data_conf = dict(data_conf)
        self.experiment_conf = dict(experiment_conf)

        self.flow = self._build_flow()

    def _build_flow(self):
        self.data_conf["base_distribution"] = self.flow_model_conf["base_distribution"]
        self.data_conf["input_dim"] = self.model_conf["latent_dim"]
        self.data_conf["n_mixtures"] = self.flow_model_conf.get("n_mixtures")

        if self.flow_model_conf["model_name"].lower() == "realnvp":
            assert self.data_conf["input_dim"] % 2 == 0, "Latent dim must be even!"
            flow = RealNVP(self.flow_model_conf, self.data_conf, self.experiment_conf)

        elif self.flow_model_conf["model_name"].lower() == "glow":
            assert self.data_conf["input_dim"] % 2 == 0, "Latent dim must be even!"
            flow = Glow(self.flow_model_conf, self.data_conf, self.experiment_conf)

        elif self.flow_model_conf["model_name"].lower() == "rqsplines":
            flow = RqSplineFlow(self.flow_model_conf, self.data_conf, self.experiment_conf)

        else:
            raise ValueError(f"Flow {self.flow_model_conf['model_name']} not implemented for VAE prior!")

        return flow

    def forward(self, x):
        z, log_det = self.flow.model.forward(x)
        return z, log_det

    def inverse(self, z):
        x, log_det = self.flow.model.inverse(z)
        return x, log_det

    def log_prob(self, x, reduction=None):
        z, log_jac = self.forward(x)
        log_jac_sum = sum(log_jac)
        nll = self.flow.model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)

        if reduction == "avg":
            log_p = torch.mean(log_jac_sum + nll)
        elif reduction == "sum":
            log_p = torch.sum(log_jac_sum + nll)
        else:
            log_p = log_jac_sum + nll

        return log_p

    def sample(self, n_samples):
        z = self.flow.model.base_distribution.sample((n_samples,))
        x, _ = self.inverse(z)
        return x
