import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn


class MOGPrior(nn.Module):
    def __init__(self, input_dim, n_mixtures, device):
        """
        References
        ----------
        [1] - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_priors_example.ipynb
        [2] - https://jmtomczak.github.io/blog/7/7_priors.html

        """
        super().__init__()
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'!")

        self.input_dim = input_dim
        self.n_mixtures = n_mixtures

        self.mu = nn.Parameter(torch.zeros(n_mixtures, input_dim, device=device), requires_grad=True)
        self.log_std = nn.Parameter(torch.randn(n_mixtures, input_dim, device=device), requires_grad=True)
        self.weights = nn.Parameter(torch.zeros(n_mixtures, device=device), requires_grad=True)

    def gaussian_logprob(self, std, mu, x):
        return -torch.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * torch.pow((x - mu) / std, 2)

    def log_prob(self, x):
        x = x.unsqueeze(0)  # 1 x B x L

        mu = self.mu.unsqueeze(1)  # C x 1 x L
        std = torch.exp(self.log_std).unsqueeze(1)  # C x 1 x L
        log_weights = F.log_softmax(self.weights, dim=0)[:, None, None]  # C x 1 x 1

        gaussian_logprob = self.gaussian_logprob(std, mu, x)  # C x B x L

        return torch.logsumexp(gaussian_logprob + log_weights, dim=0)  # B x L

    def sample(self, n):
        std = torch.exp(self.log_std)  # C x L
        log_weights = F.log_softmax(self.weights, dim=0)  # C

        logits = log_weights.unsqueeze(0)  # 1 x C

        idx = D.Categorical(logits=logits).sample(n)  # n x 1

        mu_idx = self.mu[idx].squeeze(1)  # n x L
        std_idx = std[idx].squeeze(1)  # n x L

        u = torch.randn_like(mu_idx)  # n x L

        return u * std_idx + mu_idx  # n x L


class MixtureNet(nn.Module):
    def __init__(self, input_dim, n_mixtures, net):
        """
        References
        ----------
        [1] - https://github.com/manujosephv/pytorch_tabular/blob/main/pytorch_tabular/models/mixture_density/mdn.py

        """
        super().__init__()
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.net = net
        self.epsilon = 1e-4

    def forward(self, x):
        # batch size, feature dimension, number of gaussians

        N, L, C = x.shape[0], x.shape[1], self.n_mixtures
        # net outout: (batch size, 3 * L * C) -> (batch size, C, 3 * L) -> 3 x (batch size, C, L)
        mu, std, logits = self.net(x).view(N, C, 3 * L).chunk(chunks=3, dim=-1)

        log_weights = F.log_softmax(logits, dim=1)
        log_std = torch.log(F.softplus(std) + self.epsilon)

        return mu, log_std, log_weights

    def gaussian_logprob(self, std, mu, x):
        return -torch.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * torch.pow((x - mu) / std, 2)

    def log_prob(self, x, log_weights, std, mu, other=None, repeat=True):
        if repeat:
            N, C, L = mu.shape
            x = x.repeat(1, C).view(N, C, L)

        gaussian_logprob = self.gaussian_logprob(std, mu, x)

        if other is None:
            return torch.logsumexp(gaussian_logprob + log_weights, dim=1, keepdim=False)  # N x L
        else:
            return torch.logsumexp(gaussian_logprob + log_weights + other, dim=1, keepdim=False)  # N x L

    @staticmethod
    def sample(log_weights, std, mu):
        N, C, L = log_weights.shape

        categorical = D.Categorical(logits=log_weights.view(N, L, C))  # N x L x C
        idx = categorical.sample().unsqueeze(1)  # N x 1 x L

        mu_gather = torch.gather(mu, 1, idx)  # N x 1 x L
        std_gather = torch.gather(std, 1, idx)  # N x 1 x L

        u = torch.randn_like(mu_gather)  # N x 1 x L

        sample = u * std_gather + mu_gather  # N x 1 x L

        return sample.squeeze()  # N x L
