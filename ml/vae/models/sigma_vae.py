import torch
import torch.nn as nn

from ml.vae.diststributions import log_normal_diag
from ml.vae.models.base_vae import BaseVAE, Encoder


class SigmaDecoder(nn.Module):
    def __init__(self, decoder_net, optimal_sigma=False):
        super().__init__()
        self.decoder = decoder_net
        self.optimal_sigma = optimal_sigma

        if not self.optimal_sigma:
            self.log_sigma = nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        else:
            self.log_sigma = None

    def decode(self, z):
        return self.decoder(z)

    def sample(self, z):
        outs = self.decode(z)
        x_new = torch.normal(outs, torch.exp(0.5 * self.log_sigma))
        return x_new

    def log_prob(self, x, z, log_sigma=None):
        outs = self.decode(z)
        if log_sigma is None:
            log_p = log_normal_diag(x, outs, self.log_sigma, reduction="sum", dim=-1)
        else:
            log_p = log_normal_diag(x, outs, log_sigma, reduction="sum", dim=-1)

        return log_p

    def forward(self, z, x=None, type="log_prob"):
        assert type in ["decoder", "log_prob"], "Type could be either decode or log_prob"

        if type == "log_prob":
            return self.log_prob(x, z)
        else:
            return self.sample(x)


class SigmaVAE(BaseVAE):
    def __init__(self, *args, **kwargs):
        """https://orybkin.github.io/sigma-vae/"""
        super().__init__(*args, **kwargs)
        self.optimal_sigma = self.model_conf["optimal_sigma"]
        self.shared_optimal_sigma = self.model_conf["shared_optimal_sigma"]

        encoder_net = self.create_mlp([self.input_dim] + self.model_conf["encoder_layers"] + [2 * self.latent_dim])
        decoder_net = self.create_mlp([self.latent_dim] + self.model_conf["decoder_layers"] + [self.input_dim])

        self.encoder = Encoder(encoder_net)
        self.decoder = SigmaDecoder(decoder_net, optimal_sigma=self.optimal_sigma)

    def forward(self, x, reduction="avg"):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # sigma
        if self.training:
            if self.optimal_sigma:
                recon_x = self.decoder.decode(z)

                if self.shared_optimal_sigma:
                    log_sigma = ((x - recon_x) ** 2).mean().sqrt().log()  # average over batch and features
                else:
                    log_sigma = ((x - recon_x) ** 2).mean(0).sqrt().log()  # average over batch

                self.decoder.log_sigma = log_sigma
            else:
                log_sigma = self.decoder.log_sigma
        else:
            log_sigma = self.decoder.log_sigma

        # RE
        RE = self.decoder.log_prob(x, z, log_sigma=log_sigma)

        # KL
        prior_log_prob = self.prior.log_prob(z)
        encoder_log_prob = self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)

        prior_log_prob_dim = prior_log_prob.shape[-1]
        encoder_log_prob_dim = encoder_log_prob.shape[-1]

        if prior_log_prob_dim != 1 and encoder_log_prob_dim != 1:
            KL = (prior_log_prob - encoder_log_prob).sum(-1)
        elif prior_log_prob_dim == 1 and encoder_log_prob_dim != 1:
            KL = prior_log_prob.reshape(-1) - encoder_log_prob.sum(-1)
        else:
            raise ValueError("Dimensions of prior and encoder log probs mismatch!")

        # ELBO
        if reduction == "sum":
            return -(RE + KL).sum(), RE.sum(), KL.sum()
        elif reduction == "avg":
            return -(RE + KL).mean(), RE.mean(), KL.mean()
        else:
            return -(RE + KL), RE, KL

    def sample(self, n):
        z = self.prior.sample(n)
        return self.decoder.sample(z)
