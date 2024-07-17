import torch
import torch.nn as nn

from ml.common.nn.mlp import BasicMLP, BasicResMLP
from ml.common.nn.modules import Module
from ml.vae.diststributions import log_bernoulli, log_categorical, log_normal_diag
from ml.vae.priors import FlowPrior, MOGPrior, StandardPrior


class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        """The reparameterization trick for Gaussians."""
        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)

        return mu + std * eps

    def encode(self, x):
        """This function implements the output of the encoder network (i.e., parameters of a Gaussian)."""
        h_e = self.encoder(x)  # The output of the encoder netowork of size 2M.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        # If we don't provide a mean and a log-variance, we must first calcuate it:
        if mu_e is None and log_var_e is None:
            mu_e, log_var_e = self.encode(x)

        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        """This function calculates the log-probability that is later used for calculating the ELBO in the latent space."""
        # If we provide x alone, then we can calculate a corresponsing sample:
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, forward_type="log_prob"):
        """Forward pass: it is either log-probability (by default) or sampling."""

        assert forward_type in ["encode", "log_prob"], "Type could be either encode or log_prob"

        if forward_type == "log_prob":
            return self.log_prob(x)
        else:
            return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution="normal", num_vals=None):
        super().__init__()
        self.decoder = decoder_net
        self.distribution = distribution

        # The number of possible values. This is important for the categorical distribution.
        self.num_vals = num_vals

    def decode(self, z):
        """This function calculates parameters of the likelihood function p(x|z)"""
        h_d = self.decoder(z)

        if self.distribution == "categorical":
            # We save the shapes: batch size:
            b = h_d.shape[0]
            # and the dimensionality of x.
            d = h_d.shape[1] // self.num_vals
            # Then we reshape to (Batch size, Dimensionality, Number of Values).
            h_d = h_d.view(b, d, self.num_vals)
            # To get probabilities, we apply softmax.
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]

        elif self.distribution == "bernoulli":
            # In the Bernoulli case, we have x_d \in {0,1}. Therefore, it is enough to output a single probability,
            # because p(x_d=1|z) = \theta and p(x_d=0|z) = 1 - \theta.
            mu_d = torch.sigmoid(h_d)
            return [mu_d]

        elif self.distribution == "normal":
            mu_d, log_var_d = torch.chunk(h_d, 2, dim=1)
            return [mu_d, log_var_d]

        else:
            raise ValueError

    def sample(self, z):
        """This function implements sampling from the decoder."""
        outs = self.decode(z)

        if self.distribution == "categorical":
            # We take the output of the decoder:
            mu_d = outs[0]
            # and save shapes (we will need that for reshaping).
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            # Here we use reshaping:
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            # Eventually, we sample from the categorical (the built-in PyTorch function).
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == "bernoulli":
            # In the case of Bernoulli, we don't need any reshaping
            mu_d = outs[0]
            # and we can use the built-in PyTorch sampler!
            x_new = torch.bernoulli(mu_d)

        elif self.distribution == "normal":
            mu_d, log_var_d = outs[0], outs[1]
            x_new = torch.normal(mu_d, torch.exp(0.5 * log_var_d))

        else:
            raise ValueError

        return x_new

    def log_prob(self, x, z):
        """This function calculates the conditional log-likelihood function."""
        outs = self.decode(z)

        if self.distribution == "categorical":
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction="sum", dim=-1).sum(-1)

        elif self.distribution == "bernoulli":
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction="sum", dim=-1)

        elif self.distribution == "normal":
            mu_d, log_var_d = outs[0], outs[1]
            log_p = log_normal_diag(x, mu_d, log_var_d, reduction="sum", dim=-1)

        else:
            raise ValueError

        return log_p

    def forward(self, z, x=None, forward_type="log_prob"):
        """The forward pass is either a log-prob or a sample."""

        assert forward_type in ["decoder", "log_prob"], "Type could be either decode or log_prob"

        if forward_type == "log_prob":
            return self.log_prob(x, z)
        else:
            return self.sample(z)


class BaseVAE(nn.Module):
    def __init__(self, model_conf, data_conf, experiment_conf):
        super().__init__()
        self.model_conf, self.data_conf, self.experiment_conf = model_conf, data_conf, experiment_conf
        self.device = experiment_conf["device"]

        self.input_dim = data_conf["input_dim"]
        self.latent_dim = model_conf["latent_dim"]
        self.prior_type = model_conf["prior_type"]  # hidden variable z prior

        self.likelihood_type = model_conf["likelihood_type"]  # decoder (likelihood) distribution
        self.num_vals = model_conf.get("num_vals")  # number of values for categorical distribution

        self.encoder = None
        self.decoder = None

        if self.prior_type == "standard_normal":
            self.prior = StandardPrior(self.latent_dim, device=self.device)

        elif self.prior_type == "mog":
            self.prior = MOGPrior(self.latent_dim, self.model_conf["n_mixtures"], device=self.device)

        elif self.prior_type == "flow":
            self.prior = FlowPrior(self.model_conf, self.model_conf["flow_model"], self.data_conf, self.experiment_conf)

        else:
            raise NotImplementedError("The selected prior is not implemented.")

    def create_mlp(self, layers):
        if self.model_conf["mlp_module"] == "BasicMLP":
            return BasicMLP(
                layers,
                act=self.model_conf["activation"],
                act_out=self.model_conf["act_out"],
                batchnorm=self.model_conf["batchnorm"],
                dropout=self.model_conf["dropout"],
                act_first=self.model_conf["act_first"],
            )
        elif self.model_conf["mlp_module"] == "BasicResMLP":
            return BasicResMLP(
                layers,
                act=self.model_conf["activation"],
                act_out=self.model_conf["act_out"],
                batchnorm=self.model_conf["batchnorm"],
                dropout=self.model_conf["dropout"],
                act_first=self.model_conf["act_first"],
                repeats=self.model_conf["repeats"],
            )
        else:
            raise NotImplementedError("The selected mlp module is not implemented.")

    def forward(self, x):
        raise NotImplementedError

    def sample(self, z):
        raise NotImplementedError


class VAEModule(Module):
    def __init__(self, model_conf, training_conf, model, data_conf, loss_func=None, tracker=None):
        super().__init__(model_conf, training_conf, model, loss_func, tracker)
        self.data_conf = data_conf
        self.save_hyperparameters(ignore=["loss_func", "tracker"])

    def training_step(self, batch, *args):
        x, _ = batch
        loss, _, _ = self.model.forward(x)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, *args):
        x, _ = batch
        loss, RE, KL = self.model.forward(x)

        self.log("val_loss", loss)
        self.log("val_RE", RE)
        self.log("val_KL", KL)

        return {"loss": loss}

    def test_step(self, batch, *args):
        pass
