from ml.vae.models.base_vae import BaseVAE, Decoder, Encoder


class VAE(BaseVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.likelihood_type == "normal":
            c = 2
        elif self.likelihood_type == "categorical":
            c = self.num_vals
        else:
            c = 1

        encoder_net = self.create_mlp([self.input_dim] + self.model_conf["encoder_layers"] + [2 * self.latent_dim])
        decoder_net = self.create_mlp([self.latent_dim] + self.model_conf["decoder_layers"] + [c * self.input_dim])

        self.encoder = Encoder(encoder_net)
        self.decoder = Decoder(decoder_net, self.likelihood_type, self.num_vals)

    def forward(self, x, reduction="avg"):
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # reconstruction error
        RE = self.decoder.log_prob(x, z)

        # KL divergence
        prior_log_prob = self.prior.log_prob(z)
        encoder_log_prob = self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)

        # handle the case where the prior and encoder log probs have already been summed
        prior_log_prob_dim = prior_log_prob.shape[-1]
        encoder_log_prob_dim = encoder_log_prob.shape[-1]

        # we have B x L and B x L
        if prior_log_prob_dim != 1 and encoder_log_prob_dim != 1:
            KL = (prior_log_prob - encoder_log_prob).sum(-1)
        # we have B x 1 and B x L (e.g. for flow prior)
        elif prior_log_prob_dim == 1 and encoder_log_prob_dim != 1:
            KL = prior_log_prob.reshape(-1) - encoder_log_prob.sum(-1)
        # not supported
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
