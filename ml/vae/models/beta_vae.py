from ml.vae.models.vae import VAE


class BetaVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = self.model_conf["beta"]

    def forward(self, x, reduction="avg"):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        RE = self.decoder.log_prob(x, z)

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

        KL = self.beta * KL

        # ELBO with beta
        if reduction == "sum":
            return -(RE + KL).sum(), RE.sum(), KL.sum()
        elif reduction == "avg":
            return -(RE + KL).mean(), RE.mean(), KL.mean()
        else:
            return -(RE + KL), RE, KL
