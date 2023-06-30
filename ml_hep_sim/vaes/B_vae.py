import numpy as np
import torch

from ml_hep_sim.data_utils.dataset_utils import rescale_data
from ml_hep_sim.ml_utils import mlf_loader


class BVAE:
    def __init__(self, bvae, buffer_data, n):
        self.bvae = bvae
        self.buffer_data = buffer_data
        self.n = n
        self.mu, self.logvar = self._get_mu_logvar()

    def _get_mu_logvar(self):
        with torch.no_grad():
            mu, logvar = self.bvae.encode(self.buffer_data)
        return mu, logvar

    def _sample_mu_logvar(self):
        idx = np.random.choice(np.arange(0, len(self.buffer_data), 1), self.n, replace=True)
        mu_sample, logvar_sample = self.mu[idx], self.logvar[idx]
        return mu_sample, logvar_sample

    def _sample_z(self):
        mu_sample, logvar_sample = self._sample_mu_logvar()

        var = torch.exp(0.5 * logvar_sample)
        eps = torch.randn_like(mu_sample)
        z = mu_sample + eps * var

        return z

    def generate(self):
        z = self._sample_z()
        with torch.no_grad():
            decoded = self.bvae.decode(z).detach()
        return decoded.cpu().numpy()


def bvae_generate(model_str, buffer_str, n, device="cuda", rescale=None, drop_first_col=True):
    """
    References
    ----------
    [1] - https://arxiv.org/abs/1901.00875

    """

    model = mlf_loader(model_str).eval().to(device)
    buffer_data = np.load(buffer_str).astype(np.float32)

    if drop_first_col:
        buffer_data = buffer_data[:, 1:]

    if rescale is not None:
        buffer_data, buffer_scaler = rescale_data(buffer_data, rescale_type=rescale)

    if type(buffer_data) == np.ndarray:
        buffer_data = torch.from_numpy(buffer_data).to(device)

    bvae_obj = BVAE(model.vae, buffer_data, n)

    generated = bvae_obj.generate()
    buffer_data = buffer_data.cpu().numpy()

    if rescale:
        generated = model.scalers[0].inverse_transform(generated)
        buffer_data = buffer_scaler.inverse_transform(buffer_data)
        return generated, buffer_data

    return generated, buffer_data


if __name__ == "__main__":
    g, b = bvae_generate(
        "file:///data0/jang/masters/mlruns/880809362999188590/afc63c2e2186416e8475b94cb9046fae/artifacts/Higgs_bVAE",
        "data/higgs/HIGGS_18_feature_train.npy",
        10**6,
    )

    print(g.shape, b.shape)
