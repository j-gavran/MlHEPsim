import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class Dequantization:
    def __init__(self, alpha=1e-6, device="cpu"):
        """Dequantization transform. Makes discrete values continuous.

        Parameters
        ----------
        alpha: float
            Small constant that is used to scale the original input. Prevents dealing with values very close to 0 and
            1 when inverting the sigmoid, by default 1e-6.
        device: str
            Device to use for the operations, by default "cpu".

        References
        ----------
        [1] - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html#Dequantization

        """
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.quants_lst = []

    def __call__(self, z, reverse=False):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        z = z.to(self.device)

        if not reverse:
            z = z.to(torch.int).to(torch.float32)

        z_out = torch.zeros_like(z)
        for i in range(z.shape[1]):
            z_i = z[:, i][:, None]

            if not reverse and z_i.min() < 0.0:
                raise ValueError("Negative values are not allowed! Redefine disc. values to be positive.")

            if not reverse:
                quants = z_i.max() + 1
                self.quants_lst.append(quants)
                z_transf = self.forward(z_i, quants, reverse=False)
            else:
                quants = self.quants_lst[i]
                z_transf = self.forward(z_i, quants, reverse=True)

            z_out[:, i] = z_transf.squeeze()

        return z_out.cpu().numpy()

    def forward(self, z, quants, reverse=False):
        if not reverse:
            z = self.dequant(z, quants)
            z = self.sigmoid(z, reverse=True)
        else:
            z = self.sigmoid(z, reverse=False)
            z = z * quants
            z = torch.floor(z).clamp(min=0, max=quants - 1).to(torch.int32)

        return z

    def sigmoid(self, z, reverse=False):
        # applies an invertible sigmoid transformation
        if not reverse:
            z = torch.sigmoid(z)
            # reversing scaling for numerical stability
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # scale to prevent boundaries 0 and 1
            z = torch.log(z) - torch.log(1 - z)

        return z

    def dequant(self, z, quants):
        # transform discrete values to continuous volumes
        z = z + torch.rand_like(z).detach()
        z = z / quants
        return z


class DequantizationTransform(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1e-6, device="cpu"):
        super().__init__()
        self.alpha, self.device = alpha, device
        self.dequantizer = Dequantization(alpha=alpha, device=device)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.dequantizer(X, reverse=False)

    def inverse_transform(self, X, y=None):
        return self.dequantizer(X, reverse=True)
