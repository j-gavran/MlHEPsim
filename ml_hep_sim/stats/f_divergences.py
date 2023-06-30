import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from ml_hep_sim.ml_utils import mlf_loader


def histogram_density(A, B, n_bins="auto", bin_range=None, density=None, non_zero_bins=True):
    n_features = A.shape[1]

    combined_sample = np.concatenate([A, B])

    histograms_A, histograms_B = [], []
    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(combined_sample[:, feature], bins=n_bins, range=bin_range)

        h_A, _ = np.histogram(A[:, feature], bins=bin_edges, density=density)
        h_B, _ = np.histogram(B[:, feature], bins=bin_edges, density=density)

        if non_zero_bins:
            mask = (h_A != 0.0) & (h_B != 0.0)
            h_A, h_B = h_A[mask], h_B[mask]

        histograms_A.append(h_A)
        histograms_B.append(h_B)

    return histograms_A, histograms_B


def kde_density(A, B, kernel="gaussian", bandwidth=1.0, atol=0.0, rtol=0.0):
    """https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity"""
    n_features = A.shape[1]

    kdes_A, kdes_B = [], []
    for feature in tqdm(range(n_features)):
        a, b = A[:, feature][:, None], B[:, feature][:, None]

        kde_A = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol).fit(a)
        kde_B = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol).fit(b)

        score_A, score_B = kde_A.score_samples(a), kde_B.score_samples(b)

        kdes_A.append(np.exp(score_A))
        kdes_B.append(np.exp(score_B))

    return kdes_A, kdes_B


def _get_flow_density(sample, model):
    if model.training:
        raise ValueError

    z, jac = model.flow(torch.from_numpy(sample).to(model.device))
    sum_jac = sum(jac)
    base = model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
    log_density = (base + sum_jac).cpu().numpy()

    return np.exp(log_density)


def flow_density(A, B, model_str, device="cuda"):
    model = mlf_loader(model_str).eval().to(device)
    with torch.no_grad():
        flow_A, flow_B = _get_flow_density(A, model), _get_flow_density(B, model)
        flow_A, flow_B = flow_A.flatten(), flow_B.flatten()

        mask = (flow_A != 0.0) & (flow_B != 0.0)
        flow_A, flow_B = flow_A[mask], flow_B[mask]

    return [flow_A], [flow_B]


class fDivergence:
    def __init__(
        self, P, Q, density_estimation="hist", mean_reduction=True, distance_metric=False, density_kwargs=None
    ):
        """Divergence measures between probability distributions.

        Parameters
        ----------
        P : torch.tensor or np.ndarray
            Samples from P distribution.
        Q : torch.tensor or np.ndarray
            Samples from Q distribution.
        density_estimation : str
            Histogram (hist), kernel density estimation (kde) or normalizing flow (flow).
        mean_reduction : bool
            Calculate mean over features after sum in divergences.
        distance_metric : bool
            Sqrt and divide by 2 in the final divergence result.
        density_kwargs : dict
            Dictionary of keyword arguments for density astimation functions.

        References
        ----------
        [1] - Probabilistic Machine Learning: Advanced Topics by Kevin P. Murphy
        [2] - https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition
        [3] - https://en.wikipedia.org/wiki/Hellinger_distance#Discrete_distributions
        [4] - Chi-square histogram distance: https://arxiv.org/abs/1212.0383
        [5] - Alpha divergence: https://www.ece.rice.edu/~vc3/elec633/AlphaDivergence.pdf

        """
        self.P, self.Q = self._tensor_check(P, Q)
        self.n, self.d = P.shape  # batches, features

        self.density_estimation = density_estimation
        self.mean_reduction = mean_reduction
        self.distance_metric = distance_metric

        if self.density_estimation.lower() == "flow":
            self.d = 1

        if density_kwargs is None:
            density_kwargs = {}
        self.density_kwargs = density_kwargs

        self.p, self.q = self._estimate_density()

    @staticmethod
    def _tensor_check(P, Q, sort=False):
        assert P.shape == Q.shape

        if torch.is_tensor(P) or torch.is_tensor(Q):
            P, Q = P.cpu().numpy(), Q.cpu().numpy()

        if sort:
            P, Q = np.sort(P, axis=0), np.sort(Q, axis=0)

        return P, Q

    def _estimate_density(self):
        """Desnity estimator method.

        Returns
        -------
        Tuple (p, q) of list of arrays containing density values for each feature.

        """
        if self.density_estimation.lower() == "hist":
            p, q = histogram_density(self.P, self.Q, density=True, **self.density_kwargs)
        elif self.density_estimation.lower() == "kde":
            p, q = kde_density(self.P, self.Q, **self.density_kwargs)
        elif self.density_estimation.lower() == "flow":
            p, q = flow_density(self.P, self.Q, **self.density_kwargs)
        else:
            raise ValueError

        return p, q

    def _divergence_reduction(self, div_func, *args):
        D_sum = np.zeros(self.d)

        for d in range(self.d):
            D = div_func(self.p[d], self.q[d], *args)
            D_sum[d] = np.sum(D) / len(D)

        if self.mean_reduction:
            return np.sqrt(np.mean(D_sum) / 2) if self.distance_metric else np.mean(D_sum)
        else:
            return np.sqrt(D_sum / 2) if self.distance_metric else D_sum

    def kl_divergence(self):
        div_func = lambda p, q: p * np.log(p / q)
        return self._divergence_reduction(div_func)

    def hellinger_distance(self):
        div_func = lambda p, q: (np.sqrt(p) - np.sqrt(q)) ** 2
        return self._divergence_reduction(div_func)

    def chi2_distance(self):
        div_func = lambda p, q: ((p - q) ** 2) / (p + q)
        return self._divergence_reduction(div_func)

    def alpha_divergence(self, alpha):
        # assert alpha != 0 and alpha != 1 and alpha != 0.5  # same as KL for 0 and 1, same as Hellinger for 1/2

        div_func = lambda p, q, alpha: alpha * p + (1 - alpha) * q - p**alpha * q ** (1 - alpha)
        D = self._divergence_reduction(div_func, alpha)

        return D / (alpha * (1 - alpha))
