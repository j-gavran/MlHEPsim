import logging
import multiprocessing

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from ml.common.utils.plot_utils import filter_array


class MMD:
    def __init__(self, X, Y):
        """Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.

        References
        ----------
        [1] - https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
        Y : np.ndarray or torch.Tensor

        """
        self.X, self.Y = X, Y

    def mmd_linear(self):
        """MMD using linear kernel (i.e., k(x,y) = <x,y>)

        Note that this is not the original linear MMD, only the reformulated and faster version.

        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]

        """
        delta = self.X.mean(0) - self.Y.mean(0)
        return delta.dot(delta.T)

    def mmd_rbf(self, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]

        """
        XX = metrics.pairwise.rbf_kernel(self.X, self.Y, gamma)
        YY = metrics.pairwise.rbf_kernel(self.Y, self.Y, gamma)
        XY = metrics.pairwise.rbf_kernel(self.X, self.Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def mmd_poly(self, degree=2, gamma=None, coef0=1):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]

        """
        XX = metrics.pairwise.polynomial_kernel(self.X, self.X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(self.Y, self.Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(self.X, self.Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def mmd_rbf_torch(self, sigma):
        """
        https://torchdrift.org/notebooks/note_on_mmd.html

        compare kernel MMD paper and code:
        A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012): http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

        x shape [n, d] y shape [m, d]

        """
        n, d = self.X.shape
        m, d2 = self.Y.shape
        assert d == d2
        xy = torch.cat([self.X.detach(), self.Y.detach()], dim=0)
        dists = torch.cdist(xy, xy, p=2.0)
        # we are a bit sloppy here as we just keep the diagonal and everything twice
        # note that sigma should be squared in the RBF to match the Gretton et al heuristic
        k = torch.exp((-1 / (2 * sigma**2)) * dists**2) + torch.eye(n + m) * 1e-5
        k_x = k[:n, :n]
        k_y = k[n:, n:]
        k_xy = k[:n, n:]
        # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
        # note that their code uses the biased (and differently scaled mmd)
        mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
        return mmd


def histogram_density(A, B, n_bins="auto", bin_range=None, density=None, non_zero_bins=True):
    n_features = A.shape[1]

    combined_sample = np.concatenate([A, B])

    histograms_A, histograms_B, n_bin_counts = [], [], []
    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(filter_array(combined_sample[:, feature]), bins=n_bins, range=bin_range)

        h_A, _ = np.histogram(filter_array(A[:, feature]), bins=bin_edges, density=density)
        h_B, _ = np.histogram(filter_array(B[:, feature]), bins=bin_edges, density=density)

        if non_zero_bins:
            mask = (h_A != 0.0) & (h_B != 0.0)
            h_A, h_B = h_A[mask], h_B[mask]

        histograms_A.append(h_A)
        histograms_B.append(h_B)

        if n_bins == "auto":
            n_bin_counts.append(len(h_A))

    if n_bins == "auto":
        logging.info(f"n_bins is set to 'auto', using number of bins: {n_bin_counts}")

    return histograms_A, histograms_B


def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def kde_density(A, B, kernel="gaussian", bandwidth=2.0, atol=0.005, rtol=0.05):
    """https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity"""
    n_features = A.shape[1]

    kdes_A, kdes_B = [], []
    for feature in tqdm(range(n_features), leave=False, desc="KDE"):
        a, b = filter_array(A[:, feature])[:, None], filter_array(B[:, feature])[:, None]

        kde_A = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol).fit(a)
        kde_B = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol).fit(b)

        # score_A, score_B = kde_A.score_samples(a), kde_B.score_samples(b)
        score_A, score_B = parrallel_score_samples(kde_A, a), parrallel_score_samples(kde_B, b)

        kdes_A.append(np.exp(score_A))
        kdes_B.append(np.exp(score_B))

    return kdes_A, kdes_B


class Distances:
    def __init__(
        self,
        P,
        Q,
        reduce=True,
        density_estimation="hist",
        mean_reduction=True,
        distance_metric=False,
        density_kwargs=None,
    ):
        """Divergence measures between probability distributions.

        Parameters
        ----------
        P : torch.tensor or np.ndarray
            Samples from P distribution.
        Q : torch.tensor or np.ndarray
            Samples from Q distribution.
        reduce : bool
            Reduce divergence over features.
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

        self.reduce = reduce
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
        """Check if input is tensor or numpy array and sort if needed. Clip to min shape if different."""

        if P.shape[0] != Q.shape[0]:
            min_shape = min([P.shape[0], Q.shape[0]])
            logging.warning(f"Using {min_shape} samples from P{P.shape} and Q{Q.shape}.")
            P, Q = P[:min_shape, :], Q[:min_shape, :]

        if torch.is_tensor(P) or torch.is_tensor(Q):
            P, Q = P.cpu().numpy(), Q.cpu().numpy()

        if sort:
            P, Q = np.sort(P, axis=0), np.sort(Q, axis=0)

        return P, Q

    def _estimate_density(self):
        """Density estimator method.

        Returns
        -------
        Tuple (p, q) of list of arrays containing density values for each feature.

        """
        if self.density_estimation.lower() == "hist":
            p, q = histogram_density(self.P, self.Q, density=True, **self.density_kwargs)
        elif self.density_estimation.lower() == "kde":
            p, q = kde_density(self.P, self.Q, **self.density_kwargs)
        else:
            raise ValueError

        return p, q

    def _divergence_reduction(self, div_func, *args):
        D_sum = np.zeros(self.d)

        for d in range(self.d):
            D = div_func(self.p[d], self.q[d], *args)

            D_sum[d] = np.mean(D) if type(D) is np.ndarray else D

        if self.mean_reduction:
            return np.sqrt(np.mean(D_sum) / 2) if self.distance_metric else np.mean(D_sum)
        else:
            return np.sqrt(D_sum / 2) if self.distance_metric else D_sum

    def kl_divergence(self):
        div_func = lambda p, q: p * np.log(p / q)

        if self.reduce:
            return self._divergence_reduction(div_func)
        else:
            return div_func(self.p, self.q)

    def hellinger_distance(self):
        div_func = lambda p, q: (np.sqrt(p) - np.sqrt(q)) ** 2

        if self.reduce:
            return self._divergence_reduction(div_func)
        else:
            return div_func(self.p, self.q)

    def chi2_distance(self):
        # https://stats.stackexchange.com/questions/184101/comparing-two-histograms-using-chi-square-distance
        div_func = lambda p, q: ((p - q) ** 2) / q

        if self.reduce:
            return self._divergence_reduction(div_func)
        else:
            return div_func(self.p, self.q)

    def alpha_divergence(self, alpha):
        # assert alpha != 0 and alpha != 1 and alpha != 0.5  # same as KL for 0 and 1, same as Hellinger for 1/2

        div_func = lambda p, q, alpha: alpha * p + (1 - alpha) * q - p**alpha * q ** (1 - alpha)

        if self.reduce:
            D = self._divergence_reduction(div_func, alpha)
            return D / (alpha * (1 - alpha))
        else:
            return div_func(self.p, self.q, alpha) / (alpha * (1 - alpha))

    def wasserstein_distance(self):
        div_func = lambda p, q: wasserstein_distance(p, q)

        if self.reduce:
            return self._divergence_reduction(div_func)
        else:
            return div_func(self.p, self.q)
